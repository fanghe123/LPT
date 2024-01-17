import paddle
import paddle.nn as nn
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
import os

# 定义一些初始化
trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)

# 啥都不做的 对于torch.nn.Identity
class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# 下面是DropPath, 一个正则化方法
def drop_path(x, drop_prob=0.0, training=False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class PatchEmbed(nn.Layer):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
class LayerNormChannel(nn.Layer):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, epsilon=1e-05):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=ones_)
        self.bias = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=zeros_)
        self.epsilon = epsilon

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
class Pooling(nn.Layer):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.AvgPool2D(
            kernel_size, stride=1, padding=kernel_size//2, exclusive=True)

    def forward(self, x):
        return self.pool(x) - x
class Mlp(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)     # (B, C, H, W) --> (B, C, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)     # (B, C, H, W) --> (B, C, H, W)
        x = self.drop(x)
        return x
class PoolingBlock(nn.Layer):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(kernel_size=pool_size)   # vits是msa，MLPs是mlp，这个用pool来替代
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:

            self.layer_scale_1 = paddle.create_parameter(
                shape=[dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

            self.layer_scale_2 = paddle.create_parameter(
                shape=[dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
def basic_blocks(dim, index, layers,
                 pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolingBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks
@register
@serializable
class Poolingbackbone(nn.Layer):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalizaiotn and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 pool_size=3,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=True,
                 return_idx=[0,1,2,3],
                 **kwargs):

        super().__init__()
        self.return_idx = return_idx
        self._out_channels = [64,128,320,512]
        self._out_strides = [4, 8, 16, 32]
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        ### 定义 patch embed 要调用很多次
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)

            if i >= len(layers) - 1:  # 层数够了就不搭建了，实际上就4层
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:  # 这就是红圈部分的解释，通过多次调用patchembed来降低特征，和convnet一样
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )
        self.network = nn.LayerList(network)
        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else Identity()

        self.apply(self.cls_init_weights)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_sublayer(layer_name, layer)


    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x
    def forward_tokens(self, x):
        outs = []
        self.input_proj = nn.LayerList()

        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                if idx != 0:
                    norm_layer = getattr(self, f'norm{idx}')
                    x_out = norm_layer(x)
                    outs.append(x_out)
            # output the features of four stages for dense prediction

        return outs

    def forward(self,inputs):
        x = inputs['image']
        # input embedding
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]