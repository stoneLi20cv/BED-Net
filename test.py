#!/usr/bin/env python
# coding: utf-8

# ![](https://ai-studio-static-online.cdn.bcebos.com/b93b450966a44e578e0e5090ac93fb5ab5d3e515df204734a2e5c1bdd959c07b)
# 
# In this paper, we propose a hybrid network structure, termed Conformer, to take advantage of convolutional operations and self attention mechanisms for enhanced representation learning. 
# 
# paper：[https://arxiv.org/abs/2105.03889](http://)
# 
# code：[https://github.com/pengzhiliang/Conformer](http://)

# ## 前言
# hi guy，我们又见面了，这次又来复现论文，这次主题是 CNN + ViT 的一篇作品，结构图如下
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/bdd8cb9760dc4b279f3a9469a397959ea6762e6961544710824f9b766417e1c6)
# 
# 这里唠叨一下为什么近期很多 ViT 工作都喜欢添加 CNN，主要是因为CNN具有良好的归纳偏置，它在小数据上拟合能力很强，可以弥补 ViT 易过拟合以及需要大量数据的短板，性能图如下
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/1317f79ab241423180d57bc5cbc97d47a1b67ecb58bb469f99cb0f5ef6c9d337)
# 
# 本文的思想就是作两个分支，分别是 CNN 分支和 Transformer 分支，并行同时相互补充
# 
# * 本文提出了 dual 结构，将 CNN 的局部信息和 Transformer 的全局建模融合，加强表征能力
# * 提出 FCU，用来融合 CNN 和 Transformer 特征

# ## 完整代码
# 
# 

# ### 导入所需要的包

# In[1]:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial

trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)
zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)
kaiming_normal_ = nn.initializer.KaimingNormal()


# ### 基础函数定义

# In[2]:


def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]

    return x.transpose(a)

def drop_path(x, drop_prob = 0., training = False):

    if drop_prob == 0. or not training:
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


class Identity(nn.Layer):                      

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
 
    def forward(self, input):
        return input


# ### 模型组网
# 
# 具体网络结构如下图所示
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/84cf465f7a444de8a3bde9a1f7ae676df706d86dd60940e19e479fbdedccbb87)
# 

# In[3]:


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ swapdim(k, -2, -1)) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = swapdim((attn @ v), 1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, epsilon=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Layer):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2D, epsilon=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2D(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2D(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias_attr=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2D(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.Conv2D(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias_attr=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Layer):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2D(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2D(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2)
        x = swapdim(x, 1, 2)
        x = self.ln(x)
        x = self.act(x)
        
        x_t = paddle.unsqueeze(x_t[:, 0], axis=1) 
        x = paddle.concat([x_t, x], axis=1)

        return x


class FCUUp(nn.Layer):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2D, epsilon=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2D(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = swapdim(x[:, 1:], 1, 2)
        x_r = x_r.reshape([B, C, H, W])
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=[H * self.up_stride, W * self.up_stride])


class Med_ConvBlock(nn.Layer):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2D, epsilon=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2D(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2D(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias_attr=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2D(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias_attr=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer()

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Layer):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.LayerList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Layer):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=trunc_normal_)
        self.add_parameter("cls_token", self.cls_token)

        self.trans_dpr = [x for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else Identity()
        self.pooling = nn.AdaptiveAvgPool2D(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2D(in_chans, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2D(64)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2D(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_sublayer('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_sublayer('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_sublayer('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion
                    )
            )
        self.fin_stage = fin_stage


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2D):
            ones_(m.weight)
            zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            ones_(m.weight)
            zeros_(m.bias)



    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand([B, -1, -1])

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2)
        x_t = swapdim(x_t, 1, 2)
        x_t = paddle.concat([cls_tokens, x_t], axis=1)
        x_t = self.trans_1(x_t)
        
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

        # conv classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        return conv_cls + tran_cls


# ### 模型生成

# In[4]:


def Conformer_tiny_patch16(**kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def Conformer_small_patch16(**kwargs):
    model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


def Conformer_base_patch16(**kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model


# ### 模型结构可视化

# In[5]:


paddle.Model(Conformer_base_patch16()).summary((1,3,224,224))


# ### 添加预训练权重
# 
# **ImageNet-1k validation**
# 
# | Model           | Acc@1   | Acc@5   | # Param |
# | --------------- | ------- | ------- | ------- |
# | Conformer tiny  | 81.31 % | 95.61 % | 23.53 M |
# | Conformer small | 83.14 % | 96.48 % | 37.72 M |
# | Conformer base  | 83.73 % | 96.60 % | 83.37 M |
# 
# 

# In[6]:


# conformer tiny
Conformer_tiny = Conformer_tiny_patch16()
Conformer_tiny.set_state_dict(paddle.load('/home/aistudio/data/data96103/Conformer_tiny_patch16.pdparams'))

Conformer_small = Conformer_small_patch16()
Conformer_small.set_state_dict(paddle.load('/home/aistudio/data/data96103/Conformer_small_patch16.pdparams'))

Conformer_base = Conformer_base_patch16()
Conformer_base.set_state_dict(paddle.load('/home/aistudio/data/data96103/Conformer_base_patch16.pdparams'))


# ## Cifar10 验证性能
# 
# 采用Cifar10数据集，无过多的数据增强

# ### 数据准备

# In[7]:


import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10

paddle.set_device('gpu')

#数据准备
transform = T.Compose([
    T.Resize(size=(224,224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],data_format='HWC'),
    T.ToTensor()
])

train_dataset = Cifar10(mode='train', transform=transform)
val_dataset = Cifar10(mode='test',  transform=transform)


# ### 模型准备

# In[ ]:


Conformer_tiny = Conformer_tiny_patch16(num_classes=10)
Conformer_tiny.set_state_dict(paddle.load('/home/aistudio/data/data96103/Conformer_tiny_patch16.pdparams'))
model = paddle.Model(Conformer_tiny)


# ### 开始训练
# 
# 由于时间篇幅只训练5轮，感兴趣的同学可以继续训练

# In[ ]:


model.prepare(optimizer=paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

visualdl=paddle.callbacks.VisualDL(log_dir='visual_log') # 开启训练可视化

model.fit(
    train_data=train_dataset, 
    eval_data=val_dataset, 
    batch_size=64, 
    epochs=5,
    verbose=1,
    callbacks=[visualdl] 
)


# ### 训练可视化
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/eb9bfd3d46cd42c3b55a286cf7dbc68cd99c2deab4e1468180b089dbe6b50663)
# 

# ## 总结
# 
# * 在可比较参数复杂度下，Conformer超越了CNN和vit。有很大的潜力做backbone
# * self-attention可以解决长距离依赖，但是失去局部信息
# * 在下游任务，比如目标检测分割下，Conformer 展现了良好的性能
# 
#   ![](https://ai-studio-static-online.cdn.bcebos.com/1cb9522333f1467fbe60cb6e98d926bfd2e0ccc7afcd49a79c8f92a81ac48e2b)
