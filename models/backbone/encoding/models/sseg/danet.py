###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize
from ...nn import PAM_Module
from ...nn import CAM_Module
from .base import BaseNet


__all__ = ['DANet', 'get_danet']

class DANet(BaseNet):# 整个Danet模型网络，继承BaseNet
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)# c4:模型主体部分的的基本骨架，不包括attention,的前向传播输出结果

        x = self.head(c4) # x有三个元素，第一个是是c4经过整个attention模块前向传播输出的结果，以及两个attention模块各自前向传播输出的结果
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)# 上采样最后的输出x到输入图像的尺寸
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)# outputs就是最终语义分割结果
        
class DANetHead(nn.Module):# 整个attention模块
    def __init__(self, in_channels, out_channels, norm_layer):#输入通道数：resnet101的输出通道数2048，输出通道数：要区分的类别数
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        # conv k3，inc 2048 outc 512 -> bn ->relu
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        # conv k3，inc 2048 outc 512 -> bn ->relu
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        # 位置注意力模块
        self.sa = PAM_Module(inter_channels)
        # 通道注意力模块
        self.sc = CAM_Module(inter_channels)

        # conv k3，inc 512 outc 512 -> bn ->relu
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        # conv k3，inc 512 outc 512 -> bn ->relu
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        # dp -> conv k1，inc 512 outc 类别数
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))#nn.Dropout2d随机将某个通道全部置为0
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)#输出模块融合后的结果，以及两个模块各自的结果


def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
           root='~/.encoding/models', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DANet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

