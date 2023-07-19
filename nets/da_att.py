###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        # 先经过3个卷积层生成3个新特征图BCD（尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))#a尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = Softmax(dim=-1) # 对每一行进行softmax
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # B：conv k1 通道变1/8 -> reshape操作-》B X C X (H X W)转置B X (H X W)X C
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # C：conv k1 通道变1/8 -> reshape操作
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # B转置乘C得S
        energy = torch.bmm(proj_query, proj_key)
        # 对S每一行softmax
        attention = self.softmax(energy)
        # D：reshape操作-》B X C X (H X W)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # D乘S转置得注意力之后的结果out
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out reshape操作-》B X C X H X W
        out = out.view(m_batchsize, C, height, width)
        # a尺度系数*out+A
        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        # b尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)# 对每一行进行softmax
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        # A1
        proj_query = x.view(m_batchsize, C, -1)
        # A2
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        # 这里实现了softmax用最后一维(C X C里的行)的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的c选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        # 这里X左乘A，所以X不用转置了
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

