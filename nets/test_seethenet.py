from nets import drn
from nets.CCnet import CCnetSeg_Model
from nets.danetwithtorchresnet import DANetHead, DANetwp
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models

from nets.fcn import VGGNet
from new_DANet.danet import DANet

device=torch.device('cuda')
# model = DANetHead(128, 2, nn.BatchNorm2d)
# model=DANetwp(nclass=2, backbone='vgg16',baseisperTrain=False)
model = DANet(nclass=2,baseispertrained=False)
# model=drn.drn_c_26(pretrained=False)
# model = CCnetSeg_Model(2)
# model=models.resnet101(pretrained=False)
# model=nn.Sequential(*list(model.children())[:-2])
# model = VGGNet(pretrained=False,requires_grad=True, remove_fc=True)
print(model)
model=model.to(device)
summary(model,(3,512,512))
# x=torch.rand(size=(1,2048,512,512))
#
# for layer in model:
#     x=layer(x)
#     print(layer.__class__.__name__,'   os:\t',x.shape)