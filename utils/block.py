import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils.helper_modules import Attention,ResNet,TimeEmb,SelfAttention
#from helper_modules import Attention,ResNet,TimeEmb,SelfAttention

'''

'''

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,useMaxpool = False,upscale = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useMaxpool = useMaxpool
        self.upscale = upscale
        self.resnet = ResNet(in_channels,out_channels,useMaxpool,upscale)
        #self.att = Attention(in_dim=out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        
    def forward(self,img,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        if self.useMaxpool == False:
            out = self.resnet.forward(img)
            out = torch.add(out, proj_exp)
            #out = self.att.forward(out)
            return out
        else:
            out = self.resnet.forward(img)
            out = torch.add(out, proj_exp)
            #out = self.att.forward(out)
            return out
       
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool = nn.MaxPool2d(2)
        self.inResnet = ResNet(in_channels,out_channels)
        self.outResnet = ResNet(out_channels,out_channels)
        self.att = SelfAttention(out_channels,size)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        
    def forward(self,img,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        out = self.inResnet(img)
        out = self.outResnet(out)
        out = torch.add(out, proj_exp)
        out = self.att(out)
        skip = out
        out = self.maxpool(out)
        return out,skip

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,size) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inResnet = ResNet(in_channels*2,out_channels)
        self.batchnorm = nn.BatchNorm2d(in_channels*2)
        self.att = SelfAttention(out_channels,size)
        self.outResnet = ResNet(out_channels,out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self,img,skip,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        out = self.up(img)
        out = torch.cat((out,skip),dim=1)
        out = self.batchnorm(out)
        out = self.inResnet.forward(out)
        out = self.outResnet(out)
        out = torch.add(out, proj_exp)
        out = self.att(out)
        return out

class MidBlock(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inResnet = ResNet(in_channels,out_channels)
        #self.att = Attention(in_dim=out_channels)
        self.outResnet = ResNet(out_channels,out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        
    def forward(self,img,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        
        out = self.inResnet.forward(img)
        #out = self.att.forward(out)
        out = self.outResnet(out)
        out = torch.add(out, proj_exp)
        return out
        

class TestBlock(unittest.TestCase):
    def test_forward(self):
        model = DownBlock(32,16,16)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([1,32,16,16], dtype=torch.float32)
        input_tensor1 = torch.ones([1,32,16,16], dtype=torch.float32)
        tm = torch.ones([1,32], dtype=torch.float32)
        #tm = torch.unsqueeze(tm, dim=1)
        output,_ = model.forward(input_tensor,tm)
        self.assertEqual(output.shape,(1,16,128,128))
        
if __name__ == '__main__':
    unittest.main()
        