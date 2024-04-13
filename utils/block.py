import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils.helper_modules import Attention,ResNet,TimeEmb
#from helper_modules import Attention,ResNet,TimeEmb

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
    def __init__(self,in_channels,out_channels,useMaxpool = False,upscale = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useMaxpool = useMaxpool
        self.upscale = upscale
        self.inResnet = ResNet(in_channels,out_channels,useMaxpool,upscale)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.outResnet = ResNet(out_channels,out_channels)
        #self.att = Attention(in_dim=out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        
    def forward(self,img,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        out = self.inResnet(img)
        out = torch.add(out, proj_exp)
        out = self.batchnorm(out)
        out = self.outResnet(out)
        #out = self.att.forward(out)
        return out,out

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,useMaxpool = False,upscale = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useMaxpool = useMaxpool
        self.upscale = upscale
        self.inResnet = ResNet(in_channels*2,out_channels,useMaxpool,upscale)
        self.batchnorm = nn.BatchNorm2d(in_channels*2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        #self.att = Attention(in_dim=out_channels)
        self.outResnet = ResNet(out_channels,out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        
    def forward(self,img,skip,t):
        proj = self.proj(t)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        
        out = torch.cat((img,skip),dim=1)
        out = self.batchnorm(out)
        out = self.inResnet.forward(out)
        
        out = torch.add(out, proj_exp)
        out = self.batchnorm1(out)
        #out = self.att.forward(out)
        out = self.outResnet(out)
        return out

class MidBlock(nn.Module):
    def __init__(self,in_channels,out_channels,useMaxpool = False,upscale = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.useMaxpool = useMaxpool
        self.upscale = upscale
        self.inResnet = ResNet(in_channels,out_channels,useMaxpool,upscale)
        #self.att = Attention(in_dim=out_channels)
        self.batchnorm = nn.BatchNorm2d(out_channels)
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
        
        out = torch.add(out, proj_exp)
        out = self.batchnorm(out)
        #out = self.att.forward(out)
        out = self.outResnet(out)
        return out
        

class TestBlock(unittest.TestCase):
    def test_forward(self):
        model = UpBlock(32,16,useMaxpool=True)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([1,32,64,64], dtype=torch.float32)
        input_tensor1 = torch.ones([1,32,64,64], dtype=torch.float32)
        tm = torch.ones([1,32], dtype=torch.float32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,input_tensor1,tm)
        self.assertEqual(output.shape,(1,16,32,32))
        
if __name__ == '__main__':
    unittest.main()
        