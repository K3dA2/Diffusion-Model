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
        self.timeEmb = TimeEmb()
        self.att = Attention(in_dim=out_channels)
        self.proj = nn.Sequential(
            nn.Linear(32,in_channels),
            nn.SiLU(),
            nn.Linear(in_channels,out_channels),
        )
        
    def forward(self,img,t):
        out = self.resnet.forward(img)
        emb = self.timeEmb.forward(t)
        proj = self.proj(emb)
        proj_exp = proj.view(proj.size(0),proj.size(1),1,1)
        #out += proj_exp
        out = torch.add(out, proj_exp)
        #scale = emb[:,1]
        #shift = emb[:,0]
        #out = torch.mul(out,scale[:,None,None,None]) + shift[:,None,None,None]
        out = self.att.forward(out)
        return out
    
    
class TestBlock(unittest.TestCase):
    def test_forward(self):
        model = Block(3,16,useMaxpool=True)
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([1,3,64,64], dtype=torch.float32)
        tm = torch.tensor([[1]],dtype=torch.float32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,tm)
        self.assertEqual(output.shape,(1,16,32,32))
        
if __name__ == '__main__':
    unittest.main()
        