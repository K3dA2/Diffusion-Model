import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils.block import DownBlock,UpBlock,MidBlock
from utils.helper_modules import TimeEmb

class Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downblk = DownBlock(3,64,True)
        self.downblk1 = DownBlock(64,128,True)
        self.downblk2 = DownBlock(128,256,True)
        self.midblk = MidBlock(256,256)
        self.midblk1 = MidBlock(256,256)
        self.upblk = UpBlock(256,128,upscale=True)
        self.upblk1 = UpBlock(128,64,upscale=True)
        self.upblk2 = UpBlock(64,32,upscale=True)
        self.res = nn.Conv2d(32,3,padding=1,kernel_size=3)
        self.res.weight.data.fill_(0)

        self.timeEmb = TimeEmb()
        
    def forward(self,img,t):
        t = self.timeEmb(t)
        
        out,skip = self.downblk(img,t)
        out,skip1 = self.downblk1(out,t)
        out,skip2 = self.downblk2(out,t)

        out = self.midblk(out,t)
        out = self.midblk1(out,t)

        out = self.upblk(out,skip2,t)
        out = self.upblk1(out,skip1,t)
        out = self.upblk2(out,skip,t)

        out = self.res(out)
        return out
    
class TestUnet(unittest.TestCase):
    def test_forward(self):
        model = Unet()
        #input_tensor = torch.randn(1, 16, 64, 64)
        input_tensor = torch.ones([3,3,128,128], dtype=torch.float32)
        tm = torch.tensor([[1],[2],[3]],dtype=torch.float32)
        #tm = torch.unsqueeze(tm, dim=1)
        output = model.forward(input_tensor,tm)
        self.assertEqual(output.shape,(3,3,128,128))
        
if __name__ == '__main__':
    unittest.main()