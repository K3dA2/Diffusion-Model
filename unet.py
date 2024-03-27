import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from utils.block import Block

class Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blk1 = Block(3,64)
        self.dblk1 = Block(64,128,useMaxpool=True)
        self.blk2 = Block(128,256)
        self.dblk2 = Block(256,512,useMaxpool=True)
        self.blk3 = Block(512,256)
        self.blk4 = Block(256,128,upscale=True)
        self.blk5 = Block(384,64,upscale=True)
        self.res = nn.Conv2d(128,3,padding=1,kernel_size=3)
        self.res.weight.data.fill_(0)
        
    def forward(self,img,t):
        out1 = self.blk1.forward(img,t)
        out = self.dblk1.forward(out1,t)
        out2 = self.blk2.forward(out,t)
        out = self.dblk2.forward(out2,t)
        out = self.blk3.forward(out,t)
        out = self.blk4.forward(out,t)
        out = torch.cat((out,out2),dim = 1)
        out = self.blk5.forward(out,t)
        out = torch.cat((out,out1),dim=1)
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