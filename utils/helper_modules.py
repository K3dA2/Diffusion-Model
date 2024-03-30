import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest

'''
This ResNet class is intended to be used as the smallest unit of the block class
'''
class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32, useMaxPool = True, upscale = False):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.useMaxPool = useMaxPool
        self.upscale = upscale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels // 2)
        #self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # Apply LayerNorm after conv1
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = F.layer_norm(out, out.size()[1:])
        
        # Apply LayerNorm after conv2
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        out = F.layer_norm(out, out.size()[1:])
        
        out1 = self.skip_conv(x)
        out = F.relu(self.conv3(out))
        
        if self.useMaxPool:
            out = F.max_pool2d(out + out1, 2)
        elif self.upscale:
            out = F.upsample(out + out1, scale_factor=2)
        else:
            out = F.relu(out + out1)
        
        return out

'''
Time Embedding class uses sinusodial embedding to input time information into the NN
Expected output of this module is Bx1x2 where B is the Batch size
Note to self: Should use cos as the scaling factor and sin as shifiting factor
'''
class TimeEmb(nn.Module):
    def __init__(self,h_dim=128,sf=32,maxPositions = 100_000) -> None:
        super().__init__()
        self.sf = sf
        self.h_dim = h_dim
        self.maxPos = maxPositions
    
    def forward(self,x):
        #out = self.sig_emb(x)
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                self.sf//2
            )
        ).to(x.device)
        angular_speeds = 2.0 * math.pi * frequencies
        emb = torch.concat([torch.sin(angular_speeds*x),torch.cos(angular_speeds*x)], axis = -1)
        return emb

'''
Implementation of the Attention Layer.
It uses Convolutions as the linear layer in typical attention
Expected Output is the same shape as the input
'''
class Attention(nn.Module):
    def __init__(self, num_heads = 2,in_dim=16) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(in_dim, num_heads, batch_first= True)
        self.to_qkv = nn.Conv2d(in_dim,in_dim*3, kernel_size=3, padding=1)
        self.last_layer = nn.Conv2d(in_dim,in_dim,kernel_size=3,padding=1)
        self.ln = nn.LayerNorm(in_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm([in_dim]),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self,x):
        reshaped_x = x.view(x.size(0), -1, x.size(1))  # Reshape to [batch_size, channels, -1]
        reshaped_x = self.ln(reshaped_x)
        out,_ = self.mha(reshaped_x,reshaped_x,reshaped_x)
        out = self.ff(out)
        out = out.view(x.size(0),x.size(1),x.size(2),x.size(3))
        return F.layer_norm(out+x, out.size()[1:])
        #return out
        
        
'''
Unit testing class
'''
class TestResNet(unittest.TestCase):
    def test_forward(self):
        
        model = Attention()
        input_tensor = torch.randn(1, 16, 64, 64)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape,(1,16,64,64))
        '''
        
        model = ResNet(in_channels=16,out_channels = 16,useMaxPool=False)
        input_tensor = torch.randn(1, 16, 64, 64)  # Example input with shape (batch_size, channels, height, width)
        output = model.forward(input_tensor)
        self.assertEqual(output.shape, (1, 16, 64, 64))  # Adjust the expected shape based on your model architecture
        
        
        model = TimeEmb()
        #input_tensor = torch.tensor([1.0])
        input_tensor = torch.arange(0,200)
        input_tensor = torch.unsqueeze(input_tensor,1)
        print(input_tensor.shape)
        output = model.forward(input_tensor.float())
        print(output.shape)
        self.assertEqual(output.shape,(200,32))
        print(output)
        '''
        
if __name__ == '__main__':
    unittest.main()