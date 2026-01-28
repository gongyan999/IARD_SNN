from math import nan
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
# from .library.layers import *
from typing import Dict
import sys
import scipy.io as sio
import numpy as np
import math
from thop import profile

class GroupUnit(nn.Module):
    def __init__(self, in_channels, groupConv_out_channels, conv1x1_out_channels, 
                    kernel_size, stride, padding, group, casual_dim=2, right_context=0, clamp=False, quant=False,
                    weight_clamp_val=2, bias_clamp_val=32, input_clamp_val=8, output_clamp_val=32,
                    weight_quant_bit=8, bias_quant_bit=16, input_quant_bit=8, output_quant_bit=16):
        super(GroupUnit, self).__init__()
        self.in_channels = in_channels
        self.groupConv_out_channels = groupConv_out_channels
        self.conv1x1_out_channels = conv1x1_out_channels
        self.kernel_size = kernel_size
        self.group = group
        self.casual_dim = casual_dim
        self.right_context = right_context
        self.stride = stride
        self.padding = padding
        
        self.group_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.groupConv_out_channels, 
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=self.group)
        self.active_func = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv2d(in_channels=self.groupConv_out_channels, out_channels=self.conv1x1_out_channels, 
                                    kernel_size=(1, 1), bias=False, groups=1)
        
        self.initial_parameters()
        
    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
                
    def forward(self, src, mask=None):
        out = self.group_conv(src)
        if torch.is_tensor(mask):
            mask = mask[:, :, ::mask.size(2)//out.size(2), :]
            out = out * mask
        out = self.active_func(out)
        # out = torch.clamp(out,-8,7.9375)
        out = self.conv_1x1(out)
        # out = torch.clamp(out,-8,7.9375)
        return out

class ConvUint(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, group, casual_dim=2, right_context=0, is_active=True, is_TransposeConv=False, 
                    is_clamp=False, clamp=False, quant=False,
                    weight_clamp_val=2, bias_clamp_val=32, input_clamp_val=8, output_clamp_val=32,
                    weight_quant_bit=8, bias_quant_bit=16, input_quant_bit=8, output_quant_bit=16):   
        super(ConvUint, self).__init__()
        self.is_clamp = is_clamp
        self.is_active = is_active
        self.is_TransposeConv = is_TransposeConv
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.kernel_size  = kernel_size
        self.group        = group

        if is_TransposeConv:
            self.conv = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, stride=self.kernel_size)
        else:
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=self.group)
        if self.is_active:
            self.active_func = nn.ReLU(inplace=True)
        
        self.initial_parameters()
        
    def initial_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0.1) 
            if isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0.1) 
     
    def forward(self, src, mask=None):
        out = self.conv(src)
        if torch.is_tensor(mask):
            mask = mask[:, :, ::mask.size(2)//out.size(2), :]
            out = out * mask
        if self.is_active:
            out = self.active_func(out)
        # if self.is_clamp:
        #     out = torch.clamp(out,-8,7.9375)
        return out
    
class CR(nn.Module) : 
    def __init__(self, n_band, overlap=1/3, **kwargs):
        super(CR, self).__init__()
        self.n_band = n_band
        self.overlap = overlap
        """
        if type_window == "None" :
            self.window = torch.tensor(1.0)
        elif type_window == "Rectengular" : 
            self.window = torch.kaiser_window(window_length ,beta = 0.0)
        elif type_window == "Hanning":
            self.window = torch.hann_window(window_length)
        else :
            raise NotImplementedError
        """

    def forward(self,x):
        idx = 0

        B,C,T,F = x.shape  ##  2,1,63,257
        n_freq = x.shape[3]
        sz_band = n_freq/(self.n_band*(1-self.overlap))
        sz_band = int(np.ceil(sz_band))
        y = torch.zeros(B,self.n_band*C,T,sz_band).to(x.device)
        for i in range(self.n_band):
            if idx+sz_band > F :
                sz_band = F - idx
            y[:,i*C:(i+1)*C,:,:sz_band] = x[:,:,:,idx:idx+sz_band]
            n_idx = idx + int(sz_band*(1-self.overlap))
            idx = n_idx
        return y

class CNN(nn.Module):
    def __init__(self, clamp=False, quant=False, frames_classes=9):
        super(CNN, self).__init__()
        CLAMP = clamp
        QUANT = quant
        
        # self.CR = CR(8,overlap=1/5)
        self.conv0 = ConvUint(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, group=1,
                              right_context=1, clamp=CLAMP, quant=QUANT)
        self.conv1 = ConvUint(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, group=1,
                              right_context=1, clamp=CLAMP, quant=QUANT)

        self.conv2 = ConvUint(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, group=1,
                              right_context=1, clamp=CLAMP, quant=QUANT)
        self.conv3 = ConvUint(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, group=1,
                              right_context=0, clamp=CLAMP, quant=QUANT)

        self.block1_0 = GroupUnit(in_channels=16, groupConv_out_channels=64, conv1x1_out_channels=32,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=1, clamp=CLAMP,
                                  quant=QUANT)
        self.block1_1 = GroupUnit(in_channels=32, groupConv_out_channels=64, conv1x1_out_channels=32,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        self.block1_2 = GroupUnit(in_channels=32, groupConv_out_channels=64, conv1x1_out_channels=32,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        self.block1_3 = GroupUnit(in_channels=32, groupConv_out_channels=64, conv1x1_out_channels=32,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)

        self.block2_0 = GroupUnit(in_channels=32, groupConv_out_channels=128, conv1x1_out_channels=64,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        self.block2_1 = GroupUnit(in_channels=64, groupConv_out_channels=128, conv1x1_out_channels=64,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        # self.block2_2 = GroupUnit(in_channels = 64, groupConv_out_channels = 128, conv1x1_out_channels=64, kernel_size=(3,3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP, quant=QUANT)
        # self.block2_3 = GroupUnit(in_channels = 64, groupConv_out_channels = 128, conv1x1_out_channels=64, kernel_size=(3,3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP, quant=QUANT)

        self.block3_0 = GroupUnit(in_channels=64, groupConv_out_channels=256, conv1x1_out_channels=128,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        self.block3_1 = GroupUnit(in_channels=128, groupConv_out_channels=256, conv1x1_out_channels=128,
                                  kernel_size=(3, 3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        # self.block3_2 = GroupUnit(in_channels = 128, groupConv_out_channels = 256, conv1x1_out_channels=128, kernel_size=(3,3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP, quant=QUANT)
        # self.block3_3 = GroupUnit(in_channels = 128, groupConv_out_channels = 256, conv1x1_out_channels=128, kernel_size=(3,3), stride=1, padding=1, group=4, right_context=0, clamp=CLAMP, quant=QUANT)

        self.conv2dto1d_1 = ConvUint(in_channels=896, out_channels=256, kernel_size=(1, 1), stride=1, padding=0, group=1,
                                   right_context=0, is_active=False, clamp=CLAMP, quant=QUANT)
        
        self.block4_0 = GroupUnit(in_channels=256, groupConv_out_channels=384, conv1x1_out_channels=256,
                                  kernel_size=(3, 1), stride=1, padding=(1, 0), group=4, right_context=1, clamp=CLAMP,
                                  quant=QUANT)
        self.block4_1 = GroupUnit(in_channels=256, groupConv_out_channels=384, conv1x1_out_channels=256,
                                  kernel_size=(3, 1), stride=1, padding=(1, 0), group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        # self.block4_2 = GroupUnit(in_channels = 256, groupConv_out_channels = 384, conv1x1_out_channels=256, kernel_size=(3,1), stride=1, padding=(1,0), group=4, right_context=0, clamp=CLAMP, quant=QUANT)
        self.block4_3 = GroupUnit(in_channels=256, groupConv_out_channels=384, conv1x1_out_channels=256,
                                  kernel_size=(3, 1), stride=1, padding=(1, 0), group=4, right_context=0, clamp=CLAMP,
                                  quant=QUANT)
        # self.block4_4 = GroupUnit(in_channels = 256, groupConv_out_channels = 384, conv1x1_out_channels=256, kernel_size=(3,1), stride=1, padding=(1,0), group=4, right_context=0, clamp=CLAMP, quant=QUANT)

        self.dconv = ConvUint(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=1, padding=0, group=1,
                              right_context=0, is_active=True, clamp=CLAMP, quant=QUANT)
        self.convout = ConvUint(in_channels=64, out_channels=frames_classes, kernel_size=(1, 1), stride=1, padding=0,
                                group=1,right_context=0, is_active=False, is_clamp=False, clamp=CLAMP, quant=QUANT)

    def forward(self, src, mask=None):
        '''
            input shape: N, 1, T, 40
            output shape: N, 3003, T/4
        '''
        # src = src.permute(0, 2, 3, 1)
        # src = self.CR(src)

        # xx = src.permute([0,2,1,3])
        # with open('fea', 'wb') as f:
        #     f.write(xx.detach().cpu().numpy().tobytes())

        out = self.conv0(src, mask)
        out = out + self.conv1(out, mask)
        out = F.max_pool2d(out, kernel_size=(2, 1), ceil_mode=True)
        out = self.conv2(out, mask)
        out = out + self.conv3(out, mask)
        out = F.max_pool2d(out, kernel_size=(2, 2), ceil_mode=True)

        out = self.block1_0(out, mask)
        out = out + self.block1_1(out, mask)
        out = out + self.block1_2(out, mask)

        out = out + self.block1_3(out, mask)
        out = F.max_pool2d(out, kernel_size=(1, 2), ceil_mode=True)

        out = self.block2_0(out, mask)
        out = out + self.block2_1(out, mask)
        out = out + self.block2_1(out, mask)
        out = out + self.block2_1(out, mask)
        out = F.max_pool2d(out, kernel_size=(1, 2), ceil_mode=True)

        out = self.block3_0(out, mask)
        out = out + self.block3_1(out, mask)
        out = out + self.block3_1(out, mask)
        out = out + self.block3_1(out, mask)  #([1, 128, 13, 20])
        out = F.max_pool2d(out, kernel_size=(1, 20), ceil_mode=True)  #([1, 128, 13, 1])
        # print(f"out",out.shape)
        # self.saved_feature = out.detach().cpu()
        
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(out.shape[0], out.shape[1], -1, 1)
        out = out.permute(0, 2, 1, 3).contiguous()
        # print(f"out",out.shape)
        out = self.conv2dto1d_1(out)
        # print(f"out",out.shape)
        out = out + self.block4_0(out, mask)
        out = out + self.block4_1(out, mask)
        out = out + self.block4_1(out, mask)
        out = out + self.block4_3(out, mask)
        out = out + self.block4_3(out, mask)

        dconv_out = self.dconv(out, mask)
        model_out = self.convout(dconv_out)
        model_out = torch.squeeze(model_out, 3)
        return model_out
    
if __name__ == "__main__":
    device=torch.device("cpu")
    KWS = CNN().to(device)
    x = torch.randn(1, 1, 80, 1024,device=device)
    model_out = KWS(x)
    print("model_out.shape:", model_out.shape)
