import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv_module import *
from layers.conv_utils import *
import numpy as np


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, 4, bias = False),
            nn.ReLU(),
            nn.Linear(4, 4, bias = False),
            nn.ReLU(),
            nn.Linear(4, emb_dim, bias = False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.conv1(x)
            #print(x1[0][0])
            #weight = self.conv1[0].weight.data
            #x1 = quantization_conv(x, weight, stride = 1, padding = 1, half_level=128)
            #print(x1[0][0])
            return x1

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ ResidualConvBlock(in_channels, out_channels, is_res = True), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels,if_fill = False):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias = False),
            ResidualConvBlock(out_channels, out_channels, is_res = True),
        ]
        if if_fill:
           layers[0] = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding = 0,output_padding=1, bias = False)

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        #print(self.model[0])
        #x  = conv_transpose_to_conv2d(self.model[0], x, half_level=128,)
        #print(x1[0][0])
        #x = self.model[0](x)
        #print(x[0][0])
        #x = self.model[1](x)
        return x

def cal_conv(input_feature_map, weight=None, stride = 1, padding = 0, half_level=128, isint=1):
    if weight is None:
        # 从文件中读取weight
        try:
            weight = torch.load("weight_init_cov.pt")
            print("Weight loaded from file.")
        except FileNotFoundError:
            print("Weight file not found.")
            return
    else:
        # 对weight进行量化处理
        weight, scale = data_quantization_sym(weight, half_level=half_level, isint=isint, clamp_std=3)
        out_channels, in_channels, kernel_height, kernel_width = weight.shape
        weight = weight.reshape(out_channels, kernel_height ** 2 * in_channels).transpose(1, 0).detach().numpy()
        
        # 将weight保存到文件
        torch.save(weight, "weight_init_cov.pt")
        print("Weight saved to file.")
    
    
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    output_feature_maps = []
    for feature in input_feature_map:
        out = conv2d_sim(
            input_feature_map=feature.detach().numpy(),
            weight=weight,
            repeat=None,
            stride=stride,
            kernel_size=kernel_height,
            padding=padding,
            input_half_level=half_level,
            output_half_level=half_level,
            it_time=10,
            relu=True,
            input_quant=True
        )

        output_feature_maps.append(torch.from_numpy(out))
    out = torch.stack(output_feature_maps,dim=0)
    out = out / scale
    return out.float()         


class UNet(nn.Module):
    def __init__(self, in_channels, n_feat = 16, kernel_size = 3, lattice_shape = 14):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        self.if_fill = (lattice_shape[0] % 4 != 0)

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        

        self.down1 = UnetDown(n_feat, 1 * n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        self.up0 = ResidualConvBlock(2*n_feat, 2*n_feat, is_res=True)

        
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up1 = UnetUp(4 * n_feat, n_feat, self.if_fill)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        self.out = nn.Sequential(
            nn.Conv2d(2*  n_feat, n_feat, kernel_size, 1, 1, bias = False),
            nn.BatchNorm2d(n_feat), 
            nn.ReLU(),
            nn.Conv2d( n_feat, 4, kernel_size, 1, 1, bias = False),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, self.in_channels * 2, kernel_size, 1, 1, bias = False),
        )

    def forward(self, x, t):

        x = self.init_conv(x)
        #weight1 = np.load("weights/30.net.init_conv.conv1.0.weight.npy")
        
        #print(weight1)
        #x = quantization_conv(x, weight1, stride = 1, padding = 1, half_level=128) 
        
        down1 = self.down1(x)
        #weight2 = np.load("weights/30.net.down1.model.0.conv1.0.weight.npy")
        #down1 = quantization_conv(x, weight2, stride = 1, padding = 1, half_level=128)
        #down1 = F.max_pool2d(down1, kernel_size=2)

        down2 = self.down2(down1)
        #weight3 = np.load("weights/30.net.down2.model.0.conv1.0.weight.npy")
        #down2 = quantization_conv(down1, weight3, stride = 1, padding = 1, half_level=128)
        #down2 = F.max_pool2d(down2, kernel_size=2)       

        up1 = self.up0(down2) 
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        up2 = self.up1(up1 + temb1, down2)  # add and multiply embeddings
         
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        up3 = self.up2(up2 + temb2, down1)

        out = self.out(torch.cat((up3, x), 1))
        return out

