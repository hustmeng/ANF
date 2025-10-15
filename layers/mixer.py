from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import Conv2d
import torch.nn.functional as F

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, 8, bias = False),
            nn.ReLU(),
            nn.Linear(8, 8, bias = False),
            nn.ReLU(),
            nn.Linear(8, emb_dim, bias = False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim, bias = False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,dim,  bias = False),
            
        )
    def forward(self,x):
        x=self.net(x)
        return x

class PerPixelFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias = False)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc(x)              # [B, H, W, out_channels]
        x = x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]
        return x

class MixerBlock(nn.Module):
    def __init__(self,dim,num_patch,token_dim,channel_dim,dropout=0.):
        super().__init__()
        self.token_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch,token_dim,dropout),
            Rearrange('b d n -> b n d')
 
         )
        self.channel_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim,channel_dim,dropout)
        )
        self.layer_normal = nn.BatchNorm1d(num_patch)

    def forward(self,x):
        x = x+self.token_mixer(x)
        x = x+self.channel_mixer(x)
        x = self.layer_normal(x)
        return x
    
class MLPMixer(nn.Module):
    def __init__(self,in_channels,dim,patch_size,num_patches,depth,token_dim,channel_dim,dropout=0.):
        super().__init__()

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size

        self.to_embedding=nn.Sequential(
            Conv2d(in_channels,dim,3,1,1, bias = False),
            nn.BatchNorm2d(dim),
            Conv2d(in_channels=dim,out_channels=dim,kernel_size=patch_size,stride=patch_size, bias = False),
            Rearrange('b c h w -> b (h w) c'),
        )

        
        self.mixer_blocks=nn.ModuleList([])
        self.temb_blocks = nn.ModuleList([])
        
        for _ in range(self.depth-1):
            self.mixer_blocks.append(MixerBlock(dim,self.num_patches,token_dim,channel_dim,dropout))
            self.temb_blocks.append(nn.Embedding(20,self.dim))
        
        self.out_embedding = nn.ConvTranspose2d(dim, dim*2, 3, 3, bias = False)
        
        self.regression = nn.Sequential(
                      nn.BatchNorm2d(dim*2),
                      nn.ReLU(),
                      PerPixelFC(dim*2, dim),
                      nn.BatchNorm2d(dim),
                      nn.ReLU(),
                      PerPixelFC(dim, dim),
                      nn.BatchNorm2d(dim),
                      nn.ReLU(),
                      PerPixelFC(dim, 2)
                      )
    def forward(self, x, t):       
        
        lattice = x.shape[-1]// self.patch_size
        x = self.to_embedding(x)
        

        for i, mixer_block in enumerate(self.mixer_blocks):
            temb = self.temb_blocks[i](t)
            x = x + temb
            x = mixer_block(x) 
        
        x = x.transpose(1, 2)  
         
        x = x.view(-1,self.dim,lattice,lattice)
        x = self.out_embedding(x)
        x = self.regression(x)
        return x
        


