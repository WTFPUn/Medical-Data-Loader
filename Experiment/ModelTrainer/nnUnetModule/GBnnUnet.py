from torch import nn
import torch
from .part import DoubleConv, Down, Up, OutConv, DepthwiseSeparableConv3d

from mamba.mamba_ssm import Mamba


class SSMBlock(nn.Module):
    def __init__(self, W: int, state: int, kernel_size: int = 16):
        super(SSMBlock, self).__init__()
        self.W = W
        self.state = state
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool3d(self.kernel_size, stride=self.kernel_size)
        self.mamba = Mamba(W**3, state)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume shape of x is (B, P, W, H, D)
        
        # output of pool is (B, P, W/kernel_size, H/kernel_size, D/kernel_size)
        x = self.pool(x)
        
        # output of view is (B, P, W/kernel_size * H/kernel_size * D/kernel_size)
        x = x.view(x.shape[0], x.shape[1], -1)
        
        # output of mamba is (B, P, W/kernel_size * H/kernel_size * D/kernel_size)
        x = self.mamba(x)
        
        # output of view is (B, P, W/kernel_size, H/kernel_size, D/kernel_size)
        x = x.view(x.shape[0], x.shape[1], self.W, self.W, self.W)
        return x



class GPNnUnet(nn.Module):
    def __init__(self, num_input_channels: int, num_classes: int,W: int, state: int, kernel_size: int = 16, trilinear: bool = False, use_ds_conv: bool = False, channels_multiplier: int = 1):
        super().__init__()
        _channels = (32, 64, 128, 256, 512)
        _channels = tuple([i * channels_multiplier for i in _channels])
        
        
        self.ssm = SSMBlock(W, state, kernel_size)
        self.ssm_proj = nn.Conv3d(in_channels=1, out_channels=_channels[4], kernel_size=1)
        
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.trilinear = trilinear

        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(num_input_channels, _channels[0], conv_layer=self.convtype)
        self.down = nn.ModuleList([Down(_channels[i], _channels[i + 1], conv_layer=self.convtype) for i in range(4)])
        self.up = nn.ModuleList([Up(_channels[i], _channels[i - 1], trilinear=trilinear, conv_layer=self.convtype) for i in range(4, 0, -1)])
        self.outc = OutConv(_channels[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume shape of x is (B, P, W, H, D)
        
        # output of ssm is (B, P, W, H, D)
        ssm_out = self.ssm(x)
        
        # permute dimensions from (B, P, W, H, D) to (P, B, W, H, D)
        x = x.permute(1, 0, 2, 3, 4)
        ssm_out = ssm_out.permute(1, 0, 2, 3, 4)
        
        # out shape is (P, B, num_classes, W, H, D)
        out = []
        
        # loop over P
        for i in range(x.shape[0]):
            # shape of x_i is (B, W, H, D)
            x_i = x[i]
            # shape of ssm_out_i is (B, W/16, H/16, D/16)
            ssm_out_i = ssm_out[i]
            
            # shape of encode_out is (B, 32, W/16, H/16, D/16) 
            encode_out = self.inc(x_i)
            encode_out = self.down[0](encode_out)
            encode_out = self.down[1](encode_out)
            encode_out = self.down[2](encode_out)
            encode_out = self.down[3](encode_out)
            
            # apply ssm_out_i to encode_out
            encode_out = encode_out * ssm_out_i
            
            decode_out = self.up[0](encode_out, encode_out)
            decode_out = self.up[1](decode_out, encode_out)
            decode_out = self.up[2](decode_out, encode_out)
            decode_out = self.up[3](decode_out, encode_out)
            
            # shape of logits is (B, num_classes, W, H, D)
            logits = self.outc(decode_out)
            
            # stack logits to out
            out.append(logits)
            
        out = torch.stack(out, dim=0)
            
        # permute dimensions from (P, B, num_classes, W, H, D) to (B, P, num_classes, W, H, D)
        out = out.permute(1, 0, 2, 3, 4, 5)
        return out

