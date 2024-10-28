from torch import nn
import torch
from .part import DoubleConv, Down, Up, OutConv, DepthwiseSeparableConv3d

class NnUnetBase(nn.Module):
    def __init__(self, num_input_channels: int, num_classes: int, trilinear: bool = False, use_ds_conv: bool = False, channels_multiplier: int = 1):
        super().__init__()
        _channels = (32, 64, 128, 256, 512)
        _channels = tuple([i * channels_multiplier for i in _channels])
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.trilinear = trilinear

        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(num_input_channels, _channels[0], conv_layer=self.convtype)
        self.down = nn.ModuleList([Down(_channels[i], _channels[i + 1], conv_layer=self.convtype) for i in range(4)])
        self.up = nn.ModuleList([Up(_channels[i], _channels[i - 1], trilinear=trilinear, conv_layer=self.convtype) for i in range(4, 0, -1)])
        self.outc = OutConv(_channels[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down[0](x1)
        x3 = self.down[1](x2)
        x4 = self.down[2](x3)
        x5 = self.down[3](x4)
        x = self.up[0](x5, x4)
        x = self.up[1](x, x3)
        x = self.up[2](x, x2)
        x = self.up[3](x, x1)
        logits = self.outc(x)
        return logits
    

if __name__ == '__main__':
    model = NnUnetBase(1, 3, trilinear=False, use_ds_conv= True)
    x = torch.randn((4, 1, 128, 128, 128))
    y = model(x)
    print(y.shape)
    # print(y)