import torch
from torch import nn

class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, out_chans=768, patch_size=16, img_size=224):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_chans))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, 'Input tensor shape must be of the form BxCxHxH'
        p = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        p = torch.cat((cls_tokens, p), dim=1)
        return p
    

if __name__ == "__main__":
    model = PatchEmbed()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)