from torch import nn
import torch

def patchify3D(images: torch.Tensor, patch_size: int):
    """
    Args:
        images (torch.Tensor): (B, C, W, H, D)
        patch_size (int): patch size
    Returns:
    
    """
    assert images.dim() == 5, f"images must have 5 dimensions, got {images.dim()}"
    B, C, W, H, D = images.shape 
    
    assert W == H == D, f"images must have the same width, height and depth, got {W}, {H}, {D}"
    assert W % patch_size == 0 or H % patch_size == 0 or D % patch_size == 0, f"images width, height and depth must be divisible by patch_size, got {W}, {H}, {D}"

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size, patch_size)
    return patches

class SwinTransformerBlock(nn.Module):
  	...

class SwinTransformer(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, num_classes, num_patches, patch_size, channels=3, heads_dim=64, window_size=7, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
  		super(SwinTransformer, self).__init__()
		self.patch_embed = nn.Conv3d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.num_patches = num_patches
		self.patch_size = patch_size
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
		self.pos_drop = nn.Dropout(p=drop_rate)
		
		self.layers = nn.ModuleList([
			SwinTransformerBlock(
				embed_dim=embed_dim,
				num_heads=num_heads,
				window_size=window_size,
				mlp_ratio=mlp_ratio,
				qkv_bsias=qkv_bias,
				qk_scale=qk_scale,
				drop=drop_rate,
				attn_drop=attn_drop_rate,
				drop_path=drop_path_rate,
				norm_layer=norm_layer
			) for _ in range(depths)
		])
		
		self.norm = norm_layer(embed_dim)
		self.head = nn.Linear(embed_dim, num_classes)