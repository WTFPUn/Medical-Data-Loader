import torch
import random
from typing import Tuple, Dict, Literal
import torchio as tio


class CropOrPad3D:

    def __init__(self, target_shape: Tuple[int, int, int]):
        self.transform = tio.CropOrPad(target_shape)

    def __call__(self, transform_data: Tuple[torch.tensor, ...]):
        assert len(transform_data) == 2, "Input and target voxels must be provided"
                
        input_voxel, target_voxel = transform_data[0], transform_data[1]
        
        assert input_voxel.dim() == 4, "Input voxel must be a 4D tensor(B, D, H, W)"
        assert target_voxel.dim() == 4, "Target voxel must be a 4D tensor(B, D, H, W)"

        input_voxel = self.transform(input_voxel)
        target_voxel = self.transform(target_voxel)

        return input_voxel, target_voxel

    def __repr__(self):
        return self.transform.__repr__()
