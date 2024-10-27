import torch
import random
from typing import Tuple, Dict, Literal

class RandomFlip3D:
    """
    Randomly flip the 3D voxel along specified axes for both input and target tensors.

    Args:
        axes (Tuple[int, ...], optional): Axes along which to apply random flipping.
                                         0: Depth (D), 1: Height (H), 2: Width (W)
                                         Default is (0, 1, 2), enabling flipping along all axes.
        flip_prob (float, optional): Probability of flipping along each axis. Default is 0.5.
    """

    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), flip_prob: float = 0.5):
        assert isinstance(axes, (tuple, list)), "axes should be a tuple or list of integers"
        for axis in axes:
            assert axis in (0, 1, 2), "axes can be 0 (D), 1 (H), or 2 (W)"
        assert 0.0 <= flip_prob <= 1.0, "flip_prob must be between 0 and 1"
        
        self.axes = axes
        self.flip_prob = flip_prob

    def __call__(self, transform_data: Tuple[torch.tensor, ...]):
        """
        Apply random flipping to the input and target voxels.

        Args:
            input_voxel (torch.Tensor): The input voxel tensor.
                                        Shape: (B, D, H, W) or (D, H, W)
                                        (B is the batch size, if this method is call within transformation pipeline)
            target_voxel (torch.Tensor): The target voxel tensor.
                                         Shape: (B, D, H, W) or (D, H, W)
                                            (B is the batch size, if this method is call within transformation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The transformed input and target voxels.
        """
        input_voxel, target_voxel = transform_data[0], transform_data[1]
        
        # Determine if tensors are 3D or 4D
        input_dim = input_voxel.dim()
        target_dim = target_voxel.dim()

        assert input_dim in (3, 4), "Input voxel must be a 3D or 4D tensor"
        assert target_dim in (3, 4), "Target voxel must be a 3D or 4D tensor"
        assert input_dim == target_dim, "Input and target must have the same number of dimensions"

        for axis in self.axes:
            if random.random() < self.flip_prob:
                # Calculate the dimension index considering the channel dimension for 4D tensors
                dim = axis + 1 if input_dim == 4 else axis
                input_voxel = torch.flip(input_voxel, dims=(dim,))
                target_voxel = torch.flip(target_voxel, dims=(dim,))

        return input_voxel, target_voxel

    def __repr__(self):
        axes_str = ', '.join(['D', 'H', 'W'][axis] for axis in self.axes)
        return f"{self.__class__.__name__}(axes=({axes_str}), flip_prob={self.flip_prob})"
