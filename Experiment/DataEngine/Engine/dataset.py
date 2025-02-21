from typing import List, Literal
import os

from torch.utils.data import Dataset
import torch
import numpy as np
from ..types import DatasetConfig, PatchedDatasetConfig

import torch.nn.functional as F

class MedicalDataset(Dataset):
    '''
        Dataset from torch for medical data. This dataloader return 3D image and label(3D mask)
    '''
    
    def __init__(self, num_classes: int, data_ids: List[str], data_dir: str, data_type: Literal["train", "test", "val"], dataset_config: DatasetConfig=DatasetConfig()):
        '''
            Initialize the dataset
            Args:
                data_ids (List[str]): List of data ids
                data_dir (str): Directory where the data is stored
                transform (transforms.Compose): Transform to apply on the data
                window_center (int): Window center for windowing the data
                window_width (int): Window width for windowing the data
        '''
        
        assert os.path.exists(os.path.join(data_dir, "data_npz")), "Data directory does not exist"
        assert os.path.exists(os.path.join(data_dir, "label_npz")), "Label directory does not exist"
        
        self.data_ids = data_ids
        self.data_dir = data_dir
        self.transform = dataset_config.compose[data_type] if data_type in dataset_config.compose.keys() else None
        # self.window_center = dataset_config.window_center
        # self.window_width = dataset_config.window_width
        self.gamma = dataset_config.gamma
        self.device = dataset_config.device
        self.num_classes = num_classes


    def gamma_correction(self, img: torch.Tensor, gamma: float):
        '''
            Apply gamma correction to the image.

            Args:
                img (torch.Tensor): Image tensor normalized to [0,1].
                gamma (float): Gamma correction factor.

            Returns:
                torch.Tensor: Gamma-corrected image.
        '''
        return img ** (1 / gamma)


    def preprocess(self, img: torch.Tensor):
        '''
            Apply windowing, normalization, and gamma correction to the image.

            Args:
                img (torch.Tensor): 3D image tensor.

            Returns:
                torch.Tensor: Preprocessed image tensor scaled between [0, 1].
        '''
        
        # Flatten and filter HU values > -1000
        hu_values = img.flatten()
        hu_values_filtered = hu_values[hu_values > -1000]

        # Compute mean HU value safely
        if hu_values_filtered.numel() > 0:  # Ensure non-empty tensor
            mean_value = torch.mean(hu_values_filtered)
        else:
            mean_value = torch.tensor(0.0, device=img.device)  # Default if no valid HU values

        # Select windowing based on mean HU value
        lower, upper = (100, 1600) if mean_value > 0 else (-400, 1100)

        # Apply windowing (clip)
        img = torch.clamp(img, lower, upper)

        # Normalize to [0, 1]
        img = (img - lower) / (upper - lower)

        # Apply gamma correction
        img = self.gamma_correction(img, self.gamma)
        torch.cuda.empty_cache()

        return img
        

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx: int):
        data_id = self.data_ids[idx]
        data_path = f"{self.data_dir}/data_npz/img{data_id}.npz"
        label_path = f"{self.data_dir}/label_npz/label{data_id}.npz"

        # Load data as memory-mapped arrays without changing dtype
        data = torch.from_numpy(np.load(data_path, mmap_mode='r')["arr_0"]).unsqueeze(0).float()
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        label = torch.from_numpy(np.load(label_path, mmap_mode='r')["arr_0"]).unsqueeze(0).long()

        # Preprocess data without loading the entire array into memory
        data = self.preprocess(data)

        # Apply transforms
        if self.transform:
            data, label = self.transform([data, label])
        else:
            data = torch.unsqueeze(torch.from_numpy(data), 0).float().to(self.device, non_blocking=True)
            label = torch.from_numpy(label).long().to(self.device, non_blocking=True)
        return idx, data, label

class PatchedMedicalDataset(Dataset):
    def __init__(self, num_classes: int, data_ids: List[str], data_dir: str, data_type: Literal["train", "test", "val"], dataset_config: PatchedDatasetConfig=PatchedDatasetConfig()):
        self.patch_size = dataset_config.patch_size
        self.voxel_size = dataset_config.voxel_size
        self.stride = dataset_config.stride
        self.num_classes = num_classes
        self.data_ids = data_ids
        self.data_dir = data_dir
        self.data_type = data_type
        self.device = dataset_config.device
        self.window_center = dataset_config.window_center
        self.window_width = dataset_config.window_width
        self.transform = dataset_config.compose[data_type] if data_type in dataset_config.compose.keys() else None
    
    def patchify_3d_with_positions(self, input_tensor, patch_size, voxel_size: int):
        """
        Converts a 4D tensor of shape (C, H, W, D) into patches of shape
        (C, P, patch_size, patch_size, patch_size), ensuring padding if needed, and attaches positions.

        Parameters:
            input_tensor (torch.Tensor): Input tensor of shape (C, H, W, D).
            patch_size (int): Size of the patch (assumed cubic for simplicity).
            voxel_size (int): Target size for each dimension (H, W, D).

        Returns:
            tuple: 
                - torch.Tensor: Patched tensor of shape (C, P, patch_size, patch_size, patch_size).
                - torch.Tensor: Positions tensor of shape (P, 3), where each entry is (x, y, z) position.
        """
        C, H, W, D = input_tensor.shape

        # # Check if dimensions match the voxel size, otherwise pad
        # pad_h = max(0, voxel_size - H)
        # pad_w = max(0, voxel_size - W)
        # pad_d = max(0, voxel_size - D)

        # if pad_h > 0 or pad_w > 0 or pad_d > 0:
        #     # Apply padding: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        #     input_tensor = F.pad(input_tensor, (0, pad_d, 0, pad_w, 0, pad_h))

        # Update dimensions after padding
        H, W, D = input_tensor.shape[1:]

        # Ensure H, W, D are divisible by patch_size
        assert H % patch_size == 0 and W % patch_size == 0 and D % patch_size == 0, \
            "Height, Width, and Depth must be divisible by patch_size after padding."

        # Compute the number of patches along each dimension
        num_patches_x = H // patch_size
        num_patches_y = W // patch_size
        num_patches_z = D // patch_size

        # Create position indices
        positions = []
        for x in range(num_patches_x):
            for y in range(num_patches_y):
                for z in range(num_patches_z):
                    positions.append((x, y, z))

        # Convert positions to a tensor of shape (P, 3)
        positions_tensor = torch.tensor(positions, dtype=torch.int32)

        # Reshape and permute to create patches
        patches = input_tensor.unfold(1, patch_size, patch_size)  # H -> H/p patches
        patches = patches.unfold(2, patch_size, patch_size)       # W -> W/p patches
        patches = patches.unfold(3, patch_size, patch_size)       # D -> D/p patches

        # Rearrange patches into the desired shape
        # patches shape: (C, H/p, W/p, D/p, patch_size, patch_size, patch_size)
        P = num_patches_x * num_patches_y * num_patches_z
        patches = patches.contiguous().view(C, P, patch_size, patch_size, patch_size)

        return patches, positions_tensor
    
    def preprocess(self, img: torch.Tensor):
        '''
            Apply windowing and normalization to the image.

            Args:
                img (np.ndarray): 3D image array.

            Returns:
                np.ndarray: Preprocessed image array scaled between [0, 1].
        '''
        
        lower = self.window_center - self.window_width // 2
        upper = self.window_center + self.window_width // 2
        img = torch.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
        return img
    
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx: int):
        data_id = self.data_ids[idx]
        data_path = f"{self.data_dir}/data_npz/img{data_id}.npz"
        label_path = f"{self.data_dir}/label_npz/label{data_id}.npz"

        # Load data as memory-mapped arrays without changing dtype
        data = torch.from_numpy(np.load(data_path, mmap_mode='r')["arr_0"]).unsqueeze(0).float()
        label = torch.from_numpy(np.load(label_path, mmap_mode='r')["arr_0"]).unsqueeze(0).long()

        
        # Preprocess data without loading the entire array into memory
        data = self.preprocess(data)
        
        # Apply augmentations
        if self.transform:
            data, label = self.transform([data, label])
        else:
            data = torch.unsqueeze(torch.from_numpy(data), 0).float().to(self.device, non_blocking=True)
            label = torch.from_numpy(label).long().to(self.device, non_blocking=True)
        
        print(data.shape)

        # Patchify both input data and labels
        patches, position = self.patchify_3d_with_positions(data, self.patch_size, self.voxel_size)
        label_patches, _ = self.patchify_3d_with_positions(label, self.patch_size, self.voxel_size)

        # reshape patches from (C, P, H, W, D) to (P, C, H, W, D)
        patches = patches.permute(1, 0, 2, 3, 4)
        label_patches = label_patches.permute(1, 0, 2, 3, 4)

        # When use dataloader, the shape of patches is (B, P, C, H, W, D) you need to permute to (P, B, C, H, W, D)
        return idx, patches, label_patches, position
        