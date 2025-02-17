from typing import List, Literal
import os

from torch.utils.data import Dataset
import torch
import numpy as np
from ..types import DatasetConfig

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
    