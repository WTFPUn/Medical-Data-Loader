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
        self.window_center = dataset_config.window_center
        self.window_width = dataset_config.window_width
        self.device = dataset_config.device
        self.num_classes = num_classes
        
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        '''
            Apply windowing and normalization to the image.

            Args:
                img (np.ndarray): 3D image array.

            Returns:
                np.ndarray: Preprocessed image array scaled between [0, 1].
        '''
        lower = self.window_center - self.window_width // 2
        upper = self.window_center + self.window_width // 2
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
        return img
        
    
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx: int):
        data_id = self.data_ids[idx]
        data_path = f"{self.data_dir}/data_npz/img{data_id}.npz"
        label_path = f"{self.data_dir}/label_npz/label{data_id}.npz"

        data = np.load(data_path,mmap_mode='r')["arr_0"]
        label = np.load(label_path, mmap_mode='r')["arr_0"]
        
        data = self.preprocess(data)       
        
        # Apply transforms
        if self.transform:
            data, label = self.transform([data, label])
        else:
            data = torch.unsqueeze(torch.from_numpy(data), 0).float()  # Shape: (1, W, H, D)
            label = torch.from_numpy(label.squeeze(0)).long()  # Shape: (W, H, D)

        return idx, data, label
    