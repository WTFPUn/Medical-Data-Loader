from typing import List
import os
import logging

from torch.utils.data import Dataset
from torchvision import transforms
import torch

import nibabel as nib
import SimpleITK as sitk
import numpy as np

from ..types import DatasetConfig


class MedicalDataset(Dataset):
    '''
        Dataset from torch for medical data. This dataloader return 3D image and label(3D mask)
    '''
    
    def __init__(self, data_ids: List[str], data_dir: str, transform: transforms.Compose=None, dataset_config: DatasetConfig=DatasetConfig()):
        '''
            Initialize the dataset
            Args:
                data_ids (List[str]): List of data ids
                data_dir (str): Directory where the data is stored
                transform (transforms.Compose): Transform to apply on the data
                window_center (int): Window center for windowing the data
                window_width (int): Window width for windowing the data
        '''
        self.data_ids = data_ids
        self.data_dir = data_dir
        self.transform = transform
        self.window_center = dataset_config.window_center
        self.window_width = dataset_config.window_width
        self.device = dataset_config.device
        
    def windowing(self, img, window_center, window_width) -> np.ndarray:
        upper, lower = window_center + window_width // 2, window_center - window_width // 2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        return X
    
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        data_path = f"{self.data_dir}/data/img{data_id}.nii.gz"
        label_path = f"{self.data_dir}/label/label{data_id}.nii.gz"

        data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
        data = self.windowing(data, self.window_center, self.window_width)
        
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
            
        return idx, torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(label, dtype=torch.float32, device=self.device)
        
        
        