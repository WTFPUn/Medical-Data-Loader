import logging
import threading
import os
import json
from typing import Literal

from ..types import SimpleDatasetMetaData, DatasetConfig, DatasetMetaData
from .dataset import MedicalDataset

from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np



class DataEngine:
    def __init__(self, num_classes: int, meta_data: DatasetMetaData, dataset_config: DatasetConfig=DatasetConfig(), logger: logging.Logger=logging.getLogger(__name__)):
        self.logger = logger.getChild(self.__class__.__name__)
        
        self.meta_data = meta_data
        self.dataset_path = self.meta_data.info.dataset_path
        self.dataset_config = dataset_config
        self.num_classes = num_classes
        
        self.transform_to_npz()
        
        assert isinstance(self.meta_data, SimpleDatasetMetaData) or isinstance(self.meta_data, DatasetMetaData), "meta_data must be an instance of SimpleDatasetMetaData or DatasetMetaData"

        
        logger.info("DataEngine initialized with meta data from %s", self.dataset_path, extra={"contexts": "initialize data engine"})
        
    def __niigz_to_npz_thread(self, idx: str):
        data = sitk.GetArrayFromImage(sitk.ReadImage(f"{self.dataset_path}/data/img{idx}.nii.gz"))
        label = sitk.GetArrayFromImage(sitk.ReadImage(f"{self.dataset_path}/label/label{idx}.nii.gz"))
        
        np.savez_compressed(f"{self.dataset_path}/data_npz/img{idx}.npz", data)
        np.savez_compressed(f"{self.dataset_path}/label_npz/label{idx}.npz", label)
        
        self.logger.info("Transformed %s to npz", idx, extra={"contexts": "transform to npz"})
        return True
 
    def transform_to_npz(self):
        if os.path.exists(f"{self.dataset_path}/data_npz") and os.path.exists(f"{self.dataset_path}/label_npz"):
            # I'm too lazy to check if all files are npz files :P
            self.logger.info("Data already transformed to npz", extra={"contexts": "transform to npz"})
            return
        # create datanpz and labelnpz
        os.makedirs(f"{self.dataset_path}/data_npz", exist_ok=True)
        os.makedirs(f"{self.dataset_path}/label_npz", exist_ok=True)
        
        threads = []
        for idx in self.meta_data.data.train + self.meta_data.data.test + self.meta_data.data.val:
            thread = threading.Thread(target=self.__niigz_to_npz_thread, args=(idx,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.logger.info("Transformed all nii.gz files to npz files", extra={"contexts": "transform to npz"})

    def __get_data(self, data_type: Literal["train", "test", "val"]):
        data_ids = getattr(self.meta_data.data, data_type)
        data_dir = self.dataset_path
        try:
            dataset = MedicalDataset(self.num_classes, data_ids, data_dir,data_type, self.dataset_config)
            self.logger.info("Getting data from %s set", data_type, extra={"contexts": "get data"})
            return dataset
        except Exception as e:
            self.logger.error("Failed to get data from %s set with error: %s", data_type, e, extra={"contexts": "get data"})
            return None
    
    def __get_data_for_kfold(self, data_type: Literal["train", "test", "val"], fold: int):
        data_ids = self.meta_data.data[fold].__dict__[data_type]
        data_dir = self.dataset_path
        try:
            dataset = MedicalDataset(self.num_classes, data_ids, data_dir, data_type, self.dataset_config)
            self.logger.info("Getting data from %s set of fold %s", data_type, fold, extra={"contexts": "get data"})
            return dataset
        except Exception as e:
            self.logger.error("Failed to get data from %s set of fold %s with error: %s", data_type, fold, e, extra={"contexts": "get data"})
            return None
    
    def get_dataloader(self, data_type: Literal["train", "test", "val"], batch_size: int, shuffle: bool=True, num_workers: int=0):
        data = self.__get_data(data_type)
        
        try:
            self.logger.info("Creating dataloader for %s set", data_type, extra={"contexts": "create dataloader"})
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        except Exception as e:
            self.logger.error("Failed to create dataloader for %s set with error: %s", data_type, e, extra={"contexts": "create dataloader"})
            return None
        
    def get_dataloader_for_kfold(self, data_type: Literal["train", "test", "val"], fold: int, batch_size: int, shuffle: bool=True, num_workers: int=0):
        data = self.__get_data_for_kfold(data_type, fold)
        
        try:
            self.logger.info("Creating dataloader for %s set of fold %s", data_type, fold, extra={"contexts": "create dataloader"})
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        except Exception as e:
            self.logger.error("Failed to create dataloader for %s set of fold %s with error: %s", data_type, fold, e, extra={"contexts": "create dataloader"})
            return None