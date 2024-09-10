import json
from typing import Literal

from ..types import DatasetMetaData, DatasetConfig
from .dataset import MedicalDataset
from ..log import logger

from torch.utils.data import DataLoader




class DataEngine:
    def __init__(self, meta_data_path: str, dataset_config: DatasetConfig=DatasetConfig()):
        try:
            with open(meta_data_path, "r", encoding="utf-8") as f:
                self.meta_data = DatasetMetaData.model_validate(json.load(f))
        except Exception as e:
            logger.error(e)
            return

        self.dataset_path = self.meta_data.info.dataset_path
        self.dataset_config = dataset_config
        
        logger.info("DataEngine initialized with meta data from %s", meta_data_path)

    def get_data(self, data_type: Literal["train", "test", "val"]):
        data_ids = getattr(self.meta_data.data, data_type)
        data_dir = self.dataset_path
        try:
            dataset = MedicalDataset(data_ids, data_dir, dataset_config=self.dataset_config)
            logger.info("Getting data from %s set", data_type)
            return dataset
        except Exception as e:
            logger.error("Failed to get data from %s set with error: %s", data_type, e)
            return None
    
    def get_dataloader(self, data_type: Literal["train", "test", "val"], batch_size: int, shuffle: bool=True):
        data = self.get_data(data_type)
        
        try:
            logger.info("Creating dataloader for %s set", data_type)
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        except Exception as e:
            logger.error("Failed to create dataloader for %s set with error: %s", data_type, e)
            return None