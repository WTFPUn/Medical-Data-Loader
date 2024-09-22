from ..DataEngine import MedicalDataset
from ..Generic import ModelTrainer
from .nnunet_mednext import create_mednext_v1
from ..Generic import Loss

import logging
from typing import List

import torch

generic_input, generic_output = torch.Tensor, torch.Tensor

class MedNeXt(ModelTrainer[generic_input, generic_output]):
    def __init__(self, logger: logging.Logger, losses: List[Loss[generic_input, generic_output]], num_input_channels: int, num_classes: int, model_id: str):
        super(MedNeXt, self).__init__(logger, losses, create_mednext_v1(
            num_input_channels = num_input_channels,
            num_classes = num_classes,
            model_id = model_id,             # S, B, M and L are valid model ids
            kernel_size = 3,            # 3x3x3 and 5x5x5 were tested in publication
            deep_supervision = True    # True or False 
        ))
        self.name = "MedNeXt"
    
    def train(self, train: MedicalDataset, val: MedicalDataset, weight_save_period: int) -> None:
        
        optimizer = torch.optim.adamw.AdamW(self.model.parameters(), lr=1e-4)
        loss = self.losses
        
        
        
        
        
        
        
    