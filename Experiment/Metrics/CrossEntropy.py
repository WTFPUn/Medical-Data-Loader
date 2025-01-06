import torch
import torch.nn as nn

from ..Generic import Metric

import torch

class CE(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self, weight: torch.Tensor = None):
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions and batch size for y_true to match the expected shape for nn.CrossEntropyLoss
        # y_true should be in shape (B, D*H*W)
        if y_true.dtype != torch.long:
            y_true = y_true.long()
            
        if y_pred.dtype != torch.float32:
            y_pred = y_pred.float()
        
        y_true = y_true.view(y_true.size(0), -1)
        
        # Reshape y_pred to (B, C, D*H*W)
        y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)

        # Compute the loss
        loss = self.loss_fn(y_pred, y_true)
        
        return loss
    
    def __str__(self):
        return self.__class__.__name__