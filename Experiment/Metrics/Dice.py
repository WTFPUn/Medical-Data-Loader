from ..Generic import Metric

import torch

class DSC(Metric[torch.Tensor, torch.Tensor]):
    eps = 1e-6
    
    def __init__(self, num_classes=3):
        super(DSC, self).__init__()
        self.num_classes = num_classes

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Apply softmax to y_pred
        if y_true.dtype != torch.long:
            y_true = y_true.long()
        
        if y_pred.dtype != torch.float32:
            y_pred = y_pred.float()
        
        y_true = y_true.squeeze(1)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        
        # Convert y_true to one-hot encoding
        y_true = torch.nn.functional.one_hot(y_true, self.num_classes).permute(0,4,1,2,3).contiguous()
        
        assert y_true.size() == y_pred.size(), "The size of the ground truth and the prediction should be the same."
        
        # Flatten tensors
        y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
        
        # Compute intersection and sums
        intersection = (y_pred * y_true).to(torch.float32).sum(dim=2)
        pred_sum = y_pred.to(torch.float32).sum(dim=2)
        target_sum = y_true.to(torch.float32).sum(dim=2)
        
        # Compute Dice coefficient
        dice = ((2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps))

        return dice.mean().to(torch.float16)
       

    def __str__(self):
        return self.__class__.__name__

class DSCLoss(Metric[torch.Tensor, torch.Tensor]):
    eps = 1e-6
    
    def __init__(self, num_classes: int):
        super(DSCLoss, self).__init__()
        self.num_classes = num_classes

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Apply softmax to y_pred
        if y_true.dtype != torch.long:
            y_true = y_true.long()
            
        if y_pred.dtype != torch.float32:
            y_pred = y_pred.float()
        
        y_true = y_true.squeeze(1)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        
        # Convert y_true to one-hot encoding
        y_true = torch.nn.functional.one_hot(y_true, self.num_classes).permute(0,4,1,2,3).contiguous()
        
        assert y_true.size() == y_pred.size(), "The size of the ground truth and the prediction should be the same."
        
        # Flatten tensors
        y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
        
        # Compute intersection and sums
        intersection = (y_pred * y_true).sum(dim=2)
        pred_sum = y_pred.sum(dim=2)
        target_sum = y_true.sum(dim=2)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        
        # Compute Dice loss
        dice_loss = 1.0 - dice.mean().to(torch.float16)

        return dice_loss

    def __str__(self):
        return self.__class__.__name__

