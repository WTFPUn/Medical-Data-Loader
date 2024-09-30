from ..Generic import Metric

import torch

class DSC(Metric[torch.Tensor, torch.Tensor]):
    eps = 1e-6
    
    def __init__(self):
        super(DSC, self).__init__()
        self.name = "Dice Similarity Coefficient"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Dice Similarity Coefficient between the 3D ground truth and the 3D prediction

        Args:
            y_true (torch.Tensor): The ground truth tensor(Batch Size, Class, Depth, Height, Width).
            y_pred (torch.Tensor): The prediction tensor(Batch Size, Class, Depth, Height, Width).

        Returns:
            torch.Tensor: The Dice Similarity Coefficient between the ground truth and the prediction.
        """
        y_true = torch.nn.functional.one_hot(y_true, y_pred.size(1)).permute(0,5,1,2,3,4).squeeze(2)
        
        assert y_true.size() == y_pred.size(), "The size of the ground truth and the prediction should be the same."
        
        
        pred = y_pred.contiguous().view(y_pred.size(0), y_pred.size(1), -1)
        target = y_true.contiguous().view(y_true.size(0), y_true.size(1), -1)
        
        # Compute intersection (True Positives)
        intersection = (pred * target).sum(dim=2)
        
        # Compute the sums of the predicted and target areas
        pred_sum = pred.sum(dim=2)
        target_sum = target.sum(dim=2)
        
        # Dice Coefficient formula
        dice = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        
        # divide by the number of classes
        dice = dice.mean(dim=1)
        
        return dice.mean(dim=0)
       

    def __str__(self):
        return self.name

class DSCLoss(Metric[torch.Tensor, torch.Tensor]):
    eps = 1e-6
    
    def __init__(self):
        super(DSCLoss, self).__init__()
        self.name = "Dice Similarity Coefficient"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Dice Similarity Coefficient between the 3D ground truth and the 3D prediction.

        Args:
            y_true (torch.Tensor): The ground truth tensor (B, C, D, H, W) for multi-class segmentation.
            y_pred (torch.Tensor): The prediction tensor (B, C, D, H, W) for multi-class segmentation.

        Returns:
            torch.Tensor: The Dice loss value (1 - Dice coefficient).
        """
        # assert y_true.size() == y_pred.size(), "The size of the ground truth and the prediction should be the same."
        
        # sparse to one-hot
        y_true = torch.nn.functional.one_hot(y_true, y_pred.size(1)).permute(0,5,1,2,3,4).squeeze(2)
        
        assert y_true.size() == y_pred.size(), "The size of the ground truth and the prediction should be the same."
        
        # Flatten predictions and targets into shape (B, C, -1) for easier calculation
        y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
        
        # Compute intersection (True Positives)
        intersection = (y_pred * y_true).sum(dim=2)  # Sum over spatial dimensions
        
        # Compute the sums of the predicted and target areas
        pred_sum = y_pred.sum(dim=2)
        target_sum = y_true.sum(dim=2)
        
        # Dice Coefficient formula for each class in each batch
        dice = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
        
        # Average the Dice score over the batch and classes
        dice_loss = 1.0 - dice.mean()  # Final Dice loss

        return dice_loss
       

    def __str__(self):
        return self.name
