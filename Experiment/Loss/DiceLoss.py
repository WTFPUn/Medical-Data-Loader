from ..Generic.Metric import Loss

import torch

class DiceLoss(Loss[torch.Tensor, torch.Tensor]):
    eps = 1e-6
    
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.name = "Dice Similarity Coefficient"

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Evaluate loss using the Dice Similarity Coefficient between the 3D ground truth and the 3D prediction

        Args:
            y_true (torch.Tensor): The ground truth tensor(Batch Size, Class, Depth, Height, Width).
            y_pred (torch.Tensor): The prediction tensor(Batch Size, Class, Depth, Height, Width).

        Returns:
            torch.Tensor: loss using the Dice Similarity Coefficient between the ground truth and the prediction.
        """
        
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
        
        return 1 - dice.mean(dim=0)
       

    def __str__(self):
        return self.name