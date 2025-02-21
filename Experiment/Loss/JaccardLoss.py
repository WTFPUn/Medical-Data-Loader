import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Generic import Loss


class JaccardLoss(Loss[torch.Tensor, torch.Tensor]):
    def __init__(self, smooth=1e-6):
        """
        Initializes the Jaccard Loss module.
        
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Jaccard Loss.
        
        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, W, H, D).
            y_true (torch.Tensor): Ground truth labels with shape (B, 1, W, H, D).
            
        Returns:
            torch.Tensor: Scalar Jaccard loss value.
        """
        # y_pred: (B, C, W, H, D)
        # y_true: (B, 1, W, H, D)

        # 1. Apply softmax to the predictions along the class dimension.
        y_pred = F.softmax(y_pred, dim=1)
        # y_pred remains: (B, C, W, H, D)

        # 2. Convert y_true to one-hot encoding:
        #    a. Remove the channel dimension: (B, W, H, D)
        #    b. One-hot encode to get: (B, W, H, D, C)
        #    c. Permute to obtain shape: (B, C, W, H, D)
        num_classes = y_pred.size(1)
        y_true_one_hot = F.one_hot(y_true.squeeze(1), num_classes=num_classes)  # (B, W, H, D, C)
        y_true_one_hot = y_true_one_hot.permute(0, 4, 1, 2, 3).float()             # (B, C, W, H, D)

        # 3. Compute the intersection and union:
        #    Sum over the batch and spatial dimensions.
        #    Intersection: element-wise multiplication between predictions and one-hot ground truth.
        #    Union: sum of predictions and ground truth minus their element-wise product.
        intersection = torch.sum(y_pred * y_true_one_hot, dim=[0, 2, 3, 4])  # (C,)
        union = torch.sum(y_pred + y_true_one_hot - y_pred * y_true_one_hot, dim=[0, 2, 3, 4])  # (C,)

        # 4. Compute the Jaccard index for each class.
        #    Add the smoothing term to both numerator and denominator.
        jaccard = (intersection + self.smooth) / (union + self.smooth)  # (C,)

        # 5. The Jaccard loss is defined as 1 minus the average Jaccard index over classes.
        loss = 1 - jaccard.mean()

        return loss
    
    def __str__(self):
        return self.__class__.__name__