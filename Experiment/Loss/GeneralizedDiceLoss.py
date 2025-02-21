import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Generic import Loss

class GeneralizedDiceLoss(Loss[torch.Tensor, torch.Tensor]):
    def __init__(self, epsilon=1e-6):
        """
        Initializes the Generalized Dice Loss module.
        
        Args:
            epsilon (float): Small constant to avoid division by zero.
        """
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Generalized Dice Loss.
        
        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, W, H, D)
            y_true (torch.Tensor): Ground truth labels with shape (B, 1, W, H, D)
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # y_pred: (B, C, W, H, D)
        # y_true: (B, 1, W, H, D)
        
        # Convert y_true to one-hot encoding:
        # 1. Remove the channel dimension: (B, W, H, D)
        # 2. Apply one_hot: (B, W, H, D, C)
        # 3. Permute to get shape (B, C, W, H, D)
        num_classes = y_pred.size(1)
        y_true_one_hot = F.one_hot(y_true.squeeze(1), num_classes=num_classes)  # (B, W, H, D, C)
        y_true_one_hot = y_true_one_hot.permute(0, 4, 1, 2, 3).float()             # (B, C, W, H, D)
        
        # Apply softmax to y_pred along the class dimension to get probabilities
        y_pred = F.softmax(y_pred, dim=1)  # (B, C, W, H, D)
        
        # Flatten the spatial dimensions into one dimension for both predictions and ground truth.
        # New shape: (B, C, N) where N = W * H * D
        y_pred_flat = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true_flat = y_true_one_hot.view(y_true_one_hot.size(0), y_true_one_hot.size(1), -1)
        
        # Compute per-class weights based on the ground truth volume:
        # Sum over batch and spatial dimensions, resulting in a tensor of shape (C,)
        # Then, compute weights = 1 / (sum^2 + epsilon)
        w = 1.0 / (torch.sum(y_true_flat, dim=(0, 2))**2 + self.epsilon)  # (C,)
        
        # Compute the intersection for each class:
        # Element-wise multiplication and then sum over spatial dimension, shape: (B, C)
        # Sum over the batch to obtain a tensor of shape (C,)
        intersection = torch.sum(y_pred_flat * y_true_flat, dim=2)  # (B, C)
        intersection = torch.sum(intersection, dim=0)               # (C,)
        
        # Compute the union (sum of predictions and ground truth) for each class:
        # First, sum over spatial dimension: (B, C)
        # Then, sum over the batch to obtain shape (C,)
        union = torch.sum(y_pred_flat + y_true_flat, dim=2)  # (B, C)
        union = torch.sum(union, dim=0)                      # (C,)
        
        # Compute the weighted Dice score numerator and denominator:
        numerator = 2.0 * torch.sum(w * intersection)
        denominator = torch.sum(w * union) + self.epsilon
        
        dice_score = numerator / denominator
        loss = 1.0 - dice_score
        
        return loss

    def __str__(self):
        return self.__class__.__name__