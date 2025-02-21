from ..Generic import Loss

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(Loss[torch.Tensor, torch.Tensor]):
    def __init__(self, num_classes, alpha=0.7, beta=0.3, gamma=0.75):
        """
        Initializes the FocalTverskyLoss module.

        Args:
            num_classes (int): Number of classes in the segmentation task.
            alpha (float, optional): Weight for false positives. Default is 0.7.
            beta (float, optional): Weight for false negatives. Default is 0.3.
            gamma (float, optional): Focusing parameter to adjust the emphasis on hard examples. Default is 0.75.
        """
        super(FocalTverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, y_pred, y_true):
        """
        Computes the Focal Tversky Loss.

        Args:
            y_pred (torch.Tensor): Predicted logits with shape (B, C, W, H, D), where B is batch size, C is number of classes, and W, H, D are spatial dimensions.
            y_true (torch.Tensor): Ground truth labels with shape (B, 1, W, H, D).

        Returns:
            torch.Tensor: Computed Focal Tversky Loss.
        """
        # Apply softmax to predictions along the class dimension
        y_pred = F.softmax(y_pred, dim=1)

        # Convert y_true to one-hot encoding
        y_true_one_hot = F.one_hot(y_true.squeeze(1), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Initialize the Tversky loss
        loss = 0.0

        # Compute Tversky loss for each class
        for c in range(self.num_classes):
            # Extract the predicted and true binary masks for class c
            y_pred_c = y_pred[:, c, :, :, :]  # Shape: (B, W, H, D)
            y_true_c = y_true_one_hot[:, c, :, :, :]  # Shape: (B, W, H, D)

            # Compute true positives (TP), false positives (FP), and false negatives (FN)
            TP = torch.sum(y_pred_c * y_true_c, dim=[1, 2, 3])  # Shape: (B,)
            FP = torch.sum(y_pred_c * (1 - y_true_c), dim=[1, 2, 3])  # Shape: (B,)
            FN = torch.sum((1 - y_pred_c) * y_true_c, dim=[1, 2, 3])  # Shape: (B,)

            # Compute Tversky index for class c
            Tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + 1e-6)  # Shape: (B,)

            # Compute Focal Tversky loss for class c
            focal_tversky_loss = (1 - Tversky_index) ** self.gamma  # Shape: (B,)

            # Accumulate the loss
            loss += focal_tversky_loss

        # Average the loss over all classes and batches
        loss = loss.mean() / self.num_classes  # Scalar

        return loss

    
    def __str__(self):
        return self.__class__.__name__