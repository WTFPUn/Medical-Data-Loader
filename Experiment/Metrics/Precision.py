from ..Generic import Metric

import torch

class Precision(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self):
        self.eps = 1e-6
        self.name = "Precision"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate precision for each class.

        Args:
            y_pred (torch.Tensor): The predicted logits tensor (B, C, H, W, D).
            y_true (torch.Tensor): The ground truth tensor (B, H, W, D) with class indices.

        Returns:
            torch.Tensor: The precision value.
        """
        # Get the predicted class indices
        y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: (B, H, W, D)
        
        # Flatten the tensors
        y_pred_flat = y_pred_classes.view(-1)  # Shape: (B * H * W * D)
        y_true_flat = y_true.view(-1)  # Shape: (B * H * W * D)

        # Calculate True Positives, False Positives
        tp = (y_pred_flat * y_true_flat).sum().item()
        fp = ((y_pred_flat == 1) & (y_true_flat == 0)).sum().item()

        precision = tp / (tp + fp + self.eps)
        
        return torch.tensor(precision)

    def __str__(self):
        return "Precision"
