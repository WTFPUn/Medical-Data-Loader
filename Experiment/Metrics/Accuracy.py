from ..Generic import Metric

import torch

class Accuracy(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self):
        self.name = "Accuracy"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the accuracy of the predictions.

        Args:
            y_pred (torch.Tensor): The predicted logits tensor (B, C, D, H, W).
            y_true (torch.Tensor): The ground truth tensor (B, D, H, W) with class indices.

        Returns:
            torch.Tensor: The accuracy value.
        """
        # Get the predicted class indices
        y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: (B, D, H, W)
        
        # Flatten the tensors
        y_pred_flat = y_pred_classes.view(-1)  # Shape: (B * D * H * W)
        y_true_flat = y_true.view(-1)  # Shape: (B * D * H * W)

        # Calculate accuracy
        correct = (y_pred_flat == y_true_flat).sum().item()
        total = y_true_flat.numel()
        accuracy = correct / total if total > 0 else 0

        return torch.tensor(accuracy)

    def __str__(self):
        return "Accuracy"
