from ..Generic import Metric

import torch

class Recall(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self):
        self.eps = 1e-6
        self.name = "Recall"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate recall for each class.

        Args:
            y_pred (torch.Tensor): The predicted logits tensor (B, C, H, W, D).
            y_true (torch.Tensor): The ground truth tensor (B, H, W, D) with class indices.

        Returns:
            torch.Tensor: The recall value.
        """
        # Get the predicted class indices
        y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: (B, H, W, D)
        
        # Flatten the tensors
        y_pred_flat = y_pred_classes.view(-1)  # Shape: (B * H * W * D)
        y_true_flat = y_true.view(-1)  # Shape: (B * H * W * D)

        # Calculate True Positives, False Negatives
        tp = (y_pred_flat * y_true_flat).sum().item()
        fn = ((y_pred_flat == 0) & (y_true_flat == 1)).sum().item()

        recall = tp / (tp + fn + self.eps)
        
        return torch.tensor(recall)

    def __str__(self):
        return "Recall"
