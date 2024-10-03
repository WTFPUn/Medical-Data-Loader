from ..Generic import Metric

import torch

class Accuracy(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.name = "Accuracy"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Accuracy for segmentation.

        Args:
            y_pred (torch.Tensor): Predicted logits. Shape: (B, C, W, H, D)
            y_true (torch.Tensor): Ground truth labels. Shape: (B, W, H, D)

        Returns:
            float: Accuracy score.
        """
        preds = torch.argmax(y_pred, dim=1)  # Shape: (B, W, H, D)
        correct = (preds == y_true).float()
        accuracy = correct.sum() / correct.numel()
        return accuracy
        
    def __str__(self):
        return self.name
