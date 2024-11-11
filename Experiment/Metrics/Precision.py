from ..Generic import Metric

import torch

class Precision(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self, num_classes: int):
        super(Precision, self).__init__()
        self.num_classes = num_classes

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Precision for segmentation.

        Args:
            y_pred (torch.Tensor): Predicted logits. Shape: (B, C, W, H, D)
            y_true (torch.Tensor): Ground truth labels. Shape: (B, W, H, D)

        Returns:
            float: Average Precision over all classes.
        """
        device = y_pred.device
        
        preds = torch.argmax(y_pred, dim=1)  # Shape: (B, W, H, D)
        precision = torch.tensor(0.0).to(device)
        for cls in range(self.num_classes):
            tp = ((preds == cls) & (y_true == cls)).sum().float()
            fp = ((preds == cls) & (y_true != cls)).sum().float()
            if tp + fp == 0:
                cls_precision = torch.tensor(0.0).to(device)
            else:
                cls_precision = tp / (tp + fp)
            precision += cls_precision
        precision = precision / self.num_classes
        return precision
        
    def __str__(self):
        return self.__class__.__name__
