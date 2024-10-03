from ..Generic import Metric

import torch

class Recall(Metric[torch.Tensor, torch.Tensor]):
    def __init__(self, num_classes: int):
        super(Recall, self).__init__()
        self.num_classes = num_classes
        self.name = "Recall (Sensitivity)"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Recall for segmentation.

        Args:
            y_pred (torch.Tensor): Predicted logits. Shape: (B, C, W, H, D)
            y_true (torch.Tensor): Ground truth labels. Shape: (B, W, H, D)

        Returns:
            float: Average Recall over all classes.
        """
        device = y_pred.device
        
        preds = torch.argmax(y_pred, dim=1)  # Shape: (B, W, H, D)
        recall = torch.tensor(0.0).to(device)
        for cls in range(self.num_classes):
            tp = ((preds == cls) & (y_true == cls)).sum().float()
            fn = ((preds != cls) & (y_true == cls)).sum().float()
            if tp + fn == 0:
                cls_recall = torch.tensor(0.0).to(device)
            else:
                cls_recall = tp / (tp + fn)
            recall += cls_recall
        recall = recall / self.num_classes
        return recall
        
    def __str__(self):
        return self.name
