from ..Generic import Loss

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(Loss[torch.Tensor, torch.Tensor]):
    """
    PolyLoss is a custom loss function that combines cross-entropy loss with an additional term to enhance the learning process.
        softmax (bool): If True, applies softmax to the predictions. Default is True.
        ce_weight (Optional[torch.Tensor]): A manual rescaling weight given to each class. Default is None.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
        epsilon (float): A small constant to control the contribution of the additional term. Default is 1.0.
    Methods:
        __call__(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            Computes the PolyLoss between the predicted and true labels.
                y_pred (torch.Tensor): The predicted labels. Shape should be BNH[WD], where N is the number of classes.
                y_true (torch.Tensor): The true labels. If in one-hot format, shape should be BNH[WD]. If not, shape should be B1H[WD] or BH[WD].
            Returns:
                torch.Tensor: The computed PolyLoss.
                ValueError: If `self.reduction` is not one of ["mean", "sum", "none"].
        __str__() -> str:
            Returns the name of the class.
    """
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the PolyLoss between the predicted and true labels.
            y_pred (torch.Tensor): The predicted values with shape (B, C, H, W, D) where B is the batch size,
                                   C is the number of classes, and H, W, D are spatial dimensions.
                                   The predictions can be logits or probabilities. If logits are provided,
                                   softmax will be applied.
            y_true (torch.Tensor): The ground truth values. If in one-hot format, the shape should be (B, C, H, W, D).
                                   If not one-hot encoded, the shape should be (B, 1, H, W, D) or (B, H, W, D) with
                                   binary values.
        Returns:
            torch.Tensor: The computed PolyLoss. The shape depends on the reduction method applied.
            ValueError: If `self.reduction` is not one of ["mean", "sum", "none"].
       """
        if y_true.dtype != torch.long:
            y_true = y_true.long()
        
        if y_pred.dtype != torch.float32:
            y_pred = y_pred.float()
       
        y_true = y_true.view(y_pred.size(0), -1)
        # y_true: (B, W*H*D)

        y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        # y_pred: (B, C, W*H*D)

        # Apply softmax to predictions along the class dimension
        y_pred = F.softmax(y_pred, dim=1)
        # y_pred: (B, C, W*H*D)

        # Convert y_true to one-hot encoding
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 2, 1).float()
        # y_true_one_hot: (B, W*H*D, C) -> permute to (B, C, W*H*D)

        # Calculate the probability of the true class
        pt = (y_true_one_hot * y_pred).sum(dim=1)
        # pt: (B, W*H*D)

        # Compute cross-entropy loss
        ce_loss = self.cross_entropy(y_pred, y_true.squeeze(1))
        # ce_loss: (B, W*H*D)

        # Compute PolyLoss
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        # poly_loss: (B, W*H*D)

        # Apply reduction method
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss
    
    def __str__(self):
        return self.__class__.__name__