from ..Generic import Loss
import torch
import torch.nn.functional as F

class FocalLoss(Loss[torch.Tensor, torch.Tensor]):
    def __init__(self, alpha, gamma=2.0, reduction='none', eps=None, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.num_classes = num_classes
        
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        if target.dtype != torch.long:
            target = target.long()
            
        if output.dtype != torch.float32:
            output = output.float()
            
            
        target = target.squeeze(1)
        output_softmax = F.softmax(output, dim=1)
        output_log_softmax = F.log_softmax(output, dim=1)
        
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=self.num_classes).permute((0, 4, 1, 2, 3)).contiguous().to(torch.float32)
        weight = torch.pow(1.0 - output_softmax, self.gamma)
        focal = -self.alpha * weight * output_log_softmax
        # This line is very useful, must learn einsum, bellow line equivalent to the commented line
        # loss_tmp = torch.sum(focal.to(torch.float) * one_hot_target.to(torch.float), dim=1)
        loss_tmp = torch.einsum('bc..., bc...->b...', one_hot_target, focal)
        if self.reduction == 'none':
            return loss_tmp
        elif self.reduction == 'mean':
            return torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            return torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")

    def __str__(self):
        return self.__class__.__name__
