import os
import logging
from typing import Dict, List

import numpy

from ..Generic import ModelTrainer
from ..Generic import Loss, Metric

from monai.networks.nets.swin_unetr import SwinUNETR

import torch
generic_input, generic_output = torch.Tensor, torch.Tensor


class SwinUNETRTrainer(ModelTrainer[generic_input, generic_output]):
    def __init__(
        self,
        logger: logging.Logger,
        num_classes: int,
        metrics: List[Metric[generic_input, generic_output]],
        losses: List[Loss[generic_input, generic_output]],
        name: str,
        device: torch.device,
        load_model_path: str | None,
        **kwargs,
    ):
        self.num_input_channels = kwargs.get("num_input_channels", 1)
        self.depths = kwargs.get("depths", (2,4,2,2))
        self.img_size = kwargs.get("img_size", (128, 128, 128))
        
        super(SwinUNETRTrainer, self).__init__(
            logger,
            num_classes,
            metrics,
            losses,
            name,
            device,
            load_model_path,
            **kwargs,
        )

    def set_model(self) -> torch.nn.Module:
        return SwinUNETR(img_size=self.img_size, in_channels=self.num_input_channels, out_channels=self.num_classes, depths=self.depths)