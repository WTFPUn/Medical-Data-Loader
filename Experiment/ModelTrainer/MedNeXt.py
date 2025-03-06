import os
import logging
from typing import Dict, List

import numpy

from ..DataEngine import MedicalDataset
from ..Generic import ModelTrainer, TrainConfig, ContinueTrainConfig, NewTrainConfig, PatchBaseTrainer
from .nnunet_mednext import create_mednext_v1
from ..Generic import Loss, Metric

import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
from torch.cuda.amp import autocast, GradScaler

generic_input, generic_output = torch.Tensor, torch.Tensor


class MedNeXt(ModelTrainer[generic_input, generic_output]):
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
        self.model_id = kwargs.get("model_id", "S")
        super(MedNeXt, self).__init__(
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
        return create_mednext_v1(
            num_input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            model_id=self.model_id,  # S, B, M and L are valid model ids
            kernel_size=3,  # 3x3x3 and 5x5x5 were tested in publication
            deep_supervision=False,  # True or False
        )

class PatchBaseMedNeXt(PatchBaseTrainer[generic_input, generic_output]):
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
        self.model_id = kwargs.get("model_id", "S")
        super(PatchBaseMedNeXt, self).__init__(
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
        return create_mednext_v1(
            num_input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            model_id=self.model_id,  # S, B, M and L are valid model ids
            kernel_size=3,  # 3x3x3 and 5x5x5 were tested in publication
            deep_supervision=False,  # True or False
        )