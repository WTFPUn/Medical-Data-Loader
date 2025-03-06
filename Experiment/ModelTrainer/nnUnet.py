import os
import logging
from typing import Dict, List

import numpy

from ..DataEngine import MedicalDataset
from ..Generic import ModelTrainer, TrainConfig, ContinueTrainConfig, NewTrainConfig, PatchBaseTrainer, GlobalPatchBaseTrainer
from .nnUnetModule.nnUnet import NnUnetBase
from ..Generic import Loss, Metric

import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
from torch.cuda.amp import autocast, GradScaler

generic_input, generic_output = torch.Tensor, torch.Tensor


class NnUnet(ModelTrainer[generic_input, generic_output]):
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
        self.channels_multiplier = kwargs.get("channels_multiplier", 1)
        super(NnUnet, self).__init__(
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
        return NnUnetBase(
            num_input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            trilinear=False,
            use_ds_conv=False,
            channels_multiplier=self.channels_multiplier,
        )
        
class PatchBaseNnUnet(PatchBaseTrainer[generic_input, generic_output]):
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
        self.channels_multiplier = kwargs.get("channels_multiplier", 1)
        super(PatchBaseNnUnet, self).__init__(
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
        return NnUnetBase(
            num_input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            trilinear=False,
            use_ds_conv=False,
            channels_multiplier=self.channels_multiplier,
        )
        
class GPNnUnet(GlobalPatchBaseTrainer[generic_input, generic_input]):
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
        self.channels_multiplier = kwargs.get("channels_multiplier", 1)
        super(GPNnUnet, self).__init__(
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
        return NnUnetBase(
            num_input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            trilinear=False,
            use_ds_conv=False,
            channels_multiplier=self.channels_multiplier,
        )