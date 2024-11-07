import os
import logging
from typing import Dict, List

import numpy

from ..DataEngine import MedicalDataset
from ..Generic import ModelTrainer, TrainConfig, ContinueTrainConfig, NewTrainConfig
from .nnUnetModule.nnUnet import NnUnetBase
from ..Generic import Loss, Metric

import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


generic_input, generic_output = torch.Tensor, torch.Tensor


class ResEncUnet(ModelTrainer[generic_input, generic_output]):
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
        self.n_stages = kwargs.get("n_stages", 6)
        self.features_per_stage = kwargs.get("features_per_stage", (32, 64, 128, 256, 320, 320))
        self.conv_op = kwargs.get("conv_op", nn.Conv3d)
        self.kernel_sizes = kwargs.get("kernel_sizes", 3)
        self.strides = kwargs.get("strides", (1, 2, 2, 2, 2, 2))
        self.n_blocks_per_stage = kwargs.get("n_blocks_per_stage", (1, 3, 4, 6, 6, 6))
        self.n_conv_per_stage_decoder = kwargs.get("n_conv_per_stage_decoder", (1, 1, 1, 1, 1))
        self.conv_bias = kwargs.get("conv_bias", True)
        self.norm_op = kwargs.get("norm_op", nn.InstanceNorm3d)
        self.norm_op_kwargs = kwargs.get("norm_op_kwargs", {})
        self.dropout_op = kwargs.get("dropout_op", None)
        self.nonlin = kwargs.get("nonlin", nn.LeakyReLU)
        self.nonlin_kwargs = kwargs.get("nonlin_kwargs", {'inplace': True})
        self.deep_supervision = kwargs.get("deep_supervision", False)
        
        super(ResEncUnet, self).__init__(
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
        return ResidualEncoderUNet(input_channels=self.num_input_channels, n_stages=self.n_stages, features_per_stage=self.features_per_stage,
                              conv_op=self.conv_op, kernel_sizes=self.kernel_sizes, strides=self.strides,
                              n_blocks_per_stage=self.n_blocks_per_stage, num_classes=self.num_classes,
                              n_conv_per_stage_decoder=self.n_conv_per_stage_decoder,
                              conv_bias=self.conv_bias, norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs, dropout_op=self.dropout_op,
                              nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs, deep_supervision=self.deep_supervision)