import os
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Type

from ..Metric import Loss, Metric
from ...DataEngine import MedicalDataset

from torch import nn
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import wandb

T, U = TypeVar("T"), TypeVar("U")

__all__ = [
    "ModelTrainer",
    "TrainerLoss",
    "TrainerMetric",
    "NewTrainConfig",
    "ContinueTrainConfig",
    "TrainConfig",
]


@dataclass
class NewTrainConfig:
    accumulation_steps: int
    lr: float
    weight_save_period: int
    epoch: int


@dataclass
class ContinueTrainConfig:
    accumulation_steps: int
    lr: float
    weight_save_period: int
    epoch: int
    project_name: str
    run_id: str
    current_epoch: int = 0


TrainConfig = NewTrainConfig | ContinueTrainConfig


@dataclass
class TrainerLoss(Generic[U]):
    train: List[U]
    val: List[U]
    test: List[U]


@dataclass
class TrainerMetric(Generic[U]):
    train: Dict[str, List[U]]
    val: Dict[str, List[U]]
    test: Dict[str, List[U]]


class ModelTrainer(ABC, Generic[T, U]):
    """
    A base class for training, validating, and inferring models.

    Args:
        logger (logging.Logger): The logger object for logging messages.
        losses (List[Loss[T, U]]): A list of losses to evaluate the model's performance.
        model (nn.Module): The model to be trained, validated, and inferred.

    Attributes:
        logger (logging.Logger): The logger object for logging messages.
        losses (List[Loss[T, U]]): A list of losses to evaluate the model's performance.
        model (nn.Module): The model to be trained, validated, and inferred.
        result (TrainerLoss): The result of the training process.


    """

    @abstractmethod
    def set_model(self) -> nn.Module:
        """
        Returns the model used for training.

        Returns:
            nn.Module: The model used for training.
        """
        pass

    def __init__(
        self,
        logger: logging.Logger,
        num_classes: int,
        metrics: List[Metric[T, U]],
        losses: List[Loss[T, U]],
        name: str,
        device: torch.device = "cpu",
        load_model_path: str | None = None,
        **kwargs
    ):
        self.logger = logger.getChild(self.__class__.__name__)
        self.num_classes = num_classes
        self.losses = losses
        self.metrics = metrics
        self.device = device
        # self.losses_result = TrainerLoss(train=[], val=[], test=[])
        # self.metrics_result = TrainerMetric(train={}, val={}, test={})
        self.name = name

        if load_model_path is not None:
            self.model = self.set_model()
            self.model.load_state_dict(torch.load(load_model_path))
        else:
            self.model = self.set_model()
        self.model.to(self.device)

        self.path_to_save = os.path.join(*self.logger.name.split(".")[1:], self.name)
        os.makedirs(self.path_to_save, exist_ok=True)

    def calculate_loss(self, output: T, target: T) -> U:
        """
        Calculate the loss of the model.

        Args:
            output (T): The output of the model.
            target (T): The target of the model.

        Returns:
            U: The loss of the model.

        """
        loss = 0
        for loss_fn in self.losses:
            loss += loss_fn(output, target)
        return loss

    def calculate_metrics(self, output: T, target: T) -> Dict[str, U]:
        """
        Calculate the metrics of the model.

        Args:
            output (T): The output of the model.
            target (T): The target of the model.

        Returns:
            Dict[str, U]: The metrics of the model.

        """
        metric_values = {}
        for metric in self.metrics:
            metric_values[str(metric)] = metric(output, target)
        return metric_values

    @abstractmethod
    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        config: TrainConfig,
    ) -> None:
        pass

    def infer(self, data: torch.Tensor) -> torch.Tensor:
        """
        Infer the model on the given data.

        Args:
            data (torch.Tensor): The data to infer.

        Returns:
            torch.Tensor: The output of the model.

        """
        self.model.eval()
        with torch.no_grad():
            return self.model(data)

    @abstractmethod
    def test(self, test: DataLoader) -> None:
        """
        Test the model on the given test set.

        Args:
            test (MedicalDataset): The test set to test the model on.

        Returns:
            None

        """
        pass

    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.

        Returns:
            None

        """
        torch.save(self.model.state_dict(), path)
