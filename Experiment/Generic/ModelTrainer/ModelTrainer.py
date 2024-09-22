import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict

from ..Metric import Loss
from ...DataEngine import MedicalDataset

from torch import nn
import torch

T, U = TypeVar('T'), TypeVar('U')

__all__ = ["ModelTrainer", "TrainerResult"]

class TrainerResult(Generic[U]):
    train: Dict[str, List[U]]
    val: Dict[str, List[U]]

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
        result (TrainerResult): The result of the training process.

    """

    def __init__(self, logger: logging.Logger, losses: List[Loss[T, U]], model: nn.Module):
        self.logger = logger.getChild(self.__class__.__name__)
        self.losses = losses
        self.device = "cpu"
        self.model = model
        self.result = TrainerResult()
        
    def add_config(self, logger: logging.Logger, device: torch.device) -> None:
        """
        Add configurations to the model trainer.

        Args:
            logger (logging.Logger): The logger object for logging messages.
            device (torch.device): The device to run the model.

        Returns:
            None

        """
        self.logger = logger.getChild(self.__class__.__name__)
        self.device = device
        self.model.to(device)
        self.logger.info("Model trainer configurations added.", extra={"contexts": "add configurations"})

    @abstractmethod
    def train(self, train: MedicalDataset, val: MedicalDataset, weight_save_period: int) -> None:
        pass


    @abstractmethod
    def infer(self, dataset: MedicalDataset) -> None:
        pass
    
    @abstractmethod
    def evaluate(self, dataset: MedicalDataset) -> None:
        pass

    def get_result(self):
        """
        Get the result of the training process.

        Returns:
            TrainerResult: The result of the training process.

        """
        return self.result

    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.

        Returns:
            None

        """
        torch.save(self.model.state_dict(), path)