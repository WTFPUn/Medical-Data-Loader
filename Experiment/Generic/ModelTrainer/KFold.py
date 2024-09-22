import logging
from typing import Generic, TypeVar, List, Tuple, Dict, Any, Optional, Union, Type
from abc import ABC, abstractmethod

from .ModelTrainer import ModelTrainer, TrainerResult
from ..Metric import Metric
from ...DataEngine import MedicalDataset

import torch
from torch import nn
from torch.utils.data import Subset

T, U = TypeVar('T'), TypeVar('U')

class KFoldResult(Generic[U]):
    fold: List[TrainerResult[U]]
    avg: TrainerResult[U]

class KFold(ModelTrainer, Generic[T, U]):
    def __init__(self, logger: logging.Logger, metrics: List[Metric[T, U]], model: nn.Module, k: int):
        super().__init__(logger, metrics, model)
        self.logger = logger.getChild(self.__class__.__name__)
        self.k = k
        self.result: KFoldResult[U] = KFoldResult(fold=[TrainerResult() for _ in range(k)], avg=TrainerResult)
    
    @abstractmethod
    def train(self, dataset: Subset, weight_save_period: int) -> None:
        # train for each fold
        ...
        
    @abstractmethod
    def validate(self, dataset: Subset) -> None:
        # validate for each fold
        ...
        
    def train_folds(self, dataset: MedicalDataset, weight_save_period: int):
        for i in range(self.k):
            train_indices, val_indices = self.get_fold_indices(dataset, i)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            self.train(train_dataset, weight_save_period)
            self.validate(val_dataset)
    
    def get_fold_indices(self, dataset: MedicalDataset, fold: int) -> Tuple[List[int], List[int]]:
        n = len(dataset)
        fold_size = n // self.k
        val_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_indices = list(set(range(n)) - set(val_indices))
        return train_indices, val_indices

    