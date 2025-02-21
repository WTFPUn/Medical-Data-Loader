from dataclasses import dataclass
from typing import List, Dict, Type, Generic, TypeVar

import torch


T, U = TypeVar("T"), TypeVar("U")



###########################
# from: https://peps.python.org/pep-3102/
###########################
@dataclass(kw_only=True)
class NewTrainConfig:
    accumulation_steps: int
    lr: float
    weight_save_period: int
    epoch: int
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam


@dataclass(kw_only=True)
class ContinueTrainConfig(NewTrainConfig):
    project_name: str
    run_id: str
    current_epoch: int = 0
    model_path: str = ""


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
