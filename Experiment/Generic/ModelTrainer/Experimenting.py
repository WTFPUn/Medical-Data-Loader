import logging
from typing import Generic, TypeVar, List

from ...DataEngine import DataEngine
from ..Metric import Metric
from .ModelTrainer import ModelTrainer, TrainerResult
# from .KFold import KFold, KFoldResult
from ...DataEngine.types.dataset import DatasetConfig

T, U = TypeVar('T'), TypeVar('U')

class Experimenting(Generic[T, U]):
    def __init__(self, meta_data_path: str, dataset_config: DatasetConfig, metrics: List[Metric[T, U]], logger: logging.Logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.meta_data_path = meta_data_path
        self.dataset_config = dataset_config
        self.metrics = metrics
        
        self.data_engine = DataEngine(meta_data_path, dataset_config, logger)
        self.train_method: List[ModelTrainer[T, U]] = []
        
        self.logger.info("Experimenting initialized.", extra={"contexts": "initialize experiment"})
    
    def add_trainer(self, trainer: ModelTrainer[T, U]):
        trainer.add_config(self.logger, self.dataset_config.device)
        self.train_method.append(trainer)
        
        self.logger.info(f"Added {trainer.__class__.__name__} to the experiment.", extra={"contexts": "add trainer"})
        
    def run(self):
        self.logger.info("Experimenting started.", extra={"contexts": "run experiment"})
        train_data = self.data_engine.get_data("train")
        val_data = self.data_engine.get_data("val")
        
        
