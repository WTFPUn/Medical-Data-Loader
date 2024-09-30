import logging
import json
from typing import Generic, TypeVar, List, Type

from ...DataEngine import DataEngine
from ..Metric import Metric, Loss
from .ModelTrainer import ModelTrainer, TrainConfig
# from .KFold import KFold, KFoldResult
from ...DataEngine.types.dataset import DatasetConfig

T, U = TypeVar('T'), TypeVar('U')

class Experimenting(Generic[T, U]):
    def __init__(self, experiment_name: str, meta_data_path: str, dataset_config: DatasetConfig, losses: List[Loss[T, U]], metrics: List[Metric[T, U]], num_classes: int, logger: logging.Logger):
        self.experiment_name = experiment_name
        self.logger = logger.getChild(self.__class__.__name__+"."+experiment_name)
        self.meta_data_path = meta_data_path
        self.dataset_config = dataset_config
        self.device = dataset_config.device
        self.num_classes = num_classes
        
        self.metrics = metrics
        self.losses = losses
        
        self.data_engine = DataEngine(num_classes, meta_data_path, dataset_config, logger)
        self.train_method: List[ModelTrainer[T, U]] = []
        
        self.logger.info("Experimenting initialized.", extra={"contexts": "initialize experiment"})
    
    def get_train(self):
        # for test
        train_data = self.data_engine.get_dataloader("train", 1, True, 0)
        return train_data
        
    
    def add_trainer(self, trainer: Type[ModelTrainer[T, U]], name: str, load_model_path: str | None = None, **kwargs):
        trainer = trainer(self.logger, self.num_classes, self.metrics, self.losses, name, self.device, load_model_path, **kwargs)
        self.train_method.append(trainer)
        self.logger.info(f"Added {trainer.__class__.__name__} to the experiment.", extra={"contexts": "add trainer"})
        
    def run(self, batch_size: int, num_workers: int, train_config: TrainConfig):
        self.logger.info("Experimenting started.", extra={"contexts": "run experiment"})
        train_data = self.data_engine.get_dataloader("train", batch_size, True, num_workers)
        val_data = self.data_engine.get_dataloader("val", batch_size, False, num_workers)
        test_data = self.data_engine.get_dataloader("test", batch_size, False, num_workers)
        
        for trainer in self.train_method:
            trainer.train(train_data, val_data, train_config)
            trainer.test(test_data)
            
        self.logger.info("Experimenting finished.", extra={"contexts": "finish experiment"})

    def export_result(self, json_name: str):
        metric_result = {}
        loss_result = {}
        for trainer in self.train_method:
            metric_result[str(trainer)] = trainer.metrics_result
            loss_result[str(trainer)] = trainer.losses_result
        
        with open(json_name, "w") as f:
            json.dump({"metric": metric_result, "loss": loss_result}, f)
            
        
