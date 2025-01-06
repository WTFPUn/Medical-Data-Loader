import logging
import json
import os
from typing import Generic, TypeVar, List, Type
from pydantic import ValidationError

from ...DataEngine import DataEngine
from ..Metric import Metric, Loss
from .ModelTrainer import ModelTrainer, TrainConfig
# from .KFold import KFold, KFoldResult
from ...DataEngine.types.dataset import DatasetConfig
from ...DataEngine.types.utils import SimpleDatasetMetaData, KFOLDDatasetMetaData
import numpy as np

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
        
        
        try:
            with open(meta_data_path, "r", encoding="utf-8") as f:
                self.meta_data = SimpleDatasetMetaData.model_validate(json.load(f))
        except ValidationError as e:
            try:
                with open(meta_data_path, "r", encoding="utf-8") as f:
                    self.meta_data = KFOLDDatasetMetaData.model_validate(json.load(f))
            except ValidationError as e:
                logger.error("Failed to load meta data from %s with error: %s", meta_data_path, e, extra={"contexts": "load meta data"})
                return
            except Exception as e:
                logger.error("Failed to load meta data from %s with error: %s", meta_data_path, e, extra={"contexts": "load meta data"})
                return
        except Exception as e:
            logger.error("Failed to load meta data from %s with error: %s", meta_data_path, e, extra={"contexts": "load meta data"})
            return
        
        if isinstance(self.meta_data, SimpleDatasetMetaData):
            self.add_trainer = self.__add_trainer
            self.run = self.__run
            
        else:
            self.add_trainer = self.__Kfold_add_trainer
            self.run = self.__Kfold_run
        
        self.data_engine = DataEngine(num_classes, self.meta_data, dataset_config, logger)
        self.train_method: List[ModelTrainer[T, U]] = []
        
        self.logger.info("Experimenting initialized.", extra={"contexts": "initialize experiment"})

        
        
    def get_train(self):
        # for test
        train_data = self.data_engine.get_dataloader("train", 1, True, 0)
        return train_data
        
    
    def __add_trainer(self, trainer: Type[ModelTrainer[T, U]], name: str, load_model_path: str | None = None, **kwargs):
        trainer = trainer(self.logger, self.num_classes, self.metrics, self.losses, name, self.device, load_model_path, **kwargs)
        self.train_method.append(trainer)
        self.logger.info(f"Added {trainer.__class__.__name__} to the experiment.", extra={"contexts": "add trainer"})
        
    def __run(self, batch_size: int, num_workers: int, train_config: TrainConfig):
        self.logger.info("Experimenting started.", extra={"contexts": "run experiment"})
        train_data = self.data_engine.get_dataloader("train", batch_size, True, num_workers)
        val_data = self.data_engine.get_dataloader("val", batch_size, False, num_workers)
        test_data = self.data_engine.get_dataloader("test", batch_size, False, num_workers)
        
        for trainer in self.train_method:
            trainer.train(train_data, val_data, train_config)
            trainer.test(test_data)

            
        self.logger.info("Experimenting finished.", extra={"contexts": "finish experiment"})
        
    def __Kfold_add_trainer(self, trainer: Type[ModelTrainer[T, U]], name: str, load_model_path: str | None = None, **kwargs):
        for i in range(self.meta_data.info.k):
            k_name =  f"{name}_{i}"
            ent_trainer = trainer(self.logger, self.num_classes, self.metrics, self.losses, k_name, self.device, load_model_path, **kwargs)
            self.train_method.append(ent_trainer)
            self.logger.info(f"Added {trainer.__class__.__name__} to the experiment for fold {i}.", extra={"contexts": "add trainer"})

    def __Kfold_run(self, batch_size: int, num_workers: int, train_config: TrainConfig, fixed_k: int | None = None):
        self.logger.info("Experimenting started.", extra={"contexts": "run experiment"})
        results = {
            "train": {
                l.__class__.__name__: [] for l in self.metrics
            },
            "val": {
                l.__class__.__name__: [] for l in self.metrics
            }
            
        }
        
        os.makedirs("trainer_result", exist_ok=True)
        
        
        if fixed_k is not None:
            k_list = [fixed_k]
        else:
            k_list = range(self.meta_data.info.k)
        
        for i in k_list:
            train_data = self.data_engine.get_dataloader_for_kfold("train", i, batch_size, True, num_workers)
            val_data = self.data_engine.get_dataloader_for_kfold("val", i, batch_size, False, num_workers)
            test_data = self.data_engine.get_dataloader_for_kfold("test", i, batch_size, False, num_workers)
            
            trainer = self.train_method[i]
            trainer.train(train_data, val_data, train_config)
            trainer.test(test_data)
            train_result = trainer.get_result()
            for metric in [str(l) for l in self.metrics]:
                results["train"][metric].append([train_result["train"][j]["train_"+metric] for j in range(len(train_result["train"]))])
                results["val"][metric].append([train_result["val"][j]["val_"+metric] for j in range(len(train_result["val"]))])
            
            with open("trainer_result/"+trainer.name+".json", "w") as f:
                json.dump(train_result, f)
                
            
            # for trainer in self.train_method:
                
            #     trainer.train(train_data, val_data, train_config)
            #     trainer.test(test_data)
            #     results.append(trainer.get_result())
                
        # map result to find mean of each metric
        
        if fixed_k is None:
            return
        
        os.makedirs("k_fold_result", exist_ok=True)
        # mean of each metric for each epoch
        metric_name = [str(l) for l in self.metrics]
        mean_results = {
            "train": {
                l: np.mean(results["train"][l], axis=0).tolist() for l in metric_name 
            },
            "val": {
                l: np.mean(results["val"][l], axis=0).tolist() for l in metric_name
            }
        }
        
        
        with open("k_fold_result/"+self.experiment_name+".json", "w") as f:
            json.dump(mean_results, f)
                

    def export_result(self, json_name: str):
        metric_result = {}
        loss_result = {}
        for trainer in self.train_method:
            metric_result[str(trainer)] = trainer.metrics_result
            loss_result[str(trainer)] = trainer.losses_result
        
        with open(json_name, "w") as f:
            json.dump({"metric": metric_result, "loss": loss_result}, f)
            
        
