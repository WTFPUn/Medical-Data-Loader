import os
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Type
import gc
from time import perf_counter

from ..Metric import Loss, Metric
from ...DataEngine import MedicalDataset

from torch import nn
from dataclasses import dataclass

import numpy
import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
from torch.cuda.amp import autocast, GradScaler
from .DataClass import TrainConfig, TrainerLoss, TrainerMetric, NewTrainConfig, ContinueTrainConfig
import platform
from torch.nn import functional as F

T, U = TypeVar("T"), TypeVar("U")

__all__ = [
    "PatchBaseTrainer",
]

class PatchBaseTrainer(ABC, Generic[T, U]):
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
        
        self.result = {
            "train": [],
            "val": [],
        }
        
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


    def get_result(self):
        return self.result

    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        train_config: TrainConfig,
        total_iter: int,
        total_patch: int
    ) -> None:
        if isinstance(train_config, NewTrainConfig):
            run = wandb.init(
                name=self.name,
                project="FirstPartSeniorProject",
                config={
                    "learning-rate": train_config.lr,
                    "experiment_name": self.name,
                    "architecture": self.__class__.__name__,
                    "epoch": train_config.epoch,
                    "accumulation_steps": train_config.accumulation_steps,
                    "optimizer": train_config.optimizer.__name__,
                    "losses": [str(loss) for loss in self.losses],
                },
            )
            ep_range = range(train_config.epoch)
        elif isinstance(train_config, ContinueTrainConfig):
            run = wandb.init(
                project="FirstPartSeniorProject",
                name=self.name,
                id=train_config.run_id,
                config={
                    "learning-rate": train_config.lr,
                    "experiment_name": self.name,
                    "architecture": self.__class__.__name__,
                    "epoch": train_config.epoch,
                    "accumulation_steps": train_config.accumulation_steps,
                },
                resume="allow",
            )
            ep_range = range(
                train_config.current_epoch,
                train_config.epoch + train_config.current_epoch,
            )
            self.model.load_state_dict(torch.load(train_config.model_path))
        else:
            raise TypeError("Invalid train_config type")

        optimizer = train_config.optimizer(self.model.parameters(), lr=train_config.lr)
        # scaler = torch.amp.GradScaler('cuda')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train) * train_config.epoch
        )

        accumulation_steps = train_config.accumulation_steps
        # run.watch(self.model,)
        

        # Compile the optimizer's step function
        if platform.system() == "Linux":
            @torch.compile
            def compiled_step():
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad()
        else:
            def compiled_step():
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad()

        for ep in ep_range:
            self.model.train()
            cum_loss = 0
            cum_train_metric: Dict[str, numpy.ndarray] = {}

            optimizer.zero_grad()

            with tqdm.tqdm(total=total_iter) as pbar:
                for i, (idx, input, target, position) in enumerate(train):
                    input = input.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    # reshape from (B, P, C, H, W, D) to (P, B, C, H, W, D)
                    input = input.permute(1, 0, 2, 3, 4, 5)
                    target = target.permute(1, 0, 2, 3, 4, 5)
                    
                    for j in range(input.size(0)):
                        output = self.model(input[j])
                        try:
                            loss = self.calculate_loss(output, target[j]) / accumulation_steps
                        except RuntimeError as e:
                            self.logger.error(f"Runtime error calculating loss at iteration {i}: {e}")
                            continue
                        if loss is not None and torch.isfinite(loss):
                            loss.backward()
                        cum_loss += loss.item() * accumulation_steps

                        metric_values = self.calculate_metrics(output, target[j])
                        cum_train_metric = {
                            k: v.detach().cpu().numpy() + cum_train_metric.get(k, 0)
                            for k, v in metric_values.items()
                        }
                        gc.collect()
                        pbar.update(1)
                        
                        if (i + 1) % accumulation_steps == 0:
                            compiled_step()
                    else:
                        # run after the loop
                        compiled_step()
                        
            train_output = {
                f"train_{k}": v / (len(train) * total_patch)
                for k, v in cum_train_metric.items()
            }

            self.model.eval()
            cum_val_metric = {}

            with torch.no_grad():
                for i, (idx, input, target, position) in enumerate(val):
                    input = input.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    # reshape from (B, P, C, H, W, D) to (P, B, C, H, W, D)
                    input = input.permute(1, 0, 2, 3, 4, 5)
                    target = target.permute(1, 0, 2, 3, 4, 5)
                    
                    for j in range(input.size(0)):
                        output = self.model(input[j])
                        metric_values = self.calculate_metrics(output, target[j])
                        cum_val_metric = {
                            k: v.detach().cpu().numpy() + cum_val_metric.get(k, 0)
                            for k, v in metric_values.items()
                        }
                        gc.collect()

            val_output = {
                f"val_{k}": v / (len(val) * total_patch)
                for k, v in cum_val_metric.items()
            }
            
            # merge train and val output
            merged_output = train_output | val_output
            wandb.log(merged_output)
            self.result["train"].append(train_output)
            self.result["val"].append(val_output)

            scheduler.step()  # Step the scheduler after each epoch

            # Save model weights periodically
            if ep % train_config.weight_save_period == 0 or ep == ep_range[-1]:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.path_to_save, f"model_{ep}.pth"),
                )
                
                
        run.finish()

        self.logger.info("Training finished.", extra={"contexts": "finish training"})

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

    def test(self, test: DataLoader):
        """
        Test the model on the given test set.

        Args:
            test (MedicalDataset): The test set to test the model on.

        Returns:
            None

        """
        self.model.eval()
        self.logger.info("Testing started.", extra={"contexts": "start testing"})
        
        to_return = {
            "metrics": {},
            # output is the returned value from the model
            "output": [],
            "input": [],
            "ground_truth": [],
            "infer_time": []
        }
        
        for i, (idx, input, target) in enumerate(test):
            start = perf_counter()
            output = self.model(input.to(self.device, non_blocking=True))
            end = perf_counter()
            target = target.to(self.device, non_blocking=True)
            
            # loss = self.calculate_loss(output, target).item()
            metric_values = {
                k: v.detach().cpu().numpy()
                for k, v in self.calculate_metrics(output, target).items()
            }

            # self.logger.info(
            #     f"Iteration {i}, imgid: {idx}, Metrics: {metric_values},",
            #     extra={"contexts": "test"},
            # )
            
            to_return["output"].append(output.detach().cpu().numpy())
            to_return["ground_truth"].append(target.detach().cpu().numpy())
            to_return["input"].append(input.detach().cpu().numpy())
            to_return["infer_time"].append(end - start)          
            for k, v in metric_values.items():
                to_return["metrics"][k] = to_return["metrics"].get(k, []) + [v]

        self.logger.info("Testing finished.", extra={"contexts": "finish testing"})
        
        return to_return
        
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.

        Returns:
            None

        """
        torch.save(self.model.state_dict(), path)
