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
                        cum_loss += loss.item()

                        metric_values = self.calculate_metrics(output, target[j])
                        cum_train_metric = {
                            k: v.detach().cpu().numpy() + cum_train_metric.get(k, 0)
                            for k, v in metric_values.items()
                        }
                        gc.collect()
                        pbar.update(1)
                        
                        if pbar.n % accumulation_steps == 0:
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

    @staticmethod
    def reconstruct_from_patches(patches: torch.Tensor, positions: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Reconstructs the original 4D tensor from patches and their positions.

        Parameters:
            patches (torch.Tensor): Patched tensor of shape (B, P, 1, patchH, patchW, patchD).
            positions (torch.Tensor): Positions tensor of shape (B, P, 3).
            original_shape (tuple): The desired final shape (C, finalH, finalW, finalD).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, C, finalH, finalW, finalD).
        """
        B, P, C, patchH, patchW, patchD = patches.shape
        finalH, finalW, finalD = original_shape
        C_final = 1

        reconstructed_tensor = torch.zeros((B, C_final, finalH, finalW, finalD),
                                        dtype=patches.dtype,
                                        device=patches.device)

        for b in range(B):
            for i, (x, y, z) in enumerate(positions[b]):
                reconstructed_tensor[b, :,
                                    x * patchH : (x + 1) * patchH,
                                    y * patchW : (y + 1) * patchW,
                                    z * patchD : (z + 1) * patchD] = patches[b, i]
        return reconstructed_tensor

    def infer(self, data: torch.Tensor, positions: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # shape is (P, B, C, H, W, D)
            data = data.to(self.device, non_blocking=True)
            
            # reshape from (B, P, C, H, W, D) to (P, B, C, H, W, D)
            data = data.permute(1, 0, 2, 3, 4, 5)
            outputs = []
            for i in range(data.size(0)):
                output = self.model(data[i])
                # softmax
                output = F.softmax(output, dim=1)
                # argmax
                output = torch.argmax(output, dim=1)
                outputs.append(output)
            
            # Stack outputs to get shape (P, B, H, W, D)
            outputs = torch.stack(outputs, dim=0)
            
            # Permute back to (B, P, H, W, D)
            outputs = outputs.permute(1, 0,  2, 3, 4)
            
            # Reconstruct the original shape from patches
            reconstructed_outputs = []
            for i in range(outputs.size(0)):
                reconstructed_output = self.reconstruct_patchify(outputs[i], positions[i], original_shape)
                reconstructed_outputs.append(reconstructed_output)
            
            # Stack reconstructed outputs to get final shape (B, H, W, D)
            final_output = torch.stack(reconstructed_outputs, dim=0)

            
            return final_output

    def reconstruct_patchify(self, patches: torch.Tensor, positions: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Reconstructs the original 4D tensor from patches and their positions.

        Parameters:
            patches (torch.Tensor): Patched tensor of shape (P, patch_size, patch_size, patch_size).
            positions (torch.Tensor): Positions tensor of shape (P, 3), where each entry is (x, y, z) position.
            original_shape (tuple): Original shape of the tensor (C, H, W, D).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (C, H, W, D).
        """
        H, W, D = original_shape
        patch_size = patches.shape[1]

        # Initialize an empty tensor for reconstruction
        reconstructed_tensor = torch.zeros((H, W, D), dtype=patches.dtype, device=patches.device)

        # Iterate over patches and their positions to reconstruct the original tensor
        for i, (x, y, z) in enumerate(positions):
            reconstructed_tensor[
                                 x * patch_size: (x + 1) * patch_size, 
                                 y * patch_size: (y + 1) * patch_size, 
                                 z * patch_size: (z + 1) * patch_size] = patches[i]

        return reconstructed_tensor

    def test(self, test: DataLoader):
        self.model.eval()
        self.logger.info("Testing started.", extra={"contexts": "start testing"})
        to_return = {
            "metrics": {},
            "output": [],
            "input": [],
            "ground_truth": [],
            "infer_time": []
        }
        for i, (idx, input, target, position) in enumerate(test):
            start = perf_counter()
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            input = input.permute(1, 0, 2, 3, 4, 5)
            target = target.permute(1, 0, 2, 3, 4, 5)
            for j in range(input.size(0)):
                output = self.model(input[j])
                metric_values = {
                    k: v.detach().cpu().numpy()
                    for k, v in self.calculate_metrics(output, target[j]).items()
                }
                to_return["output"].append(output.detach().cpu().numpy())
                to_return["ground_truth"].append(target[j].detach().cpu().numpy())
                to_return["input"].append(input[j].detach().cpu().numpy())
                for k, v in metric_values.items():
                    to_return["metrics"][k] = to_return["metrics"].get(k, []) + [v]
                gc.collect()
            end = perf_counter()
            to_return["infer_time"].append(end - start)
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
