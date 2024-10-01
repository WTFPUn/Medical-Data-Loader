from math import e
import os
import logging
from typing import Dict, List

import numpy

from ..DataEngine import MedicalDataset
from ..Generic import ModelTrainer, TrainConfig, ContinueTrainConfig, NewTrainConfig
from .nnunet_mednext import create_mednext_v1
from ..Generic import Loss, Metric

import torch
from torch.utils.data import DataLoader
import wandb
import tqdm

generic_input, generic_output = torch.Tensor, torch.Tensor


class MedNeXt(ModelTrainer[generic_input, generic_output]):
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
        self.model_id = kwargs.get("model_id", "S")
        super(MedNeXt, self).__init__(
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
        return create_mednext_v1(
            num_input_channels=self.num_input_channels,
            num_classes=self.num_classes,
            model_id=self.model_id,  # S, B, M and L are valid model ids
            kernel_size=3,  # 3x3x3 and 5x5x5 were tested in publication
            deep_supervision=False,  # True or False
        )

    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        train_config: TrainConfig,
    ) -> None:
        
        if isinstance(train_config, NewTrainConfig):
            run = wandb.init(
            project="SeniorProject",
            config={
                "learning-rate": train_config.lr,
                "experiment_name": self.name,
                "architecture": self.__class__.__name__,
                "epoch": train_config.epoch,
                "accumulation_steps": train_config.accumulation_steps 
            }
            )
            ep_range = range(train_config.epoch)
        elif isinstance(train_config, ContinueTrainConfig):
            run = wandb.init(
            project="SeniorProject",
            id=train_config.run_id,
            config={
                "learning-rate": train_config.lr,
                "experiment_name": self.name,
                "architecture": self.__class__.__name__,
                "epoch": train_config.epoch,
                "accumulation_steps": train_config.accumulation_steps 
            },
            resume="allow"
        )
            ep_range = range(train_config.current_epoch, train_config.epoch + train_config.current_epoch)

        optimizer = train_config.optimizer(self.model.parameters(), lr=train_config.lr)
        run.watch(self.model, log="all")
        
        for ep in ep_range:
            # Training loop
            self.model.train()
            cum_loss = 0
            cum_train_metric: List[Dict[str, numpy.ndarray]] = []
            cum_val_metric: List[Dict[str, numpy.ndarray]] = []

            optimizer.zero_grad()  # Zero out gradients before starting
            for i, (idx, input, target) in tqdm.tqdm(enumerate(train), total=len(train)):
                with torch.amp.autocast(self.device):  # Mixed precision context
                    output = self.model(input)

                    # softmax
                    output = torch.nn.functional.softmax(output, dim=1)

                    loss = (
                        self.calculate_loss(output, target) / accumulation_steps
                    )  # Scale loss by accumulation steps
                    cum_loss += loss.item()

                # Backward pass with scaled gradients
                scaler.scale(loss).backward()

                # Accumulate gradients and step every 'accumulation_steps' iterations
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)  # Apply scaled gradients to optimizer
                    scaler.update()  # Update scaler for next iteration
                    optimizer.zero_grad()  # Reset gradients after each step
                
                metric_values = { k: v.detach().cpu().numpy() for k, v in self.calculate_metrics(output, target).items() }
                metric_values["loss"] = loss.item()
                
                # wandb.log(metric_values)
                cum_train_metric.append(metric_values)

                # Free memory
                del output, loss, metric_values
                
            train_output = {k: sum([m[k] for m in cum_train_metric]) / len(cum_train_metric) for k in cum_train_metric[0].keys()} 
            
            self.model.eval()

            # Validation loop
            with torch.no_grad():
                for i, (idx, input, target) in tqdm.tqdm(enumerate(val), total=len(val)):
                    output = self.model(
                        input
                    ).detach()  # Detach the tensor to prevent gradient computation
                    
                    metric_values = { k: v.detach().cpu().numpy() for k, v in self.calculate_metrics(output, target).items() }

                    cum_val_metric.append(metric_values)

                    self.logger.info(
                        f"Epoch {ep}, Iteration {i}, imgid: {idx}, Metrics: {metric_values}",
                        extra={"contexts": "val"},
                    )
                    # Free memory
                    del output, metric_values
                    
            val_output = {k: sum([m[k] for m in cum_val_metric]) / len(cum_val_metric) for k in cum_val_metric[0].keys()}
            # merge the val and train output by adding the prefix "train_" and "val_" to the keys
            merged_output = {f"train_{k}": v for k, v in train_output.items()}
            merged_output.update({f"val_{k}": v for k, v in val_output.items()})
            wandb.log(merged_output)
            
            scheduler.step()  # Step the scheduler after each epoch

            # Save model weights periodically
            if ep % train_config.weight_save_period == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.path_to_save, f"model_{ep}.pth"),
                )

            # # Clear GPU cache after every epoch to free up unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info("Training finished.", extra={"contexts": "finish training"})

    def test(self, test: DataLoader) -> None:
        self.model.eval()
        self.logger.info("Testing started.", extra={"contexts": "start testing"})
        for i, (idx, input, target) in enumerate(test):
            output = self.model(input).detach()
            loss = self.calculate_loss(output, target).item()
            metric_values = { k: v.detach().cpu().numpy() for k, v in self.calculate_metrics(output, target).items() }

            self.logger.info(
                f"Iteration {i}, imgid: {idx}, Loss: {loss}, Metrics: {metric_values},",
                extra={"contexts": "test"},
            )

        self.logger.info("Testing finished.", extra={"contexts": "finish testing"})
