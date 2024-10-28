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
from torch.cuda.amp import autocast, GradScaler

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
                    "accumulation_steps": train_config.accumulation_steps,
                    "optimizer": train_config.optimizer.__name__,
                },
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
                    "accumulation_steps": train_config.accumulation_steps,
                },
                resume="allow",
            )
            ep_range = range(
                train_config.current_epoch,
                train_config.epoch + train_config.current_epoch,
            )
            self.model.load_state_dict(torch.load(train_config.model_path))

        optimizer = train_config.optimizer(self.model.parameters(), lr=train_config.lr)
        scaler = GradScaler()  # Mixed precision scaler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train) * train_config.epoch
        )  # Example scheduler

        accumulation_steps = train_config.accumulation_steps
        run.watch(self.model, log="all")

        for ep in ep_range:
            # Training loop
            self.model.train()
            cum_loss = 0
            cum_train_metric: Dict[str, numpy.ndarray] = {}
            
            optimizer.zero_grad()  # Zero out gradients before starting
            for i, (idx, input, target) in tqdm.tqdm(
                enumerate(train), total=len(train)
            ):
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with autocast():
                    output = self.model(input)
                    try:
                        loss = self.calculate_loss(output, target) / accumulation_steps
                    except RuntimeError as e:
                        self.logger.error(f"Runtime error calculating loss at iteration {i}: {e}")
                        continue

                if loss is not None and torch.isfinite(loss):
                    scaler.scale(loss).backward()
                else:
                    self.logger.warning(f"Invalid loss at iteration {i}. Skipping gradient update.")
                    continue

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                cum_loss += loss.item() * accumulation_steps

                metric_values = self.calculate_metrics(output, target)
                # cum_train_metric.append({k: v.detach() for k, v in metric_values.items()})
                cum_train_metric = {
                    k: v.detach().cpu().numpy() + cum_train_metric.get(k, 0)
                    for k, v in metric_values.items()
                }

            train_output = {
                f"train_{k}": v / len(train)
                for k, v in cum_train_metric.items()
            }

            self.model.eval()
            cum_val_metric = {}

            # Validation loop
            with torch.no_grad():
                for i, (idx, input, target) in tqdm.tqdm(
                    enumerate(val), total=len(val)
                ):  
                    input = input.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    output = self.model(input)

                    metric_values = self.calculate_metrics(output, target)
                    cum_val_metric = {
                        k: v.detach().cpu().numpy() + cum_val_metric.get(k, 0)
                        for k, v in metric_values.items()
                    }

            val_output = {
                f"val_{k}": v / len(val)
                for k, v in cum_val_metric.items()
            }
            
            # merge train and val output
            merged_output = train_output | val_output
            wandb.log(merged_output)

            scheduler.step()  # Step the scheduler after each epoch

            # Save model weights periodically
            if ep % train_config.weight_save_period == 0 or ep == ep_range[-1]:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.path_to_save, f"model_{ep}.pth"),
                )
                
                
        run.finish()

        self.logger.info("Training finished.", extra={"contexts": "finish training"})

    def test(self, test: DataLoader) -> None:
        self.model.eval()
        self.logger.info("Testing started.", extra={"contexts": "start testing"})
        for i, (idx, input, target) in enumerate(test):
            output = self.model(input.to(self.device, non_blocking=True)).detach()
            loss = self.calculate_loss(output, target).item()
            metric_values = {
                k: v.detach().cpu().numpy()
                for k, v in self.calculate_metrics(output, target).items()
            }

            self.logger.info(
                f"Iteration {i}, imgid: {idx}, Loss: {loss}, Metrics: {metric_values},",
                extra={"contexts": "test"},
            )

        self.logger.info("Testing finished.", extra={"contexts": "finish testing"})
