from Experiment import (
    Experimenting,
    DatasetConfig,
    PatchedDatasetConfig,
    DSC,
    DSCLoss,
    CE,
    Accuracy,
    Precision,
    Recall,
    NewTrainConfig,
    logger,
    # MedNeXt,
    # ResEncUnet,
    # NnUnet,
    RandomFlip3D,
    Resize,
    CropOrPad3D,
    # SwinUNETRTrainer
)
from Experiment.ModelTrainer import (
    MedNeXt,
    ResEncUnet,
    NnUnet,
    SwinUNETRTrainer,
    PatchBaseNnUnet,
)

from itertools import product

from Experiment.Loss.FocalLoss import FocalLoss
from Experiment.Loss.PolyLoss import PolyLoss
from Experiment.Loss.FocalTverskyLoss import FocalTverskyLoss
from Experiment.Loss.GeneralizedDiceLoss import GeneralizedDiceLoss
from Experiment.Loss.JaccardLoss import JaccardLoss

import torch.nn as nn
import torch
from torchvision import datasets, transforms
import os
import argparse

os.environ['WANDB_MODE'] = 'offline'

CE_weight = torch.tensor([0.00644722, 0.41434646, 0.57920632]).to("cuda")

torch.manual_seed(0)
datasetConfig = PatchedDatasetConfig(
    window_center=600,
    window_width=500,
    patch_size=128,
    voxel_size=768,
    stride=0,
    device="cuda",
    compose={
        "train": transforms.Compose(
            [
                RandomFlip3D(axes=(0, 1, 2), flip_prob=0.5),
                CropOrPad3D((768, 768, 768)),
            ]
        ),
        "val": transforms.Compose(
            [
                CropOrPad3D((768, 768, 768)),
            ]
        ),
        "test": transforms.Compose(
            [
                CropOrPad3D((768, 768, 768)),
            ]
        ),
    },
)

dist_base = [CE(CE_weight), FocalLoss(alpha=0.5, gamma=2, reduction='mean', num_classes=3), PolyLoss()]
region_base = [GeneralizedDiceLoss(), JaccardLoss(), FocalTverskyLoss(alpha=0.3, beta=0.7, num_classes=3, gamma=0.75)]

products = product(dist_base, region_base)
output = []
for product in products:
    output.append(product)

parser = argparse.ArgumentParser(description="Run medical data loader experiment")
parser.add_argument('--loss', type=int, required=True, help='Loss index')
# parser.add_argument('--kfold', type=int, required=False, help='run specific kfold')
args = parser.parse_args()

if isinstance(args.loss, int) or args.loss < 0 or args.loss >= len(products):
    raise ValueError("Invalid loss index")

experimentOne = Experimenting[torch.tensor, torch.tensor](
    "kfold_test",
    "kfold.json",
    datasetConfig,
    output[args.loss],
    [DSC(),Precision(3), Recall(3)],
    3,
    logger,
)

experimentOne.add_trainer(
    PatchBaseNnUnet,
    f"loss_test{'_'.join([loss.__str__() for loss in output[args.loss]])}",
    num_input_channels=1,
    channels_multiplier=2,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experimentOne.run(
    batch_size=2,
    num_workers=0,
    train_config=NewTrainConfig(
        epoch=20,
        weight_save_period=4,
        accumulation_steps=16,
        lr=1e-4,
        optimizer=torch.optim.AdamW
    ),
    fixed_k=0,
)

# to run: python script.py --loss 0
