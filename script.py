from Experiment import (
    Experimenting,
    DatasetConfig,
    DSC,
    DSCLoss,
    CE,
    Accuracy,
    Precision,
    Recall,
    IoU,
    NewTrainConfig,
    ContinueTrainConfig,
    logger,
    MedNeXt,
    ResEncUnet,
    NnUnet,
    RandomFlip3D,
    Resize,
    SwinUNETRTrainer
)
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import os
import argparse

os.environ['WANDB_MODE'] = 'offline'

CE_weight = torch.tensor([0.00644722, 0.41434646, 0.57920632]).to("cuda")

torch.manual_seed(0)
datasetConfig = DatasetConfig(
    window_center=400,
    window_width=1000,
    device="cuda",
    compose={
        "train": transforms.Compose(
            [
                Resize((128)),
                RandomFlip3D(axes=(0, 1, 2), flip_prob=0.5),
            ]
        ),
        "val": transforms.Compose(
            [
                Resize((128)),
            ]
        ),
        "test": transforms.Compose(
            [
                Resize((128)),
            ]
        ),
    },
)

parser = argparse.ArgumentParser(description="Run medical data loader experiment")
parser.add_argument('--model', type=str, required=True, help='Model name to run (MedNeXt, ResEncUnet, NnUnet)')
parser.add_argument('--kfold', type=int, required=False, help='run specific kfold')
args = parser.parse_args()
experimentOne = Experimenting[torch.tensor, torch.tensor](
    "kfold_test",
    "split_k.json",
    datasetConfig,
    [DSCLoss(3), CE(CE_weight)],
    [DSC(), CE(CE_weight), Accuracy(), Precision(3), Recall(3)],
    3,
    logger,
)

if args.model == "MedNeXt":
    experimentOne.add_trainer(
        MedNeXt,
        "MedNeXt_M",
        num_input_channels=1,
        model_id="M",
    )
elif args.model == "ResEncUnet":
    experimentOne.add_trainer(
        ResEncUnet,
        "ResEncUnetM",
        n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage= (1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={}, dropout_op=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False
    )
elif args.model == "NnUnet":
    experimentOne.add_trainer(
        NnUnet,
        "nnUnet_8xdim",
        num_input_channels=1,
        channels_multiplier=8,
    )
elif args.model == "SwinUNETR":
    experimentOne.add_trainer(
        SwinUNETRTrainer,
        "SwinUNETR",
        num_input_channels=1,
        depths=(2,4,2,2),
        img_size=(128, 128, 128)
    )
else:
    raise ValueError(f"Unknown model name: {args.model}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experimentOne.run(
    batch_size=1,
    num_workers=0,
    train_config=NewTrainConfig(
        epoch=100,
        weight_save_period=10,
        accumulation_steps=16,
        lr=1e-4,
        optimizer=torch.optim.AdamW
    ),
    fixed_k=args.kfold if args.kfold is not None else None
)

# to run: python script.py --model MedNeXt --kfold 0