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
)
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import os

# os.environ['WANDB_MODE'] = 'offline'

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

experimentOne = Experimenting[torch.tensor, torch.tensor](
    "before_i_die",
    "split.json",
    datasetConfig,
    [DSCLoss(3), CE()],
    [DSC(), CE(), Accuracy(), Precision(3), Recall(3)],
    3,
    logger,
)
# experimentOne.add_trainer(
#     ResEncUnet,
#     "ResEncUnetM",
#     n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
#     conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
#     n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
#     n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
#     conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={}, dropout_op=None,
#     nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False
# )
experimentOne.add_trainer(
    MedNeXt,
    "MedNeXt",
    num_input_channels=1,
    model_id="M",
)

# experimentOne.add_trainer(
#     NnUnet,
#     "nnUnet_8xdim",
#     num_input_channels=1,
#     channels_multiplier=8,
# )

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
)
# experimentOne.run(
#     batch_size=1,
#     num_workers=0,
#     train_config=ContinueTrainConfig(
#         epoch=100,
#         weight_save_period=10,
#         accumulation_steps=16,
#         lr=1e-4,
#         optimizer=torch.optim.Adam,
#         current_epoch=15,
#         run_id="p0euuug4",
#         model_path="Experimenting/before_i_die/MedNeXt/mednext_first/model_15.pth",
#         project_name="SeniorProject",
#     ),
# )
