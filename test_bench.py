from Experiment.Generic.Benchmark import BenchmarkGeneric, ModelTrainerInit
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
    SwinUNETRTrainer,
    UmambaEnc
)
import torch.nn as nn
import torch
from torchvision import datasets, transforms

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

metrics = [DSC()]
benchmark = BenchmarkGeneric[torch.tensor, torch.tensor](
    "test", "split.json", datasetConfig, metrics, 3, logger
)

benchmark.add_trainer(
    ModelTrainerInit(
        model=NnUnet,
        model_name="nnunet",
        experiment_name="test",
        model_path="infer/nnunet.pth",
        epochs=99,
    ),
    num_input_channels=1,
    channels_multiplier=2,
)


benchmark.add_trainer(
    ModelTrainerInit(
        model=ResEncUnet,
        model_name="resencunet",
        experiment_name="test",
        model_path="infer/resenc.pth",
        epochs=99,
    ),
    n_stages=6,
    features_per_stage=(32, 64, 128, 256, 320, 320),
    conv_op=nn.Conv3d,
    kernel_sizes=3,
    strides=(1, 2, 2, 2, 2, 2),
    n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
    n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={},
    dropout_op=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
    deep_supervision=False,
)

benchmark.add_trainer(
    ModelTrainerInit(
        model=SwinUNETRTrainer,
        model_name="swinunetr",
        experiment_name="test",
        model_path="infer/swinunetr.pth",
        epochs=99,
    ),
    num_input_channels=1,
    depths=(2,4,2,2),
    img_size=(128, 128, 128),
)

benchmark.add_trainer(
    ModelTrainerInit(
        model=UmambaEnc,
        model_name="umambaenc",
        experiment_name="test",
        model_path="infer/swinunetr.pth",
        epochs=99,
    ),
    num_input_channels=1,
    img_size=(128, 128, 128)
)

benchmark.add_trainer(
    ModelTrainerInit(
        model=MedNeXt,
        model_name="mednext",
        experiment_name="test",
        model_path="infer/mednext.pth",
        epochs=99,
    ),
    num_input_channels=1,
    model_id="M",
)