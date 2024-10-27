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
    RandomFlip3D,
    Resize,
)
import torch
from torchvision import datasets, transforms

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
experimentOne.add_trainer(
    MedNeXt,
    "mednext_second",
    num_input_channels=1,
    model_id="S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experimentOne.run(
    batch_size=1,
    num_workers=0,
    train_config=NewTrainConfig(
        epoch=100,
        weight_save_period=10,
        accumulation_steps=16,
        lr=1e-4,
        optimizer=torch.optim.Adam,
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
