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
    logger,
    MedNeXt,
    RandomFlip3D,
    Resize
)
import torch
from torchvision import datasets, transforms

torch.manual_seed(0)
datasetConfig = DatasetConfig(
    window_center=400,
    window_width=1000,
    device="cuda",
    compose= {
        "train": transforms.Compose([
            transforms.Lambda(lambda x: (torch.tensor(x[0]).unsqueeze(0).float(), torch.tensor(x[1]).unsqueeze(0).float())),
            Resize((128)),
            RandomFlip3D(axes=(0, 1, 2), flip_prob=0.5),
        ]),
        "val": transforms.Compose([
            transforms.Lambda(lambda x: (torch.tensor(x[0]).unsqueeze(0).float(), torch.tensor(x[1]).unsqueeze(0).float())),
            Resize((128)),
        ]),
        "test": transforms.Compose([
            transforms.Lambda(lambda x: (torch.tensor(x[0]).unsqueeze(0).float(), torch.tensor(x[1]).unsqueeze(0).float())),
            Resize((128)),
            RandomFlip3D(axes=(0, 1, 2), flip_prob=0.5),
        ]),
    }
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
    "mednext_first",
    num_input_channels=1,
    model_id="S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experimentOne.run(
    batch_size=1,
    num_workers=0,
    train_config=NewTrainConfig(
        epoch=40, weight_save_period=1, accumulation_steps=16, lr=1e-4,optimizer=torch.optim.Adam
    ),
)
