import os
import json
import random
import re

from .types.utils import *
from .log import logger
from .types.utils import KFOLDDatasetMetaData
from .types.utils import KFOLDInfoMetaData



__all__ = ["get_split_meta_data", "sample_file"]



def get_split_meta_data(
    dataset_path,
    output_path: str,
    split_ratio: SplitRatio,
    seed: int = 42,
):
    """
    **Split the data into training and testing data and write the meta data into a json file.**

    ---
    Formats:
    - Path between data and label is seperate to 2 different folders, raw nii files are in data folder and labels are in label folder.
    - Name of raw nii files are `img{id}.nii` and labels are `label{id}.nii` where id is the number of the file and contain 4 digits.

    ---
    Parameters:
    - dataset_path: Path to the dataset
    - split_ratio: Tuple of 2 or 3 floats, first float is the ratio of training data, second float is the ratio of testing data, third float is the ratio of validation data
    - seed: Seed for the random number generator
    - output_path: Path to the output folder

    ---
    Returns:
    - None
    """

    random.seed(seed)

    raw_data_path = os.path.join(dataset_path, "data")
    label_data_path = os.path.join(dataset_path, "label")

    # check label is subset of data
    raw_data_id = [re.findall(r"\d+", file)[0] for file in os.listdir(raw_data_path)]
    label_data_id = [
        re.findall(r"\d+", file)[0] for file in os.listdir(label_data_path)
    ]

    matched_id = set(raw_data_id).intersection(set(label_data_id))

    logger.info(f"Found {len(matched_id)} matched ID between data and label")
    if len(matched_id) == 0:
        logger.error("No matched ID found between data and label")
        return
    elif len(matched_id) != len(raw_data_id):
        logger.warning(f"Only {len(matched_id)} out of {len(raw_data_id)} matched")

    matched_id = list(matched_id)

    # split the data
    random.shuffle(matched_id)
    if len(split_ratio) == 2:
        train_ratio, test_ratio = split_ratio
        val_ratio = 0
    else:
        train_ratio, test_ratio, val_ratio = split_ratio

    train_id = matched_id[: int(len(matched_id) * train_ratio)]
    test_id = matched_id[
        int(len(matched_id) * train_ratio) : int(
            len(matched_id) * (train_ratio + test_ratio)
        )
    ]
    val_id = matched_id[int(len(matched_id) * (train_ratio + test_ratio)) :]

    try:
        meta_data = DatasetMetaData(
            info=InfoMetaData(
                dataset_path=dataset_path, split_ratio=split_ratio, seed=seed
            ),
            data=DataMetaData(train=train_id, test=test_id, val=val_id),
        )
    except Exception as e:
        logger.error(e)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_data.model_dump(), f, indent=4)

    logger.info(f"Done Spliting! Meta data written to {output_path}")
    return


    

def sample_file(directory, n=100):
    """
    **Generate and create file  for testing get_split_meta_data function**

    ---
    Parameters:
    - n: Number of files to generate

    ---
    Returns:
    - None

    """

    os.makedirs(directory, exist_ok=True)

    raw_data_path = os.path.join(directory, "data")
    label_data_path = os.path.join(directory, "label")

    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(label_data_path, exist_ok=True)

    # format of number is 0001, 0002, 0003, ...
    for i in range(n):
        with open(f"{raw_data_path}/img{i:04d}.nii", "a") as f:
            f.write("This is a raw data file")
        with open(f"{label_data_path}/label{i:04d}.nii", "a") as f:
            f.write("This is a label file")

    return

def get_kfold_meta_data(
    dataset_path,
    output_path: str,
    k: int,
    split_ratio: SplitRatio,
    seed: int = 42,
):
    """
    **Split the data into training and testing data and write the meta data into a json file.**

    ---
    Formats:
    - Path between data and label is seperate to 2 different folders, raw nii files are in data folder and labels are in label folder.
    - Name of raw nii files are `img{id}.nii` and labels are `label{id}.nii` where id is the number of the file and contain 4 digits.

    ---
    Parameters:
    - dataset_path: Path to the dataset
    - k: Number of folds
    - seed: Seed for the random number generator
    - output_path: Path to the output folder

    ---
    Returns:
    - None
    """

    random.seed(seed)

    raw_data_path = os.path.join(dataset_path, "data")
    label_data_path = os.path.join(dataset_path, "label")

    # Check that labels are a subset of data
    raw_data_files = os.listdir(raw_data_path)
    label_data_files = os.listdir(label_data_path)

    raw_data_id = [re.findall(r"\d+", file)[0] for file in raw_data_files]
    label_data_id = [re.findall(r"\d+", file)[0] for file in label_data_files]

    matched_id = set(raw_data_id).intersection(set(label_data_id))

    print(f"Found {len(matched_id)} matched IDs between data and label")
    if len(matched_id) == 0:
        print("No matched IDs found between data and label")
        return
    elif len(matched_id) != len(raw_data_id):
        print(f"Only {len(matched_id)} out of {len(raw_data_id)} IDs matched")

    matched_id = list(matched_id)

    # Shuffle the matched IDs
    random.shuffle(matched_id)

    # Split the data into training and testing sets
    train_ratio, test_ratio = split_ratio
    total_ids = len(matched_id)
    train_size = int(total_ids * train_ratio)
    test_size = int(total_ids * test_ratio)

    train_ids = matched_id[:train_size]
    test_ids = matched_id[train_size:train_size + test_size]

    # Ensure train_ids are shuffled
    random.shuffle(train_ids)

    # Split train_ids into K folds
    fold_size = len(train_ids) // k
    folds_ids = []
    for i in range(k):
        if i < k - 1:
            fold_ids = train_ids[i * fold_size: (i + 1) * fold_size]
        else:
            # Last fold takes the remaining IDs
            fold_ids = train_ids[i * fold_size:]
        folds_ids.append(fold_ids)

    # Create folds with train, test, and val IDs
    folds = []
    for i in range(k):
        val_ids = folds_ids[i]
        train_ids_fold = [id_ for idx, fold in enumerate(folds_ids) if idx != i for id_ in fold]
        fold = DataMetaData(train=train_ids_fold, test=test_ids, val=val_ids)
        folds.append(fold)

    # Create the metadata object
    try:
        info = KFOLDInfoMetaData(
            dataset_path=dataset_path,
            seed=seed,
            k=k,
            split_ratio=split_ratio
        )
        meta_data = KFOLDDatasetMetaData(
            info=info,
            data=folds
        )
    except Exception as e:
        print(f"Error creating metadata: {e}")
        return

    # Write the metadata to the output path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_data.dict(), f, indent=4)

    print(f"Done splitting! Metadata written to {output_path}")

def sample_kfold_file(directory, n=100):
    """
    **Generate and create file  for testing get_split_meta_data function**

    ---
    Parameters:
    - n: Number of files to generate

    ---
    Returns:
    - None

    """

    os.makedirs(directory, exist_ok=True)

    raw_data_path = os.path.join(directory, "data")
    label_data_path = os.path.join(directory, "label")

    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(label_data_path, exist_ok=True)

    # format of number is 0001, 0002, 0003, ...
    for i in range(n):
        with open(f"{raw_data_path}/img{i:04d}.nii", "a") as f:
            f.write("This is a raw data file")
        with open(f"{label_data_path}/label{i:04d}.nii", "a") as f:
            f.write("This is a label file")

    return