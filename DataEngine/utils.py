import os
import json
import random
import re
import logging

from .types.utils import *

import threading
from zipfile import ZipFile


__all__ = ["get_split_meta_data", "sample_file"]

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


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

    with open(output_path, "w") as f:
        json.dump(meta_data.dict(), f, indent=4)

    logger.info(f"Meta data written to {output_path}")
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


def unzip_file(zip_path, output_path):
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    return


def unzip_file_by_dir(directory):
    # use threading to unzip files
    thread_num = threading.active_count() // 2
    thread_list = []
    for file in os.listdir(directory):
        if file.endswith(".zip"):
            thread = threading.Thread(
                target=unzip_file, args=(os.path.join(directory, file), directory)
            )
            thread_list.append(thread)
            thread.start()
            if len(thread_list) == thread_num:
                for t in thread_list:
                    t.join()
                thread_list = []
