from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Union
from typing_extensions import TypedDict

__all__ = [
    "DatasetMetaData",
    "InfoMetaData",
    "DataMetaData",
    "SplitRatio",
]

ratio_with_test = Tuple[float, float, float]
ratio_without_test = Tuple[float, float]

SplitRatio = Union[ratio_with_test, ratio_without_test]


class InfoMetaData(BaseModel):
    dataset_path: str
    split_ratio: Union[ratio_with_test, ratio_without_test]
    seed: int


class DataMetaData(BaseModel):
    train: List[str]
    test: List[str]
    val: List[str]


class DatasetMetaData(BaseModel):
    info: InfoMetaData
    data: DataMetaData
