from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Union
from typing_extensions import TypedDict

__all__ = [
    "InfoMetaData",
    "DataMetaData",
    "SimpleDatasetMetaData",
    "KFOLDInfoMetaData",
    "KFOLDDatasetMetaData",
    "RatioWithValidate",
    "SplitRatio",
    "DatasetMetaData",
]

RatioWithValidate = Tuple[float, float, float]
RatioWithoutValidate = Tuple[float, float]

SplitRatio = Union[RatioWithValidate, RatioWithoutValidate]


class InfoMetaData(BaseModel):
    dataset_path: str
    split_ratio: Union[RatioWithValidate, RatioWithoutValidate]
    seed: int


class DataMetaData(BaseModel):
    train: List[str]
    test: List[str]
    val: List[str]


class SimpleDatasetMetaData(BaseModel):
    info: InfoMetaData
    data: DataMetaData
    
class KFOLDInfoMetaData(BaseModel):
    dataset_path: str
    seed: int
    k: int
    split_ratio: RatioWithoutValidate
    
Fold = DataMetaData
    
class KFOLDDatasetMetaData(BaseModel):
    info: KFOLDInfoMetaData
    data: List[Fold]

DatasetMetaData =  Union[SimpleDatasetMetaData, KFOLDDatasetMetaData]