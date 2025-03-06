from typing import Literal, Type, Any
from typing_extensions import TypedDict

from pydantic import BaseModel
from torchvision.transforms import Compose

class DatasetConfig(BaseModel):
    window_center: int = 40
    window_width: int = 80
    device: Literal["cpu", "cuda"] = "cpu"
    compose: None | Any = None
    gamma: float = 0.35
    
class PatchedDatasetConfig(BaseModel):
    compose: None | Any = None
    device: Literal["cpu", "cuda"] = "cpu"
    window_width: int = 80
    window_center: int = 40
    patch_size: int = 64
    stride: int = 32
    voxel_size: int = 800
    gamma: float = 0.35
