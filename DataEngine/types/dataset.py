from typing import Literal, Type, Any
from typing_extensions import TypedDict

from pydantic import BaseModel
from torchvision.transforms import Compose

class DatasetConfig(BaseModel):
    window_center: int = 40
    window_width: int = 80
    device: Literal["cpu", "cuda"] = "cpu"
    compose: None | Any = None