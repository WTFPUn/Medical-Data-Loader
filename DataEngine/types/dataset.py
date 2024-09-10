from pydantic import BaseModel

from typing import Literal

class DatasetConfig(BaseModel):
    window_center: int = 40
    window_width: int = 80
    device: Literal["cpu", "cuda"] = "cpu"