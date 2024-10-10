from torchio.transforms import Resize as ResizeTorchio
from typing import Tuple, Dict, Literal
import torch


class Resize:
    def __init__(self, target_shape: Tuple[int, int, int]):
        self.target_shape = target_shape
        self.resize = ResizeTorchio(target_shape)

    def __call__(
        self, transform_data: Tuple[torch.tensor, ...]
    ) -> Tuple[torch.tensor, torch.tensor]:

        data, label = self.resize(transform_data[0]), self.resize(transform_data[1])
        return data, label
