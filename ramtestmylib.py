import torchio as tio
from memory_profiler import profile
from torchvision.transforms import Compose
import torch
from typing import Tuple, Dict, Literal
from torchio.transforms import Resize as ResizeTorchio, SpatialTransform, CropOrPad, Resample
from torchio.typing import TypeSpatialShape
import numpy as np
import warnings
from torchio.utils import to_tuple
from torchio.data.subject import Subject
import psutil
import matplotlib.pyplot as plt
import gc

class TestResize(SpatialTransform):
    """Resample images so the output shape matches the given target shape.

    The field of view remains the same.

    .. warning:: In most medical image applications, this transform should not
        be used as it will deform the physical object by scaling anistropically
        along the different dimensions. The solution to change an image size is
        typically applying :class:`~torchio.transforms.Resample` and
        :class:`~torchio.transforms.CropOrPad`.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`. The size of dimensions set to
            -1 will be kept.
        image_interpolation: See :ref:`Interpolation`.
        label_interpolation: See :ref:`Interpolation`.
    """

    def __init__(
        self,
        target_shape: TypeSpatialShape,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = np.asarray(to_tuple(target_shape, length=3))
        self.image_interpolation = self.parse_interpolation(
            image_interpolation,
        )
        self.label_interpolation = self.parse_interpolation(
            label_interpolation,
        )
        self.args_names = [
            'target_shape',
            'image_interpolation',
            'label_interpolation',
        ]

    # @profile
    def apply_transform(self, subject: Subject) -> Subject:
        shape_in = np.asarray(subject.spatial_shape)
        shape_out = self.target_shape
        negative_mask = shape_out == -1
        shape_out[negative_mask] = shape_in[negative_mask]
        spacing_in = np.asarray(subject.spacing)
        spacing_out = shape_in / shape_out * spacing_in
        resample = Resample(
            spacing_out,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
        )
        resampled = resample(subject)
        assert isinstance(resampled, Subject)
        # Sometimes, the output shape is one voxel too large
        # Probably because Resample uses np.ceil to compute the shape
        if not resampled.spatial_shape == tuple(shape_out):
            message = (
                f'Output shape {resampled.spatial_shape}'
                f' != target shape {tuple(shape_out)}. Fixing with CropOrPad'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            crop_pad = CropOrPad(shape_out)  # type: ignore[arg-type]
            resampled = crop_pad(resampled)
        assert isinstance(resampled, Subject)
        return resampled

class Resize:
    def __init__(self, target_shape: Tuple[int, int, int], copy: bool = True):
        self.target_shape = target_shape
        self.resize = TestResize(target_shape, copy=copy)
        
    def __call__(
        self, transform_data: Tuple[torch.tensor, ...]
    ) -> Tuple[torch.tensor, torch.tensor]:

        return self.resize(transform_data[0]), self.resize(transform_data[1])
    
    
def run_main(
    length: int = 100,
    copy: bool = False
):
    transform = Compose([Resize((128, 128, 128), copy=copy)])
    memory_usages = []
    
    for i in range(length):
        data = transform([torch.randn(1, 600, 600, 600), torch.randn(1, 600, 600, 600)])
        unreachable = gc.collect()
        print("Processed subject")
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 ** 2)  # Convert to MB
        print(f"Iteration {i}: Memory usage: {memory_usage:.2f} MB with {unreachable} unreachable objects")
        print(f"output shape: {data[0].shape}")
        memory_usages.append(memory_usage)
        
    plt.plot(memory_usages)
    plt.xlabel('Iteration')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Iterations')
    plt.show()
        
        
if __name__ == "__main__":
    run_main()