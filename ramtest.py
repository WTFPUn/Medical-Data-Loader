import torchio as tio
from memory_profiler import profile
import torch

class DummyTransform(tio.Transform):

    @profile
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:

        for img in subject.get_images_dict().values():
            # Make shallow copy - that must be the memory leak
            if isinstance(img, tio.Image):
                new_img = img.data.clone().float().numpy()
                new_img += 1
                new_img = torch.as_tensor(new_img)
                img.set_data(new_img)
            else:
                raise ValueError(f"Expected data to be of type np.ndarray or torch.Tensor but got {type(img.data)}")
        return subject
        
        
def run_main(
    length: int = 4,
    copy_compose: bool = False,
):
    transforms = tio.Compose([
        DummyTransform(copy=False, include=["t1c", "t1n", "t2w", "t2f"]),
        DummyTransform(copy=False, include=["t1c", "t1n", "t2w", "t2f"]),
        DummyTransform(copy=False, include=["t1c", "t1n", "t2w", "t2f"]),
    ], copy=copy_compose)

    # roughly 3GB in size (4 * 0.75GB)
    subject = tio.Subject(**{seq: tio.ScalarImage(tensor=torch.randn(1, 1024, 1024, 192)) for seq in ["t1c", "t1n", "t2w", "t2f"]})

    subjects_dataset = tio.SubjectsDataset(
        subjects=length * [subject],
        transform=transforms
    )

    subjects_loader = tio.SubjectsLoader(
        dataset=subjects_dataset,
        batch_size=1,
        num_workers=0  # increasing it will multiply the memory leak per worker
    )

    for subject in subjects_loader:
        print("Processed subject")

if __name__ == "__main__":
    run_main()