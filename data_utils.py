from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


BINARY_NAMES = ("NORMAL", "PATHOLOGY")  # 0 = NORMAL, 1 = DME ∪ DRUSEN


def pathology_label_from_folder(folder_name: str) -> int:
    u = folder_name.upper()
    if u == "NORMAL":
        return 0
    if u in ("DME", "DRUSEN"):
        return 1
    raise ValueError(f"Unknown class folder: {folder_name}")


def collect_image_paths(data_root: Path) -> tuple[list[Path], list[int]]:
    """Pool train/ and test/ class folders into one list (paths relabeled by folder)."""
    srinivasan = data_root / "Srinivasan"
    paths: list[Path] = []
    labels: list[int] = []
    for split in ("train", "test"):
        split_dir = srinivasan / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            y = pathology_label_from_folder(class_dir.name)
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                    paths.append(p)
                    labels.append(y)
    return paths, labels


def stratified_split(
    paths: Sequence[Path],
    labels: Sequence[int],
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return train indices and test indices (stratified)."""
    idx = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=np.asarray(labels),
        random_state=seed,
    )
    return train_idx.tolist(), test_idx.tolist()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_rgb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


# --- Classifier transforms (ImageNet stats, 128x128) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform_classifier() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform_classifier() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImagePathDataset(Dataset):
    """Binary labels; loads RGB; applies transform on tensor path."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        indices: Sequence[int] | None,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.indices = list(indices) if indices is not None else list(range(len(paths)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        j = self.indices[i]
        img = pil_rgb(self.paths[j])
        x = self.transform(img)
        return x, int(self.labels[j])


# --- SRGAN: HR 128, LR 32 (tensor in [0,1] for GAN; separate from classifier norm) ---


def hr_transform_gan() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )


def gan_tensor_to_minus1_1(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0 - 1.0


def minus1_1_to_gan_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t + 1.0) * 0.5


class SRGANPairDataset(Dataset):
    """HR at 128 for discriminator / content; LR is 32x32 downsample of same tensor."""

    def __init__(self, paths: Sequence[Path], indices: Sequence[int]) -> None:
        self.paths = list(paths)
        self.indices = list(indices)
        self.hr_tf = hr_transform_gan()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = self.indices[i]
        hr = self.hr_tf(pil_rgb(self.paths[j]))
        lr = torch.nn.functional.interpolate(
            hr.unsqueeze(0),
            size=(32, 32),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        return lr, hr


def iter_indices_batches(indices: Sequence[int], batch_size: int, shuffle: bool, seed: int) -> Iterator[list[int]]:
    order = list(indices)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for k in range(0, len(order), batch_size):
        yield order[k : k + batch_size]
