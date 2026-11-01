"""Dataset input and image transforms."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Build train-only augmentation or deterministic validation/test preprocessing."""
    if train:
        image_ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
        ]
    else:
        image_ops = [transforms.Resize((image_size, image_size))]
    return transforms.Compose(
        [*image_ops, transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class EthiopianFoodDataset(Dataset):
    """Reads image paths and labels from the prepared project manifest."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        class_names: list[str],
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(data_dir).resolve()
        self.transform = transform
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        manifest = self.root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Prepared manifest not found: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No examples found for split '{split}' in {manifest}")
        unknown = sorted({row["label"] for row in self.rows} - self.class_to_idx.keys())
        if unknown:
            raise ValueError(f"Labels missing from class list: {unknown}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = (self.root / row["path"]).resolve()
        if self.root not in image_path.parents:
            raise ValueError("Image path escapes the prepared data directory")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[row["label"]], row["path"]
