"""Checkpoint loading and single-image inference."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ethiopian_food_vision.data import make_transform
from ethiopian_food_vision.models import build_model


class FoodPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=True)
        self.classes = list(checkpoint["classes"])
        self.image_size = int(checkpoint["image_size"])
        self.model = build_model(checkpoint["model_name"], len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.transform = make_transform(self.image_size, train=False)

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        if image is None:
            raise ValueError("Provide an image to classify")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor)[0], dim=0)
        k = min(max(1, top_k), len(self.classes))
        scores, indices = torch.topk(probabilities, k=k)
        return [
            (self.classes[int(index)], float(score))
            for score, index in zip(scores, indices, strict=True)
        ]
