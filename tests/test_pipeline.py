from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image, ImageDraw

from ethiopian_food_vision.data import EthiopianFoodDataset, make_transform
from ethiopian_food_vision.models import build_model
from scripts.prepare_dataset import prepare

CLASSES = [
    "beyaynetu",
    "chechebsa",
    "doro_wat",
    "firfir",
    "genfo",
    "kikil",
    "kitfo",
    "shekla_tibs",
    "shiro_wat",
    "tihlo",
    "tire_siga",
]


def _test_image_bytes(seed: int) -> bytes:
    image = Image.new("RGB", (48, 48), (seed * 13 % 255, seed * 29 % 255, seed * 47 % 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((seed % 20, seed % 20, 30, 30), outline=(255, 255, 255), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_prepare_deduplicates_and_makes_stratified_splits(tmp_path: Path):
    parquet_path = tmp_path / "fixture.parquet"
    output_dir = tmp_path / "prepared"
    rows = []
    seed = 0
    for label in CLASSES:
        for _ in range(10):
            rows.append(
                {"image": {"bytes": _test_image_bytes(seed), "path": f"{seed}.png"}, "text": label}
            )
            seed += 1
    rows.append(rows[0])  # exact duplicate should not cross a split
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)

    summary = prepare(parquet_path, output_dir, download=False)
    assert summary["source_rows"] == 111
    assert summary["exact_duplicate_rows_removed"] == 1
    assert summary["readable_unique_images"] == 110
    assert (
        summary["split_sizes"]["train"]
        + summary["split_sizes"]["valid"]
        + summary["split_sizes"]["test"]
        == 110
    )
    assert set(summary["classes"]) == set(CLASSES)
    manifest = (output_dir / "manifest.csv").read_text(encoding="utf-8")
    assert manifest.count("test\n") >= 1
    assert json.loads((output_dir / "dataset_summary.json").read_text())["seed"] == 42


def test_dataset_loads_rgb_image_and_class_index(tmp_path: Path):
    image_path = tmp_path / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), (100, 20, 10)).save(image_path)
    (tmp_path / "manifest.csv").write_text(
        "path,label,sha256,split\nimages/sample.jpg,beyaynetu,dummy,train\n", encoding="utf-8"
    )
    dataset = EthiopianFoodDataset(
        tmp_path, "train", CLASSES, transform=make_transform(64, train=False)
    )
    image, label, relative_path = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert label == CLASSES.index("beyaynetu")
    assert relative_path == "images/sample.jpg"


def test_dataset_rejects_path_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.jpg"
    Image.new("RGB", (10, 10)).save(outside)
    (tmp_path / "manifest.csv").write_text(
        "path,label,sha256,split\n../outside.jpg,beyaynetu,dummy,train\n", encoding="utf-8"
    )
    dataset = EthiopianFoodDataset(tmp_path, "train", CLASSES)
    with pytest.raises(ValueError, match="escapes"):
        dataset[0]


def test_models_return_one_logit_per_food_class():
    model = build_model("baseline", len(CLASSES), pretrained=False)
    logits = model(torch.zeros(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, len(CLASSES))

    transfer_model = build_model("mobilenet_v3_small", len(CLASSES), pretrained=False)
    transfer_logits = transfer_model(torch.zeros(2, 3, 224, 224))
    assert tuple(transfer_logits.shape) == (2, len(CLASSES))
