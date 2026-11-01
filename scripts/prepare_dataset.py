"""Download and prepare the public Ethiopian foods dataset for local experiments.

The upstream Hub card says the images were collected from social media but does
not declare a license. This script is for local research only; it does not grant
redistribution rights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

DATASET_REPO = "Tinsae/Ethiopian-foods"
PARQUET_URL = (
    "https://huggingface.co/datasets/Tinsae/Ethiopian-foods/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
SPLIT_SEED = 42


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"Using cached source parquet: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ethiopian-food-vision/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
        total = response.headers.get("Content-Length")
        copied = 0
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            copied += len(chunk)
            if total:
                print(f"Downloaded {copied / int(total):.0%}", end="\r", flush=True)
    temporary.replace(destination)
    print(f"Downloaded source parquet: {destination} ({destination.stat().st_size:,} bytes)")


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        content = value.get("bytes")
        if content is None:
            raise ValueError("Parquet image row has no embedded image bytes")
        return content
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"Unsupported image value type: {type(value).__name__}")


def _label(row: dict[str, Any]) -> str:
    for key in ("text", "classes", "label", "class"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "_")
    raise ValueError(f"Could not find a string class label in columns {list(row)}")


def prepare(source: Path, out_dir: Path, download: bool = True) -> dict[str, Any]:
    if download:
        _download(PARQUET_URL, source)
    if not source.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {source}")

    rows = pq.read_table(source).to_pylist()
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    duplicates = 0
    corrupt = 0
    conflicts: set[str] = set()

    for index, row in enumerate(rows):
        label = _label(row)
        raw = _image_bytes(row.get("image"))
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            duplicates += 1
            if seen[digest] != label:
                conflicts.add(digest)
            continue
        seen[digest] = label
        path = image_dir / f"{digest}.jpg"
        try:
            with Image.open(BytesIO(raw)) as image:
                image.convert("RGB").save(path, format="JPEG", quality=95)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupt += 1
            print(f"Skipping unreadable image at source row {index}: {exc}")
            continue
        entries.append(
            {"path": path.relative_to(out_dir).as_posix(), "label": label, "sha256": digest}
        )

    if conflicts:
        entries = [entry for entry in entries if entry["sha256"] not in conflicts]
    if not entries:
        raise RuntimeError("No readable images were extracted")

    labels = [entry["label"] for entry in entries]
    counts = Counter(labels)
    if min(counts.values()) < 3:
        raise ValueError(f"At least three unique images per class are needed: {counts}")

    indices = list(range(len(entries)))
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.30, random_state=SPLIT_SEED, stratify=labels
    )
    remainder_labels = [labels[index] for index in remainder_idx]
    valid_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=SPLIT_SEED,
        stratify=remainder_labels,
    )
    split_of = {index: "train" for index in train_idx}
    split_of.update({index: "valid" for index in valid_idx})
    split_of.update({index: "test" for index in test_idx})
    for index, entry in enumerate(entries):
        entry["split"] = split_of[index]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "sha256", "split"])
        writer.writeheader()
        writer.writerows(entries)

    split_counts = {
        split: dict(
            sorted(Counter(entry["label"] for entry in entries if entry["split"] == split).items())
        )
        for split in ("train", "valid", "test")
    }
    summary = {
        "source_dataset": DATASET_REPO,
        "source_url": "https://huggingface.co/datasets/Tinsae/Ethiopian-foods",
        "source_rows": len(rows),
        "readable_unique_images": len(entries),
        "exact_duplicate_rows_removed": duplicates,
        "label_conflicts_removed": len(conflicts),
        "corrupt_images_skipped": corrupt,
        "seed": SPLIT_SEED,
        "split_sizes": {
            split: sum(entry["split"] == split for entry in entries)
            for split in ("train", "valid", "test")
        },
        "split_class_counts": split_counts,
        "classes": sorted(counts),
        "license_status": "Not declared on the source Hub dataset card; local research only.",
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/ethiopian-foods.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--no-download", action="store_true", help="Use an already downloaded parquet file"
    )
    args = parser.parse_args()
    prepare(args.source, args.output, download=not args.no_download)


if __name__ == "__main__":
    main()
