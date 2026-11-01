"""Train a scratch baseline or a transfer-learned MobileNetV3 classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from ethiopian_food_vision.data import EthiopianFoodDataset, make_transform
from ethiopian_food_vision.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_classes(data_dir: Path) -> list[str]:
    summary = json.loads((data_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    return list(summary["classes"])


def _loader(data_dir: Path, split: str, classes: list[str], image_size: int, batch_size: int):
    dataset = EthiopianFoodDataset(
        data_dir, split, classes, transform=make_transform(image_size, train=(split == "train"))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _labels_for_split(data_dir: Path, split: str, classes: list[str]) -> list[str]:
    dataset = EthiopianFoodDataset(data_dir, split, classes)
    return [classes[label] for _, label, _ in dataset]


def _metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def _predict(model, loader, device, classes: list[str]):
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    paths: list[str] = []
    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            predictions.extend(probs.argmax(dim=1).tolist())
            probabilities.extend(probs.tolist())
            targets.extend(labels.tolist())
            paths.extend(batch_paths)
    return targets, predictions, probabilities, paths


def _write_test_reports(output_dir: Path, classes: list[str], y_true, y_pred, probs, paths) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(y_true, y_pred, classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    metrics["test_examples"] = len(y_true)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *classes])
        for class_name, row in zip(classes, cm, strict=True):
            writer.writerow([class_name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(title="Held-out test confusion matrix", xlabel="Predicted class", ylabel="True class")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[row, col] > (cm.max() / 2 if cm.size else 0) else "black",
            )
    fig.colorbar(image, ax=ax, label="Image count")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    errors_path = output_dir / "error_analysis.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "true_label", "predicted_label", "confidence", "top3"])
        for true_idx, pred_idx, scores, path in zip(y_true, y_pred, probs, paths, strict=True):
            top = np.argsort(scores)[::-1][:3]
            top3 = " | ".join(f"{classes[int(i)]}:{scores[int(i)]:.4f}" for i in top)
            writer.writerow(
                [path, classes[true_idx], classes[pred_idx], f"{scores[pred_idx]:.6f}", top3]
            )
    return metrics


def _set_transfer_phase(model, phase: str) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if phase == "finetune":
        # Unfreeze only the final three feature blocks, keeping early visual features fixed.
        for block in list(model.features.children())[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def _optimizer(model, model_name: str, phase: str):
    if model_name == "baseline":
        return torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    if phase == "head":
        return torch.optim.AdamW(model.classifier.parameters(), lr=8e-4, weight_decay=1e-4)
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": feature_params, "lr": 8e-5},
            {"params": model.classifier.parameters(), "lr": 4e-4},
        ],
        weight_decay=1e-4,
    )


def train(args) -> dict:
    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    classes = _read_classes(data_dir)
    train_loader = _loader(data_dir, "train", classes, args.image_size, args.batch_size)
    valid_loader = _loader(data_dir, "valid", classes, args.image_size, args.batch_size)
    test_loader = _loader(data_dir, "test", classes, args.image_size, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, len(classes), pretrained=args.model == "mobilenet_v3_small").to(
        device
    )

    freeze_epochs = (
        min(args.freeze_epochs, max(args.epochs - 1, 0)) if args.model != "baseline" else 0
    )
    if args.model != "baseline":
        _set_transfer_phase(model, "head")
    phase = "head" if args.model != "baseline" else "baseline"
    optimizer = _optimizer(model, args.model, phase)

    train_counts = {name: 0 for name in classes}
    for label in _labels_for_split(data_dir, "train", classes):
        train_counts[label] += 1
    weights = torch.tensor(
        [len(train_loader.dataset) / max(len(classes) * train_counts[name], 1) for name in classes],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    best_score = -1.0
    history: list[dict] = []
    best_path = output_dir / "best.pt"

    for epoch in range(args.epochs):
        if args.model != "baseline" and epoch == freeze_epochs and freeze_epochs < args.epochs:
            _set_transfer_phase(model, "finetune")
            optimizer = _optimizer(model, args.model, "finetune")
            phase = "finetune"
        model.train()
        loss_sum = 0.0
        train_true: list[int] = []
        train_pred: list[int] = []
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            train_true.extend(labels.tolist())
            train_pred.extend(logits.detach().argmax(dim=1).tolist())
        val_true, val_pred, _, _ = _predict(model, valid_loader, device, classes)
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": loss_sum / len(train_loader.dataset),
            "train_accuracy": float(accuracy_score(train_true, train_pred)),
            "val_accuracy": float(accuracy_score(val_true, val_pred)),
            "val_macro_f1": float(f1_score(val_true, val_pred, average="macro", zero_division=0)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_macro_f1"] > best_score:
            best_score = row["val_macro_f1"]
            torch.save(
                {
                    "model_name": args.model,
                    "classes": classes,
                    "image_size": args.image_size,
                    "state_dict": model.state_dict(),
                    "seed": args.seed,
                    "best_val_macro_f1": best_score,
                    "epoch": epoch + 1,
                },
                best_path,
            )

    (results_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "training_history.csv").write_text(
        "epoch,phase,train_loss,train_accuracy,val_accuracy,val_macro_f1\n"
        + "".join(
            f"{r['epoch']},{r['phase']},{r['train_loss']:.6f},{r['train_accuracy']:.6f},"
            f"{r['val_accuracy']:.6f},{r['val_macro_f1']:.6f}\n"
            for r in history
        ),
        encoding="utf-8",
    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    y_true, y_pred, probs, paths = _predict(model, test_loader, device, classes)
    metrics = _write_test_reports(results_dir, classes, y_true, y_pred, probs, paths)
    metrics["model"] = args.model
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics["best_val_macro_f1"] = checkpoint["best_val_macro_f1"]
    metrics["device"] = str(device)
    (results_dir / "run_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "TEST",
        json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, ensure_ascii=False),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts/baseline")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--model", choices=["baseline", "mobilenet_v3_small"], default="baseline")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
