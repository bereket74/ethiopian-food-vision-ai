"""Build aggregate, image-free benchmark artifacts from completed model runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAMES = ("baseline", "mobilenet_v3_small")
DISPLAY_NAMES = {
    "baseline": "Scratch CNN",
    "mobilenet_v3_small": "MobileNetV3-Small (ImageNet transfer)",
}


def build(results_dir: Path, data_dir: Path, reports_dir: Path) -> dict:
    dataset = json.loads((data_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    runs = {}
    for model_name in MODEL_NAMES:
        summary = json.loads(
            (results_dir / model_name / "run_summary.json").read_text(encoding="utf-8")
        )
        runs[model_name] = {
            "display_name": DISPLAY_NAMES[model_name],
            "best_epoch": summary["best_epoch"],
            "validation_macro_f1": summary["best_val_macro_f1"],
            "test_examples": summary["test_examples"],
            "test_accuracy": summary["accuracy"],
            "test_macro_f1": summary["macro_f1"],
            "test_balanced_accuracy": summary["balanced_accuracy"],
            "per_class": {
                name: {
                    "support": int(summary["per_class"][name]["support"]),
                    "precision": summary["per_class"][name]["precision"],
                    "recall": summary["per_class"][name]["recall"],
                    "f1": summary["per_class"][name]["f1-score"],
                }
                for name in dataset["classes"]
            },
        }

    report = {
        "dataset": {
            "source_dataset": dataset["source_dataset"],
            "source_rows": dataset["source_rows"],
            "readable_unique_images": dataset["readable_unique_images"],
            "exact_duplicate_rows_removed": dataset["exact_duplicate_rows_removed"],
            "corrupt_images_skipped": dataset["corrupt_images_skipped"],
            "split_seed": dataset["seed"],
            "split_sizes": dataset["split_sizes"],
            "class_counts_by_split": dataset["split_class_counts"],
            "license_status": dataset["license_status"],
        },
        "selection_metric": "validation macro-F1",
        "test_split_used_for_checkpoint_selection": False,
        "runs": runs,
        "absolute_improvement_transfer_minus_baseline": {
            metric: runs["mobilenet_v3_small"][f"test_{metric}"]
            - runs["baseline"][f"test_{metric}"]
            for metric in ("accuracy", "macro_f1", "balanced_accuracy")
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (reports_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "best_epoch",
                "validation_macro_f1",
                "test_accuracy",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_examples",
            ]
        )
        for _name, run in runs.items():
            writer.writerow(
                [
                    run["display_name"],
                    run["best_epoch"],
                    f"{run['validation_macro_f1']:.6f}",
                    f"{run['test_accuracy']:.6f}",
                    f"{run['test_macro_f1']:.6f}",
                    f"{run['test_balanced_accuracy']:.6f}",
                    run["test_examples"],
                ]
            )

    with (reports_dir / "mobilenet_v3_small_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "test_support", "precision", "recall", "f1"])
        for name, metric in runs["mobilenet_v3_small"]["per_class"].items():
            writer.writerow(
                [
                    name,
                    metric["support"],
                    f"{metric['precision']:.6f}",
                    f"{metric['recall']:.6f}",
                    f"{metric['f1']:.6f}",
                ]
            )

    metric_names = ("test_accuracy", "test_macro_f1", "test_balanced_accuracy")
    labels = ("Accuracy", "Macro-F1", "Balanced accuracy")
    values = np.array([[runs[name][metric] for metric in metric_names] for name in MODEL_NAMES])
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(x - width / 2, values[0], width, label=DISPLAY_NAMES[MODEL_NAMES[0]], color="#7a8ca5")
    ax.bar(x + width / 2, values[1], width, label=DISPLAY_NAMES[MODEL_NAMES[1]], color="#df8b36")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on held-out test split")
    ax.set_xticks(x, labels)
    ax.set_title("Ethiopian Food Vision: model comparison")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(reports_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    for model_name in MODEL_NAMES:
        shutil.copy2(
            results_dir / model_name / "confusion_matrix.png",
            reports_dir / f"{model_name}_confusion_matrix.png",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report = build(args.results_dir, args.data_dir, args.output_dir)
    print(json.dumps({"runs": report["runs"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
