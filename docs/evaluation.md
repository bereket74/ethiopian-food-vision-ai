# Evaluation and model comparison

## Dataset and split

The local dataset pipeline read 1,097 rows from the public Hugging Face mirror. Four rows were exact byte duplicates, leaving 1,093 readable unique images; no corrupt images were skipped. A deterministic stratified split (seed 42) produced 765 training, 164 validation, and 164 held-out test images. The smallest class, Tihlo, has only 30 examples overall and four in test. The source repository has one `train` partition, so these splits were generated locally.

The source card says images were collected from social media and does not declare a dataset license. All data and trained checkpoints remain local; these metrics do not establish rights to redistribute either. See [`sources.md`](sources.md).

## Held-out results

The scratch CNN and transfer-learning model use the identical split and train-only augmentation. The best epoch is selected by validation macro-F1; the test split is not used for checkpoint selection.

| Model | Best epoch | Validation macro-F1 | Test accuracy | Test macro-F1 | Test balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 5 / 8 | 0.527 | 45.1% | 0.437 | 0.448 |
| MobileNetV3-Small, ImageNet transfer | 7 / 8 | 0.907 | 80.5% | 0.770 | 0.794 |

On this split, transfer learning improved accuracy by 35.4 percentage points and macro-F1 by 0.334. The validation-to-test macro-F1 gap for the transfer model is substantial; the test set is small, so the results should be treated as an initial benchmark rather than evidence of robust generalization.

The [model comparison chart](../reports/model_comparison.png), [MobileNet confusion matrix](../reports/mobilenet_v3_small_confusion_matrix.png), [baseline confusion matrix](../reports/baseline_confusion_matrix.png), aggregate [summary JSON](../reports/benchmark_summary.json), and MobileNet [per-class metrics CSV](../reports/mobilenet_v3_small_per_class.csv) provide the saved evidence.

## Error analysis

MobileNet misclassified 32 of 164 test images. The largest individual off-diagonal confusion counts were Kitfo→Tihlo (3), Kikil→Firfir (3), Firfir→Chechebsa (3), and Chechebsa→Firfir (3). A high-confidence error was Genfo→Tihlo at 0.899 model probability, showing that the displayed scores should not be assumed calibrated. Tihlo had 1.00 recall but only 0.31 precision on four test examples (F1 0.471), so the apparent recall is especially uncertain. Common class pairs and low-support classes are summarized in [`error_analysis_summary.md`](../reports/error_analysis_summary.md); all per-image predictions are saved locally in `results/mobilenet_v3_small/error_analysis.csv`.

The baseline misclassified 90 of 164 test images. Its weakest classes included Tihlo (F1 0.167), Genfo (0.286), and Shekla Tibs (0.286). Full per-image and per-class outputs are retained locally under `results/`.

## Inference smoke test

The saved MobileNet checkpoint loaded successfully. On one held-out Beyaynetu image, the inference function ranked Beyaynetu first with probability 0.998. The local Gradio interface also constructed successfully. This is a single-example smoke test, not an additional metric or user study.

## Limitations and next evaluation steps

- The dataset has only about 1.1K images, uneven class counts, and only four Tihlo images in the test set.
- Images were collected from social media; image rights and source grouping are unknown. Exact byte duplicates were removed, but near-duplicates could still cross splits.
- A random split from one source collection can overstate performance on new restaurants, lighting, regions, dishes, or phone cameras. An independently collected, permission-cleared test set is needed.
- The system predicts one of 11 labels even for non-food or unfamiliar inputs. It has no out-of-distribution rejection, and its probability scores have not been calibrated.
- Before any public model or Space release, establish dataset reuse and derivative-model rights. The present images and checkpoints are not distributed.
