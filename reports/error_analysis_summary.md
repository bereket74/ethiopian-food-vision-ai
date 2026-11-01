# Error analysis summary

The transfer model made 32 errors on the 164-image test split (80.5% accuracy). The largest off-diagonal counts were:

- Kitfo → Tihlo: 3 images
- Kikil → Firfir: 3 images
- Firfir → Chechebsa: 3 images
- Chechebsa → Firfir: 3 images

The two-way Firfir/Chechebsa confusion accounts for six cases. One Genfo image was assigned Tihlo with a model score of 0.899, a reminder that high scores are not calibrated confidence. This project did not inspect enough examples to attribute the errors to particular visual causes.

Tihlo has only four test images: recall is 1.00, precision 0.31, and F1 0.471. These estimates are unstable at that support. Chechebsa has eight test images (F1 0.625). The widest class-level concern is therefore limited and imbalanced coverage, not a claim that the model reliably distinguishes every dish.

For a stronger follow-up, collect a permission-cleared, independently sourced test set with more examples per class and grouped splits by restaurant, photographer, or source post. Review calibration and add a reject/unknown option before real-world use. Per-image predictions and the full confusion matrix are retained locally in `results/mobilenet_v3_small/`.
