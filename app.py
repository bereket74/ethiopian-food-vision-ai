"""Run the local Gradio inference interface after training."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from ethiopian_food_vision.predict import FoodPredictor

CHECKPOINT = Path(os.environ.get("FOOD_VISION_CHECKPOINT", "artifacts/mobilenet_v3_small/best.pt"))


def create_demo(checkpoint: Path = CHECKPOINT) -> gr.Interface:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. "
            "Train the transfer model first; see README.md."
        )
    predictor = FoodPredictor(checkpoint)

    def classify(image):
        if image is None:
            return {}
        return {
            label.replace("_", " "): score for label, score in predictor.predict(image, top_k=3)
        }

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Upload a food photo"),
        outputs=gr.Label(num_top_classes=3, label="Top predictions"),
        title="Ethiopian Food Vision",
        description=(
            "Experimental image classification across 11 Ethiopian dishes. Predictions are not "
            "verified for dietary, medical, allergen, or nutrition decisions. This prototype has "
            "no out-of-distribution detector, so unfamiliar foods may receive a confident label."
        ),
        article=(
            "Local research demo. Upstream image reuse rights are unresolved; "
            "see `docs/sources.md`. "
            "Do not upload private images or rely on predictions for safety-critical decisions."
        ),
    )


if __name__ == "__main__":
    create_demo().launch(server_name="127.0.0.1", server_port=7860, share=False)
