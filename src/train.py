"""Stage 3: building segmentation with a lightweight pixel-wise CNN.

Replaces the original SVM classifier with a tiny fully-convolutional network
that classifies every pixel as building or background. The CNN is trained on
3 HLP images (using the red hand-painted mask as ground truth) and then
applied to all dataset images.

Design: CPU only, no BatchNorm, no inplace ops - keeps 8-bit quantization
accurate. ``torch`` is imported lazily so non-CNN stages never pay the cost.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import RunPaths

logger = logging.getLogger("skyscrapper.train")

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

CONTOURS_FILE = "building_contours.csv"
METRICS_FILE = "segmentation_metrics.json"


def _find_dataset_images(dataset_path: str) -> List[str]:
    """Recursively find all image files under ``dataset_path``."""
    image_files = []
    for root, _dirs, files in os.walk(dataset_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return image_files


def stage_train_and_segment(run: RunPaths) -> Dict[str, str]:
    """Train the pixel-wise CNN on HLP red masks and segment all dataset images."""
    import json as _json

    from src import config
    from src.cnn import (
        find_hlp_images,
        save_cnn,
        segment_image_files,
        train_cnn,
    )

    # --- Select training images ---
    hlp_images = find_hlp_images(run.inputs_hlp)
    if not hlp_images:
        raise FileNotFoundError(f"No HLP images found under {run.inputs_hlp}")

    training_names = sorted(hlp_images.keys())[: config.CNN_TRAIN_IMAGES]
    logger.info("Training images (%d): %s", len(training_names), training_names)

    # Save training images list for downstream stages
    Path(run.out_models).mkdir(parents=True, exist_ok=True)
    with open(run.training_images_path, "w") as f:
        _json.dump({"training_images": training_names}, f)

    # --- Train CNN ---
    import torch

    device = torch.device("cpu")
    model, train_stats = train_cnn(
        run.inputs_hlp,
        training_names,
        max_side=config.CNN_MAX_SIDE,
        patch_size=config.CNN_PATCH_SIZE,
        patches_per_image=config.CNN_PATCHES_PER_IMAGE,
        batch_size=config.CNN_BATCH_SIZE,
        epochs=config.CNN_EPOCHS,
        lr=config.CNN_LR,
        base_channels=config.CNN_BASE_CHANNELS,
        seed=config.CNN_SEED,
        device=device,
    )

    # --- Save CNN model ---
    save_cnn(model, run.cnn_model_path, base=config.CNN_BASE_CHANNELS, in_channels=config.CNN_IN_CHANNELS)

    # --- Segment all dataset images ---
    image_paths = _find_dataset_images(run.inputs_dataset)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {run.inputs_dataset}")

    contours_df, metrics = segment_image_files(
        model,
        image_paths,
        run.out_masks,
        max_side=config.CNN_MAX_SIDE,
        threshold=config.CNN_THRESHOLD,
        device=device,
    )

    contours_path = os.path.join(run.out_predictions, CONTOURS_FILE)
    Path(run.out_predictions).mkdir(parents=True, exist_ok=True)
    contours_df.to_csv(contours_path, index=False)

    metrics.update(
        {
            "model_type": "pixel-wise CNN",
            "cnn": train_stats,
            "parameters": {
                "patch_size": config.CNN_PATCH_SIZE,
                "batch_size": config.CNN_BATCH_SIZE,
                "epochs": config.CNN_EPOCHS,
                "learning_rate": config.CNN_LR,
                "base_channels": config.CNN_BASE_CHANNELS,
                "min_area": 200,
                "min_solidity": 0.4,
                "max_axis_ratio": 8.0,
            },
        }
    )
    metrics_path = os.path.join(run.out_models, METRICS_FILE)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Segmentation done: %d images, %d buildings",
        metrics["images_processed"],
        metrics["total_buildings_detected"],
    )
    return {"contours": contours_path, "metrics": metrics_path, "model": run.cnn_model_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit("This module is not meant to be run directly; import and call stage_* functions.")
