"""Stage 4 + Stage 6: finalize results and compare with HLP annotations.

Logic ported from ``code/code/04_save_results.py`` and ``code/code/06_hlp.py``
of the notebook workflow. Stage 5 (excel->csv) is optional and skipped because
no Excel reference file is used.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import RunPaths

logger = logging.getLogger("skyscrapper.evaluate")

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

CONTOURS_FILE = "building_contours.csv"
METRICS_FILE = "segmentation_metrics.json"
FINAL_CONTOURS_FILE = "building_contours_final.csv"
FINAL_METRICS_FILE = "segmentation_metrics_final.json"
COMPARISON_CSV = "comparison_results.csv"
RMS_JSON = "rms_metrics.json"

RED_LOWER1 = np.array([0, 70, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 70, 50])
RED_UPPER2 = np.array([180, 255, 255])


# ===========================================================================
# Stage 4 - save final segmentation results
# ===========================================================================
def enrich_contours(contours_path: str) -> pd.DataFrame:
    df = pd.read_csv(contours_path)
    df["run_timestamp"] = datetime.now().isoformat()
    df["pipeline_version"] = "v2.0_cnn_segmentation"
    return df


def enrich_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    metrics["run_timestamp"] = datetime.now().isoformat()
    metrics["pipeline_version"] = "v2.0_cnn_segmentation"
    metrics["notes"] = (
        "Building segmentation with a lightweight pixel-wise CNN (CPU) followed "
        "by morphological post-processing and contour filtering."
    )
    return metrics


def create_visualizations(run: RunPaths, n: int = 5) -> int:
    """Side-by-side original | mask | overlay for the first n dataset images."""
    Path(run.out_visualizations).mkdir(parents=True, exist_ok=True)

    image_files = []
    for root, _dirs, files in os.walk(run.inputs_dataset):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append((os.path.join(root, f), f))

    if not image_files:
        logger.warning("No images found for visualization.")
        return 0

    samples = image_files[:n]
    logger.info("Creating visualizations for %d images...", len(samples))
    created = 0
    for img_path, img_name in samples:
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(run.out_masks, f"{base_name}_mask.png")
        if not os.path.isfile(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        h, w = img.shape[:2]
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_rgb = cv2.resize(mask_rgb, (w, h))
        comparison = np.hstack([img, mask_rgb, blended])

        vis_path = os.path.join(run.out_visualizations, f"{base_name}_comparison.png")
        cv2.imwrite(vis_path, comparison)
        created += 1

    logger.info("Visualizations saved to %s (%d files)", run.out_visualizations, created)
    return created


def stage_finalize(run: RunPaths) -> Dict[str, str]:
    contours_path = os.path.join(run.out_predictions, CONTOURS_FILE)
    metrics_path = os.path.join(run.out_models, METRICS_FILE)
    for required in [contours_path, metrics_path]:
        if not os.path.isfile(required):
            raise FileNotFoundError(f"Missing required input from stage 3: {required}")

    contours_df = enrich_contours(contours_path)
    metrics = enrich_metrics(metrics_path)

    final_contours = os.path.join(run.out_predictions, FINAL_CONTOURS_FILE)
    contours_df.to_csv(final_contours, index=False)
    logger.info("Final contours saved to %s (%d rows)", final_contours, len(contours_df))

    final_metrics = os.path.join(run.out_models, FINAL_METRICS_FILE)
    with open(final_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final metrics saved to %s", final_metrics)

    created = create_visualizations(run, n=5)

    logger.info("SEGMENTATION RUN SUMMARY: total_buildings=%d images=%s area_ratio=%.2f%%",
                len(contours_df),
                contours_df["image_name"].nunique() if "image_name" in contours_df.columns else "N/A",
                metrics.get("building_area_ratio", 0) * 100)
    return {
        "final_contours": final_contours,
        "final_metrics": final_metrics,
        "visualizations": created,
    }


# ===========================================================================
# Stage 6 - HLP comparison
# ===========================================================================
def extract_red_mask(hlp_img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(hlp_img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hlp_img, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return red_mask


def calculate_rms(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1.astype(np.float32) / 255.0
    m2 = mask2.astype(np.float32) / 255.0
    rms = np.sqrt(np.mean((m1 - m2) ** 2))
    return float(rms)


def calculate_metrics(detected: np.ndarray, ground_truth: np.ndarray) -> dict:
    det = (detected > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)

    intersection = np.sum(det & gt)
    union = np.sum(det | gt)

    iou = intersection / max(union, 1)
    precision = intersection / max(np.sum(det), 1)
    recall = intersection / max(np.sum(gt), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    correct = np.sum(det == gt)
    total = det.shape[0] * det.shape[1]
    accuracy = correct / total

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pixel_accuracy": float(accuracy),
        "detected_pixels": int(np.sum(det)),
        "ground_truth_pixels": int(np.sum(gt)),
        "intersection_pixels": int(intersection),
    }


def create_visualization(original: np.ndarray, detected_mask: np.ndarray,
                         ground_truth_mask: np.ndarray, img_name: str) -> np.ndarray:
    h, w = original.shape[:2]

    det_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    det_rgb[detected_mask > 0] = [255, 100, 100]

    gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gt_rgb[ground_truth_mask > 0] = [100, 255, 100]

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    det_bin = (detected_mask > 0).astype(np.uint8)
    gt_bin = (ground_truth_mask > 0).astype(np.uint8)

    overlap = det_bin & gt_bin
    overlay[overlap > 0] = [255, 255, 255]

    det_only = det_bin & ~gt_bin
    overlay[det_only > 0] = [255, 150, 150]

    gt_only = gt_bin & ~det_bin
    overlay[gt_only > 0] = [150, 255, 150]

    det_rgb = cv2.resize(det_rgb, (w, h))
    gt_rgb = cv2.resize(gt_rgb, (w, h))
    overlay = cv2.resize(overlay, (w, h))

    comparison = np.hstack([original, det_rgb, gt_rgb, overlay])
    return comparison


def find_hlp_images(hlp_path: str) -> Dict[str, str]:
    images: Dict[str, str] = {}
    if not os.path.isdir(hlp_path):
        raise FileNotFoundError(f"HLP path does not exist: {hlp_path}")
    for f in os.listdir(hlp_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            base_name = os.path.splitext(f)[0].rstrip(".")
            images[base_name] = os.path.join(hlp_path, f)
    return images


def compare_masks(run: RunPaths) -> Tuple[pd.DataFrame, dict]:
    Path(run.out_hlp).mkdir(parents=True, exist_ok=True)
    Path(run.out_hlp_visualizations).mkdir(parents=True, exist_ok=True)

    training_names: List[str] = []
    if Path(run.training_images_path).is_file():
        with open(run.training_images_path) as f:
            training_names = json.load(f).get("training_images", [])

    hlp_images = find_hlp_images(run.inputs_hlp)

    detected_masks = {}
    if os.path.isdir(run.out_masks):
        for f in os.listdir(run.out_masks):
            if f.endswith("_mask.png"):
                base_name = f.replace("_mask.png", "")
                detected_masks[base_name] = os.path.join(run.out_masks, f)

    if not detected_masks:
        raise FileNotFoundError(f"No detected masks found in {run.out_masks} (run stage 3 first)")

    logger.info("Found %d detected masks, %d HLP images", len(detected_masks), len(hlp_images))

    matching = set(detected_masks.keys()) & set(hlp_images.keys())
    if training_names:
        before_count = len(matching)
        matching = matching - set(training_names)
        logger.info("Excluded %d training images from comparison", before_count - len(matching))
    logger.info("Matching images for evaluation: %d", len(matching))

    results = []
    all_rms = []

    for idx, base_name in enumerate(sorted(matching), 1):
        logger.info("  Comparing %d / %d: %s", idx, len(matching), base_name)
        det_mask = cv2.imread(detected_masks[base_name], cv2.IMREAD_GRAYSCALE)

        hlp_img = cv2.imread(hlp_images[base_name])
        if hlp_img is None:
            logger.warning("  Cannot read HLP image: %s", hlp_images[base_name])
            continue

        gt_mask = extract_red_mask(hlp_img)
        if det_mask.shape != gt_mask.shape:
            det_mask = cv2.resize(det_mask, (gt_mask.shape[1], gt_mask.shape[0]))

        rms = calculate_rms(det_mask, gt_mask)
        all_rms.append(rms)
        metrics = calculate_metrics(det_mask, gt_mask)

        result = {
            "image_name": base_name + ".png",
            "rms_error": rms,
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "detected_pixels": metrics["detected_pixels"],
            "ground_truth_pixels": metrics["ground_truth_pixels"],
        }
        results.append(result)

        original = cv2.imread(hlp_images[base_name])
        vis = create_visualization(original, det_mask, gt_mask, base_name)
        vis_path = os.path.join(run.out_hlp_visualizations, f"{base_name}_comparison.png")
        cv2.imwrite(vis_path, vis)

    results_df = pd.DataFrame(results)

    overall_metrics = {
        "total_images_compared": len(results),
        "mean_rms_error": float(np.mean(all_rms)) if all_rms else 0,
        "std_rms_error": float(np.std(all_rms)) if all_rms else 0,
        "min_rms_error": float(np.min(all_rms)) if all_rms else 0,
        "max_rms_error": float(np.max(all_rms)) if all_rms else 0,
        "mean_iou": float(results_df["iou"].mean()) if not results_df.empty else 0,
        "mean_precision": float(results_df["precision"].mean()) if not results_df.empty else 0,
        "mean_recall": float(results_df["recall"].mean()) if not results_df.empty else 0,
        "mean_f1": float(results_df["f1"].mean()) if not results_df.empty else 0,
    }
    return results_df, overall_metrics


def stage_hlp_comparison(run: RunPaths) -> Dict[str, str]:
    """Run stage 6; returns paths of comparison CSV and RMS metrics JSON."""
    results_df, metrics = compare_masks(run)

    comp_path = os.path.join(run.out_hlp, COMPARISON_CSV)
    results_df.to_csv(comp_path, index=False)
    logger.info("Comparison results saved to %s", comp_path)

    rms_path = os.path.join(run.out_hlp, RMS_JSON)
    with open(rms_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("RMS metrics saved to %s", rms_path)

    logger.info("HLP COMPARISON: images=%d mean_rms=%.4f mean_iou=%.4f "
                "mean_precision=%.4f mean_recall=%.4f mean_f1=%.4f",
                metrics["total_images_compared"], metrics["mean_rms_error"],
                metrics["mean_iou"], metrics["mean_precision"],
                metrics["mean_recall"], metrics["mean_f1"])
    return {"comparison_csv": comp_path, "rms_metrics_json": rms_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit("This module is not meant to be run directly; import and call stage_* functions.")
