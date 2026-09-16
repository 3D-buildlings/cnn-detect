"""Stage 1 + Stage 2: image inspection and hand-crafted feature extraction.

Logic ported from ``code/code/01_load_data.py`` and ``code/code/02_preprocess.py``
of the notebook workflow. Google-Drive mounting and in-script pip installs are
removed; every function is importable and takes explicit paths.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from src.config import RunPaths

logger = logging.getLogger("skyscrapper.data")

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Stage 2 constants
TARGET_SIZE = (256, 256)
TILE_SIZE = 256
NORMALIZE = True

METADATA_FILE = "image_metadata.csv"
FEATURES_FILE = "features.csv"


# ===========================================================================
# Stage 1 - data loading & inspection
# ===========================================================================
def inspect_image(path: str) -> Optional[dict]:
    """Return basic properties of an image, or None if unreadable."""
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        return {
            "file_name": os.path.basename(path),
            "width": w,
            "height": h,
            "channels": channels,
            "dtype": str(img.dtype),
            "min_val": int(img.min()),
            "max_val": int(img.max()),
            "is_corrupted": False,
        }
    except Exception:
        return {
            "file_name": os.path.basename(path),
            "width": None,
            "height": None,
            "channels": None,
            "dtype": None,
            "min_val": None,
            "max_val": None,
            "is_corrupted": True,
        }


def find_image_files(dataset_path: str) -> List[str]:
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    files = []
    for root, _dirs, names in os.walk(dataset_path):
        for f in names:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No image files found under {dataset_path}")
    return files


def load_and_inspect(dataset_path: str) -> pd.DataFrame:
    """Walk the dataset directory, read every image, collect metadata."""
    image_files = find_image_files(dataset_path)

    ext_counts: Dict[str, int] = {}
    for fp in image_files:
        ext = os.path.splitext(fp)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    logger.info("Found %d image files under %s (extensions: %s)",
                len(image_files), dataset_path, ext_counts)

    records = []
    corrupted = []
    for idx, fp in enumerate(image_files, 1):
        meta = inspect_image(fp)
        if meta is None:
            corrupted.append(os.path.basename(fp))
            continue
        meta["relative_path"] = os.path.relpath(fp, dataset_path)
        records.append(meta)

    df = pd.DataFrame(records)
    logger.info("Successfully read: %d images", len(df))
    if corrupted:
        logger.warning("Corrupted / unreadable files (%d): %s", len(corrupted), corrupted)
    else:
        logger.info("No corrupted files detected.")

    if not df.empty:
        logger.info("Image size stats (width x height): min=%s max=%s mean=(%.1f, %.1f)",
                    (df["width"].min(), df["height"].min()),
                    (df["width"].max(), df["height"].max()),
                    df["width"].mean(), df["height"].mean())
        logger.info("Channel counts: %s", df["channels"].value_counts().to_dict())
    return df


def save_metadata(df: pd.DataFrame, output_dir: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / METADATA_FILE
    df.to_csv(out_path, index=False)
    logger.info("Metadata saved to %s (%d rows)", out_path, len(df))
    return str(out_path)


def show_samples(dataset_path: str, output_dir: str, n: int = 4) -> Optional[str]:
    """Save a few sample images to <output_dir>/sample_images.png (best-effort)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Could not display samples (matplotlib missing): %s", exc)
        return None
    try:
        image_files = find_image_files(dataset_path)
        samples = image_files[:n]
        if not samples:
            return None
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
        if len(samples) == 1:
            axes = [axes]
        for ax, fp in zip(axes, samples):
            img = cv2.imread(fp)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(os.path.basename(fp), fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        out_file = out_dir / "sample_images.png"
        fig.savefig(out_file, dpi=100)
        plt.close(fig)
        logger.info("Sample images saved to %s", out_file)
        return str(out_file)
    except Exception as exc:
        logger.warning("Could not display samples: %s", exc)
        return None


def stage_load_and_inspect(run: RunPaths) -> str:
    """Run stage 1; returns the metadata CSV path."""
    df = load_and_inspect(run.inputs_dataset)
    metadata_csv = save_metadata(df, run.out_metadata)
    show_samples(run.inputs_dataset, run.out_metadata, n=4)
    return metadata_csv


# ===========================================================================
# Stage 2 - preprocessing & feature extraction
# ===========================================================================
def check_tiling_needed(width: int, height: int, tile_size: int) -> bool:
    return (width > tile_size * 2) or (height > tile_size * 2)


def tile_image(img: np.ndarray, tile_size: int) -> List[np.ndarray]:
    h, w = img.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y: y + tile_size, x: x + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros(
                    (tile_size, tile_size, tile.shape[2] if tile.ndim == 3 else 1),
                    dtype=tile.dtype,
                )
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            tiles.append(tile)
    return tiles


def extract_features(img: np.ndarray) -> dict:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    features: dict = {}

    for c, name in enumerate(["r", "g", "b"]):
        ch = img[:, :, c].astype(np.float32)
        features[f"mean_{name}"] = ch.mean()
        features[f"std_{name}"] = ch.std()
        features[f"min_{name}"] = ch.min()
        features[f"max_{name}"] = ch.max()

    for c, name in enumerate(["r", "g", "b"]):
        ch = img[:, :, c]
        hist, _ = np.histogram(ch, bins=8, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        for i in range(8):
            features[f"hist_{name}_{i}"] = hist[i]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    features["edge_density"] = (mag > 30).mean()
    features["edge_mean"] = mag.mean()
    features["edge_std"] = mag.std()

    ys, xs = np.mgrid[0: gray.shape[0], 0: gray.shape[1]]
    total = gray.sum() + 1e-8
    features["centroid_x"] = (xs * gray).sum() / total / gray.shape[1]
    features["centroid_y"] = (ys * gray).sum() / total / gray.shape[0]

    return features


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def preprocess(metadata_path: str, dataset_path: str) -> pd.DataFrame:
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path} (run stage 1 first)")

    meta_df = pd.read_csv(metadata_path)
    logger.info("Loaded metadata for %d images", len(meta_df))

    feature_rows = []
    tile_count = 0

    for _idx, row in meta_df.iterrows():
        img_path = os.path.join(dataset_path, row["relative_path"])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Skipping unreadable: %s", img_path)
            continue

        img = to_rgb(img)

        if NORMALIZE:
            img = img.astype(np.float32) / 255.0

        w, h = int(row["width"]), int(row["height"])

        if check_tiling_needed(w, h, TILE_SIZE):
            tiles = tile_image((img * 255).astype(np.uint8) if NORMALIZE else img, TILE_SIZE)
            for t_idx, tile in enumerate(tiles):
                feat = extract_features(tile)
                feat["image_name"] = row["file_name"]
                feat["tile_id"] = t_idx
                feature_rows.append(feat)
            tile_count += len(tiles)
        else:
            resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            feat = extract_features((resized * 255).astype(np.uint8) if NORMALIZE else resized)
            feat["image_name"] = row["file_name"]
            feat["tile_id"] = 0
            feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)
    logger.info("Extracted features: %d rows (from %d images, %d tiles)",
                len(feat_df), len(meta_df), tile_count)
    return feat_df


def save_features(df: pd.DataFrame, output_dir: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / FEATURES_FILE
    df.to_csv(out_path, index=False)
    logger.info("Features saved to %s (%d rows, %d cols)", out_path, len(df), len(df.columns))
    return str(out_path)


def stage_preprocess(run: RunPaths) -> str:
    """Run stage 2; returns the features CSV path."""
    metadata_csv = Path(run.out_metadata) / METADATA_FILE
    feat_df = preprocess(str(metadata_csv), run.inputs_dataset)
    return save_features(feat_df, run.out_features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit("This module is not meant to be run directly; import and call stage_* functions.")
