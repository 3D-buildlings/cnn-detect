"""Lightweight pixel-wise CNN for building segmentation (CPU only).

The model is a tiny fully-convolutional network that classifies every pixel as
* building (1) or background (0). It is trained on HLP images, using the red
hand-painted mask as the ground truth, and then applied to the whole dataset.

Design notes
------------
* CPU only - small channel counts, no huge dense layers, no CUDA dependency.
* Fully convolutional - the same weights accept any image size, so the network
  is effectively a per-pixel classifier that sees a local receptive field.
* No BatchNorm / no inplace ops - keeps the later 8-bit quantization step
  simple and accurate.

``torch`` is imported lazily inside the functions so that stages which do not
need the CNN (e.g. ``fetch_dataset``) never pay the import cost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("skyscrapper.cnn")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Red mask extraction (HSV thresholds; red wraps around 0/180)
RED_LOWER1 = np.array([0, 70, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 70, 50])
RED_UPPER2 = np.array([180, 255, 255])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(base: int = 16, in_channels: int = 3):
    """Build a fresh (untrained) lightweight pixel-wise CNN on CPU."""
    import torch.nn as nn

    class PixelSegCNN(nn.Module):
        """Tiny fully-convolutional pixel classifier."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, base, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(base, base, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
                nn.Conv2d(base, base * 2, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(base * 2, base * 2, 3, padding=1),
                nn.ReLU(inplace=False),
            )
            self.head = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(base * 2, base, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(base, 1, 1),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return PixelSegCNN()


def save_cnn(model, path: Path, base: int = 16, in_channels: int = 3) -> None:
    """Persist the CNN weights + architecture metadata (no pickled classes)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixel_seg_cnn_v1",
            "state_dict": model.state_dict(),
            "base": base,
            "in_channels": in_channels,
        },
        str(path),
    )
    logger.info("CNN model saved to %s", path)


def load_cnn(path: Path, device=None):
    """Load a CNN saved by :func:`save_cnn`."""
    import torch

    device = device or torch.device("cpu")
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model = build_model(checkpoint.get("base", 16), checkpoint.get("in_channels", 3))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Training data (HLP images + red masks)
# ---------------------------------------------------------------------------
def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def extract_red_mask(hlp_img: np.ndarray) -> np.ndarray:
    """Extract red-painted (building) pixels from an HLP image.

    Returns a binary ``uint8`` mask (255 = building, 0 = background).
    """
    import cv2

    hsv = cv2.cvtColor(hlp_img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return red_mask


def find_hlp_images(hlp_path: str) -> Dict[str, str]:
    """Return ``{base_name: path}`` for images directly inside ``hlp_path``."""
    import os

    images: Dict[str, str] = {}
    if not os.path.isdir(hlp_path):
        return images
    for f in sorted(os.listdir(hlp_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            base_name = os.path.splitext(f)[0].rstrip(".")
            images[base_name] = os.path.join(hlp_path, f)
    return images


def prepare_pairs(hlp_path: str, names: list, max_side: int = 256) -> list:
    """Load ``(rgb_float, binary_mask)`` pairs for the given HLP image names."""
    import cv2

    images = find_hlp_images(hlp_path)
    pairs = []
    for name in names:
        path = images.get(name)
        if path is None:
            logger.warning("  HLP image not found: %s", name)
            continue
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("  Cannot read HLP image: %s", path)
            continue
        mask = extract_red_mask(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize_max_side(rgb, max_side)
        mask = _resize_max_side(mask, max_side)
        pairs.append(
            (
                rgb.astype(np.float32) / 255.0,
                (mask > 0).astype(np.float32),
            )
        )
    return pairs


def _make_dataset(pairs: list, patch_size: int, patches_per_image: int):
    import torch
    from torch.utils.data import Dataset

    class HlpSegDataset(Dataset):
        def __init__(self) -> None:
            self.pairs = pairs
            self.patch_size = patch_size
            self.patches_per_image = patches_per_image

        def __len__(self) -> int:
            return max(len(self.pairs), 1) * self.patches_per_image

        def __getitem__(self, idx):
            img, mask = self.pairs[idx % len(self.pairs)]
            ps = self.patch_size
            h, w = img.shape[:2]

            if h < ps or w < ps:
                pad_h = max(0, ps - h)
                pad_w = max(0, ps - w)
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
                h, w = img.shape[:2]

            y = int(np.random.randint(0, h - ps + 1))
            x = int(np.random.randint(0, w - ps + 1))
            patch = img[y : y + ps, x : x + ps]
            patch_mask = mask[y : y + ps, x : x + ps]

            if np.random.rand() < 0.5:
                patch = patch[:, ::-1]
                patch_mask = patch_mask[:, ::-1]
            if np.random.rand() < 0.5:
                patch = patch[::-1, :]
                patch_mask = patch_mask[::-1, :]

            patch = np.ascontiguousarray(patch.transpose(2, 0, 1))
            patch_mask = np.ascontiguousarray(patch_mask[None, ...])
            return (
                torch.from_numpy(patch),
                torch.from_numpy(patch_mask),
            )

    return HlpSegDataset()


def train_cnn(
    hlp_path: str,
    training_names: list,
    *,
    max_side: int = 256,
    patch_size: int = 64,
    patches_per_image: int = 16,
    batch_size: int = 16,
    epochs: int = 8,
    lr: float = 1e-3,
    base_channels: int = 16,
    seed: int = 42,
    device=None,
) -> Tuple:
    """Train the pixel-wise CNN on HLP pairs and return ``(model, stats)``."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or torch.device("cpu")
    pairs = prepare_pairs(hlp_path, training_names, max_side=max_side)
    if not pairs:
        raise RuntimeError(f"No usable HLP images found under {hlp_path}")

    dataset = _make_dataset(pairs, patch_size, patches_per_image)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = build_model(base=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    positives = float(sum(float(m.sum()) for _, m in pairs))
    total = float(sum(m.size for _, m in pairs))
    pos_weight = torch.tensor(
        [(total - positives) / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.info(
        "Training pixel-wise CNN on %d HLP image(s), patch=%d, epochs=%d, pos_weight=%.2f",
        len(pairs),
        patch_size,
        epochs,
        float(pos_weight.item()),
    )

    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        avg_loss = running / max(batches, 1)
        history.append(avg_loss)
        logger.info("  epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

    model.eval()
    stats = {
        "epochs": epochs,
        "final_loss": history[-1] if history else None,
        "loss_history": history,
        "patch_size": patch_size,
        "base_channels": base_channels,
        "training_images": list(training_names),
        "device": str(device),
    }
    return model, stats


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _to_tensor(rgb_float: np.ndarray):
    import torch

    chw = np.ascontiguousarray(rgb_float.transpose(2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def predict_prob(model, rgb: np.ndarray, max_side: int = 256, device=None) -> np.ndarray:
    """Return the per-pixel building probability map for an RGB image."""
    import cv2
    import torch

    device = device or torch.device("cpu")
    h, w = rgb.shape[:2]
    resized = _resize_max_side(rgb, max_side)
    rh, rw = resized.shape[:2]

    pad_h = (2 - rh % 2) % 2
    pad_w = (2 - rw % 2) % 2
    if pad_h or pad_w:
        resized = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)))

    tensor = _to_tensor(resized.astype(np.float32) / 255.0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = prob[:rh, :rw]
    if (rh, rw) != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological clean-up of a binary mask (close then open)."""
    import cv2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def predict_mask(
    model, rgb: np.ndarray, max_side: int = 256, device=None, threshold: float = 0.5
) -> np.ndarray:
    """Segment one RGB image -> cleaned binary mask (255 = building)."""
    prob = predict_prob(model, rgb, max_side=max_side, device=device)
    mask = (prob >= threshold).astype(np.uint8) * 255
    return clean_mask(mask)


def find_buildings(mask: np.ndarray, img_area: int) -> list:
    """Detect building contours and filter them by area/solidity/axis ratio."""
    import cv2

    MIN_AREA = 200
    MAX_AREA_RATIO = 0.5
    MIN_SOLIDITY = 0.4
    MAX_AXIS_RATIO = 8.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        if area > img_area * MAX_AREA_RATIO:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        axis_ratio = max(bw, bh) / (min(bw, bh) + 1)
        if axis_ratio > MAX_AXIS_RATIO:
            continue

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + bw // 2, y + bh // 2

        buildings.append(
            {
                "building_id": f"B{i + 1:03d}",
                "contour_index": i,
                "area_px": int(area),
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": bw,
                "bbox_height": bh,
                "solidity": round(float(solidity), 3),
                "axis_ratio": round(float(axis_ratio), 3),
            }
        )
    return buildings


def segment_image_files(
    model,
    image_paths: list,
    masks_dir: str,
    *,
    max_side: int = 256,
    threshold: float = 0.5,
    device=None,
) -> Tuple:
    """Segment every image, saving masks and returning ``(contours_df, metrics)``."""
    import cv2
    import pandas as pd

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_pixels = 0
    building_pixels = 0
    processed = 0
    failed = 0

    for idx, img_path in enumerate(image_paths, 1):
        img_path = Path(img_path)
        if idx % 10 == 0 or idx == len(image_paths):
            logger.info("  segmenting %d/%d: %s", idx, len(image_paths), img_path.name)

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            failed += 1
            logger.warning("  cannot read: %s", img_path)
            continue

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 1:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = predict_mask(model, rgb, max_side=max_side, threshold=threshold, device=device)
        processed += 1
        h, w = rgb.shape[:2]

        cv2.imwrite(str(masks_dir / f"{img_path.stem}_mask.png"), mask)

        total_pixels += h * w
        building_pixels += int(np.sum(mask > 0))

        for building in find_buildings(mask, h * w):
            building["image_name"] = img_path.name
            rows.append(building)

    contours_df = pd.DataFrame(rows)
    metrics = {
        "total_images": len(image_paths),
        "images_processed": processed,
        "images_failed": failed,
        "total_buildings_detected": int(len(contours_df)),
        "avg_buildings_per_image": round(len(contours_df) / max(processed, 1), 2),
        "total_pixels": int(total_pixels),
        "building_pixels": int(building_pixels),
        "building_area_ratio": round(building_pixels / max(total_pixels, 1), 4),
        "model_type": "pixel-wise CNN",
        "device": "cpu",
    }
    return contours_df, metrics
