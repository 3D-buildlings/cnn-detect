"""8-bit quantization of the trained pixel-wise CNN (CPU only).

What is quantized
-----------------
* **Weights** -> signed 8-bit integers (``int8``).
* **Inputs / activations** -> unsigned 8-bit integers (``uint8``). The first
  ``QuantizeLinear`` node in the exported graph turns the float input into
  ``uint8`` using a calibrated scale / zero-point, so the network effectively
  consumes 8-bit inputs.

The primary path is ONNX Runtime *static* post-training quantization (weights +
activations), which is the standard way to get a real 8-bit CPU model. If the
ONNX toolchain is unavailable, the code falls back to PyTorch eager static
quantization.

After quantizing, the stage re-runs inference with the quantized model so the
whole downstream flow (masks, contours, metrics) is exercised on 8-bit weights
and 8-bit activations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("skyscrapper.quantize")


# ---------------------------------------------------------------------------
# ONNX export (float model)
# ---------------------------------------------------------------------------
def export_onnx(model, onnx_path: Path, in_channels: int = 3, patch_size: int = 64) -> Path:
    """Export the float CNN to ONNX with dynamic batch/height/width axes."""
    import torch

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(1, in_channels, patch_size, patch_size)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )
    logger.info("Float ONNX model exported to %s", onnx_path)
    return onnx_path


# ---------------------------------------------------------------------------
# Calibration data (representative inputs -> used for activation ranges)
# ---------------------------------------------------------------------------
def build_calibration_arrays(
    hlp_path: str,
    training_names: list,
    *,
    patch_size: int = 64,
    count: int = 24,
    seed: int = 42,
) -> list:
    """Return a list of ``(1, 3, ps, ps)`` float32 arrays for calibration."""
    from .cnn import prepare_pairs

    pairs = prepare_pairs(hlp_path, training_names)
    if not pairs:
        raise RuntimeError(f"No HLP images available for calibration under {hlp_path}")

    rng = np.random.default_rng(seed)
    arrays = []
    while len(arrays) < count:
        img, _mask = pairs[len(arrays) % len(pairs)]
        h, w = img.shape[:2]
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
            h, w = img.shape[:2]
        y = int(rng.integers(0, h - patch_size + 1))
        x = int(rng.integers(0, w - patch_size + 1))
        patch = img[y : y + patch_size, x : x + patch_size]
        arrays.append(np.ascontiguousarray(patch.transpose(2, 0, 1))[None].astype(np.float32))
    logger.info("Built %d calibration samples (patch=%d)", len(arrays), patch_size)
    return arrays


# ---------------------------------------------------------------------------
# Static 8-bit quantization (ONNX Runtime)
# ---------------------------------------------------------------------------
def _quantize_static_onnx(onnx_path: Path, quantized_path: Path, calib_arrays: list) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _CalibReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._input_name = "input"
            self._items = [{self._input_name: array} for array in calib_arrays]
            self._iterator = iter(self._items)

        def get_next(self):
            return next(self._iterator, None)

        def rewind(self) -> None:
            self._iterator = iter(self._items)

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        calibration_data_reader=_CalibReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    logger.info("ONNX Runtime static 8-bit quantization written to %s", quantized_path)


def _quantize_static_torch(model, calib_arrays: list, torch_path: Path):
    """Fallback: PyTorch eager static quantization (qnnpack, CPU)."""
    import torch
    from torch.ao.quantization import convert, get_default_qconfig, prepare

    model = model.cpu().eval()
    model.qconfig = get_default_qconfig("qnnpack")
    prepared = prepare(model, inplace=False)
    with torch.no_grad():
        for array in calib_arrays:
            prepared(torch.from_numpy(array))
    quantized = convert(prepared, inplace=False)
    scripted = torch.jit.script(quantized)
    torch_path = Path(torch_path)
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(torch_path))
    logger.info("PyTorch static quantized model written to %s", torch_path)
    return torch_path


# ---------------------------------------------------------------------------
# Input quantization parameters
# ---------------------------------------------------------------------------
def input_quant_params(quantized_path: Path, input_name: str = "input"):
    """Read the scale / zero-point used to quantize the graph input."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return None, None

    model = onnx.load(str(quantized_path))
    initializers = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and input_name in node.input:
            scale = initializers.get(node.input[1])
            zero_point = initializers.get(node.input[2])
            if scale is not None:
                return float(np.ravel(scale)[0]), (
                    int(np.ravel(zero_point)[0]) if zero_point is not None else 0
                )
    return None, None


def quantize_input_uint8(image_float: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize a float image (NCHW, [0, 1]) to ``uint8`` using the graph params."""
    scaled = np.rint(image_float / scale) + zero_point
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Float vs quantized error
# ---------------------------------------------------------------------------
def _compare_outputs(onnx_path: Path, quantized_path: Path, calib_arrays: list) -> dict:
    import onnxruntime as ort

    sess_float = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    input_name = sess_float.get_inputs()[0].name

    errors = []
    for array in calib_arrays[: min(len(calib_arrays), 8)]:
        out_f = sess_float.run(None, {input_name: array})[0]
        out_q = sess_quant.run(None, {input_name: array})[0]
        errors.append(np.abs(out_f - out_q))

    stacked = np.concatenate([e.ravel() for e in errors]) if errors else np.array([0.0])
    return {
        "mean_absolute_error": float(stacked.mean()),
        "max_absolute_error": float(stacked.max()),
        "compared_samples": len(errors),
    }


# ---------------------------------------------------------------------------
# Quantized inference (masks + contours + metrics)
# ---------------------------------------------------------------------------
def _onnx_predict_mask(
    session, rgb: np.ndarray, input_name: str, output_name: str,
    threshold: float, max_side: int = 256,
):
    import cv2

    from .cnn import _resize_max_side, clean_mask

    h, w = rgb.shape[:2]
    resized = _resize_max_side(rgb, max_side)
    rh, rw = resized.shape[:2]
    pad_h = (2 - rh % 2) % 2
    pad_w = (2 - rw % 2) % 2
    if pad_h or pad_w:
        resized = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)))

    array = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    logits = session.run([output_name], {input_name: array})[0]
    prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    prob = prob[:rh, :rw]
    if (rh, rw) != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= threshold).astype(np.uint8) * 255
    return clean_mask(mask)


def segment_with_quantized(
    quantized_path: Path,
    image_paths: list,
    masks_dir: str,
    *,
    hlp_path: str | None = None,
    max_side: int = 256,
    threshold: float = 0.5,
) -> Tuple:
    """Run 8-bit inference over the dataset and collect contours + agreement."""
    import cv2
    import onnxruntime as ort
    import pandas as pd

    from .cnn import find_buildings, find_hlp_images, extract_red_mask

    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    hlp_images = find_hlp_images(hlp_path) if hlp_path else {}

    rows = []
    agreements = []
    hlp_scores = []
    processed = 0

    for idx, img_path in enumerate(image_paths, 1):
        img_path = Path(img_path)
        if idx % 10 == 0 or idx == len(image_paths):
            logger.info("  quantized inference %d/%d: %s", idx, len(image_paths), img_path.name)

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 1:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        processed += 1

        mask = _onnx_predict_mask(session, rgb, input_name, output_name, threshold, max_side)
        cv2.imwrite(str(masks_dir / f"{img_path.stem}_mask.png"), mask)

        h, w = rgb.shape[:2]
        for building in find_buildings(mask, h * w):
            building["image_name"] = img_path.name
            rows.append(building)

        hlp_path_match = hlp_images.get(img_path.stem)
        if hlp_path_match is not None:
            bgr = cv2.imread(hlp_path_match, cv2.IMREAD_COLOR)
            if bgr is not None:
                gt = extract_red_mask(bgr)
                if gt.shape[:2] != mask.shape[:2]:
                    gt = cv2.resize(gt, (w, h))
                det = (mask > 0).astype(np.uint8)
                gt_bin = (gt > 0).astype(np.uint8)
                intersection = int(np.sum(det & gt_bin))
                union = int(np.sum(det | gt_bin))
                iou = intersection / max(union, 1)
                precision = intersection / max(int(np.sum(det)), 1)
                recall = intersection / max(int(np.sum(gt_bin)), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                hlp_scores.append({"iou": iou, "precision": precision, "recall": recall, "f1": f1})

    contours_df = pd.DataFrame(rows)
    summary = {
        "quantized_images_processed": processed,
        "quantized_buildings_detected": int(len(contours_df)),
    }
    if hlp_scores:
        summary["quantized_mean_iou"] = float(np.mean([s["iou"] for s in hlp_scores]))
        summary["quantized_mean_f1"] = float(np.mean([s["f1"] for s in hlp_scores]))
        summary["quantized_mean_precision"] = float(
            np.mean([s["precision"] for s in hlp_scores])
        )
        summary["quantized_mean_recall"] = float(
            np.mean([s["recall"] for s in hlp_scores])
        )
    return contours_df, summary, agreements


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def quantize_model_8bit(
    model,
    onnx_path: Path,
    quantized_path: Path,
    hlp_path: str,
    training_names: list,
    *,
    in_channels: int = 3,
    patch_size: int = 64,
    calib_samples: int = 24,
    seed: int = 42,
    fallback_torch_path: Path | None = None,
) -> dict:
    """Quantize ``model`` to 8 bits (weights + activations/inputs).

    Returns a metrics dictionary describing the quantization (sizes, errors and
    the input quantization parameters).
    """
    onnx_path = Path(onnx_path)
    quantized_path = Path(quantized_path)

    export_onnx(model, onnx_path, in_channels=in_channels, patch_size=patch_size)
    calib_arrays = build_calibration_arrays(
        hlp_path, training_names,
        patch_size=patch_size, count=calib_samples, seed=seed,
    )

    original_size = onnx_path.stat().st_size
    try:
        _quantize_static_onnx(onnx_path, quantized_path, calib_arrays)
        engine = "onnxruntime-static-qdq"
        quantized_model_path = quantized_path
        error = _compare_outputs(onnx_path, quantized_path, calib_arrays)
        scale, zero_point = input_quant_params(quantized_path)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("ONNX static quantization failed (%s); using PyTorch fallback", exc)
        fallback_torch_path = fallback_torch_path or quantized_path.with_suffix(".pt")
        quantized_model_path = _quantize_static_torch(model, calib_arrays, fallback_torch_path)
        engine = "torch-static-qnnpack"
        error = {"mean_absolute_error": None, "max_absolute_error": None, "compared_samples": 0}
        scale, zero_point = None, None

    quantized_size = Path(quantized_model_path).stat().st_size
    metrics = {
        "quantization_bits": 8,
        "engine": engine,
        "weights_dtype": "int8",
        "activations_dtype": "uint8",
        "input_dtype": "uint8",
        "input_scale": scale,
        "input_zero_point": zero_point,
        "calibration_samples": len(calib_arrays),
        "float_model_path": str(onnx_path),
        "quantized_model_path": str(quantized_model_path),
        "float_model_size_mb": round(original_size / (1024 * 1024), 4),
        "quantized_model_size_mb": round(quantized_size / (1024 * 1024), 4),
        "compression_ratio": round(original_size / max(quantized_size, 1), 3),
        "compared_samples": error["compared_samples"],
        "mean_absolute_error": error["mean_absolute_error"],
        "max_absolute_error": error["max_absolute_error"],
    }
    logger.info(
        "8-bit quantization done: %.3f MB -> %.3f MB (x%.2f), MAE=%s",
        metrics["float_model_size_mb"],
        metrics["quantized_model_size_mb"],
        metrics["compression_ratio"],
        metrics["mean_absolute_error"],
    )
    return metrics
