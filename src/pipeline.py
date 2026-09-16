"""End-to-end pipeline runner.

Orchestrates every stage in the same order as the original notebook:

    fetch inputs (MinIO) -> load/inspect -> preprocess -> train/segment
        -> quantize -> finalize results -> HLP comparison -> MLflow logging
        -> upload outputs

The Airflow DAG splits these steps into individual tasks; this module keeps a
single-process ``run_pipeline()`` so the same code can be executed from the
notebook (``mlops.ipynb``) and from tests.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from src import config, data, evaluate, storage, track, train

logger = logging.getLogger("skyscrapper.pipeline")


def run_pipeline(
    run_id: str,
    base_dir: str | None = None,
    *,
    fetch_from_minio: bool = True,
    upload_outputs: bool = True,
) -> Dict[str, Any]:
    """Run the whole skyscrapper MLOps pipeline for ``run_id``."""
    run = config.RunPaths.build(run_id, base_dir)
    run.create_dirs()

    if fetch_from_minio:
        client = storage.default_client()
        client.download_prefix(config.DATASET_PREFIX, run.inputs_dataset)
        client.download_prefix(config.HLP_PREFIX, run.inputs_hlp)

    data.stage_load_and_inspect(run)
    data.stage_preprocess(run)
    train.stage_train_and_segment(run)

    # Quantize the trained CNN to INT8
    _run_quantize(run)

    evaluate.stage_finalize(run)
    evaluate.stage_hlp_comparison(run)
    summary = track.log_run_to_mlflow(run)

    if upload_outputs:
        s3_prefix = upload_outputs_to_minio(run)
        summary["s3_uploaded_prefix"] = s3_prefix

    return summary


def _run_quantize(run: config.RunPaths) -> None:
    """Quantize the trained CNN to INT8 and re-run inference."""
    from src import cnn, quantize

    if not Path(run.cnn_model_path).is_file():
        logger.warning("CNN model not found at %s; skipping quantization", run.cnn_model_path)
        return

    training_names = []
    if Path(run.training_images_path).is_file():
        with open(run.training_images_path) as f:
            training_names = json.load(f).get("training_images", [])

    model = cnn.load_cnn(run.cnn_model_path)

    quant_metrics = quantize.quantize_model_8bit(
        model,
        onnx_path=Path(run.cnn_onnx_path),
        quantized_path=Path(run.quantized_model_path),
        hlp_path=run.inputs_hlp,
        training_names=training_names,
        in_channels=config.CNN_IN_CHANNELS,
        patch_size=config.CNN_PATCH_SIZE,
        calib_samples=config.QUANT_CALIB_SAMPLES,
        seed=config.CNN_SEED,
    )

    image_paths = []
    for root, _dirs, files in os.walk(run.inputs_dataset):
        for f in files:
            if os.path.splitext(f)[1].lower() in data.IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    contours_df, summary, _ = quantize.segment_with_quantized(
        Path(quant_metrics["quantized_model_path"]),
        image_paths,
        run.out_quantized_masks,
        hlp_path=run.inputs_hlp,
        max_side=config.CNN_MAX_SIDE,
        threshold=config.CNN_THRESHOLD,
    )

    contours_path = os.path.join(run.out_predictions, "building_contours_quantized.csv")
    contours_df.to_csv(contours_path, index=False)

    quant_metrics["run_id"] = run.run_id
    quant_metrics.update(summary)

    with open(run.quantization_metrics_path, "w") as f:
        json.dump(quant_metrics, f, indent=2)

    logger.info("Quantization done: %s, x%.2f compression",
                quant_metrics.get("engine", "n/a"),
                quant_metrics.get("compression_ratio", 0))


def upload_outputs_to_minio(run: config.RunPaths, extra: Dict[str, Any] | None = None) -> str:
    """Upload the whole per-run ``outputs`` tree under ``runs/<run_id>/``."""
    client = storage.default_client()
    prefix = config.s3_run_prefix(run.run_id)

    client.upload_dir(run.outputs_dir, prefix)
    objects = client.list_objects(prefix)
    manifest = {
        "dag_run_id": run.run_id,
        "local_outputs_dir": run.outputs_dir,
        "s3_runs_uri": config.s3_runs_uri(run.run_id),
        "object_count": len(objects),
        "objects": sorted(objects),
    }
    if extra:
        manifest["extra"] = extra
    storage.write_run_manifest(client, run.run_id, manifest)
    logger.info("All outputs uploaded to %s (%d objects)",
                config.s3_runs_uri(run.run_id), len(objects))
    return prefix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline <run_id>")
    summary = run_pipeline(sys.argv[1])
    print(summary)
