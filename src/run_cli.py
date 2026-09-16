#!/usr/bin/env python3
"""Stage runner used by the Airflow DAG.

Each Airflow task shells out to this script (``python run_cli.py <stage>
<dag_run_id>``). Running as a separate process makes the pipeline independent of
the import/sys.path restrictions of the Airflow task runner: the script inserts
its own project root on ``sys.path`` (``sys.path[0]`` is always the script's
directory), exactly like ``seed_minio.py``.

Usage:
    python run_cli.py <stage> <run_id>

Stages:
    fetch_dataset | fetch_hlp | prepare_data | train_segmentation |
    quantize_model | finalize_results | compare_hlp | log_mlflow |
    upload_outputs | report
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# sys.path[0] == directory of this script == coder/mlops2 (project root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, data, evaluate, pipeline, storage, track, train  # noqa: E402

logger = logging.getLogger("run_cli")


def _run(run_id: str) -> config.RunPaths:
    run = config.RunPaths.build(run_id)
    run.create_dirs()
    return run


def fetch_dataset(run: config.RunPaths) -> None:
    client = storage.default_client()
    try:
        files = client.download_prefix(config.DATASET_PREFIX, run.inputs_dataset)
    except storage.StorageError as exc:
        print(f"ERROR: {exc}")
        print("Upload the initial data first (seed the 'dataset/' prefix in MinIO).")
        sys.exit(1)
    print(f"Fetched {len(files)} dataset objects -> {run.inputs_dataset}")


def fetch_hlp(run: config.RunPaths) -> None:
    client = storage.default_client()
    try:
        files = client.download_prefix(config.HLP_PREFIX, run.inputs_hlp)
    except storage.StorageError as exc:
        print(f"ERROR: {exc}")
        print("Upload the initial data first (seed the 'hlp/' prefix in MinIO).")
        sys.exit(1)
    print(f"Fetched {len(files)} hlp objects -> {run.inputs_hlp}")


def prepare_data(run: config.RunPaths) -> None:
    metadata_csv = data.stage_load_and_inspect(run)
    features_csv = data.stage_preprocess(run)
    print(f"metadata: {metadata_csv}")
    print(f"features: {features_csv}")


def train_segmentation(run: config.RunPaths) -> None:
    paths = train.stage_train_and_segment(run)
    print(f"train outputs: {paths}")


def quantize_model(run: config.RunPaths) -> None:
    """Quantize the trained CNN to INT8 and re-run inference."""
    from src import cnn, quantize

    if not Path(run.cnn_model_path).is_file():
        print(f"ERROR: CNN model not found: {run.cnn_model_path} (run train_segmentation first)")
        sys.exit(1)

    # Load training images list
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

    # Re-run inference with quantized model
    image_paths = []
    for root, _dirs, files in os.walk(run.inputs_dataset):
        for f in files:
            if os.path.splitext(f)[1].lower() in data.IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))

    from src.cnn import find_hlp_images

    hlp_images = find_hlp_images(run.inputs_hlp)
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

    print(f"quantized inference: {summary.get('quantized_buildings_detected', 0)} buildings")
    print(f"quantized model: {quant_metrics['quantized_model_size_mb']:.3f} MB "
          f"(x{quant_metrics['compression_ratio']:.2f} compression)")


def finalize_results(run: config.RunPaths) -> None:
    paths = evaluate.stage_finalize(run)
    print(f"finalize outputs: {paths}")


def compare_hlp(run: config.RunPaths) -> None:
    paths = evaluate.stage_hlp_comparison(run)
    print(f"hlp comparison outputs: {paths}")


def log_mlflow(run: config.RunPaths) -> None:
    summary = track.log_run_to_mlflow(run)
    print(f"mlflow_run_id: {summary['mlflow_run_id']}")
    print(f"experiment: {summary['experiment_name']}")
    print(f"tracking_uri: {summary['mlflow_tracking_uri']}")


def upload_outputs(run: config.RunPaths) -> None:
    prefix = pipeline.upload_outputs_to_minio(run)
    print(f"uploaded outputs -> s3://{config.minio_bucket()}/{prefix}")


def report(run: config.RunPaths) -> None:
    summary = track.load_summary(run)
    metrics = summary.get("metrics", {})
    metrics_lines = "\n".join(f"    {k}: {v}" for k, v in metrics.items()) or "    (none)"

    quant = {}
    if Path(run.quantization_metrics_path).is_file():
        with open(run.quantization_metrics_path) as f:
            quant = json.load(f)

    block = "\n".join([
        "=" * 60,
        "MLOps Pipeline Completed Successfully",
        "=" * 60,
        f"Airflow DAG Run ID: {run.run_id}",
        "",
        f"MLflow Experiment: {summary.get('experiment_name', config.mlflow_experiment_name())}",
        f"MLflow Run ID: {summary.get('mlflow_run_id', 'N/A')}",
        f"MLflow Tracking URI: {summary.get('mlflow_tracking_uri', config.mlflow_tracking_uri())}",
        "",
        "Metrics:",
        metrics_lines,
        "",
        "Quantization:",
        f"  Engine     : {quant.get('engine', 'n/a')}",
        f"  FP32 size  : {quant.get('float_model_size_mb', 'n/a')} MB",
        f"  INT8 size  : {quant.get('quantized_model_size_mb', 'n/a')} MB",
        f"  Compression: x{quant.get('compression_ratio', 'n/a')}",
        f"  MAE        : {quant.get('mean_absolute_error', 'n/a')}",
        "",
        f"Artifacts: {summary.get('s3_runs_uri', config.s3_runs_uri(run.run_id))}",
        "=" * 60,
    ])
    print(block)


STAGES = {
    "fetch_dataset": fetch_dataset,
    "fetch_hlp": fetch_hlp,
    "prepare_data": prepare_data,
    "train_segmentation": train_segmentation,
    "quantize_model": quantize_model,
    "finalize_results": finalize_results,
    "compare_hlp": compare_hlp,
    "log_mlflow": log_mlflow,
    "upload_outputs": upload_outputs,
    "report": report,
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: run_cli.py <stage> <dag_run_id>")
        return 2
    stage, run_id = sys.argv[1], sys.argv[2]
    if stage not in STAGES:
        print(f"unknown stage: {stage}. Valid: {sorted(STAGES)}")
        return 2
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run = _run(run_id)
    print(f"--- stage: {stage} | run_id: {run_id} ---")
    STAGES[stage](run)
    print(f"--- stage {stage} complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
