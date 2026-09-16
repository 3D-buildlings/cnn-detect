"""MLflow experiment tracking for one pipeline execution.

Logic follows the notebook (cell 9) and ``code/code/07_mlflow.py`` but logs to a
central MLflow tracking server (``MLFLOW_TRACKING_URI``) instead of a local
SQLite file. Each Airflow DAG run is logged as one MLflow run whose name equals
the DAG run id, so runs can always be traced back to the DAG execution.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src import config
from src.config import RunPaths

logger = logging.getLogger("skyscrapper.track")

warnings.filterwarnings("ignore")

METRIC_KEYS_HLP = ["mean_rms_error", "mean_iou", "mean_precision", "mean_recall",
                   "mean_f1", "total_images_compared", "std_rms_error",
                   "min_rms_error", "max_rms_error"]
METRIC_KEYS_SEG = ["total_images", "images_processed", "images_failed",
                   "total_buildings_detected", "avg_buildings_per_image",
                   "building_pixels", "building_area_ratio"]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_hlp_metrics(run: RunPaths) -> Dict[str, Any]:
    return _load_json(str(Path(run.out_hlp) / "rms_metrics.json"))


def _load_seg_metrics(run: RunPaths) -> Dict[str, Any]:
    return _load_json(str(Path(run.out_models) / "segmentation_metrics_final.json"))


def _load_quant_metrics(run: RunPaths) -> Dict[str, Any]:
    if Path(run.quantization_metrics_path).is_file():
        return _load_json(run.quantization_metrics_path)
    return {}


def _log_artifacts(mlflow, run: RunPaths) -> None:
    """Log representative artifacts produced by the pipeline."""
    candidates: List[tuple] = [
        ("predictions", Path(run.out_predictions) / "building_contours_final.csv"),
        ("predictions", Path(run.out_predictions) / "building_contours.csv"),
        ("metrics", Path(run.out_hlp) / "rms_metrics.json"),
        ("metrics", Path(run.out_hlp) / "comparison_results.csv"),
        ("metrics", Path(run.out_models) / "segmentation_metrics.json"),
        ("metrics", Path(run.out_models) / "segmentation_metrics_final.json"),
        ("metrics", Path(run.quantization_metrics_path)),
    ]
    logged = 0
    for artifact_path, file in candidates:
        if file.is_file():
            mlflow.log_artifact(str(file), artifact_path=artifact_path)
            logged += 1

    for model_artifact in [run.cnn_model_path, run.quantized_model_path]:
        if Path(model_artifact).is_file():
            mlflow.log_artifact(str(model_artifact), artifact_path="models")
            logged += 1

    vis_dir = Path(run.out_hlp_visualizations)
    if vis_dir.is_dir():
        for img in sorted(vis_dir.glob("*_comparison.png"))[:5]:
            mlflow.log_artifact(str(img), artifact_path="plots")
            logged += 1
    logger.info("Logged %d artifacts to MLflow.", logged)


def log_run_to_mlflow(run: RunPaths, extra_tags: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Log parameters/metrics/artifacts for this run to the MLflow server."""
    import mlflow
    from mlflow.exceptions import MlflowException

    tracking_uri = config.mlflow_tracking_uri()
    experiment_name = config.mlflow_experiment_name()

    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment = mlflow.set_experiment(experiment_name)
    except MlflowException as exc:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' could not be "
                           f"prepared at {tracking_uri}: {exc}") from exc

    hlp_metrics = _load_hlp_metrics(run)
    seg_metrics = _load_seg_metrics(run)
    quant_metrics = _load_quant_metrics(run)

    run_name = f"dag_{config.sanitize_run_id(run.run_id)}"
    with mlflow.start_run(run_name=run_name, tags={"dag_run_id": run.run_id}):
        run_id = mlflow.active_run().info.run_id
        logger.info("MLflow run started: id=%s experiment=%s tracking=%s",
                    run_id, experiment_name, tracking_uri)

        # --- Parameters ---
        params = {
            "model_type": "pixel-wise CNN",
            "device": "cpu",
            "training_images": str(config.CNN_TRAIN_IMAGES),
            "architecture": "PixelSegCNN (conv-relu-pool-conv-relu-upsample)",
            "patch_size": str(config.CNN_PATCH_SIZE),
            "batch_size": str(config.CNN_BATCH_SIZE),
            "epochs": str(config.CNN_EPOCHS),
            "learning_rate": str(config.CNN_LR),
            "base_channels": str(config.CNN_BASE_CHANNELS),
            "threshold": str(config.CNN_THRESHOLD),
            "min_area": "200",
            "min_solidity": "0.4",
            "max_axis_ratio": "8.0",
            "quantization": f"{config.QUANT_BITS}-bit (int8 weights / uint8 activations)",
            "quantization_engine": quant_metrics.get("engine", "n/a"),
            "pipeline_version": "v2.0_cnn_8bit",
        }
        mlflow.log_params(params)
        logger.info("Parameters logged (%d).", len(params))

        # --- Metrics (HLP comparison) ---
        for key in METRIC_KEYS_HLP:
            if key in hlp_metrics and hlp_metrics[key] is not None:
                mlflow.log_metric(key, float(hlp_metrics[key]))

        # --- Metrics (segmentation) ---
        for key in METRIC_KEYS_SEG:
            if key in seg_metrics and seg_metrics[key] is not None:
                mlflow.log_metric(f"seg_{key}", float(seg_metrics[key]))

        # --- Metrics (quantization) ---
        for key in (
            "mean_absolute_error",
            "max_absolute_error",
            "compression_ratio",
            "quantized_mean_iou",
            "quantized_mean_f1",
            "quantized_mean_precision",
            "quantized_mean_recall",
            "quantized_buildings_detected",
        ):
            if key in quant_metrics and quant_metrics[key] is not None:
                mlflow.log_metric(f"quant_{key}", float(quant_metrics[key]))

        # --- Model sizes ---
        if "float_model_size_mb" in quant_metrics:
            mlflow.log_metric("fp32_model_size_mb", quant_metrics["float_model_size_mb"])
        if "quantized_model_size_mb" in quant_metrics:
            mlflow.log_metric("int8_model_size_mb", quant_metrics["quantized_model_size_mb"])

        # --- Tags ---
        mlflow.set_tag("pipeline_version", "v2.0_cnn_8bit")
        mlflow.set_tag("dataset", "skyscrapper_tehran")
        mlflow.set_tag("method", "pixelwise_cnn_cpu")
        mlflow.set_tag("quantization", f"{config.QUANT_BITS}-bit (int8 weights / uint8 activations)")
        mlflow.set_tag("dag_run_id", run.run_id)
        if extra_tags:
            mlflow.set_tags(extra_tags)

        # --- Artifacts ---
        _log_artifacts(mlflow, run)

    logger.info("MLflow run completed: %s", run_id)

    # Read back the logged metrics so the report task can print them.
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    logged_run = client.get_run(run_id)

    def _metric_value(v: Any) -> Any:
        return getattr(v, "value", v)

    metrics_summary = {
        k: _metric_value(v) for k, v in sorted(logged_run.data.metrics.items())
    }

    summary = {
        "dag_run_id": run.run_id,
        "run_dir": run.work_dir,
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "mlflow_run_id": run_id,
        "mlflow_tracking_uri": tracking_uri,
        "logged_at": datetime.now().isoformat(),
        "metrics": metrics_summary,
        "s3_run_prefix": config.s3_run_prefix(run.run_id),
        "s3_runs_uri": config.s3_runs_uri(run.run_id),
    }

    Path(run.out_summary).mkdir(parents=True, exist_ok=True)
    summary_path = Path(run.out_summary) / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Run summary written to %s", summary_path)
    return summary


def load_summary(run: RunPaths) -> Dict[str, Any]:
    summary_path = Path(run.out_summary) / "run_summary.json"
    if summary_path.is_file():
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit("This module is not meant to be run directly; import and call log_run_to_mlflow().")
