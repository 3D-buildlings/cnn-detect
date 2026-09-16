"""Central configuration for the skyscrapper MLOps pipeline.

Everything is resolved from environment variables at runtime (never hard-coded
secrets). The same variables are exported by the Airflow docker-compose so the
values used by the DAG, the seed script and the notebook stay in sync.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Environment variable names (spec / requirement 9)
# ---------------------------------------------------------------------------
ENV_MINIO_ENDPOINT = "MINIO_ENDPOINT"
ENV_MINIO_ACCESS_KEY = "MINIO_ACCESS_KEY"
ENV_MINIO_SECRET_KEY = "MINIO_SECRET_KEY"
ENV_MINIO_BUCKET = "MINIO_BUCKET"
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_MLFLOW_EXPERIMENT = "MLFLOW_EXPERIMENT_NAME"

# ---------------------------------------------------------------------------
# Pipeline environment variables
# ---------------------------------------------------------------------------
ENV_RUNS_DIR = "MLOPS_RUNS_DIR"
ENV_INPUT_ROOT = "MLOPS_INPUT_ROOT"  # local (seeded) copy of dataset/ + hlp/

# ---------------------------------------------------------------------------
# Defaults (non-secret, overridable through the environment)
# ---------------------------------------------------------------------------
DEFAULT_MINIO_BUCKET = "mlops"
DEFAULT_EXPERIMENT_NAME = "skyscrapper_building_segmentation"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_INPUT_ROOT = "."  # contains dataset/ and hlp/ folders when seeding locally

_MINIO_PREFIX_DATASET = "dataset"
_MINIO_PREFIX_HLP = "hlp"
_MINIO_PREFIX_RUNS = "runs"
_MINIO_PREFIX_MLFLOW = "mlflow-artifacts"

DATASET_PREFIX = _MINIO_PREFIX_DATASET
HLP_PREFIX = _MINIO_PREFIX_HLP
RUNS_PREFIX = _MINIO_PREFIX_RUNS
MLFLOW_ARTIFACT_PREFIX = _MINIO_PREFIX_MLFLOW

# ---------------------------------------------------------------------------
# CNN defaults (CPU only, lightweight)
# ---------------------------------------------------------------------------
CNN_IN_CHANNELS = int(os.environ.get("CNN_IN_CHANNELS", "3"))
CNN_BASE_CHANNELS = int(os.environ.get("CNN_BASE_CHANNELS", "16"))
CNN_PATCH_SIZE = int(os.environ.get("CNN_PATCH_SIZE", "64"))
CNN_PATCHES_PER_IMAGE = int(os.environ.get("CNN_PATCHES_PER_IMAGE", "16"))
CNN_BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", "16"))
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS", "8"))
CNN_LR = float(os.environ.get("CNN_LR", "1e-3"))
CNN_TRAIN_IMAGES = int(os.environ.get("CNN_TRAIN_IMAGES", "3"))
CNN_MAX_SIDE = int(os.environ.get("CNN_MAX_SIDE", "256"))
CNN_THRESHOLD = float(os.environ.get("CNN_THRESHOLD", "0.5"))
CNN_SEED = int(os.environ.get("CNN_SEED", "42"))

# ---------------------------------------------------------------------------
# 8-bit quantization defaults
# ---------------------------------------------------------------------------
QUANT_BITS = int(os.environ.get("QUANT_BITS", "8"))
QUANT_CALIB_SAMPLES = int(os.environ.get("QUANT_CALIB_SAMPLES", "24"))


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, trimming surrounding whitespace."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


# ---------------------------------------------------------------------------
# MinIO connection settings
# ---------------------------------------------------------------------------
def minio_endpoint() -> str:
    return (get_env(ENV_MINIO_ENDPOINT, "") or "http://minio:9000").rstrip("/")


def minio_access_key() -> str:
    return get_env(ENV_MINIO_ACCESS_KEY, "") or ""


def minio_secret_key() -> str:
    return get_env(ENV_MINIO_SECRET_KEY, "") or ""


def minio_bucket() -> str:
    return get_env(ENV_MINIO_BUCKET, "") or DEFAULT_MINIO_BUCKET


# ---------------------------------------------------------------------------
# MLflow settings
# ---------------------------------------------------------------------------
def mlflow_tracking_uri() -> str:
    return get_env(ENV_MLFLOW_TRACKING_URI, "") or "http://mlflow:5000"


def mlflow_experiment_name() -> str:
    return get_env(ENV_MLFLOW_EXPERIMENT, "") or DEFAULT_EXPERIMENT_NAME


# ---------------------------------------------------------------------------
# Local filesystem layout for a single run
# ---------------------------------------------------------------------------
def runs_dir() -> str:
    return get_env(ENV_RUNS_DIR, "") or DEFAULT_RUNS_DIR


def input_root() -> str:
    return get_env(ENV_INPUT_ROOT, "") or DEFAULT_INPUT_ROOT


def sanitize_run_id(run_id: str) -> str:
    """Make an Airflow ``dag_run_id`` safe to use as an S3 key / dir name."""
    cleaned = re.sub(r"[:/\\+ ]", "-", run_id.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned or "run"


def s3_run_prefix(run_id: str) -> str:
    """S3 prefix holding every artifact of one pipeline execution."""
    return f"{RUNS_PREFIX}/{sanitize_run_id(run_id)}"


def s3_runs_uri(run_id: str) -> str:
    """Human readable s3 URI of the run directory."""
    return f"s3://{minio_bucket()}/{s3_run_prefix(run_id)}"


# ---------------------------------------------------------------------------
# Per-run working layout on the local filesystem
# ---------------------------------------------------------------------------
@dataclass
class RunPaths:
    """Local directory layout of a single pipeline execution."""

    run_id: str
    work_dir: str = ""
    inputs_dataset: str = ""
    inputs_hlp: str = ""
    out_metadata: str = ""
    out_features: str = ""
    out_masks: str = ""
    out_quantized_masks: str = ""
    out_predictions: str = ""
    out_models: str = ""
    out_visualizations: str = ""
    out_hlp: str = ""
    out_hlp_visualizations: str = ""
    out_mlruns: str = ""
    out_summary: str = ""
    inputs_dir: str = ""
    outputs_dir: str = ""

    @classmethod
    def build(cls, run_id: str, base_dir: str | None = None) -> "RunPaths":
        base = base_dir or runs_dir()
        safe = sanitize_run_id(run_id)
        work = str(Path(base) / safe)

        def p(*parts: str) -> str:
            return str(Path(work, *parts))

        return cls(
            run_id=run_id,
            work_dir=work,
            inputs_dir=p("inputs"),
            inputs_dataset=p("inputs", "dataset"),
            inputs_hlp=p("inputs", "hlp"),
            outputs_dir=p("outputs"),
            out_metadata=p("outputs", "metadata"),
            out_features=p("outputs", "features"),
            out_masks=p("outputs", "masks"),
            out_quantized_masks=p("outputs", "quantized_masks"),
            out_predictions=p("outputs", "predictions"),
            out_models=p("outputs", "models"),
            out_visualizations=p("outputs", "visualizations"),
            out_hlp=p("outputs", "hlp"),
            out_hlp_visualizations=p("outputs", "hlp", "visualizations"),
            out_mlruns=p("outputs", "mlruns"),
            out_summary=p("outputs", "summary"),
        )

    @property
    def cnn_model_path(self) -> str:
        return str(Path(self.out_models) / "cnn_model.pkl")

    @property
    def cnn_onnx_path(self) -> str:
        return str(Path(self.out_models) / "cnn_model.onnx")

    @property
    def quantized_model_path(self) -> str:
        return str(Path(self.out_models) / "cnn_model_quantized.onnx")

    @property
    def quantization_metrics_path(self) -> str:
        return str(Path(self.out_models) / "quantization_metrics.json")

    @property
    def training_images_path(self) -> str:
        return str(Path(self.out_models) / "training_images.json")

    def create_dirs(self) -> None:
        for d in [
            self.inputs_dataset,
            self.inputs_hlp,
            self.out_metadata,
            self.out_features,
            self.out_masks,
            self.out_quantized_masks,
            self.out_predictions,
            self.out_models,
            self.out_visualizations,
            self.out_hlp_visualizations,
            self.out_mlruns,
            self.out_summary,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "work_dir": self.work_dir,
            "inputs_dataset": self.inputs_dataset,
            "inputs_hlp": self.inputs_hlp,
            "inputs_dir": self.inputs_dir,
            "outputs_dir": self.outputs_dir,
            "out_metadata": self.out_metadata,
            "out_features": self.out_features,
            "out_masks": self.out_masks,
            "out_quantized_masks": self.out_quantized_masks,
            "out_predictions": self.out_predictions,
            "out_models": self.out_models,
            "out_visualizations": self.out_visualizations,
            "out_hlp": self.out_hlp,
            "out_mlruns": self.out_mlruns,
            "out_summary": self.out_summary,
        }
