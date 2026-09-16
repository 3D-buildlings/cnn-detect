"""Skyscrapper building-segmentation MLOps source package.

The Airflow DAG ``skyscrapper_mlops_pipeline_cnn`` shells out to ``run_cli.py``
which dispatches to the stage implementations in the source modules.

Modules
-------
``config``    : central configuration (paths, MinIO/S3, MLflow, CNN, quantization)
``data``      : Stage 1 + 2 (image inspection, feature extraction)
``train``     : Stage 3 (CNN training + dataset segmentation)
``quantize``  : INT8 quantization of the trained CNN
``evaluate``  : Stage 4 + 6 (finalize results, HLP comparison)
``track``     : MLflow experiment tracking
``storage``   : MinIO/S3 storage helpers
``pipeline``  : end-to-end pipeline runner
``run_cli``   : ``python run_cli.py <stage> <run_id>`` entry point
"""

from __future__ import annotations

__all__ = [
    "config",
    "data",
    "train",
    "quantize",
    "evaluate",
    "track",
    "storage",
    "pipeline",
    "run_cli",
]
