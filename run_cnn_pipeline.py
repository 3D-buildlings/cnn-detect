#!/usr/bin/env python3
"""cnn-detect building-segmentation pipeline - GitHub execution entrypoint.

This file is executed by the Airflow **GitHub CI runner** inside an isolated
virtual environment (``.venv/bin/python run_cnn_pipeline.py``). It is a plain
Python program: it does **not** import Airflow and it does **not** assume it
lives under ``/opt/airflow/dags``. All paths are relative to this file or
resolved from environment variables, so the same checkout runs anywhere.

Workflow (``src/pipeline.run_pipeline``)
----------------------------------------
1. Fetch the ``dataset`` and ``hlp`` inputs from MinIO into the run workspace.
2. Inspect the images and extract hand-crafted features.
3. Train a CPU pixel-wise CNN on the HLP red masks and segment every tile.
4. Quantize the CNN to 8 bits and re-run inference.
5. Finalize results and compare detected masks with the HLP annotations.
6. Log params, metrics and artifacts to MLflow.
7. Upload everything under ``s3://<bucket>/runs/<run_id>/``.

Configuration (environment variables, no secrets hard-coded)
------------------------------------------------------------
* ``MINIO_ENDPOINT`` / ``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY`` / ``MINIO_BUCKET``
* ``MLFLOW_TRACKING_URI`` / ``MLFLOW_EXPERIMENT_NAME``
* ``MLOPS_RUNS_DIR``   local scratch directory for this run (default ``./runs``)

The Airflow runner injects ``DAG_RUN_ID`` (plus ``GITHUB_CI_*`` metadata). The run
id can also be passed explicitly with ``--run-id``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Portable project root: this file's directory (works from a GitHub checkout).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep local outputs inside the checkout unless the runner overrides the path.
os.environ.setdefault("MLOPS_RUNS_DIR", str(PROJECT_ROOT / "runs"))

logger = logging.getLogger("cnn_detect.entrypoint")


def resolve_run_id(cli_run_id: str | None) -> str:
    """Resolve the run id from the CLI, the Airflow context or a timestamp."""
    if cli_run_id:
        return cli_run_id
    for name in ("DAG_RUN_ID", "AIRFLOW_CTX_DAG_RUN_ID", "GITHUB_CI_RUN_ID"):
        value = os.environ.get(name)
        if value:
            return value
    from datetime import datetime, timezone

    return "local__" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S_%f")


def print_banner(run_id: str) -> None:
    logger.info("=" * 68)
    logger.info("cnn-detect building-segmentation pipeline (GitHub execution)")
    logger.info("=" * 68)
    logger.info("Run ID           : %s", run_id)
    logger.info("Project root     : %s", PROJECT_ROOT)
    logger.info("Repository       : %s", os.environ.get("GITHUB_CI_REPOSITORY", "(local)"))
    logger.info("Branch           : %s", os.environ.get("GITHUB_CI_BRANCH", "(local)"))
    logger.info("Commit           : %s", os.environ.get("GITHUB_CI_COMMIT", "(local)"))
    logger.info("Runs dir         : %s", os.environ.get("MLOPS_RUNS_DIR"))
    logger.info("MinIO endpoint   : %s", os.environ.get("MINIO_ENDPOINT", "(default)"))
    logger.info("MLflow tracking  : %s", os.environ.get("MLFLOW_TRACKING_URI", "(default)"))
    logger.info("=" * 68)


def run_full_pipeline(run_id: str, *, fetch_from_minio: bool, upload_outputs: bool) -> int:
    from src import config, pipeline

    summary = pipeline.run_pipeline(
        run_id,
        fetch_from_minio=fetch_from_minio,
        upload_outputs=upload_outputs,
    )

    metrics = summary.get("metrics", {}) or {}
    logger.info("=" * 68)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 68)
    logger.info("MLflow experiment: %s", summary.get("experiment_name", config.mlflow_experiment_name()))
    logger.info("MLflow run id    : %s", summary.get("mlflow_run_id", "N/A"))
    logger.info("Tracking URI     : %s", summary.get("mlflow_tracking_uri", config.mlflow_tracking_uri()))
    logger.info("Artifacts        : %s", summary.get("s3_runs_uri", config.s3_runs_uri(run_id)))
    if metrics:
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info("  %s: %s", key, value)
    logger.info("=" * 68)
    return 0


def run_single_stage(run_id: str, stage: str) -> int:
    from src import config, run_cli

    run = config.RunPaths.build(run_id)
    run.create_dirs()
    logger.info("Running single stage '%s' for run '%s'", stage, run_id)
    run_cli.STAGES[stage](run)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cnn-detect MLOps pipeline entrypoint.")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (defaults to DAG_RUN_ID / AIRFLOW_CTX_DAG_RUN_ID).")
    parser.add_argument("--stage", default=None,
                        help="Run a single pipeline stage instead of the whole pipeline.")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip downloading inputs from MinIO.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading outputs back to MinIO.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_id = resolve_run_id(args.run_id)
    print_banner(run_id)

    if args.stage:
        from src.run_cli import STAGES

        if args.stage not in STAGES:
            logger.error("Unknown stage '%s'. Valid stages: %s", args.stage, sorted(STAGES))
            return 2
        return run_single_stage(run_id, args.stage)

    return run_full_pipeline(
        run_id,
        fetch_from_minio=not args.no_fetch,
        upload_outputs=not args.no_upload,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - report a clean, non-zero failure
        logger.exception("cnn-detect MLOps pipeline failed.")
        sys.exit(1)
