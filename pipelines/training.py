"""Prepare one local training snapshot and run the team's existing MLflow trainers."""

from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from load_rds import connect_to_rds
from pipelines.data_preparation import build_order_features
from pipelines.database import read_labelled_history

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAM_EXPERIMENT = "SP-Buy-Fraud-Detection"


def resolve_trainer():
    from compiled_models import validate_tracking_uri

    validate_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "").strip())
    trainer = Path(os.getenv("MLFLOW_TRAINER_PATH", str(PROJECT_ROOT / "compiled_models.py")))
    if not trainer.is_file():
        raise FileNotFoundError(
            "The MLflow trainer is missing. Set "
            "MLFLOW_TRAINER_PATH to the team runner compiled_models.py. "
            "Use retrain=false to load data only."
        )
    return trainer.resolve()


def retrain_history(flow_run_id, logger):
    trainer = resolve_trainer()
    with connect_to_rds() as connection:
        features = build_order_features(read_labelled_history(connection))

    if features["is_fraud"].nunique() != 2:
        raise ValueError("Training history must contain both fraud and non-fraud labels")

    with TemporaryDirectory(prefix="fraud-training-") as directory:
        snapshot = Path(directory) / "base_processed.csv"
        trainer_copy = Path(directory) / "compiled_models.py"
        shutil.copyfile(trainer, trainer_copy)
        features.to_csv(snapshot, index=False)
        digest = sha256()
        with snapshot.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = digest.hexdigest()

        environment = os.environ.copy()
        environment.update(
            TRAINING_PROJECT_ROOT=str(PROJECT_ROOT),
            FRAUD_DATA_PATH=str(snapshot),
            FRAUD_ETL_RUN_ID=flow_run_id,
            FRAUD_DATA_SHA256=fingerprint,
            MLFLOW_TRACKING_URI=os.environ["MLFLOW_TRACKING_URI"].strip(),
            MLFLOW_EXPERIMENT_NAME=os.getenv("MLFLOW_EXPERIMENT_NAME", TEAM_EXPERIMENT),
            FRAUD_TRAINING_ROWS=str(len(features)),
            # ETL retrains the team models and records metrics without publishing new MLflow models.
            LOG_MODEL_ARTIFACTS="0",
            PYTHONIOENCODING="utf-8",
        )
        logger.info("Training all six models on %s labelled orders. MLflow etl_run_id: %s", f"{len(features):,}", flow_run_id)
        with subprocess.Popen(
            [sys.executable, "-u", "-m", "pipelines.run_team_training", str(trainer_copy)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                logger.info("%s", line.rstrip())
            if process.wait() != 0:
                raise RuntimeError(f"Team retraining failed. See this Prefect run and MLflow etl_run_id {flow_run_id}.")

    return {"etl_run_id": flow_run_id, "dataset_sha256": fingerprint, "training_rows": len(features)}
