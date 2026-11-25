"""Call the MLflow branch trainer and attach Prefect lineage to its original runs."""

import importlib.util
import os
from pathlib import Path
import sys

from mlflow import MlflowClient


def main():
    path = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("compiled_models", path)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    print(f"MLflow comparison_batch: {trainer.COMPARISON_BATCH}")

    try:
        trainer.main()
    finally:
        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])
        if experiment:
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.comparison_batch = '{trainer.COMPARISON_BATCH}'",
            )
            for run in runs:
                for key, value in {
                    "etl_run_id": os.environ["FRAUD_ETL_RUN_ID"],
                    "dataset_sha256": os.environ["FRAUD_DATA_SHA256"],
                    "training_rows": os.environ["FRAUD_TRAINING_ROWS"],
                    "training_source": "rds_cleaned_history",
                }.items():
                    client.set_tag(run.info.run_id, key, value)


if __name__ == "__main__":
    main()
