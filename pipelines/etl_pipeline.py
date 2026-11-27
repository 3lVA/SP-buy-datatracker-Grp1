import os
from pathlib import Path
from time import time_ns

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.entities import Metric
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from data_cleaning import clean_batch, load_batch
from load_rds import connect_to_rds, load_cleaned_tables
from pipelines.data_preparation import validate_schema
from pipelines.database import get_table_counts, verify_labels
from pipelines.training import TEAM_EXPERIMENT, resolve_trainer, retrain_history

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@task(retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE, persist_result=False)
def extract(customers, orders, labels):
    logger = get_run_logger()
    datasets = load_batch(customers, orders, labels)
    for name, df in datasets.items():
        logger.info("%s: %s rows extracted", name, f"{len(df):,}")
    return datasets


@task(cache_policy=NO_CACHE, persist_result=False)
def validate(datasets):
    return validate_schema(datasets)


@task(cache_policy=NO_CACHE, persist_result=False)
def transform(datasets):
    cleaned, audit = clean_batch(datasets["Customers"], datasets["Orders"], datasets["Labels"])
    get_run_logger().info("\n%s", audit.to_string())
    return cleaned


@task(cache_policy=NO_CACHE)
def load(cleaned):
    logger = get_run_logger()
    with connect_to_rds() as connection:
        before = get_table_counts(connection)
        report = load_cleaned_tables(connection, cleaned)
        after = get_table_counts(connection)
        verified = verify_labels(connection)
    logger.info("Before ingestion: %s", before)
    logger.info("After ingestion: %s", after)
    logger.info("Write report: %s", report)
    logger.info("Verified labels: %s", f"{verified:,}")
    return {"before": before, "after": after, "writes": report, "verified_labels": verified}


@task(retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def log_ingestion(flow_run_id, batch):
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"].strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", TEAM_EXPERIMENT)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)

    tags = {"etl_run_id": flow_run_id, "run_type": "etl_ingestion"}
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.etl_run_id = '{flow_run_id}' AND tags.run_type = 'etl_ingestion'",
        max_results=1,
    )
    run = runs[0] if runs else client.create_run(
        experiment_id, tags=tags, run_name=f"ETL_Ingestion_{flow_run_id}"
    )
    if run.info.status != "FINISHED":
        counts = {"verified_labels": batch["verified_labels"]}
        for stage in ("before", "after"):
            counts.update({f"{table}_{stage}": count for table, count in batch[stage].items()})
        counts.update({
            name.lower().replace(" ", "_"): count for name, count in batch["writes"].items()
        })
        timestamp = time_ns() // 1_000_000
        client.log_batch(
            run.info.run_id,
            metrics=[Metric(name, count, timestamp, step=0) for name, count in counts.items()],
            synchronous=True,
        )
        client.set_terminated(run.info.run_id)
    get_run_logger().info("MLflow ingestion audit run: %s", run.info.run_id)
    return run.info.run_id


@task(cache_policy=NO_CACHE)
def train(flow_run_id):
    return retrain_history(flow_run_id, get_run_logger())


@flow(name="automated-data-ingestion")
def etl_pipeline(customers_url: str, orders_url: str, labels_url: str, retrain: bool = True):
    if not os.getenv("MLFLOW_TRACKING_URI", "").strip():
        raise ValueError("Set MLFLOW_TRACKING_URI in the project .env or process environment")
    if retrain:
        resolve_trainer()
    datasets = extract(customers_url, orders_url, labels_url)
    cleaned = transform(validate(datasets))
    batch = load(cleaned)
    report = {
        **batch["writes"],
        "ingestion_mlflow_run_id": log_ingestion(flow_run.id, batch),
    }
    if retrain:
        report["training"] = train(flow_run.id)
    return report


if __name__ == "__main__":
    etl_pipeline.serve(name="data-ingestion", global_limit=1)
