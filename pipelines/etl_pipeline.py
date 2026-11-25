from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from data_cleaning import clean_batch, load_batch
from load_rds import connect_to_rds, load_cleaned_tables
from pipelines.data_preparation import validate_schema
from pipelines.database import get_table_counts, verify_labels
from pipelines.training import resolve_trainer, retrain_history

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
    return report


@task(cache_policy=NO_CACHE)
def train(flow_run_id):
    return retrain_history(flow_run_id, get_run_logger())


@flow(name="automated-data-ingestion")
def etl_pipeline(customers_url: str, orders_url: str, labels_url: str, retrain: bool = True):
    if retrain:
        resolve_trainer()
    datasets = extract(customers_url, orders_url, labels_url)
    cleaned = transform(validate(datasets))
    report = load(cleaned)
    if retrain:
        report["training"] = train(flow_run.id)
    return report


if __name__ == "__main__":
    etl_pipeline.serve(name="data-ingestion", global_limit=1)
