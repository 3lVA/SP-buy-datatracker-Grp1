"""Restore a Prefect SQLite snapshot and checkpoint changes to private S3."""

from contextlib import closing
import logging
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from sqlalchemy.engine import make_url

LOG = logging.getLogger("prefect-persistence")
CHECKPOINT_SECONDS = 30


class SnapshotConflict(RuntimeError):
    """The current container no longer owns the latest S3 snapshot."""


def check_history(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
        if database.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("Prefect SQLite integrity check failed")
        runs = database.execute("SELECT COUNT(*) FROM flow_run").fetchone()[0]
    if not runs:
        raise ValueError("Prefect snapshot contains no flow runs")
    return runs


def restore(s3, bucket, key, database_path):
    with TemporaryDirectory(dir=database_path.parent) as directory:
        pending = Path(directory) / "prefect.db"
        response = s3.get_object(Bucket=bucket, Key=key)
        with closing(response["Body"]) as source, pending.open("wb") as target:
            shutil.copyfileobj(source, target)
        runs = check_history(pending)
        os.replace(pending, database_path)
    LOG.info("Restored %d Prefect runs from S3", runs)
    return response["ETag"]


def checkpoint(s3, bucket, key, database, etag):
    with TemporaryDirectory() as directory:
        snapshot = Path(directory) / "prefect.db"
        with closing(sqlite3.connect(snapshot)) as target:
            database.backup(target)
        check_history(snapshot)
        try:
            with snapshot.open("rb") as body:
                response = s3.put_object(
                    Bucket=bucket, Key=key, Body=body, IfMatch=etag, ServerSideEncryption="AES256"
                )
        except ClientError as error:
            if error.response["Error"]["Code"] in {
                "PreconditionFailed", "412", "ConditionalRequestConflict", "409",
            }:
                raise SnapshotConflict("The S3 Prefect snapshot changed. Refusing to overwrite it.") from error
            raise
    LOG.info("Saved Prefect history checkpoint to S3")
    return response["ETag"]


def stop_server(server):
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def server_started():
    try:
        with urlopen("http://127.0.0.1:4200/api/health", timeout=1) as response:
            return response.status == 200
    except (URLError, TimeoutError):
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    auth = os.environ.get("PREFECT_SERVER_API_AUTH_STRING", "")
    user, _, password = auth.partition(":")
    if not user or not password:
        raise ValueError("PREFECT_SERVER_API_AUTH_STRING must contain a username and password separated by a colon")
    api = urlsplit(os.environ["PREFECT_UI_API_URL"])
    if (
        api.scheme != "https" or not api.hostname or api.path.rstrip("/") != "/api"
        or api.username or api.password or api.query or api.fragment
    ):
        raise ValueError("PREFECT_UI_API_URL must be an HTTPS address ending in /api")
    config = make_url(os.environ["PREFECT_SERVER_DATABASE_CONNECTION_URL"])
    database_path = Path(config.database or "")
    if config.drivername != "sqlite+aiosqlite" or not database_path.is_absolute() or config.query:
        raise ValueError("Use an absolute SQLite file URL for PREFECT_SERVER_DATABASE_CONNECTION_URL")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    bucket = os.environ["PREFECT_PERSISTENCE_BUCKET"]
    prefix = os.environ["PREFECT_PERSISTENCE_PREFIX"].strip("/")
    if not bucket or not prefix or not prefix.startswith("prefect-history/"):
        raise ValueError("Set a bucket and a dedicated PREFECT_PERSISTENCE_PREFIX under prefect-history/")
    key = f"{prefix}/prefect.db"
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        config=Config(connect_timeout=3, read_timeout=5, retries={"total_max_attempts": 2}),
    )
    etag = restore(s3, bucket, key, database_path)
    with closing(sqlite3.connect(database_path, timeout=5)) as database:
        saved_version = database.execute("PRAGMA data_version").fetchone()[0]
        server = subprocess.Popen(
            ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"],
            start_new_session=True,
        )
        ready = False
        next_checkpoint = monotonic() + CHECKPOINT_SECONDS
        try:
            while not stop.wait(2) and server.poll() is None:
                ready = ready or server_started()
                if not ready or monotonic() < next_checkpoint:
                    continue
                next_checkpoint = monotonic() + CHECKPOINT_SECONDS
                version = database.execute("PRAGMA data_version").fetchone()[0]
                if version == saved_version:
                    continue
                try:
                    etag = checkpoint(s3, bucket, key, database, etag)
                except SnapshotConflict:
                    raise
                except Exception:
                    LOG.exception("Could not save Prefect history to S3. Retrying next interval.")
                else:
                    saved_version = version
            stop_server(server)
            if ready and database.execute("PRAGMA data_version").fetchone()[0] != saved_version:
                checkpoint(s3, bucket, key, database, etag)
            return 0 if stop.is_set() else (server.returncode or 1)
        finally:
            stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
