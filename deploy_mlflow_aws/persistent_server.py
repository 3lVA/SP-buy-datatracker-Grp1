"""Restore MLflow metadata before startup and checkpoint it to S3 while serving."""

from contextlib import closing
import hashlib
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import sqlite3
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic

BUCKET = os.environ["MLFLOW_PERSISTENCE_BUCKET"]
ALLOW_EMPTY_START = os.environ.get("MLFLOW_ALLOW_EMPTY_START", "0") == "1"
PREFIX = os.environ.get("MLFLOW_PERSISTENCE_PREFIX", "mlflow-backups/group1-auto").strip("/")
SEED_KEY = os.environ.get("MLFLOW_SEED_KEY", "mlflow-backups/group1-seed.tar.gz")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
SNAPSHOT_KEY = f"{PREFIX}/mlflow.db"
ARTIFACT_PREFIX = f"{PREFIX}/artifacts"
DB = Path("/mlflow/mlflow.db")
POLL_SECONDS = 2
MAX_SNAPSHOT_INTERVAL = 60
RETRY_SECONDS = 10
LOG = logging.getLogger("mlflow-persistence")


class SnapshotConflict(RuntimeError):
    """Another container has updated the S3 snapshot."""


def read_database(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)


def check_database(path):
    with closing(read_database(path)) as db:
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("SQLite integrity check failed")
        return db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def error_code(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code")


def seed_from_archive(s3, directory):
    archive_path = directory / "seed.tar.gz"
    s3.download_file(BUCKET, SEED_KEY, str(archive_path))
    with tarfile.open(archive_path, "r:gz") as archive:
        files = archive.getmembers()
        for member in files:
            path = PurePosixPath(member.name)
            if path not in (PurePosixPath("mlflow.db"), PurePosixPath("artifacts")) and (
                not path.parts or path.parts[0] != "artifacts" or ".." in path.parts
            ):
                raise ValueError(f"Unexpected seed entry {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Seed entry is not an ordinary file or directory: {member.name}")
        if len({member.name for member in files}) != len(files):
            raise ValueError("Duplicate seed archive entries")
        for member in files:
            parts = PurePosixPath(member.name).parts
            if any(":" in part or chr(92) in part for part in parts):
                raise ValueError(f"Unexpected seed entry {member.name}")
            target = directory.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    database = directory / "mlflow.db"
    runs = check_database(database)
    if runs == 0:
        raise ValueError("The seed backup contains no runs")
    artifacts = directory / "artifacts"
    files = sorted(file for file in artifacts.rglob("*") if file.is_file())
    if not files:
        raise ValueError("The seed backup contains no model or run files")
    for file in files:
        key = f"{ARTIFACT_PREFIX}/{file.relative_to(artifacts).as_posix()}"
        s3.upload_file(str(file), BUCKET, key, ExtraArgs={"ServerSideEncryption": "AES256"})
    LOG.info("Seeded %d runs and %d artifact files from S3", runs, len(files))
    return database


def restore(s3):
    with TemporaryDirectory(dir=DB.parent) as temporary:
        directory = Path(temporary)
        pending = directory / "remote.db"
        try:
            response = s3.get_object(Bucket=BUCKET, Key=SNAPSHOT_KEY)
        except Exception as exc:
            if error_code(exc) not in {"NoSuchKey", "404", "NotFound"}:
                raise
            if ALLOW_EMPTY_START:
                # Explicit first-start mode for a new bucket; never use on AccessDenied.
                subprocess.run(
                    ["mlflow", "db", "upgrade", f"sqlite:///{pending}"],
                    check=True,
                )
                check_database(pending)
                etag, digest = publish(s3, pending, None, None)
                os.replace(pending, DB)
                LOG.info("Initialised a new MLflow database and saved its first S3 checkpoint")
                return etag, digest
            seed = seed_from_archive(s3, directory)
            etag, digest = publish(s3, seed, None, None)
            os.replace(seed, DB)
            return etag, digest
        with closing(response["Body"]) as body, pending.open("wb") as output:
            for chunk in body.iter_chunks():
                output.write(chunk)
        runs = check_database(pending)
        if not runs and not ALLOW_EMPTY_START:
            raise ValueError("S3 snapshot shows no runs; refusing to start with an empty database")
        LOG.info("Restored %d runs from S3", runs)
        with pending.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        os.replace(pending, DB)
        return response["ETag"], digest


def publish(s3, snapshot, etag, previous_digest):
    with snapshot.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
        if digest == previous_digest:
            return etag, digest
        data.seek(0)
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            result = s3.put_object(
                Bucket=BUCKET, Key=SNAPSHOT_KEY, Body=data, ServerSideEncryption="AES256", **condition
            )
        except Exception as exc:
            if error_code(exc) in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
                raise SnapshotConflict("Another process updated the S3 database snapshot") from exc
            raise
        LOG.info("Saved SQLite checkpoint to S3")
        return result["ETag"], digest


def checkpoint(s3, db, etag, digest):
    with TemporaryDirectory(dir=DB.parent) as temporary:
        snapshot = Path(temporary) / "mlflow.db"
        with closing(sqlite3.connect(snapshot)) as copy:
            db.backup(copy)
        check_database(snapshot)
        return publish(s3, snapshot, etag, digest)


def run_state(db):
    return db.execute("PRAGMA data_version").fetchone()[0], db.execute(
        "SELECT COUNT(*), COALESCE(MAX(end_time), 0) FROM runs WHERE status IN ('FINISHED','FAILED','KILLED')"
    ).fetchone()


def stop_server(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=7)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def main():
    import boto3
    from botocore.config import Config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s3 = boto3.client("s3", region_name=REGION, config=Config(connect_timeout=4, read_timeout=10, retries={"total_max_attempts": 2}))
    etag, digest = restore(s3)
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    with closing(read_database(DB)) as db:
        saved_version, saved_runs = run_state(db)
        server = subprocess.Popen([
            "mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1",
            "--backend-store-uri", f"sqlite:///{DB}", "--serve-artifacts",
            "--artifacts-destination", f"s3://{BUCKET}/{ARTIFACT_PREFIX}",
            "--allowed-hosts", "*", "--cors-allowed-origins", "*",
        ], start_new_session=True)
        last_saved = monotonic()
        retry_after = 0
        try:
            while not stop.wait(POLL_SECONDS) and server.poll() is None:
                now = monotonic()
                version, runs = run_state(db)
                if now >= retry_after and (runs != saved_runs or (version != saved_version and now - last_saved >= MAX_SNAPSHOT_INTERVAL)):
                    try:
                        etag, digest = checkpoint(s3, db, etag, digest)
                    except SnapshotConflict:
                        raise
                    except Exception:
                        LOG.exception("S3 checkpoint failed; will retry")
                        retry_after = now + RETRY_SECONDS
                    else:
                        last_saved = now
                        saved_version, saved_runs = version, runs
            stop_server(server)
            if run_state(db)[0] != saved_version:
                checkpoint(s3, db, etag, digest)
            return 0 if stop.is_set() else (server.returncode or 1)
        finally:
            stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())


