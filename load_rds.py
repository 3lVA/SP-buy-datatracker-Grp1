"""Load cleaned CSV tables into MySQL, with a replay check in the write report."""

import argparse
from decimal import Decimal
from itertools import islice
import os

import pandas as pd

from data_cleaning import clean_batch, load_batch

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# =====================================================
# Define database columns
# =====================================================
CUSTOMER_COLUMNS = (
    "country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
    "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
    "total_payment_last_50days", "num_associated_customers", "first_order_datetime",
)
ORDER_COLUMNS = (
    "country_code", "order_id", "collect_type", "payment_method", "order_value",
    "num_items_ordered", "refund_value", "order_date",
)
LABEL_COLUMNS = ("country_code", "order_id", "customer_id", "is_fraud")
MONEY_COLUMNS = {"total_payment_last_50days", "order_value", "refund_value"}


# =====================================================
# Connect to MySQL
# =====================================================
def connect_to_rds():
    import mysql.connector

    return mysql.connector.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", "3306")),
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        database=os.environ.get("RDS_DATABASE", "sp_buy_datatracker"),
        ssl_ca=os.environ["RDS_SSL_CA"],
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        connection_timeout=10,
        autocommit=False,
        sql_mode="STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION",
    )


# =====================================================
# Convert and write rows in batches
# =====================================================
def mysql_rows(frame, columns):
    # Convert pandas missing values and scalars to Connector/Python types.
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                value = None
            elif column in MONEY_COLUMNS:
                value = Decimal(str(value))
            elif isinstance(value, pd.Timestamp):
                value = value.date() if column == "order_date" else value.to_pydatetime()
            elif hasattr(value, "item"):
                value = value.item()
            values.append(value)
        yield tuple(values)


def write_rows(cursor, table, columns, frame, *, update_existing):
    column_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    statement = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"
    if update_existing:
        updates = ", ".join(f"{column} = incoming.{column}" for column in columns[2:])
        statement += f" AS incoming ON DUPLICATE KEY UPDATE {updates}"

    rows = mysql_rows(frame, columns)
    affected = 0
    while batch := list(islice(rows, 2_000)):
        cursor.executemany(statement, batch)
        affected += cursor.rowcount
    return affected


# =====================================================
# Load customers, orders and labels
# =====================================================
def load_cleaned_tables(connection, tables):
    """Write one clean batch. Later batches update existing keys."""
    with connection.cursor() as cursor:
        try:
            # Stage labels so they can match parents already in the database.
            cursor.execute("DROP TEMPORARY TABLE IF EXISTS incoming_labels")
            cursor.execute("CREATE TEMPORARY TABLE incoming_labels LIKE labels")

            customer_writes = write_rows(
                cursor, "customers", CUSTOMER_COLUMNS, tables["Customers"], update_existing=True
            )
            order_writes = write_rows(
                cursor, "orders", ORDER_COLUMNS, tables["Orders"], update_existing=True
            )
            write_rows(cursor, "incoming_labels", LABEL_COLUMNS, tables["Labels"], update_existing=False)

            # Check both parent tables after inserting this batch's customers and orders.
            cursor.execute("""
                INSERT INTO labels (country_code, order_id, customer_id, is_fraud)
                SELECT incoming.country_code, incoming.order_id,
                       incoming.customer_id, incoming.is_fraud
                FROM (
                    SELECT l.country_code, l.order_id, l.customer_id, l.is_fraud
                    FROM incoming_labels AS l
                    JOIN orders AS o
                      ON o.country_code = l.country_code AND o.order_id = l.order_id
                    JOIN customers AS c
                      ON c.country_code = l.country_code AND c.customer_id = l.customer_id
                ) AS incoming
                ON DUPLICATE KEY UPDATE customer_id = incoming.customer_id,
                                        is_fraud = incoming.is_fraud
            """)
            label_writes = cursor.rowcount
            cursor.execute("""
                SELECT COUNT(*)
                FROM incoming_labels AS l
                LEFT JOIN orders AS o
                  ON o.country_code = l.country_code AND o.order_id = l.order_id
                LEFT JOIN customers AS c
                  ON c.country_code = l.country_code AND c.customer_id = l.customer_id
                WHERE o.order_id IS NULL OR c.customer_id IS NULL
            """)
            unmatched_labels = cursor.fetchone()[0]
            cursor.execute("DROP TEMPORARY TABLE incoming_labels")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    # MySQL reports 1 for an insert, 2 for a changed update and 0 for an unchanged row.
    return {
        "Customer affected rows": customer_writes,
        "Order affected rows": order_writes,
        "Label affected rows": label_writes,
        "Labels skipped because a parent was missing": unmatched_labels,
    }


# =====================================================
# Run the connection check or load a batch
# =====================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only test the Python connection")
    for table in ("customers", "orders", "labels"):
        parser.add_argument(f"--{table}", help=f"Local path or public URL for the {table} CSV")
    args = parser.parse_args()
    if not args.check and not all((args.customers, args.orders, args.labels)):
        parser.error("Specify all three CSV sources, or run with --check.")

    if not args.check:
        raw = load_batch(args.customers, args.orders, args.labels)
        tables, audit = clean_batch(raw["Customers"], raw["Orders"], raw["Labels"])
        print(audit.to_string())

    with connect_to_rds() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise RuntimeError("Python connection check returned an unexpected result.")
        print("Python connection to MySQL verified.")
        if not args.check:
            report = load_cleaned_tables(connection, tables)
            for name, value in report.items():
                print(f"{name}: {value:,}")


if __name__ == "__main__":
    main()
