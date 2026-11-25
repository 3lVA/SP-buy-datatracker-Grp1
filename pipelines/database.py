import pandas as pd


def get_table_counts(connection):

    counts = {}

    with connection.cursor() as cursor:

        for table in ("customers", "orders", "labels"):

            cursor.execute(
                f"SELECT COUNT(*) FROM {table}"
            )

            counts[table] = cursor.fetchone()[0]

    return counts


def verify_labels(connection):

    with connection.cursor() as cursor:

        cursor.execute("""
            SELECT COUNT(*)
            FROM labels AS l
            JOIN orders AS o
                ON o.country_code = l.country_code
                AND o.order_id = l.order_id
            JOIN customers AS c
                ON c.country_code = l.country_code
                AND c.customer_id = l.customer_id
        """)

        matched = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM labels")
        total = cursor.fetchone()[0]

    if matched != total:
        raise ValueError("Label verification failed")

    return matched
def read_labelled_history(connection):
    """Read all labelled orders from a single, consistently ordered MySQL snapshot."""
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT l.country_code, l.order_id, l.customer_id, o.order_date,
                   o.collect_type, o.payment_method, o.order_value, o.num_items_ordered,
                   l.is_fraud, c.first_order_datetime
            FROM labels AS l
            JOIN orders AS o
              ON o.country_code = l.country_code AND o.order_id = l.order_id
            JOIN customers AS c
              ON c.country_code = l.country_code AND c.customer_id = l.customer_id
            ORDER BY l.country_code, l.order_id
        """)
        rows = cursor.fetchall()
        columns = cursor.column_names

    if not rows:
        raise ValueError("RDS has no matched labelled orders to train on")
    data = pd.DataFrame.from_records(rows, columns=columns)
    for column in ("order_value", "num_items_ordered"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data
