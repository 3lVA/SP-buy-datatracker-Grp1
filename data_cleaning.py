"""CSV loading and cleaning rules from the data preparation notebook."""

import pandas as pd


# =====================================================
# Read the source CSV files
# =====================================================
def load_batch(customers_csv_source, orders_csv_source, labels_csv_source):
    """Read customer, order, and label CSVs from local paths or public URLs."""
    return {
        "Customers": pd.read_csv(
            customers_csv_source,
            dtype={"country_code": "string", "customer_id": "string", "mobile_verified": "string"},
        ),
        "Orders": pd.read_csv(
            orders_csv_source,
            dtype={"country_code": "string", "order_id": "string"},
        ),
        "Labels": pd.read_csv(
            labels_csv_source,
            dtype={"country_code": "string", "order_id": "string", "customer_id": "string"},
        ),
    }


# =====================================================
# Standardise and validate source values
# =====================================================
def normalise_country_codes(country_codes):
    return country_codes.str.strip().str.upper()


def parse_batch_dates(date_values):
    """Parse the source date formats and leave invalid dates missing."""
    return pd.to_datetime(date_values, format="mixed", errors="coerce")


def clean_table_keys(raw_table, key_columns):
    table = raw_table.copy()
    table["country_code"] = normalise_country_codes(table["country_code"].astype("string"))
    for column in key_columns:
        if column != "country_code":
            table[column] = table[column].astype("string").str.strip()
    invalid_key = ~table["country_code"].str.fullmatch(r"[A-Z]{2}", na=False)
    invalid_key |= table[key_columns].fillna("").eq("").any(axis=1)
    return table.loc[~invalid_key].copy(), {
        "Input rows": len(raw_table),
        "Invalid key rows": int(invalid_key.sum()),
    }


def resolve_duplicate_keys(table, key_columns, audit):
    # Compare standardised values before deciding whether records disagree.
    distinct_rows = table.drop_duplicates()
    conflicting_keys = distinct_rows.duplicated(key_columns, keep=False)
    audit["Repeated rows removed"] = len(table) - len(distinct_rows)
    audit["Conflicting key rows"] = int(conflicting_keys.sum())
    return distinct_rows.loc[~conflicting_keys].copy(), audit


# =====================================================
# Clean the customers table
# =====================================================
def clean_customer_records(raw_customers):
    customers, audit = clean_table_keys(raw_customers, ["country_code", "customer_id"])
    # Convert the supported verification strings to nullable booleans.
    verification = customers["mobile_verified"].astype("string").str.strip().str.lower()
    mapped = verification.map({"true": True, "yes": True, "false": False})
    audit["Unexpected verification values"] = int((verification.notna() & mapped.isna()).sum())
    customers["mobile_verified"] = mapped.astype("boolean")
    # Parse first-order dates and count non-missing values that failed to parse.
    first_order = parse_batch_dates(customers["first_order_datetime"])
    audit["Unparseable first-order dates"] = int(
        (customers["first_order_datetime"].notna() & first_order.isna()).sum()
    )
    customers["first_order_datetime"] = first_order
    return resolve_duplicate_keys(customers, ["country_code", "customer_id"], audit)


# =====================================================
# Clean the orders table
# =====================================================
def clean_order_records(raw_orders):
    orders, audit = clean_table_keys(raw_orders, ["country_code", "order_id"])
    # Parse order dates, keep the calendar day, and filter rows without a usable date.
    orders["order_date"] = parse_batch_dates(orders["order_date"]).dt.normalize()
    invalid_date = orders["order_date"].isna()
    audit["Missing or invalid order dates"] = int(invalid_date.sum())
    orders = orders.loc[~invalid_date].copy()

    # Convert item counts to numbers so invalid text and fractional counts can be flagged.
    item_counts = pd.to_numeric(orders["num_items_ordered"], errors="coerce")
    valid_counts = item_counts.gt(0) & item_counts.mod(1).eq(0) & item_counts.lt(2**63)
    audit["Invalid item counts set to missing"] = int(
        (orders["num_items_ordered"].notna() & ~valid_counts).sum()
    )
    orders["num_items_ordered"] = item_counts.where(valid_counts).astype("Int64")
    # Keep finite, non-negative amounts and set invalid amounts to missing.
    for column in ("order_value", "refund_value"):
        amounts = pd.to_numeric(orders[column], errors="coerce")
        valid_amount = amounts.ge(0) & amounts.lt(float("inf"))
        audit[f"Invalid {column} set to missing"] = int(
            (orders[column].notna() & ~valid_amount).sum()
        )
        orders[column] = amounts.where(valid_amount)
    # Standardise category casing and whitespace, then treat blanks as missing.
    for column in ("payment_method", "collect_type"):
        orders[column] = orders[column].astype("string").str.strip().str.lower().replace("", pd.NA)
    return resolve_duplicate_keys(orders, ["country_code", "order_id"], audit)


# =====================================================
# Clean the labels table
# =====================================================
def clean_label_records(raw_labels):
    labels, audit = clean_table_keys(raw_labels, ["country_code", "order_id"])
    labels["customer_id"] = labels["customer_id"].astype("string").str.strip()
    missing_customer = labels["customer_id"].fillna("").eq("")
    # Convert the target to a number and keep only 0 or 1.
    fraud_values = pd.to_numeric(labels["is_fraud"], errors="coerce")
    invalid_label = ~fraud_values.isin([0, 1])
    audit["Missing customer IDs"] = int(missing_customer.sum())
    audit["Invalid or missing fraud labels"] = int(invalid_label.sum())
    valid_rows = ~(missing_customer | invalid_label)
    labels = labels.loc[valid_rows].copy()
    labels["is_fraud"] = fraud_values.loc[valid_rows].astype("int8")
    return resolve_duplicate_keys(labels, ["country_code", "order_id"], audit)


# =====================================================
# Clean the batch and report changes
# =====================================================
def clean_batch(raw_customers, raw_orders, raw_labels):
    """Apply the same cleaning rules to any batch loaded from CSV."""
    # Run the three cleaners and collect their row counts in one audit table.
    results = {
        "Customers": clean_customer_records(raw_customers),
        "Orders": clean_order_records(raw_orders),
        "Labels": clean_label_records(raw_labels),
    }
    cleaned = {name: table for name, (table, _) in results.items()}
    audit = pd.DataFrame.from_dict({
        name: {**counts, "Cleaned rows": len(table)}
        for name, (table, counts) in results.items()
    }, orient="index").astype("Int64")
    return cleaned, audit
