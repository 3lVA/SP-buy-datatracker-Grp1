import pandas as pd

from load_rds import (
    CUSTOMER_COLUMNS,
    ORDER_COLUMNS,
    LABEL_COLUMNS
)


SCHEMAS = {
    "Customers": CUSTOMER_COLUMNS,
    "Orders": ORDER_COLUMNS,
    "Labels": LABEL_COLUMNS
}


def validate_schema(datasets):

    for name, df in datasets.items():

        expected = set(SCHEMAS[name])
        actual = set(df.columns)

        missing = expected - actual
        unexpected = actual - expected

        if missing or unexpected:
            raise ValueError(
                f"{name}: Missing={missing}, "
                f"Unexpected={unexpected}"
            )

        if df.empty:
            raise ValueError(f"{name} dataset is empty")

    return datasets
# The training snapshot uses the same feature definitions as the preparation notebook.
def build_order_features(labelled_orders):
    model_data = labelled_orders[[
        "country_code", "order_id", "customer_id", "order_date",
        "collect_type", "payment_method", "order_value", "num_items_ordered", "is_fraud",
    ]].copy()
    model_data["order_date"] = pd.to_datetime(model_data["order_date"]).dt.normalize()
    first_order_day = pd.to_datetime(labelled_orders["first_order_datetime"]).dt.normalize()

    days = (model_data["order_date"] - first_order_day).dt.days
    model_data["days_since_first_order"] = days.where(days.ge(0)).astype("Int64")
    positive_count = model_data["num_items_ordered"].where(model_data["num_items_ordered"].gt(0))
    model_data["value_per_item"] = model_data["order_value"].div(positive_count)
    model_data["order_weekday"] = model_data["order_date"].dt.day_name()
    model_data["payment_method"] = model_data["payment_method"].fillna("Unknown")
    return model_data
