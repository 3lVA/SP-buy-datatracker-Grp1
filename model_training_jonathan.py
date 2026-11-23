import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


DATA_PATH = Path(os.getenv("FRAUD_DATA_PATH", "model_data.csv"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Fraud Detection")

SEED = 42
DEFAULT_THRESHOLD = 0.5

NUMERIC = [
    "order_value",
    "num_items_ordered",
    "days_since_first_order",
    "value_per_item",
]
CATEGORICAL = [
    "country_code",
    "collect_type",
    "payment_method",
    "order_weekday",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_fraud"
ORDER_KEYS = ["country_code", "order_id"]
LOG_FIELDS = ["order_value", "num_items_ordered"]
INTERACTION_FIELDS = [*LOG_FIELDS, "days_since_first_order"]


# ==========================================
# 1. LOAD DATASET
# ==========================================

def load_data():
    orders = pd.read_csv(DATA_PATH, dtype={key: "string" for key in ORDER_KEYS})
    required = set(FEATURES + ORDER_KEYS + ["order_date", TARGET])
    missing = sorted(required - set(orders.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if orders.empty:
        raise ValueError("The dataset contains no orders.")

    orders["order_date"] = pd.to_datetime(
        orders["order_date"], errors="raise"
    ).dt.normalize()
    if orders["order_date"].isna().any():
        raise ValueError("Every order must have a date.")

    for key in ORDER_KEYS:
        if orders[key].isna().any() or orders[key].str.strip().eq("").any():
            raise ValueError(f"Missing or blank order key: {key}")

    if orders.duplicated(ORDER_KEYS).any():
        raise ValueError("Duplicate (country_code, order_id) keys.")
    if not orders[TARGET].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0 and 1.")

    orders[TARGET] = orders[TARGET].astype(int)
    orders[NUMERIC] = orders[NUMERIC].apply(
        pd.to_numeric, errors="raise"
    ).astype(float)
    if np.isinf(orders[NUMERIC].to_numpy()).any():
        raise ValueError("Numeric features contain infinity.")

    for column in CATEGORICAL:
        orders[column] = (
            orders[column].astype("string").fillna("Unknown").astype(object)
        )

    return orders


# ==========================================
# 2. PREPROCESSING
# ==========================================

def make_encoder(dense=False):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=not dense,
            dtype=np.float32 if dense else np.float64,
        )),
    ])


def make_xgb_preprocessor():
    return Pipeline([
        ("columns", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            ("categorical", make_encoder(dense=True), CATEGORICAL),
        ])),
        ("float32", FunctionTransformer(np.asarray, kw_args={"dtype": np.float32})),
    ])


def log_order_values(values):
    if (values < 0).any():
        raise ValueError("log1p fields must be nonnegative.")
    return np.log1p(values)


def tenure_products(values):
    order_value, item_count, tenure = values.T
    return np.column_stack([order_value * tenure, item_count * tenure])


def numeric_pipeline(use_log=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_log:
        steps.append(("log1p", FunctionTransformer(log_order_values)))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


# ==========================================
# 3. CONFIGURE MLFLOW
# ==========================================

def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


# ==========================================
# 4. MODEL FUNCTIONS
# ==========================================

def create_xgb_model(params, early_stopping):
    callbacks = None
    if early_stopping:
        callbacks = [EarlyStopping(
            rounds=50,
            metric_name="aucpr",
            data_name="validation_1",
            maximize=True,
            save_best=True,
        )]
    return XGBClassifier(**params, callbacks=callbacks)


def create_lr_model(C, class_weight, use_log, use_interactions):
    if use_log:
        other_numeric = [column for column in NUMERIC if column not in LOG_FIELDS]
        transforms = [
            ("log_numeric", numeric_pipeline(use_log=True), LOG_FIELDS),
            ("other_numeric", numeric_pipeline(), other_numeric),
        ]
    else:
        transforms = [("numeric", numeric_pipeline(), NUMERIC)]

    if use_interactions:
        interaction_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("centre_and_scale", StandardScaler()),
            ("products", FunctionTransformer(tenure_products)),
            ("scale_products", StandardScaler()),
        ])
        transforms.append(("interactions", interaction_pipeline, INTERACTION_FIELDS))

    transforms.append(("categorical", make_encoder(), CATEGORICAL))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transforms)),
        ("model", LogisticRegression(
            C=C, class_weight=class_weight, solver="lbfgs", max_iter=1000,
        )),
    ])


class FraudModel(mlflow.pyfunc.PythonModel):
    """Serve the fitted pipeline with its validation-tuned cutoff."""

    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, context, model_input, params=None):
        inputs = model_input.loc[:, FEATURES].copy()
        for column in CATEGORICAL:
            inputs[column] = (
                inputs[column].astype("string").fillna("Unknown").astype(object)
            )

        scores = self.pipeline.predict_proba(inputs)[:, 1].astype(float)
        return pd.DataFrame({
            "fraud_score": scores,
            "is_fraud": (scores >= self.threshold).astype("int64"),
        }, index=model_input.index)


# ==========================================
# 5. EVALUATION AND THRESHOLD TUNING
# ==========================================

def evaluate(y, scores, threshold):
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y, predictions),
        "average_precision": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "flagged_rate": predictions.mean(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(
        y, np.asarray(scores, dtype=float)
    )

    # The last precision/recall pair has no corresponding threshold.
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )

    # Prefer the highest threshold when multiple cutoffs give the same F1.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


# ==========================================
# 6. COMMON CHRONOLOGICAL SPLIT
# ==========================================

def split_data(orders):
    orders = orders.sort_values(["order_date", *ORDER_KEYS], kind="stable")
    dates = orders["order_date"]
    validation_start = dates.iloc[int(len(orders) * 0.70)]
    test_start = dates.iloc[int(len(orders) * 0.85)]

    # Keep orders from the same date in the same partition.
    parts = {
        "train": orders.loc[dates < validation_start],
        "validation": orders.loc[(dates >= validation_start) & (dates < test_start)],
        "test": orders.loc[dates >= test_start],
    }

    for name, part in parts.items():
        if part[TARGET].nunique() != 2:
            raise ValueError(f"The {name} partition must contain both classes.")
        print(f"{name}: {len(part):,} orders")

    return parts


# ==========================================
# 7. XGBOOST BASELINE AND STAGED TUNING
# ==========================================

def train_xgboost(X_train, y_train, X_val, y_val):
    preprocessor = make_xgb_preprocessor()
    train_encoded = preprocessor.fit_transform(X_train)
    val_encoded = preprocessor.transform(X_val)

    base_params = {
        "objective": "binary:logistic",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "scale_pos_weight": 1.0,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "tree_method": "hist",
        "eval_metric": ["logloss", "aucpr"],
        "random_state": SEED,
        "n_jobs": 4,
    }

    trials = []
    best = None

    def fit_candidate(name, params, early_stopping=False):
        nonlocal best

        model = create_xgb_model(params, early_stopping)
        model.fit(
            train_encoded,
            y_train,
            eval_set=[(train_encoded, y_train), (val_encoded, y_val)],
            verbose=False,
        )

        scores = model.predict_proba(val_encoded)[:, 1]
        metrics = evaluate(y_val, scores, DEFAULT_THRESHOLD)
        logged_params = {
            **params,
            "early_stopping": early_stopping,
            "trees_retained": model.get_booster().num_boosted_rounds(),
        }
        trials.append({"run": name, **logged_params, **metrics})
        print(f"{name}: validation AP={metrics['average_precision']:.4f}", flush=True)

        # Match the notebook's AP ranking and run-name tie-breaker.
        key = (-metrics["average_precision"], name)
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "run": name,
                "model": model,
                "params": params,
                "log_params": logged_params,
                "scores": scores,
            }

    fit_candidate("XGB_baseline", base_params)

    ratio = int((y_train == 0).sum()) / int((y_train == 1).sum())
    weights = {"none": 1.0, "intermediate": 3.0, "training_ratio": ratio}

    for depth, (weight_name, weight) in product((3, 4, 6), weights.items()):
        if (depth, weight_name) == (6, "none"):
            continue
        fit_candidate(
            f"XGB_depth{depth}_weight_{weight_name}",
            {**base_params, "max_depth": depth, "scale_pos_weight": weight},
        )

    fit_candidate(
        "XGB_slow_learning",
        {**best["params"], "learning_rate": 0.05, "n_estimators": 1000},
        early_stopping=True,
    )
    fit_candidate(
        "XGB_sampling",
        {**best["params"], "subsample": 0.8, "colsample_bytree": 0.8},
        early_stopping=best["log_params"]["early_stopping"],
    )

    parent_params = best["params"]
    early_stopping = best["log_params"]["early_stopping"]
    regularisation = {
        "XGB_lambda5": {"reg_lambda": 5.0},
        "XGB_child_weight5": {"min_child_weight": 5.0},
        "XGB_gamma1": {"gamma": 1.0},
    }
    for name, change in regularisation.items():
        fit_candidate(name, {**parent_params, **change}, early_stopping)

    best["pipeline"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best.pop("model")),
    ])
    return best, pd.DataFrame(trials)


# ==========================================
# 8. LOGISTIC REGRESSION BASELINE AND GRID SEARCH
# ==========================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    trials = []
    best = None
    grid = product(
        (False, True),
        (False, True),
        (None, "balanced"),
        (0.01, 0.1, 1.0, 10.0),
    )

    for use_log, use_interactions, class_weight, C in grid:
        transform = "log1p" if use_log else "original"
        interactions = "on" if use_interactions else "off"
        weight = class_weight or "none"
        name = f"LR_{transform}_int-{interactions}_weight-{weight}_C-{C:g}"
        if (use_log, use_interactions, class_weight, C) == (False, False, None, 1.0):
            name = "LR_baseline"

        params = {
            "C": C,
            "class_weight": class_weight,
            "use_log": use_log,
            "use_interactions": use_interactions,
            "solver": "lbfgs",
            "max_iter": 1000,
        }

        pipeline = create_lr_model(C, class_weight, use_log, use_interactions)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            with threadpool_limits(limits=4, user_api="blas"):
                pipeline.fit(X_train, y_train)

        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        scores = pipeline.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, scores, DEFAULT_THRESHOLD)
        trials.append({"run": name, **params, "converged": converged, **metrics})
        print(f"{name}: validation AP={metrics['average_precision']:.4f}", flush=True)

        key = (metrics["average_precision"], metrics["f1"])
        if converged and (best is None or key > best["key"]):
            best = {
                "key": key,
                "run": name,
                "pipeline": pipeline,
                "log_params": params,
                "scores": scores,
            }

    if best is None:
        raise RuntimeError("No Logistic Regression candidate converged.")

    return best, pd.DataFrame(trials)


# ==========================================
# 9. LOG FINALISTS TO MLFLOW
# ==========================================

def model_requirements(model_name):
    packages = ["cloudpickle", "numpy", "pandas", "scikit-learn"]
    if model_name == "XGBoost":
        try:
            version("xgboost")
        except PackageNotFoundError:
            packages.append("xgboost-cpu")
        else:
            packages.append("xgboost")
    return [f"{package}=={version(package)}" for package in packages]


def log_finalist(name, candidate, trials, parts, comparison_id, is_winner):
    threshold = candidate["threshold"]
    serving_model = FraudModel(candidate["pipeline"], threshold)

    test = parts["test"]
    test_predictions = serving_model.predict(None, test[FEATURES])
    test_metrics = evaluate(
        test[TARGET], test_predictions["fraud_score"].to_numpy(), threshold
    )

    params = {
        "model_type": name,
        "selected_run": candidate["run"],
        "split_method": "chronological_holdout",
        "decision_threshold": threshold,
        "threshold_metric": "validation_fraud_f1",
        **candidate["log_params"],
    }
    for split, part in parts.items():
        params[f"{split}_samples"] = len(part)
        params[f"{split}_start"] = str(part["order_date"].min().date())
        params[f"{split}_end"] = str(part["order_date"].max().date())

    with mlflow.start_run(run_name=f"{name}_Chronological_Holdout") as run:
        mlflow.set_tags({
            "dataset": DATA_PATH.name,
            "comparison_id": comparison_id,
            "selected_for_final_model": str(is_winner).lower(),
        })
        mlflow.log_params({
            key: value if value is not None else "none"
            for key, value in params.items()
        })

        for split, metrics in [
            ("validation", candidate["metrics"]),
            ("test", test_metrics),
        ]:
            mlflow.log_metrics({
                f"{split}_{key}": float(value)
                for key, value in metrics.items()
            })

        mlflow.log_text(
            trials.to_csv(index=False), "validation_trials_at_0_5.csv"
        )

        matrix = np.array([
            [test_metrics["tn"], test_metrics["fp"]],
            [test_metrics["fn"], test_metrics["tp"]],
        ])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Genuine", "Fraud"],
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name} test confusion matrix (cutoff {threshold:.4f})")
        fig.tight_layout()
        mlflow.log_figure(fig, "test_confusion_matrix.png")
        plt.close(fig)

        sample = parts["validation"][FEATURES].head(20)
        signature = infer_signature(sample, serving_model.predict(None, sample))
        model_info = mlflow.pyfunc.log_model(
            name="fraud_model",
            python_model=serving_model,
            signature=signature,
            pip_requirements=model_requirements(name),
        )

        print(f"{name} run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")

    return test_metrics


# ==========================================
# 10. COMPARE BOTH MODELS AND SELECT THE WINNER
# ==========================================

def main():
    parts = split_data(load_data())
    configure_mlflow()

    train, val = parts["train"], parts["validation"]
    args = (train[FEATURES], train[TARGET], val[FEATURES], val[TARGET])

    results = {
        "XGBoost": train_xgboost(*args),
        "Logistic Regression": train_logistic_regression(*args),
    }

    for candidate, _ in results.values():
        candidate["threshold"] = tune_threshold(val[TARGET], candidate["scores"])
        candidate["metrics"] = evaluate(
            val[TARGET], candidate["scores"], candidate["threshold"]
        )

    # Select the winner before scoring the test orders.
    winner = max(
        results,
        key=lambda name: (
            results[name][0]["metrics"]["f1"],
            results[name][0]["metrics"]["recall"],
        ),
    )

    comparison_id = uuid4().hex
    comparison = []

    for name, (candidate, trials) in results.items():
        test_metrics = log_finalist(
            name, candidate, trials, parts, comparison_id, name == winner
        )
        comparison.append({
            "model": name,
            "run": candidate["run"],
            "cutoff": candidate["threshold"],
            "validation_f1": candidate["metrics"]["f1"],
            "validation_recall": candidate["metrics"]["recall"],
            "test_f1": test_metrics["f1"],
            "test_recall": test_metrics["recall"],
        })

    table = pd.DataFrame(comparison).sort_values(
        ["validation_f1", "validation_recall"], ascending=False
    )
    print("\nFinal model comparison")
    print(table.round(4).to_string(index=False))
    print(f"\nFinal model: {winner} at cutoff {results[winner][0]['threshold']:.6f}")


if __name__ == "__main__":
    main()