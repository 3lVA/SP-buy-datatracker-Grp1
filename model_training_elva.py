# ==========================================
# IMPORTS
# ==========================================
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. DATASET FEATURES
# ==========================================
NUMERICAL_FEATURES = [
    "order_value",
    "num_items_ordered",
    "days_since_first_order",
    "value_per_item",
]
CATEGORICAL_FEATURES = [
    "country_code",
    "collect_type",
    "payment_method",
    "order_weekday",
]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"
SEED = 42


# ==========================================
# 2. PREPROCESSING
# ==========================================
# Fit on training data, separately within each CV fold.
def create_preprocessor(scale_numeric=False):
    numerical_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numerical_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numerical_steps), NUMERICAL_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ==========================================
# 3. CONFIGURE MLFLOW
# ==========================================
def configure_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("Fraud Detection")
    print("MLflow URI:", mlflow.get_tracking_uri(), flush=True)


# ==========================================
# 4. MODEL FUNCTIONS
# ==========================================
# Baseline and Tuned models differ only in class weighting.
def create_LR_model(class_weight=None):
    return Pipeline(
        [
            ("preprocessor", create_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=SEED,
                ),
            ),
        ]
    )


def create_RF_model(class_weight=None):
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=10,
                    class_weight=class_weight,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================
# Fraud (1) is the positive class for precision, recall and F1.
def evaluate_model(model, X_eval, y_eval):
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = model.predict(X_eval)
    matrix = confusion_matrix(y_eval, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "precision": precision_score(y_eval, predictions, zero_division=0),
        "recall": recall_score(y_eval, predictions, zero_division=0),
        "f1": f1_score(y_eval, predictions, zero_division=0),
        "accuracy": accuracy_score(y_eval, predictions),
        "average_precision": average_precision_score(y_eval, probabilities),
        "roc_auc": roc_auc_score(y_eval, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, matrix


def plot_confusion_matrices(run_name, matrices):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cell_labels = [["TN", "FP"], ["FN", "TP"]]

    for ax, (name, matrix) in zip(axes, matrices.items()):
        chart = ConfusionMatrixDisplay(
            matrix, display_labels=["Legitimate (0)", "Fraud (1)"]
        )
        chart.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        for row in range(2):
            for column in range(2):
                chart.text_[row, column].set_text(
                    f"{cell_labels[row][column]}\n{matrix[row, column]:,}"
                )
        ax.set_title(name.capitalize())

    fig.suptitle(f"{run_name} - Confusion Matrices")
    fig.text(
        0.5,
        0.02,
        "TN: correctly cleared | FP: false alarm | FN: missed fraud | TP: fraud caught",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    return fig


# ==========================================
# 6. TRAINING AND MLFLOW LOGGING
# ==========================================
def run_experiment(
    run_name, model, X_train, X_val, X_test, y_train, y_val, y_test, use_kfold=False
):
    data_sets = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }
    for name, (_, y_part) in data_sets.items():
        if y_part.nunique() != 2:
            raise ValueError(f"The {name} partition must contain both classes.")

    with mlflow.start_run(run_name=run_name):
        classifier = model.named_steps["classifier"]
        model_params = classifier.get_params()
        mlflow.log_params(
            {
                "model_type": type(classifier).__name__,
                "split_method": (
                    "stratified_kfold" if use_kfold else "chronological_holdout"
                ),
                "training_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "decision_threshold": 0.50,
                "stratified": use_kfold,
                **{
                    f"{name}_samples": len(X_part)
                    for name, (X_part, _) in data_sets.items()
                },
                **{
                    key: "none" if model_params[key] is None else model_params[key]
                    for key in (
                        "C",
                        "solver",
                        "max_iter",
                        "class_weight",
                        "random_state",
                        "n_estimators",
                        "max_depth",
                        "min_samples_leaf",
                    )
                    if key in model_params
                },
            }
        )
        mlflow.set_tag("dataset", "base_processed.csv")
        mlflow.set_tag("classifier", type(classifier).__name__)

        if use_kfold:
            # Five-fold CV uses ONLY the 70% training partition.
            # Each fold refits its own imputation, encoding and scaling.
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            scoring = {
                "precision": make_scorer(precision_score, zero_division=0),
                "recall": make_scorer(recall_score, zero_division=0),
                "f1": make_scorer(f1_score, zero_division=0),
                "accuracy": "accuracy",
                "average_precision": "average_precision",
                "roc_auc": "roc_auc",
            }
            mlflow.log_params({"n_splits": 5, "cv_scope": "training_70_percent_only"})
            print(f"\n{run_name}: starting stratified five-fold CV", flush=True)
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=skf,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
                verbose=2,
            )
            for metric in scoring:
                scores = cv_results[f"test_{metric}"]
                mlflow.log_metrics(
                    {
                        f"cv_{metric}": float(scores.mean()),
                        f"cv_{metric}_std": float(scores.std()),
                    }
                )
                for fold, score in enumerate(scores, start=1):
                    mlflow.log_metric(f"fold_{fold}_{metric}", float(score))
                print(
                    f"{metric}: {np.round(scores, 4)} | "
                    f"mean={scores.mean():.4f}, std={scores.std():.4f}"
                )

        # Final fit uses the full 70%; validation and test remain held out.
        print(f"\nFitting {run_name} on {len(X_train):,} training orders", flush=True)
        model.fit(X_train, y_train)
        results = {}
        matrices = {}
        for name, (X_part, y_part) in data_sets.items():
            metrics, matrix = evaluate_model(model, X_part, y_part)
            results[name] = metrics
            matrices[name] = matrix
            mlflow.log_metrics(
                {f"{name}_{key}": float(value) for key, value in metrics.items()}
            )

        # Show train/validation/test together, with labels inside every cell.
        fig = plot_confusion_matrices(run_name, matrices)
        mlflow.log_figure(fig, "confusion_matrices.png")
        plt.show(block=False)
        plt.pause(0.1)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )
        print(
            f"\n{run_name} metrics:\n{pd.DataFrame(results).T.round(4).to_string()}",
            flush=True,
        )

    return {
        "model": run_name,
        **{f"validation_{key}": value for key, value in results["validation"].items()},
        **{f"test_{key}": value for key, value in results["test"].items()},
    }


def main():
    # Allow MLflow's Unicode run links in Windows terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    configure_mlflow()

    # ==========================================
    # 7. LOAD DATASET
    # ==========================================
    # Read the labelled CSV already exported by the EDA notebook.
    df = pd.read_csv(Path(__file__).with_name("base_processed.csv"))
    df["order_date"] = pd.to_datetime(df["order_date"], errors="raise")
    df[TARGET] = df[TARGET].astype(int)
    df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].astype(float)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("Unknown").astype(object)
    print("Dataset shape:", df.shape)
    print("Fraud distribution:\n", df[TARGET].value_counts(normalize=True))

    # ==========================================
    # 8. LOGISTIC REGRESSION - STRATIFIED 5-FOLD
    # ==========================================
    # Stratified 70/30, then split the 30% equally into 15/15.
    orders = df.sort_values(["country_code", "order_id"])
    X = orders[FEATURES]
    y = orders[TARGET]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.50,
        stratify=y_holdout,
        random_state=SEED,
    )
    print("\nLR training shapes:", X_train.shape, y_train.shape)
    print("LR validation shapes:", X_val.shape, y_val.shape)
    print("LR testing shapes:", X_test.shape, y_test.shape)
    comparison = [
        run_experiment(
            "LR_Baseline",
            create_LR_model(),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            use_kfold=True,
        ),
        run_experiment(
            "LR_TUNED",
            create_LR_model("balanced"),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            use_kfold=True,
        ),
    ]

    # ==========================================
    # 9. RANDOM FOREST - CHRONOLOGICAL HOLDOUT
    # ==========================================
    # Earliest 70% training, next 15% validation, latest 15% test.
    orders = df.sort_values(["order_date", "country_code", "order_id"])
    X = orders[FEATURES]
    y = orders[TARGET]
    train_end = int(len(orders) * 0.70)
    validation_end = int(len(orders) * 0.85)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    print("\nRF training shapes:", X_train.shape, y_train.shape)
    print("RF validation shapes:", X_val.shape, y_val.shape)
    print("RF testing shapes:", X_test.shape, y_test.shape)
    comparison.extend(
        [
            run_experiment(
                "RF_Baseline",
                create_RF_model(),
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ),
            run_experiment(
                "RF_Tuned",
                create_RF_model("balanced"),
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ),
        ]
    )

    # ==========================================
    # 10. BASELINE AND TUNED MODEL COMPARISON
    # ==========================================
    # Use validation for model choices; test metrics are final reports.
    comparison = pd.DataFrame(comparison).set_index("model")
    for split in ("validation", "test"):
        columns = [
            f"{split}_{metric}"
            for metric in (
                "precision",
                "recall",
                "f1",
                "accuracy",
                "average_precision",
                "roc_auc",
            )
        ]
        print(
            f"\n{split.upper()} MODEL COMPARISON\n",
            comparison[columns].round(4).to_string(),
        )
    print("\nCompare baseline/tuned models within each algorithm using validation metrics.")
    print(
        "LR and RF use different test sets, so their test results are not a controlled comparison."
    )
    print("All experiments logged to MLflow:", mlflow.get_tracking_uri())
    # Keep all four model figures open after training finishes.
    plt.show()


if __name__ == "__main__":
    main()
