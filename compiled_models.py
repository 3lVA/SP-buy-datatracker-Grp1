# MERGED TRAINING SCRIPT — MICOLE + ELVA + JONATHAN
# ------------------------------------------------------------
# Usage: python compiled_models_all_members.py
# File 1: Elva 
# File 2: Jonathan 
# File 3: Micole 
#


# ==========================================
# IMPORTS
# ==========================================

import os
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-aws-eb-group1-env.eba-k9axgp3x.us-east-1.elasticbeanstalk.com/")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SP-Buy-Fraud-Detection")
LOG_MODEL_ARTIFACTS = os.getenv("LOG_MODEL_ARTIFACTS", "1") == "1"

SEED = 42
TARGET = "is_fraud"
# Identifies runs produced by the same invocation for MLflow comparison.
COMPARISON_BATCH = uuid4().hex

DATA_PATH = Path(os.getenv("FRAUD_DATA_PATH", str(Path(__file__).with_name("base_processed.csv"))))


# ==========================================
# 2. DATASET FEATURES
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


# ==========================================
# 3. MLFLOW CONFIGURATION
# ==========================================

def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", EXPERIMENT_NAME)


# ==========================================
# 4. PREPROCESSING
# ==========================================

def create_preprocessor(scale_numeric=False):

    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(numerical_steps),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# ==========================================
# 5. MODEL CREATION FUNCTIONS
# ==========================================

# ----- Logistic Regression -----

def create_LR_model(class_weight=None):

    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(scale_numeric=True),
            ),
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


# ----- Random Forest (File 1) -----

def create_RF_model(class_weight=None):

    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(),
            ),
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


# ----- XGBoost (File 3) -----

def create_XGB_model():

    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ----- Random Forest (File 3) -----

def create_RFC_model():

    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_eval, y_eval):

    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    matrix = confusion_matrix(
        y_eval,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(
            y_eval, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_eval, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_eval, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_eval, probabilities
        ),
    }

    metrics["average_precision"] = average_precision_score(y_eval, probabilities)
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics, matrix


# ==========================================
# 7. COMMON MLFLOW MODEL LOGGING
# ==========================================

def log_sklearn_model(model, model_name, input_example, trusted_types=None):
    """Log fitted sklearn/XGBoost pipelines without skops trust-list failures.

    cloudpickle can serialize sklearn pipelines that contain numpy dtypes,
    transformers and XGBoost estimators. Only load model artifacts from
    MLflow servers and runs you trust (pickle-based files can execute code).
    The trusted_types argument is retained for compatibility with call sites.
    """
    if not LOG_MODEL_ARTIFACTS:
        mlflow.set_tag("model_artifact_logged", "false")
        return
    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        input_example=input_example.head(5).copy(),
        serialization_format="cloudpickle",
    )
    mlflow.set_tag("model_artifact_logged", "true")


# ==========================================
# 9. FILE 1 EXPERIMENT RUNNER
# ==========================================

def run_experiment(
    run_name,
    model,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    use_kfold=False,
):

    data_sets = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for name, (_, y_part) in data_sets.items():

        if y_part.nunique() != 2:
            raise ValueError(
                f"The {name} partition must contain both classes."
            )

    with mlflow.start_run(run_name="Elva_" + run_name):
        mlflow.set_tags({"team_member": "Elva", "evaluation_group": "elva_lr_stratified" if use_kfold else "elva_rf_chronological", "comparison_batch": COMPARISON_BATCH, "split_design": "stratified_random_70_15_15" if use_kfold else "chronological_rows_70_15_15", "controlled_comparison": "false"})

        classifier = model.named_steps["classifier"]
        model_params = classifier.get_params()

        mlflow.log_params(
            {
                "model_type": type(classifier).__name__,
                "split_method": (
                    "stratified_kfold"
                    if use_kfold
                    else "chronological_holdout"
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
                    key: (
                        "none"
                        if model_params[key] is None
                        else model_params[key]
                    )
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

        mlflow.set_tag(
            "dataset", "base_processed.csv"
        )

        mlflow.set_tag(
            "classifier", type(classifier).__name__
        )

        # ----- Stratified 5-Fold CV -----

        if use_kfold:

            skf = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=SEED,
            )

            scoring = {
                "precision": make_scorer(
                    precision_score,
                    zero_division=0,
                ),
                "recall": make_scorer(
                    recall_score,
                    zero_division=0,
                ),
                "f1": make_scorer(
                    f1_score,
                    zero_division=0,
                ),
                "accuracy": "accuracy",
                "average_precision": "average_precision",
                "roc_auc": "roc_auc",
            }

            mlflow.log_params(
                {
                    "n_splits": 5,
                    "cv_scope": "training_70_percent_only",
                }
            )

            print(
                f"\n{run_name}: Starting stratified 5-fold CV"
            )

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

                scores = cv_results[
                    f"test_{metric}"
                ]

                mlflow.log_metrics(
                    {
                        f"cv_{metric}": float(
                            scores.mean()
                        ),
                        f"cv_{metric}_std": float(
                            scores.std()
                        ),
                    }
                )

                for fold, score in enumerate(
                    scores, start=1
                ):

                    mlflow.log_metric(
                        f"fold_{fold}_{metric}",
                        float(score),
                    )

                print(
                    f"{metric}: {np.round(scores, 4)} | "
                    f"mean={scores.mean():.4f}, "
                    f"std={scores.std():.4f}"
                )

        # ----- Final Model Training -----

        print(
            f"\nFitting {run_name} "
            f"on {len(X_train):,} training orders"
        )

        model.fit(X_train, y_train)

        results = {}

        # ----- Train, Validation and Test Evaluation -----

        for name, (X_part, y_part) in data_sets.items():

            metrics, matrix = evaluate_model(
                model,
                X_part,
                y_part,
            )

            # Average Precision retained from File 1
            probabilities = model.predict_proba(
                X_part
            )[:, 1]

            metrics["average_precision"] = (
                average_precision_score(
                    y_part,
                    probabilities,
                )
            )

            tn, fp, fn, tp = matrix.ravel()

            metrics.update(
                {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

            results[name] = metrics

            mlflow.log_metrics(
                {
                    f"{name}_{key}": float(value)
                    for key, value in metrics.items()
                }
            )

        # Confusion matrix counts (tn/fp/fn/tp) were logged as metrics above.
        # MLflow can compare scalar metrics across runs without matplotlib.

        # ----- Log Model -----

        log_sklearn_model(
            model,
            "model",
            X_train,
            [
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )

        print(
            f"\n{run_name} metrics:\n",
            pd.DataFrame(results).T.round(4),
        )

    return {
        "model": run_name,

        **{
            f"validation_{key}": value
            for key, value in results["validation"].items()
        },

        **{
            f"test_{key}": value
            for key, value in results["test"].items()
        },
    }


# ==========================================
# 10. FILE 3 EXPERIMENT RUNNER
# ==========================================

def run_file3_experiment(
    run_name,
    model,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    model_type,
    model_params,
    trusted_types,
):

    with mlflow.start_run(run_name="Micole_" + run_name):
        mlflow.set_tags({"team_member": "Micole", "evaluation_group": "micole_shared_stratified", "comparison_batch": COMPARISON_BATCH, "split_design": "stratified_random_70_15_15", "controlled_comparison": "false"})

        model.fit(X_train, y_train)

        # Validation evaluation
        val_metrics, val_matrix = evaluate_model(
            model,
            X_val,
            y_val,
        )

        # Test evaluation
        test_metrics, test_matrix = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # Standardised MLflow parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "split_method": "stratified_holdout",
                "training_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "stratified": True,
                "random_state": SEED,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "testing_samples": len(X_test),
                **model_params,
            }
        )

        # Standardised metric naming
        mlflow.log_metrics(
            {
                f"validation_{key}": float(value)
                for key, value in val_metrics.items()
            }
        )

        mlflow.log_metrics(
            {
                f"test_{key}": float(value)
                for key, value in test_metrics.items()
            }
        )

        mlflow.set_tag(
            "dataset", "base_processed.csv"
        )

        mlflow.set_tag(
            "classifier", model_type
        )

        # Test confusion-matrix counts are already logged as metrics.

        # Log fitted model
        log_sklearn_model(
            model,
            "model",
            X_train,
            trusted_types,
        )

        print(f"\n{run_name} Validation Metrics:")

        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        print(f"\n{run_name} Test Metrics:")

        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")

    return {
        "model": run_name,
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
    }


# ==========================================
# 11. MAIN FUNCTION
# ==========================================


# ============================================================
# MEMBER WORKFLOWS — original splits, configurations, and CV
# ============================================================

def prepare_elva_data(df):
    # Elva's original preprocessing.
    df_file1 = df.copy()

    df_file1["order_date"] = pd.to_datetime(
        df_file1["order_date"],
        errors="raise",
    )

    df_file1[TARGET] = df_file1[TARGET].astype(int)

    df_file1[NUMERICAL_FEATURES] = (
        df_file1[NUMERICAL_FEATURES].astype(float)
    )

    df_file1[CATEGORICAL_FEATURES] = (
        df_file1[CATEGORICAL_FEATURES]
        .fillna("Unknown")
        .astype(object)
    )
    return df_file1


def train_elva_logistic_regression(df_file1):
    comparison = []
    print("\n========== LOGISTIC REGRESSION ==========")

    orders = df_file1.sort_values(
        ["country_code", "order_id"]
    )

    X = orders[FEATURES]
    y = orders[TARGET]

    # Original 70/30 split
    X_train, X_holdout, y_train, y_holdout = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=SEED,
        )
    )

    # Original 15/15 split
    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_holdout,
            y_holdout,
            test_size=0.50,
            stratify=y_holdout,
            random_state=SEED,
        )
    )

    print("\nLR Training:", X_train.shape)
    print("LR Validation:", X_val.shape)
    print("LR Testing:", X_test.shape)

    # Only Elva's tuned Logistic Regression is trained and logged.
    comparison.append(
        run_experiment(
            "LR_Tuned_Stratif5Fold",
            create_LR_model("balanced"),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            use_kfold=True,
        )
    )
    return comparison


def train_elva_random_forest(df_file1):
    comparison = []
    print("\n========== RANDOM FOREST ==========")

    orders = df_file1.sort_values(
        ["order_date", "country_code", "order_id"]
    )

    X = orders[FEATURES]
    y = orders[TARGET]

    # Original chronological split
    train_end = int(len(orders) * 0.70)
    validation_end = int(len(orders) * 0.85)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    print("\nRF Training:", X_train.shape)
    print("RF Validation:", X_val.shape)
    print("RF Testing:", X_test.shape)

    # Only Elva's tuned Random Forest is trained and logged.
    comparison.append(
        run_experiment(
            "RF_Tuned_Chronological",
            create_RF_model("balanced"),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )
    )
    return comparison


def prepare_micole_data(df):
    # Micole's original preprocessing.
    # Missing categorical values are handled
    # inside its preprocessing pipeline.

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    print("\n========== MICOLE MODELS ==========")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ==========================================
    # COMMON 85/15 DEVELOPMENT/TEST SPLIT
    # ==========================================

    # Original File 3 split.
    # Both XGBoost and Random Forest use
    # the exact same independent test set.

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=SEED,
            stratify=y,
        )
    )

    print("\nDevelopment samples:", len(X_dev))
    print("Testing samples:", len(X_test))
    return X_dev, X_test, y_dev, y_test


def train_micole_xgboost(micole_data):
    X_dev, X_test, y_dev, y_test = micole_data
    comparison = []
    print("\n========== XGBOOST ==========")

    # Original File 3 development split.
    # 70% training and 15% validation overall.

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=15/85,
            random_state=SEED,
            stratify=y_dev,
        )
    )

    print("\nXGBoost Training:", X_train.shape)
    print("XGBoost Validation:", X_val.shape)
    print("XGBoost Testing:", X_test.shape)

    comparison.append(
        run_file3_experiment(
            "XGBoost_Stratified_Holdout",
            create_XGB_model(),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            "XGBoost",
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            [
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
    )
    return comparison


def train_micole_random_forest(micole_data):
    X_dev, X_test, y_dev, y_test = micole_data
    comparison = []
    print("\n========== MICOLE RANDOM FOREST ==========")

    skf = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    with mlflow.start_run(
        run_name="Micole_RF_Balanced_Stratif5Fold"
    ):
        mlflow.set_tags({"team_member": "Micole", "evaluation_group": "micole_shared_stratified", "comparison_batch": COMPARISON_BATCH, "split_design": "stratified_random_dev85_test15_cv3", "controlled_comparison": "false"})

        rf_model = create_RFC_model()

        # Original CV on the full 85% development set.
        cv_results = cross_validate(
            rf_model,
            X_dev,
            y_dev,
            cv=skf,
            scoring=scoring,
            n_jobs=1,
        )

        # Calculate mean CV metrics
        cv_metrics = {
            metric: float(
                np.mean(cv_results[f"test_{metric}"])
            )
            for metric in scoring
        }

        # Log CV metrics
        mlflow.log_metrics(
            {
                f"cv_{key}": value
                for key, value in cv_metrics.items()
            }
        )

        # Log individual folds
        for i in range(skf.get_n_splits()):

            mlflow.log_metrics(
                {
                    f"fold_{i+1}_{metric}": float(
                        cv_results[f"test_{metric}"][i]
                    )
                    for metric in scoring
                }
            )

        # Original final fit on full 85% development set
        rf_model.fit(X_dev, y_dev)

        # Independent test evaluation
        rf_test_metrics, rf_test_matrix = evaluate_model(
            rf_model,
            X_test,
            y_test,
        )

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "RandomForest",
                "split_method": "stratified_kfold",
                "training_fraction": 0.85,
                "test_fraction": 0.15,
                "stratified": True,
                "random_state": SEED,
                "n_splits": skf.get_n_splits(),
                "training_samples": len(X_dev),
                "testing_samples": len(X_test),
                "n_estimators": rf_model.named_steps["classifier"].n_estimators,
                "max_depth": rf_model.named_steps["classifier"].max_depth,
                "min_samples_leaf": rf_model.named_steps["classifier"].min_samples_leaf,
                "class_weight": rf_model.named_steps["classifier"].class_weight,
            }
        )

        # Log test metrics
        mlflow.log_metrics(
            {
                f"test_{key}": float(value)
                for key, value in rf_test_metrics.items()
            }
        )

        mlflow.set_tag(
            "dataset", "base_processed.csv"
        )

        mlflow.set_tag(
            "classifier", "RandomForest"
        )

        # Test confusion-matrix counts are already logged as metrics.

        # Log fitted model
        log_sklearn_model(
            rf_model,
            "model",
            X_dev,
            [
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print("\nRandom Forest CV Metrics:")

        for key, value in cv_metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nRandom Forest Test Metrics:")

        for key, value in rf_test_metrics.items():
            print(f"{key}: {value:.4f}")

        comparison.append(
            {
                "model": "RF_Balanced_Stratif5Fold",
                **{
                    f"test_{key}": value
                    for key, value in rf_test_metrics.items()
                },
            }
        )
    return comparison

def prepare_jonathan_context():
    """Build Jonathan's independent chronological holdout and tuning helpers.

    Kept in its own function scope to avoid collisions with the
    preprocessing, metrics, model builders and main() above.
    """
    import os
    import warnings
    from importlib.metadata import PackageNotFoundError, version
    from itertools import product
    from pathlib import Path
    from uuid import uuid4


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


    DATA_PATH = Path(os.getenv("FRAUD_DATA_PATH", str(Path(__file__).with_name("base_processed.csv"))))

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

        with mlflow.start_run(run_name=f"Jonathan_{name.replace(' ', '_')}_Chronological_Holdout") as run:
            mlflow.set_tags({
                "dataset": DATA_PATH.name,
                "comparison_id": comparison_id,
                "selected_for_final_model": str(is_winner).lower(),
                "team_member": "Jonathan",
                "evaluation_group": "jonathan_chronological",
                "controlled_comparison": "false",
                "comparison_batch": COMPARISON_BATCH,
                "split_design": "chronological_dates_70_15_15_approx",
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

            # Scalar metrics include TN, FP, FN and TP for each split.
            # MLflow compares these metrics; it does not create confusion matrix
            # image artifacts from scalar counts automatically.

            if LOG_MODEL_ARTIFACTS:
                sample = parts["validation"][FEATURES].head(20)
                signature = infer_signature(sample, serving_model.predict(None, sample))
                model_info = mlflow.pyfunc.log_model(
                    name="fraud_model",
                    python_model=serving_model,
                    signature=signature,
                    pip_requirements=model_requirements(name),
                )
                mlflow.set_tag("model_artifact_logged", "true")
                print(f"Model URI: {model_info.model_uri}")
            else:
                mlflow.set_tag("model_artifact_logged", "false")

            print(f"{name} run ID: {run.info.run_id}")

        return test_metrics


    parts = split_data(load_data())
    return {
        "parts": parts,
        "train_xgboost": train_xgboost,
        "train_logistic_regression": train_logistic_regression,
        "tune_threshold": tune_threshold,
        "evaluate": evaluate,
        "log_finalist": log_finalist,
    }


def train_jonathan_xgboost(context):
    """Run Jonathan's original staged XGBoost search."""
    parts = context["parts"]
    train, val = parts["train"], parts["validation"]
    return context["train_xgboost"](
        train[FEATURES], train[TARGET], val[FEATURES], val[TARGET]
    )


def train_jonathan_logistic_regression(context):
    """Run Jonathan's original 64-combination LR grid search."""
    parts = context["parts"]
    train, val = parts["train"], parts["validation"]
    return context["train_logistic_regression"](
        train[FEATURES], train[TARGET], val[FEATURES], val[TARGET]
    )


def finalize_jonathan_experiments(context, xgb_result, lr_result):
    """Tune each cutoff; select winner using validation only; log test once."""
    parts = context["parts"]
    val = parts["validation"]
    results = {"XGBoost": xgb_result, "Logistic Regression": lr_result}
    for candidate, _ in results.values():
        candidate["threshold"] = context["tune_threshold"](
            val[TARGET], candidate["scores"]
        )
        candidate["metrics"] = context["evaluate"](
            val[TARGET], candidate["scores"], candidate["threshold"]
        )
    winner = max(
        results,
        key=lambda name: (
            results[name][0]["metrics"]["f1"],
            results[name][0]["metrics"]["recall"],
        ),
    )
    comparison_id = uuid4().hex
    rows = []
    for name, (candidate, trials) in results.items():
        test_metrics = context["log_finalist"](
            name, candidate, trials, parts, comparison_id, name == winner
        )
        rows.append({
            "model": name, "run": candidate["run"],
            "cutoff": candidate["threshold"],
            "validation_f1": candidate["metrics"]["f1"],
            "validation_recall": candidate["metrics"]["recall"],
            "test_f1": test_metrics["f1"],
            "test_recall": test_metrics["recall"],
        })
    print("\nJonathan's final model comparison")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"Jonathan's selected model: {winner}")
    return rows

# ============================================================
# Comparison of ORIGINAL member experiments (no extra split)
# ============================================================
def compare_original_experiments():
    """Report metrics grouped by member and original split design.

    CAUTION: Since held-out samples and hyperparameters differ between
    workflows, the resulting comparison is descriptive, not a controlled
    measure of split effects.
    """
    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("MLflow experiment not found")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.comparison_batch = '{COMPARISON_BATCH}'",
        output_format="pandas",
    )
    if runs.empty:
        print("No matching runs were logged")
        return

    names = {
        "tags.team_member": "member",
        "tags.evaluation_group": "evaluation_group",
        "tags.split_design": "split_design",
        "params.model_type": "model_type",
        "params.decision_threshold": "threshold",
    }
    metric_keys = ["f1", "recall", "precision", "accuracy", "roc_auc", "average_precision", "fn", "fp"]
    cols = ["run_id", "tags.mlflow.runName"] + list(names) + [
        f"metrics.{split}_{metric}"
        for split in ("validation", "test") for metric in metric_keys
    ]
    for col in cols:
        if col not in runs:
            runs[col] = np.nan
    report = runs[cols].rename(columns={
        "tags.mlflow.runName": "run_name",
        **names,
        **{f"metrics.{sp}_{met}": f"{sp}_{met}" for sp in ("validation", "test") for met in metric_keys},
    })
    # All original member runs; no benchmark copies are added.
    report = report.sort_values(["member", "run_name"], na_position="last")
    report_path = Path(__file__).with_name("original_split_model_comparison.csv")
    report.to_csv(report_path, index=False)
    show = ["member", "run_name", "split_design", "validation_f1", "test_f1", "test_recall", "test_average_precision", "test_fn"]
    print("\n=== ORIGINAL SPLITS — DESCRIPTIVE COMPARISON ===")
    print(report[show].round(4).to_string(index=False))
    print("NOTE: These results use DIFFERENT test sets, and may use different tuning/thresholds.")
    print("They cannot isolate the effect of data splitting alone.")

    with mlflow.start_run(run_name="Original_Splits_Comparison_Summary"):
        mlflow.set_tags({
            "run_role": "comparison_summary", "comparison_batch": COMPARISON_BATCH,
            "controlled_comparison": "false", "description": "Original member splits preserved",
        })
        mlflow.log_artifact(str(report_path), artifact_path="comparison")
    return report


# ============================================================
# SINGLE ENTRY POINT: ALL THREE TEAM MEMBERS
# ============================================================
def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Place base_processed.csv beside the script or set FRAUD_DATA_PATH."
        )
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET, "order_date", "order_id"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("is_fraud must only contain binary 0/1 labels")
    configure_mlflow()
    print("Dataset shape:", df.shape)
    print("Fraud distribution:\n", df[TARGET].value_counts(normalize=True))

    # Elva — Logistic Regression and Random Forest, original splits
    elva_data = prepare_elva_data(df)
    train_elva_logistic_regression(elva_data)
    train_elva_random_forest(elva_data)

    # Micole — XGBoost and Random Forest, original shared splits
    micole_data = prepare_micole_data(df)
    train_micole_xgboost(micole_data)
    train_micole_random_forest(micole_data)

    # Jonathan — shared chronological partition, full tuning preserved
    jonathan_context = prepare_jonathan_context()
    jonathan_xgb = train_jonathan_xgboost(jonathan_context)
    jonathan_lr = train_jonathan_logistic_regression(jonathan_context)
    finalize_jonathan_experiments(jonathan_context, jonathan_xgb, jonathan_lr)

    # MLflow already supports cross-run metric charts; save comparison CSV.
    compare_original_experiments()
    print(f"\nAll experiments completed. Open {EXPERIMENT_NAME} in MLflow.")


if __name__ == "__main__":
    main()
