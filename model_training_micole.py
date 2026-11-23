import os 
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow 
import mlflow.sklearn 
import mlflow.xgboost
import matplotlib.pyplot as plt 

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

# Update to your cleaned dataset CSV path
df = pd.read_csv("base_processed.csv")

features = [
    "country_code",
    "collect_type",
    "payment_method",
    "order_value",
    "num_items_ordered",
    "days_since_first_order",
    "value_per_item",
    "order_weekday"
]

X = df[features].copy()
y = df["is_fraud"].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFraud distribution:")
print(y.value_counts(normalize=True))


# ==========================================
# 2. PREPROCESSING
# ==========================================

numerical_features = [
    "order_value",
    "num_items_ordered",
    "days_since_first_order",
    "value_per_item"
]

categorical_features = [
    "country_code",
    "collect_type",
    "payment_method",
    "order_weekday"
]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# 3. CONFIGURE MLFLOW
# ==========================================

MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Fraud Detection")

print("\nMLflow URI:", mlflow.get_tracking_uri())


# ==========================================
# 4. MODEL FUNCTIONS
# ==========================================

def create_XGB_model():
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb)
    ])

    return model


def create_RFC_model():
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    return model


def createStratifiedKFold():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_eval, y_eval):

    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

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
        "roc_auc": roc_auc_score(y_eval, probabilities)
    }

    return metrics


# ==========================================
# 6. COMMON TEST SET
# ==========================================

# Both classifiers use the same 15% test set.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

print("\nDevelopment samples:", len(X_dev))
print("Testing samples:", len(X_test))


# ==========================================
# 7. XGBOOST - STRATIFIED HOLDOUT
# ==========================================

print("\n========== XGBOOST ==========")

# Split the 85% development set into
# 70% training and 15% validation overall.
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=15/85,
    random_state=42,
    stratify=y_dev
)

with mlflow.start_run(
    run_name="XGBoost_Stratified_Holdout"
):

    xgb_model = create_XGB_model()

    # Train model
    xgb_model.fit(X_train, y_train)

    # Validation metrics
    val_metrics = evaluate_model(
        xgb_model, X_val, y_val
    )

    # Final test metrics
    xgb_test_metrics = evaluate_model(
        xgb_model, X_test, y_test
    )

    # Log parameters
    mlflow.log_params({
        "model_type": "XGBoost",
        "split_method": "stratified_holdout",
        "training_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "stratified": True,
        "random_state": 42,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "testing_samples": len(X_test),
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })

    # Log validation metrics
    mlflow.log_metrics({
        f"validation_{k}": float(v)
        for k, v in val_metrics.items()
    })

    # Log test metrics
    mlflow.log_metrics({
        f"test_{k}": float(v)
        for k, v in xgb_test_metrics.items()
    })

    mlflow.sklearn.log_model(
            sk_model=xgb_model,
            name="xgboost_model",
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )

    # Log model tags
    mlflow.set_tag("dataset", "fraud")
    mlflow.set_tag("classifier", "XGBoost")

    # Display validation metrics
    print("\nXGBoost Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    # Display test metrics
    print("\nXGBoost Test Metrics:")
    for k, v in xgb_test_metrics.items():
        print(f"{k}: {v:.4f}")

# ==========================================
# 8. RANDOM FOREST - STRATIFIED K-FOLD
# ==========================================

print("\n========== RANDOM FOREST ==========")

skf = createStratifiedKFold()

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

with mlflow.start_run(
    run_name="RandomForest_Stratified_KFold"
):

    rf_model = create_RFC_model()

    # Perform 5-fold cross-validation
    cv_results = cross_validate(
        rf_model,
        X_dev,
        y_dev,
        cv=skf,
        scoring=scoring,
        n_jobs=1
    )

    # Calculate mean CV metrics
    cv_metrics = {
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "precision": np.mean(cv_results["test_precision"]),
        "recall": np.mean(cv_results["test_recall"]),
        "f1": np.mean(cv_results["test_f1"]),
        "roc_auc": np.mean(cv_results["test_roc_auc"])
    }

    # Train final model on full development set
    rf_model.fit(X_dev, y_dev)

    # Evaluate using independent test set
    rf_test_metrics = evaluate_model(
        rf_model, X_test, y_test
    )

    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForest",
        "split_method": "stratified_kfold",
        "n_splits": 5,
        "training_fraction": 0.85,
        "test_fraction": 0.15,
        "stratified": True,
        "random_state": 42,
        "training_samples": len(X_dev),
        "testing_samples": len(X_test),
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "class_weight": "balanced"
    })

    # Log mean cross-validation metrics
    mlflow.log_metrics({
        f"cv_{k}": float(v)
        for k, v in cv_metrics.items()
    })

    # Log individual fold metrics
    for i in range(5):
        mlflow.log_metrics({
            f"fold_{i+1}_accuracy":
                float(cv_results["test_accuracy"][i]),
            f"fold_{i+1}_precision":
                float(cv_results["test_precision"][i]),
            f"fold_{i+1}_recall":
                float(cv_results["test_recall"][i]),
            f"fold_{i+1}_f1":
                float(cv_results["test_f1"][i]),
            f"fold_{i+1}_roc_auc":
                float(cv_results["test_roc_auc"][i])
        })

    # Log test metrics
    mlflow.log_metrics({
        f"test_{k}": float(v)
        for k, v in rf_test_metrics.items()
    })

    # Log model artifact
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        name="random_forest_model",
        serialization_format="skops",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )

    mlflow.set_tag("dataset", "fraud")
    mlflow.set_tag("classifier", "RandomForest")

    print("\nRandom Forest Cross-Validation Metrics:")
    for k, v in cv_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nRandom Forest Test Metrics:")
    for k, v in rf_test_metrics.items():
        print(f"{k}: {v:.4f}")


# ==========================================
# 9. COMPARE BOTH MODELS
# ==========================================

comparison = pd.DataFrame(
    [xgb_test_metrics, rf_test_metrics],
    index=["XGBoost", "Random Forest"]
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4))

print("\nAll experiments logged to MLflow!")
print("MLflow URI:", MLFLOW_URI)
