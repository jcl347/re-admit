"""
train.py — The ONLY file you modify during autoresearch experiments.

This file defines the model architecture, hyperparameters, and training loop.
Everything is fair game: model type, ensembling, feature engineering,
hyperparameter tuning, preprocessing, etc.

The autoresearch loop will modify this file, run it, and evaluate results.
If results improve (higher AUROC), the change is kept. Otherwise, reverted.

Current model: XGBoost baseline
"""

import time
import os
import tracemalloc
import numpy as np
from prepare import (
    preprocess_data,
    evaluate_cv,
    print_results,
    TIME_BUDGET_SECONDS,
    RANDOM_SEED,
)

# ─── Model Configuration ─────────────────────────────────────────────────────
# Modify anything below this line.

MODEL_TYPE = "xgboost"  # Options: xgboost, lightgbm, catboost, sklearn_rf, sklearn_lr, ensemble

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "scale_pos_weight": 1,  # will be auto-set based on class imbalance
    "random_state": RANDOM_SEED,
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
}


# ─── Training Function ───────────────────────────────────────────────────────

def train_and_predict(X_train, y_train, X_val):
    """
    Train a model and return predictions on the validation set.

    Args:
        X_train: training features (n_train, n_features)
        y_train: training labels (n_train,)
        X_val: validation features (n_val, n_features)

    Returns:
        y_pred_proba: predicted probabilities for class 1 (n_val,)
    """
    if MODEL_TYPE == "xgboost":
        from xgboost import XGBClassifier
        params = XGBOOST_PARAMS.copy()
        # Auto-set scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        params["scale_pos_weight"] = neg_count / pos_count
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

    elif MODEL_TYPE == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

    elif MODEL_TYPE == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

    elif MODEL_TYPE == "sklearn_rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

    elif MODEL_TYPE == "sklearn_lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            C=1.0,
        )
        model.fit(X_train_s, y_train)
        y_pred_proba = model.predict_proba(X_val_s)[:, 1]

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    return y_pred_proba


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Model: {MODEL_TYPE}")
    print(f"[train] Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"[train] Time budget: {TIME_BUDGET_SECONDS}s")
    print(f"[train] Running {len(fold_indices)}-fold cross-validation...")

    # Track time and memory
    tracemalloc.start()
    start_time = time.time()

    results = evaluate_cv(X, y, fold_indices, train_and_predict)

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    # Print parseable results
    print_results(results, elapsed, peak_mb)
