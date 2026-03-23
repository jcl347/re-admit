"""
train.py — The ONLY file you modify during autoresearch experiments.

Current model: Stacking ensemble (CatBoost + LightGBM + XGBoost) with LR meta-learner
"""

import time
import tracemalloc
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from prepare import (
    preprocess_data,
    evaluate_cv,
    print_results,
    TIME_BUDGET_SECONDS,
    RANDOM_SEED,
)


def train_and_predict(X_train, y_train, X_val):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count

    # Base models
    models = [
        ("cb", CatBoostClassifier(
            iterations=1000, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, border_count=128,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED, verbose=0, thread_count=-1,
        )),
        ("lgb", LGBMClassifier(
            n_estimators=1000, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            min_child_samples=20, scale_pos_weight=spw,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=RANDOM_SEED,
            eval_metric="auc", tree_method="hist", n_jobs=-1,
        )),
    ]

    # Level 1: Generate out-of-fold predictions for stacking
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    meta_train = np.zeros((len(X_train), len(models)))
    meta_val = np.zeros((len(X_val), len(models)))

    for i, (name, model) in enumerate(models):
        # Out-of-fold predictions on training set
        oof_preds = cross_val_predict(
            model, X_train, y_train, cv=inner_cv,
            method="predict_proba", n_jobs=1
        )[:, 1]
        meta_train[:, i] = oof_preds

        # Refit on full training set for val predictions
        if name == "xgb":
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        meta_val[:, i] = model.predict_proba(X_val)[:, 1]

    # Level 2: Meta-learner (logistic regression on stacked predictions)
    scaler = StandardScaler()
    meta_train_s = scaler.fit_transform(meta_train)
    meta_val_s = scaler.transform(meta_val)

    meta_model = LogisticRegression(
        random_state=RANDOM_SEED, C=1.0, max_iter=1000
    )
    meta_model.fit(meta_train_s, y_train)
    y_pred_proba = meta_model.predict_proba(meta_val_s)[:, 1]

    return y_pred_proba


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Model: Stacking (CB+LGB+XGB -> LR)")
    print(f"[train] Features: {X.shape[1]}, Samples: {X.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    results = evaluate_cv(X, y, fold_indices, train_and_predict)

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print_results(results, elapsed, peak_mb)
