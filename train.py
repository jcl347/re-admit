"""
train.py — The ONLY file you modify during autoresearch experiments.

Current model: Weighted average ensemble (CatBoost + LightGBM + XGBoost)
with optimized weights from inner CV
"""

import time
import tracemalloc
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count

    # Base models
    cb = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, border_count=128,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED, verbose=0, thread_count=-1,
    )
    lgb = LGBMClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, num_leaves=63,
        min_child_samples=20, scale_pos_weight=spw,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=RANDOM_SEED,
        eval_metric="auc", tree_method="hist", n_jobs=-1,
    )

    models = [cb, lgb, xgb]

    # Find optimal weights using inner CV on training set
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = [np.zeros(len(X_train)) for _ in models]

    for fold_train, fold_val in inner_cv.split(X_train, y_train):
        for i, model in enumerate(models):
            if isinstance(model, type(xgb)):
                model.fit(X_train[fold_train], y_train[fold_train], verbose=False)
            else:
                model.fit(X_train[fold_train], y_train[fold_train])
            oof_preds[i][fold_val] = model.predict_proba(X_train[fold_val])[:, 1]

    # Optimize weights to maximize AUROC
    def neg_auroc(weights):
        w = np.array(weights)
        w = w / w.sum()  # normalize
        blended = sum(w[i] * oof_preds[i] for i in range(len(models)))
        return -roc_auc_score(y_train, blended)

    result = minimize(neg_auroc, [0.45, 0.30, 0.25],
                      bounds=[(0.01, 1.0)] * 3,
                      method='Nelder-Mead')
    best_w = np.array(result.x)
    best_w = best_w / best_w.sum()
    print(f"    Optimal weights: CB={best_w[0]:.3f} LGB={best_w[1]:.3f} XGB={best_w[2]:.3f}")

    # Refit on full training set
    for i, model in enumerate(models):
        if isinstance(model, type(xgb)):
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)

    # Predict with optimal weights
    val_preds = [model.predict_proba(X_val)[:, 1] for model in models]
    y_pred_proba = sum(best_w[i] * val_preds[i] for i in range(len(models)))

    return y_pred_proba


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Model: Optimized Weighted Ensemble (CB+LGB+XGB)")
    print(f"[train] Features: {X.shape[1]}, Samples: {X.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    results = evaluate_cv(X, y, fold_indices, train_and_predict)

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print_results(results, elapsed, peak_mb)
