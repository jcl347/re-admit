"""
train.py — The ONLY file you modify during autoresearch experiments.

Current model: sklearn GradientBoosting (dominant in prev ensemble) tuned solo
"""

import time
import tracemalloc
import numpy as np
from prepare import (
    preprocess_data,
    evaluate_cv,
    print_results,
    TIME_BUDGET_SECONDS,
    RANDOM_SEED,
)


def train_and_predict(X_train, y_train, X_val):
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Model: sklearn GradientBoosting (tuned)")
    print(f"[train] Features: {X.shape[1]}, Samples: {X.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    results = evaluate_cv(X, y, fold_indices, train_and_predict)

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print_results(results, elapsed, peak_mb)
