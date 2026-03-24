"""
train.py — The ONLY file you modify during autoresearch experiments.

Current model: GBM with ICD-9 disease groups ADDED as extra features (not replacing).
Also adds is_dead flag from discharge_disposition_id.
"""

import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from prepare import (
    preprocess_data,
    evaluate_cv,
    print_results,
    TIME_BUDGET_SECONDS,
    RANDOM_SEED,
    evaluate,
)
from sklearn.metrics import roc_auc_score

MODEL_SEED = RANDOM_SEED

# Feature indices
IDX_DIAG1 = 13
IDX_DIAG2 = 14
IDX_DIAG3 = 15
IDX_DISCHARGE = 4


def icd9_to_group(code_str):
    """Map an ICD-9 code string to one of 9 disease categories."""
    if pd.isna(code_str) or code_str == '?' or code_str == 'nan':
        return 8
    code = str(code_str).strip()
    if code.startswith('E') or code.startswith('V'):
        return 8
    try:
        num = float(code)
    except ValueError:
        return 8
    if 250 <= num < 251:
        return 0  # Diabetes
    elif (390 <= num <= 459) or (785 <= num < 786):
        return 1  # Circulatory
    elif (460 <= num <= 519) or (786 <= num < 787):
        return 2  # Respiratory
    elif (520 <= num <= 579) or (787 <= num < 788):
        return 3  # Digestive
    elif 800 <= num <= 999:
        return 4  # Injury
    elif 710 <= num <= 739:
        return 5  # Musculoskeletal
    elif (580 <= num <= 629) or (788 <= num < 789):
        return 6  # Genitourinary
    elif 140 <= num <= 239:
        return 7  # Neoplasms
    else:
        return 8  # Other


# Pre-compute extra features once
_extra_features = None

def get_extra_features():
    """Load raw CSV and compute ICD-9 disease groups + is_dead flag."""
    global _extra_features
    if _extra_features is not None:
        return _extra_features

    raw_path = Path.home() / '.cache/re-admit/diabetic_data.csv'
    df = pd.read_csv(raw_path, usecols=['diag_1', 'diag_2', 'diag_3', 'discharge_disposition_id'])

    g1 = df['diag_1'].apply(icd9_to_group).values.astype(np.float64)
    g2 = df['diag_2'].apply(icd9_to_group).values.astype(np.float64)
    g3 = df['diag_3'].apply(icd9_to_group).values.astype(np.float64)

    # Flag: patient died or went to hospice (can't be readmitted)
    # discharge_disposition_id: 11=Expired, 13=Hospice/home, 14=Hospice/medical,
    # 19=Expired at home, 20=Expired in medical facility, 21=Expired place unknown
    dead_codes = {11, 13, 14, 19, 20, 21}
    is_dead = df['discharge_disposition_id'].isin(dead_codes).astype(np.float64).values

    # Primary diagnosis is diabetes flag
    is_diab_primary = (g1 == 0).astype(np.float64)

    # Number of diabetes-related diagnoses (0-3)
    n_diab = ((g1 == 0).astype(int) + (g2 == 0).astype(int) + (g3 == 0).astype(int)).astype(np.float64)

    _extra_features = np.column_stack([g1, g2, g3, is_dead, is_diab_primary, n_diab])
    return _extra_features


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Loading ICD-9 disease groups + extra features...")
    extra = get_extra_features()

    # Augment X with extra features
    X_aug = np.hstack([X, extra])

    print(f"[train] Model: GBM + ICD-9 groups + is_dead + diabetes flags")
    print(f"[train] Features: {X_aug.shape[1]} ({X.shape[1]} base + {extra.shape[1]} new)")
    print(f"[train] Samples: {X_aug.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    # Custom CV loop using augmented features
    all_metrics = []
    all_y_true = []
    all_y_proba = []

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        from sklearn.ensemble import GradientBoostingClassifier

        X_train = X_aug[train_idx]
        X_val = X_aug[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientBoostingClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=MODEL_SEED,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        fold_metrics = evaluate(y_val, y_pred_proba)
        all_metrics.append(fold_metrics)
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_pred_proba.tolist())

        print(f"  Fold {fold_i+1}/{len(fold_indices)}: "
              f"AUROC={fold_metrics['auroc']:.4f} "
              f"F1={fold_metrics['f1']:.4f} "
              f"Acc={fold_metrics['accuracy']:.4f}")

    result = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)
    result["auroc_pooled"] = roc_auc_score(all_y_true, all_y_proba)

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print_results(result, elapsed, peak_mb)
