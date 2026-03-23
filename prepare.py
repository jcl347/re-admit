"""
prepare.py — Data preparation, preprocessing, and evaluation for readmission prediction.

THIS FILE IS READ-ONLY. Do not modify.

Dataset: UCI Diabetes 130-US Hospitals for Years 1999-2008
Source: https://archive.ics.uci.edu/dataset/296
Paper: Strack et al. (2014) "Impact of HbA1c Measurement on Hospital Readmission Rates"

Task: Binary classification — predict 30-day hospital readmission
Target: 1 = readmitted within 30 days, 0 = not readmitted within 30 days
"""

import os
import hashlib
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

# ─── Constants (fixed, do not change) ─────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5
TIME_BUDGET_SECONDS = 300  # 5 minutes per experiment
CACHE_DIR = Path.home() / ".cache" / "re-admit"
DATA_FILE = CACHE_DIR / "diabetic_data.csv"
PROCESSED_FILE = CACHE_DIR / "processed_data.npz"
METADATA_FILE = CACHE_DIR / "metadata.json"

# Columns to drop (identifiers, redundant, or too sparse)
DROP_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "weight",           # 97% missing
    "payer_code",       # 40% missing, not clinically useful
    "medical_specialty", # 49% missing
    "citoglipton",      # near-zero variance
    "examide",          # near-zero variance
]

# Medication columns (binary: changed/not)
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# ─── Data Download ────────────────────────────────────────────────────────────

def download_data():
    """Download the UCI Diabetes 130-US Hospitals dataset."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print(f"[prepare] Data already cached at {DATA_FILE}")
        return

    print("[prepare] Downloading UCI Diabetes 130-US Hospitals dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=296)
        df = dataset.data.original
        df.to_csv(DATA_FILE, index=False)
        print(f"[prepare] Saved {len(df)} records to {DATA_FILE}")
    except Exception as e:
        print(f"[prepare] ucimlrepo failed ({e}), trying direct download...")
        import urllib.request
        import zipfile
        import io
        url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
        response = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in z.namelist():
            if "diabetic_data.csv" in name:
                with z.open(name) as f:
                    df = pd.read_csv(f)
                    df.to_csv(DATA_FILE, index=False)
                    print(f"[prepare] Saved {len(df)} records to {DATA_FILE}")
                    break


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_data(force=False):
    """
    Preprocess the raw dataset into ML-ready features and labels.

    Returns:
        X: np.ndarray of shape (n_samples, n_features) — float64
        y: np.ndarray of shape (n_samples,) — int (0 or 1)
        feature_names: list of str
        fold_indices: list of (train_idx, val_idx) tuples for 5-fold CV
    """
    if PROCESSED_FILE.exists() and METADATA_FILE.exists() and not force:
        print("[prepare] Loading cached processed data...")
        data = np.load(PROCESSED_FILE)
        meta = json.loads(METADATA_FILE.read_text())
        fold_indices = [(np.array(f["train"]), np.array(f["val"])) for f in meta["folds"]]
        return data["X"], data["y"], meta["feature_names"], fold_indices

    download_data()
    print("[prepare] Preprocessing data...")
    df = pd.read_csv(DATA_FILE)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # ── Target: binary 30-day readmission ──
    # Original: '<30', '>30', 'NO'
    df["readmitted_30d"] = (df["readmission_full"] if "readmission_full" in df.columns
                            else df["readmitted"]).map({"<30": 1, ">30": 0, "NO": 0})
    df = df.dropna(subset=["readmitted_30d"])

    # ── Drop columns ──
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    if "readmitted" in df.columns:
        df = df.drop(columns=["readmitted"])

    y = df["readmitted_30d"].astype(int).values
    df = df.drop(columns=["readmitted_30d"])

    # ── ICD-9 diagnosis codes: extract first 3 chars as category ──
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str[:3]

    # ── Encode categoricals ──
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # ── Fill remaining NaN with median ──
    df = df.fillna(df.median())

    # ── Convert to numpy ──
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)

    # ── Stratified K-Fold (fixed seed) ──
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = [(train_idx.tolist(), val_idx.tolist())
                    for train_idx, val_idx in skf.split(X, y)]

    # ── Cache ──
    np.savez_compressed(PROCESSED_FILE, X=X, y=y)
    meta = {
        "feature_names": feature_names,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "positive_rate": float(y.mean()),
        "folds": [{"train": f[0], "val": f[1]} for f in fold_indices],
    }
    METADATA_FILE.write_text(json.dumps(meta))
    print(f"[prepare] Processed: {X.shape[0]} samples, {X.shape[1]} features, "
          f"positive rate: {y.mean():.3f}")

    # Convert back to numpy arrays
    fold_indices = [(np.array(f[0]), np.array(f[1])) for f in fold_indices]
    return X, y, feature_names, fold_indices


# ─── Evaluation (ground truth metric — DO NOT MODIFY) ─────────────────────────

def evaluate(y_true, y_pred_proba, y_pred_class=None):
    """
    Evaluate readmission predictions. This is the ground truth evaluation.

    Args:
        y_true: true binary labels (0/1)
        y_pred_proba: predicted probabilities for class 1
        y_pred_class: predicted class labels (optional, derived from proba if None)

    Returns:
        dict with all metrics
    """
    if y_pred_class is None:
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auroc": roc_auc_score(y_true, y_pred_proba),
        "auprc": average_precision_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred_class),
        "f1": f1_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class, zero_division=0),
        "recall": recall_score(y_true, y_pred_class, zero_division=0),
    }
    return metrics


def evaluate_cv(X, y, fold_indices, train_and_predict_fn):
    """
    Run 5-fold cross-validation and return aggregated metrics.

    This is the PRIMARY evaluation function. The key metric is AUROC.

    Args:
        X: feature matrix
        y: labels
        fold_indices: list of (train_idx, val_idx) tuples
        train_and_predict_fn: callable(X_train, y_train, X_val) -> y_pred_proba

    Returns:
        dict with mean and std of each metric across folds
    """
    all_metrics = []
    all_y_true = []
    all_y_proba = []

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_pred_proba = train_and_predict_fn(X_train, y_train, X_val)
        fold_metrics = evaluate(y_val, y_pred_proba)
        all_metrics.append(fold_metrics)
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_pred_proba.tolist())

        print(f"  Fold {fold_i+1}/{len(fold_indices)}: "
              f"AUROC={fold_metrics['auroc']:.4f} "
              f"F1={fold_metrics['f1']:.4f} "
              f"Acc={fold_metrics['accuracy']:.4f}")

    # Aggregate
    result = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)

    # Also compute pooled AUROC (all predictions together)
    result["auroc_pooled"] = roc_auc_score(all_y_true, all_y_proba)

    return result


def print_results(results, training_seconds, peak_memory_mb=0, num_params=0):
    """Print results in a standardized format that can be parsed by the autoresearch loop."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    # Key metrics in parseable format (one per line, like Karpathy's format)
    print(f"auroc: {results['auroc_mean']:.6f}")
    print(f"auroc_std: {results['auroc_std']:.6f}")
    print(f"auroc_pooled: {results['auroc_pooled']:.6f}")
    print(f"auprc: {results['auprc_mean']:.6f}")
    print(f"f1: {results['f1_mean']:.6f}")
    print(f"accuracy: {results['accuracy_mean']:.6f}")
    print(f"precision: {results['precision_mean']:.6f}")
    print(f"recall: {results['recall_mean']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_memory_mb: {peak_memory_mb:.1f}")
    print(f"num_params: {num_params}")
    print("=" * 70)


# ─── Main: prepare data ──────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y, feature_names, fold_indices = preprocess_data(force=True)
    print(f"\nDataset ready:")
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(feature_names[:5])}...)")
    print(f"  Positive rate (30-day readmission): {y.mean():.3f}")
    print(f"  Folds: {len(fold_indices)}")
    print(f"\nCached at: {CACHE_DIR}")
    print("You can now run: python train.py")
