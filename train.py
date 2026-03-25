"""
train.py — The ONLY file you modify during autoresearch experiments.

Experiment 61: Ensemble of CatBoost + sklearn GBM.
CatBoost handles categoricals natively; GBM uses numeric features differently.
Blend their predictions for better generalization.
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


_cached = {}

def get_extra_features():
    """Load raw CSV and compute extra features for GBM (numpy arrays)."""
    if 'extra' in _cached:
        return _cached['extra']

    raw_path = Path.home() / '.cache/re-admit/diabetic_data.csv'
    raw = pd.read_csv(raw_path, usecols=[
        'diag_1', 'diag_2', 'diag_3', 'discharge_disposition_id',
        'medical_specialty'
    ])

    g1 = raw['diag_1'].apply(icd9_to_group).values.astype(np.float64)
    g2 = raw['diag_2'].apply(icd9_to_group).values.astype(np.float64)
    g3 = raw['diag_3'].apply(icd9_to_group).values.astype(np.float64)

    dead_codes = {11, 13, 14, 19, 20, 21}
    is_dead = raw['discharge_disposition_id'].isin(dead_codes).astype(np.float64).values
    is_diab_primary = (g1 == 0).astype(np.float64)
    n_diab = ((g1 == 0).astype(int) + (g2 == 0).astype(int) + (g3 == 0).astype(int)).astype(np.float64)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    med_spec = le.fit_transform(raw['medical_specialty'].fillna('?').astype(str)).astype(np.float64)

    _cached['extra'] = np.column_stack([g1, g2, g3, is_dead, is_diab_primary, n_diab, med_spec])
    _cached['raw'] = raw
    return _cached['extra']


def build_catboost_df(X, feature_names):
    """Build DataFrame for CatBoost with proper categorical types."""
    df = pd.DataFrame(X, columns=feature_names)

    raw_path = Path.home() / '.cache/re-admit/diabetic_data.csv'
    raw = pd.read_csv(raw_path, usecols=[
        'diag_1', 'diag_2', 'diag_3', 'discharge_disposition_id',
        'medical_specialty'
    ])

    df['diag_group_1'] = raw['diag_1'].apply(icd9_to_group).astype(int).astype(str)
    df['diag_group_2'] = raw['diag_2'].apply(icd9_to_group).astype(int).astype(str)
    df['diag_group_3'] = raw['diag_3'].apply(icd9_to_group).astype(int).astype(str)

    dead_codes = {11, 13, 14, 19, 20, 21}
    df['is_dead'] = raw['discharge_disposition_id'].isin(dead_codes).astype(int).astype(str)
    df['is_diab_primary'] = (df['diag_group_1'] == '0').astype(int).astype(str)
    df['n_diab_diag'] = (
        (df['diag_group_1'] == '0').astype(int) +
        (df['diag_group_2'] == '0').astype(int) +
        (df['diag_group_3'] == '0').astype(int)
    ).astype(str)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['medical_specialty'] = le.fit_transform(raw['medical_specialty'].fillna('?').astype(str))
    df['medical_specialty'] = df['medical_specialty'].astype(int).astype(str)

    # Interaction features
    df['inpatient_x_meds'] = df['number_inpatient'] * df['num_medications']
    df['inpatient_x_time'] = df['number_inpatient'] * df['time_in_hospital']
    df['meds_x_time'] = df['num_medications'] * df['time_in_hospital']
    df['emergency_x_inpatient'] = df['number_emergency'] * df['number_inpatient']
    df['num_total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone']
    df['n_active_meds'] = sum((df[col] != 0).astype(int) for col in med_cols if col in df.columns)

    cat_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id',
                'admission_source_id', 'diag_1', 'diag_2', 'diag_3',
                'max_glu_serum', 'A1Cresult',
                'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone', 'change', 'diabetesMed']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(int).astype(str)

    added_cat = ['diag_group_1', 'diag_group_2', 'diag_group_3',
                 'is_dead', 'is_diab_primary', 'n_diab_diag', 'medical_specialty']
    all_cat = [c for c in cat_cols if c in df.columns] + added_cat

    return df, all_cat


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Building features...")
    extra = get_extra_features()
    X_gbm = np.hstack([X, extra])  # For GBM: numeric augmented
    # Add interaction features for GBM too
    inpatient = X[:, 12]  # number_inpatient
    meds = X[:, 9]        # num_medications
    time_hosp = X[:, 6]   # time_in_hospital
    emergency = X[:, 11]  # number_emergency
    outpatient = X[:, 10] # number_outpatient
    X_gbm = np.hstack([X_gbm,
        (inpatient * meds).reshape(-1, 1),
        (inpatient * time_hosp).reshape(-1, 1),
        (meds * time_hosp).reshape(-1, 1),
        (emergency * inpatient).reshape(-1, 1),
        (outpatient + emergency + inpatient).reshape(-1, 1),
    ])

    df_cb, cat_features = build_catboost_df(X, feature_names)

    print(f"[train] Model: ENSEMBLE (CatBoost + GBM)")
    print(f"[train] CatBoost features: {df_cb.shape[1]} ({len(cat_features)} cat)")
    print(f"[train] GBM features: {X_gbm.shape[1]}")

    tracemalloc.start()
    start_time = time.time()

    all_metrics = []
    all_y_true = []
    all_y_proba = []

    CATBOOST_WEIGHT = 0.6  # CatBoost gets more weight (it's stronger)

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        from catboost import CatBoostClassifier, Pool
        from sklearn.ensemble import GradientBoostingClassifier

        y_train, y_val = y[train_idx], y[val_idx]

        # --- CatBoost ---
        df_train_cb = df_cb.iloc[train_idx]
        df_val_cb = df_cb.iloc[val_idx]
        train_pool = Pool(df_train_cb, label=y_train, cat_features=cat_features)
        val_pool = Pool(df_val_cb, cat_features=cat_features)

        cb_model = CatBoostClassifier(
            iterations=3000,
            depth=6,
            learning_rate=0.03,
            rsm=0.8,
            l2_leaf_reg=1,
            min_data_in_leaf=20,
            random_seed=MODEL_SEED,
            verbose=0,
            eval_metric='AUC',
            task_type='CPU',
        )
        cb_model.fit(train_pool)
        cb_proba = cb_model.predict_proba(val_pool)[:, 1]

        # --- GBM ---
        X_train_gbm = X_gbm[train_idx]
        X_val_gbm = X_gbm[val_idx]

        gbm_model = GradientBoostingClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.8,
            random_state=MODEL_SEED,
        )
        gbm_model.fit(X_train_gbm, y_train)
        gbm_proba = gbm_model.predict_proba(X_val_gbm)[:, 1]

        # --- Blend ---
        y_pred_proba = CATBOOST_WEIGHT * cb_proba + (1 - CATBOOST_WEIGHT) * gbm_proba

        fold_metrics = evaluate(y_val, y_pred_proba)
        all_metrics.append(fold_metrics)
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_pred_proba.tolist())

        # Also log individual model performance
        cb_auroc = roc_auc_score(y_val, cb_proba)
        gbm_auroc = roc_auc_score(y_val, gbm_proba)
        print(f"  Fold {fold_i+1}/{len(fold_indices)}: "
              f"AUROC={fold_metrics['auroc']:.4f} "
              f"(CB={cb_auroc:.4f} GBM={gbm_auroc:.4f}) "
              f"F1={fold_metrics['f1']:.4f}")

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
