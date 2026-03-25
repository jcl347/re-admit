"""
train.py — The ONLY file you modify during autoresearch experiments.

Experiment 67: Stacking meta-learner with CatBoost + GBM base models.
Generate OOF predictions from both models, then train LR on stacked features.
"""

import gc
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

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


def build_dataframe(X, feature_names):
    """Build a pandas DataFrame with proper dtypes for CatBoost."""
    df = pd.DataFrame(X, columns=feature_names)

    raw_path = Path.home() / '.cache/re-admit/diabetic_data.csv'
    raw = pd.read_csv(raw_path, usecols=[
        'diag_1', 'diag_2', 'diag_3', 'discharge_disposition_id',
        'medical_specialty'
    ])

    # ICD-9 disease group features
    df['diag_group_1'] = raw['diag_1'].apply(icd9_to_group).astype(int).astype(str)
    df['diag_group_2'] = raw['diag_2'].apply(icd9_to_group).astype(int).astype(str)
    df['diag_group_3'] = raw['diag_3'].apply(icd9_to_group).astype(int).astype(str)

    # Dead/hospice flag
    dead_codes = {11, 13, 14, 19, 20, 21}
    df['is_dead'] = raw['discharge_disposition_id'].isin(dead_codes).astype(int).astype(str)

    # Diabetes flags
    df['is_diab_primary'] = (df['diag_group_1'] == '0').astype(int).astype(str)
    df['n_diab_diag'] = (
        (df['diag_group_1'] == '0').astype(int) +
        (df['diag_group_2'] == '0').astype(int) +
        (df['diag_group_3'] == '0').astype(int)
    ).astype(str)

    # Medical specialty
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['medical_specialty'] = le.fit_transform(raw['medical_specialty'].fillna('?').astype(str))
    df['medical_specialty'] = df['medical_specialty'].astype(int).astype(str)

    del raw
    gc.collect()

    # Interaction features (numeric)
    df['inpatient_x_meds'] = df['number_inpatient'] * df['num_medications']
    df['inpatient_x_time'] = df['number_inpatient'] * df['time_in_hospital']
    df['meds_x_time'] = df['num_medications'] * df['time_in_hospital']
    df['emergency_x_inpatient'] = df['number_emergency'] * df['number_inpatient']
    df['num_total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

    # Active medications count
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone']
    df['n_active_meds'] = sum((df[col] != 0).astype(int) for col in med_cols if col in df.columns)

    # Categorical columns for CatBoost
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

    print(f"[train] Building DataFrame with features...")
    df, cat_features = build_dataframe(X, feature_names)

    del X
    gc.collect()

    print(f"[train] Model: Stacking (CatBoost + GBM) -> LR meta-learner")
    print(f"[train] Features: {df.shape[1]} ({len(cat_features)} categorical)")
    print(f"[train] Samples: {df.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    all_metrics = []
    all_y_true = []
    all_y_proba = []

    # Get numeric-only version for GBM (can't use string categoricals)
    numeric_cols = [c for c in df.columns if c not in cat_features]
    # For GBM, convert cat cols back to numeric
    df_numeric = df.copy()
    for col in cat_features:
        if col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        from catboost import CatBoostClassifier, Pool

        # --- CatBoost base model ---
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(df_train, label=y_train, cat_features=cat_features)
        val_pool = Pool(df_val, cat_features=cat_features)

        cb_model = CatBoostClassifier(
            iterations=4000,
            depth=6,
            learning_rate=0.02,
            rsm=0.8,
            l2_leaf_reg=1,
            min_data_in_leaf=20,
            random_seed=MODEL_SEED,
            verbose=0,
            eval_metric='AUC',
            task_type='CPU',
        )
        cb_model.fit(train_pool)
        cb_val_pred = cb_model.predict_proba(val_pool)[:, 1]

        # CatBoost OOF for inner stacking: use inner 3-fold CV on training data
        from sklearn.model_selection import StratifiedKFold
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_SEED)
        cb_oof_train = np.zeros(len(train_idx))

        for inner_train, inner_val in inner_skf.split(df_train, y_train):
            inner_train_pool = Pool(df_train.iloc[inner_train], label=y_train[inner_train], cat_features=cat_features)
            inner_val_pool = Pool(df_train.iloc[inner_val], cat_features=cat_features)
            inner_cb = CatBoostClassifier(
                iterations=2000,
                depth=6,
                learning_rate=0.03,
                rsm=0.8,
                l2_leaf_reg=1,
                min_data_in_leaf=20,
                random_seed=MODEL_SEED,
                verbose=0,
                task_type='CPU',
            )
            inner_cb.fit(inner_train_pool)
            cb_oof_train[inner_val] = inner_cb.predict_proba(inner_val_pool)[:, 1]
            del inner_cb, inner_train_pool, inner_val_pool
            gc.collect()

        del cb_model, train_pool, val_pool
        gc.collect()

        # --- GBM base model ---
        X_train_num = df_numeric.iloc[train_idx].values.astype(np.float64)
        X_val_num = df_numeric.iloc[val_idx].values.astype(np.float64)

        gbm_model = GradientBoostingClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            max_features=0.8,
            random_state=MODEL_SEED,
        )
        gbm_model.fit(X_train_num, y_train)
        gbm_val_pred = gbm_model.predict_proba(X_val_num)[:, 1]

        # GBM OOF for inner stacking
        gbm_oof_train = np.zeros(len(train_idx))
        for inner_train, inner_val in inner_skf.split(X_train_num, y_train):
            inner_gbm = GradientBoostingClassifier(
                n_estimators=1000,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.8,
                max_features=0.8,
                random_state=MODEL_SEED,
            )
            inner_gbm.fit(X_train_num[inner_train], y_train[inner_train])
            gbm_oof_train[inner_val] = inner_gbm.predict_proba(X_train_num[inner_val])[:, 1]
            del inner_gbm
            gc.collect()

        del gbm_model
        gc.collect()

        # --- Meta-learner (LR) ---
        meta_train = np.column_stack([cb_oof_train, gbm_oof_train])
        meta_val = np.column_stack([cb_val_pred, gbm_val_pred])

        lr_meta = LogisticRegression(C=1.0, random_state=MODEL_SEED, max_iter=1000)
        lr_meta.fit(meta_train, y_train)
        y_pred_proba = lr_meta.predict_proba(meta_val)[:, 1]

        fold_metrics = evaluate(y_val, y_pred_proba)
        all_metrics.append(fold_metrics)
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_pred_proba.tolist())

        print(f"  Fold {fold_i+1}/{len(fold_indices)}: "
              f"AUROC={fold_metrics['auroc']:.4f} "
              f"F1={fold_metrics['f1']:.4f} "
              f"Acc={fold_metrics['accuracy']:.4f}"
              f" (CB={roc_auc_score(y_val, cb_val_pred):.4f}"
              f" GBM={roc_auc_score(y_val, gbm_val_pred):.4f})")

        del df_train, df_val, meta_train, meta_val
        gc.collect()

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
