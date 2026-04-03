"""
train.py — The ONLY file you modify during autoresearch experiments.

Experiment 118: Single powerful CatBoost — CB1 only with 6000 iters, lr=0.01.
Drop all other models to save memory for more iterations on the best model.
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
        return 0
    elif (390 <= num <= 459) or (785 <= num < 786):
        return 1
    elif (460 <= num <= 519) or (786 <= num < 787):
        return 2
    elif (520 <= num <= 579) or (787 <= num < 788):
        return 3
    elif 800 <= num <= 999:
        return 4
    elif 710 <= num <= 739:
        return 5
    elif (580 <= num <= 629) or (788 <= num < 789):
        return 6
    elif 140 <= num <= 239:
        return 7
    else:
        return 8


def build_dataframe(X, feature_names):
    """Build a pandas DataFrame with proper dtypes for CatBoost."""
    df = pd.DataFrame(X, columns=feature_names)

    raw_path = Path.home() / '.cache/re-admit/diabetic_data.csv'
    raw = pd.read_csv(raw_path, usecols=[
        'diag_1', 'diag_2', 'diag_3', 'discharge_disposition_id',
        'medical_specialty', 'payer_code', 'weight'
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

    df['diag_pattern'] = df['diag_group_1'] + '_' + df['diag_group_2'] + '_' + df['diag_group_3']

    discharge_map = {}
    for c in range(30):
        if c in {1}: discharge_map[c] = '0'
        elif c in {6}: discharge_map[c] = '1'
        elif c in {2, 10, 16, 27}: discharge_map[c] = '2'
        elif c in {3, 4, 5}: discharge_map[c] = '3'
        elif c in {7}: discharge_map[c] = '4'
        elif c in {11, 13, 14, 19, 20, 21}: discharge_map[c] = '5'
        else: discharge_map[c] = '6'
    df['discharge_group'] = raw['discharge_disposition_id'].map(discharge_map).fillna('6').astype(str)

    df['n_unique_diag_groups'] = (
        df[['diag_group_1', 'diag_group_2', 'diag_group_3']].nunique(axis=1)
    )

    df['diag1_x_admit'] = df['diag_group_1'] + '_' + df['admission_type_id'].astype(int).astype(str)
    df['diag1_x_discharge'] = df['diag_group_1'] + '_' + df['discharge_group']

    # Payer code (insurance type, 40% missing)
    from sklearn.preprocessing import LabelEncoder as LE2
    le2 = LE2()
    df['payer_code'] = le2.fit_transform(raw['payer_code'].fillna('?').astype(str))
    df['payer_code'] = df['payer_code'].astype(int).astype(str)

    # Weight recorded flag (97% missing — missingness is informative)
    df['weight_recorded'] = (raw['weight'] != '?').astype(int).astype(str)

    # Payer × primary diagnosis group interaction
    df['payer_x_diag1'] = df['payer_code'] + '_' + df['diag_group_1']

    del raw
    gc.collect()

    df['num_total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

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
                 'is_dead', 'is_diab_primary', 'n_diab_diag', 'medical_specialty',
                 'diag_pattern', 'discharge_group', 'diag1_x_admit', 'diag1_x_discharge',
                 'payer_code', 'weight_recorded', 'payer_x_diag1']
    all_cat = [c for c in cat_cols if c in df.columns] + added_cat

    return df, all_cat


def greedy_ensemble(preds_list, y_true, n_rounds=50):
    """AutoGluon-style greedy ensemble selection.

    Each round: try adding each model's predictions to the current ensemble,
    keep the one that improves AUROC the most. Weights are determined by
    selection frequency.
    """
    n_models = len(preds_list)
    selected = []
    best_auroc = 0

    for _ in range(n_rounds):
        best_i = 0
        best_round_auroc = 0
        for i in range(n_models):
            candidate = selected + [i]
            # Average predictions of selected models
            blend = np.mean([preds_list[j] for j in candidate], axis=0)
            auroc = roc_auc_score(y_true, blend)
            if auroc > best_round_auroc:
                best_round_auroc = auroc
                best_i = i
        selected.append(best_i)
        best_auroc = best_round_auroc

    # Convert selection frequency to weights
    weights = np.zeros(n_models)
    for i in selected:
        weights[i] += 1
    weights /= weights.sum()

    return weights, best_auroc


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Building DataFrame with features...")
    df, cat_features = build_dataframe(X, feature_names)

    del X
    gc.collect()

    print(f"[train] Model: Single CatBoost (6000 iters, lr=0.01)")
    print(f"[train] Features: {df.shape[1]} ({len(cat_features)} categorical)")
    print(f"[train] Samples: {df.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    all_metrics = []
    all_y_true = []
    all_y_proba = []

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        from catboost import CatBoostClassifier, Pool

        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        preds = {}  # model_name -> val predictions

        # --- Model 1: CatBoost (langevin, strongest) ---
        train_pool = Pool(df_train, label=y_train, cat_features=cat_features)
        val_pool = Pool(df_val, cat_features=cat_features)

        cb_model = CatBoostClassifier(
            iterations=6000, depth=6, learning_rate=0.01, rsm=0.8,
            l2_leaf_reg=5, min_data_in_leaf=20, random_seed=MODEL_SEED,
            verbose=0, eval_metric='AUC', task_type='CPU',
            bagging_temperature=0.5, random_strength=0.5,
            langevin=True, diffusion_temperature=10000,
        )
        cb_model.fit(train_pool)
        y_pred_proba = cb_model.predict_proba(val_pool)[:, 1]
        del cb_model, train_pool, val_pool
        gc.collect()

        fold_metrics = evaluate(y_val, y_pred_proba)
        all_metrics.append(fold_metrics)
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_pred_proba.tolist())

        print(f"  Fold {fold_i+1}/{len(fold_indices)}: "
              f"AUROC={fold_metrics['auroc']:.4f}"
        )

        del df_train, df_val
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
