"""
train.py — The ONLY file you modify during autoresearch experiments.

Experiment 100: Remove numeric interaction features (CatBoost can learn them).
Keep only the categorical interactions that actually help.
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

    # Diagnosis combination pattern (e.g., "1_0_8" = Circulatory+Diabetes+Other)
    df['diag_pattern'] = df['diag_group_1'] + '_' + df['diag_group_2'] + '_' + df['diag_group_3']

    # Discharge disposition grouping (clinically meaningful groups)
    # 1=home, 2=short-term hospital, 3=SNF, 5=other facility, 6=home health
    # 7=AMA, 11/13/14/19/20/21=dead/hospice (already captured in is_dead)
    discharge_map = {}
    for c in range(30):
        if c in {1}: discharge_map[c] = '0'       # Home
        elif c in {6}: discharge_map[c] = '1'      # Home with home health
        elif c in {2, 10, 16, 27}: discharge_map[c] = '2'  # Another hospital/facility
        elif c in {3, 4, 5}: discharge_map[c] = '3'  # SNF/ICF/other facility
        elif c in {7}: discharge_map[c] = '4'       # AMA (left against advice - high risk!)
        elif c in {11, 13, 14, 19, 20, 21}: discharge_map[c] = '5'  # Dead/hospice
        else: discharge_map[c] = '6'                # Other
    df['discharge_group'] = raw['discharge_disposition_id'].map(discharge_map).fillna('6').astype(str)

    # Number of unique diagnosis groups (diversity of conditions)
    df['n_unique_diag_groups'] = (
        df[['diag_group_1', 'diag_group_2', 'diag_group_3']].nunique(axis=1)
    )

    # Primary diagnosis × admission type (clinical pathway pattern)
    df['diag1_x_admit'] = df['diag_group_1'] + '_' + df['admission_type_id'].astype(int).astype(str)
    # Primary diagnosis × discharge group (outcome pathway)
    df['diag1_x_discharge'] = df['diag_group_1'] + '_' + df['discharge_group']

    del raw
    gc.collect()

    # Keep only total visits (useful for XGBoost which can't split on categoricals)
    df['num_total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

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
                 'is_dead', 'is_diab_primary', 'n_diab_diag', 'medical_specialty',
                 'diag_pattern', 'discharge_group', 'diag1_x_admit', 'diag1_x_discharge']
    all_cat = [c for c in cat_cols if c in df.columns] + added_cat

    return df, all_cat


if __name__ == "__main__":
    print(f"[train] Loading data...")
    X, y, feature_names, fold_indices = preprocess_data()

    print(f"[train] Building DataFrame with features...")
    df, cat_features = build_dataframe(X, feature_names)

    del X
    gc.collect()

    # Build numeric version for XGBoost
    df_numeric = df.copy()
    for col in cat_features:
        if col in df_numeric.columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)

    print(f"[train] Model: CatBoost + XGBoost ensemble (0.7/0.3 blend)")
    print(f"[train] Features: {df.shape[1]} ({len(cat_features)} categorical)")
    print(f"[train] Samples: {df.shape[0]}")

    tracemalloc.start()
    start_time = time.time()

    all_metrics = []
    all_y_true = []
    all_y_proba = []

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        from catboost import CatBoostClassifier, Pool
        from xgboost import XGBClassifier

        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost with native categoricals
        train_pool = Pool(df_train, label=y_train, cat_features=cat_features)
        val_pool = Pool(df_val, cat_features=cat_features)

        cb_model = CatBoostClassifier(
            iterations=4000,
            depth=6,
            learning_rate=0.02,
            rsm=0.8,
            l2_leaf_reg=5,
            min_data_in_leaf=20,
            random_seed=MODEL_SEED,
            verbose=0,
            eval_metric='AUC',
            task_type='CPU',
            bagging_temperature=0.5,
            random_strength=0.5,
            langevin=True,
            diffusion_temperature=10000,
        )
        cb_model.fit(train_pool)
        cb_pred = cb_model.predict_proba(val_pool)[:, 1]

        del cb_model
        gc.collect()

        # Second CatBoost with different config for diversity
        cb_model2 = CatBoostClassifier(
            iterations=3000,
            depth=6,
            learning_rate=0.03,
            rsm=0.7,
            l2_leaf_reg=3,
            min_data_in_leaf=25,
            random_seed=123,
            verbose=0,
            eval_metric='AUC',
            task_type='CPU',
            bagging_temperature=0.3,
        )
        cb_model2.fit(train_pool)
        cb_pred2 = cb_model2.predict_proba(val_pool)[:, 1]

        del cb_model2, train_pool, val_pool
        gc.collect()

        # XGBoost on numeric features
        X_train_num = df_numeric.iloc[train_idx].values.astype(np.float64)
        X_val_num = df_numeric.iloc[val_idx].values.astype(np.float64)

        xgb_model = XGBClassifier(
            n_estimators=3000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=MODEL_SEED,
            eval_metric='auc',
            verbosity=0,
        )
        xgb_model.fit(X_train_num, y_train)
        xgb_pred = xgb_model.predict_proba(X_val_num)[:, 1]

        del xgb_model
        gc.collect()

        # 3-model blend: CB1(langevin) 0.50 + CB2(no-langevin) 0.30 + XGB 0.20
        y_pred_proba = 0.50 * cb_pred + 0.30 * cb_pred2 + 0.20 * xgb_pred

        fold_metrics = evaluate(y_val, y_pred_proba)
        all_metrics.append(fold_metrics)
        all_y_true.extend(y_val.tolist())
        all_y_proba.extend(y_pred_proba.tolist())

        cb_auroc = roc_auc_score(y_val, cb_pred)
        cb2_auroc = roc_auc_score(y_val, cb_pred2)
        xgb_auroc = roc_auc_score(y_val, xgb_pred)
        print(f"  Fold {fold_i+1}/{len(fold_indices)}: "
              f"AUROC={fold_metrics['auroc']:.4f} "
              f"(CB1={cb_auroc:.4f} CB2={cb2_auroc:.4f} XGB={xgb_auroc:.4f})")

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
