# CLAUDE.md — Project Context for Autoresearch

## Goal
Reach AUROC 0.70 on the UCI Diabetes 130-US Hospitals 30-day readmission prediction task.
Current best: **AUROC 0.6879** (CatBoost + GBM ensemble with native categorical handling + ICD-9 groups + interactions + medical_specialty).

## Key Paper: Strack et al. (2014)
**Citation:** Strack, B., DeShazo, J.P., Gennings, C., et al. "Impact of HbA1c Measurement on Hospital Readmission Rates." *BioMed Research International*, 2014, 781670.

### Their ICD-9 Diagnosis Grouping (Table 2)
The original paper groups the ~700+ unique ICD-9 3-digit codes in diag_1, diag_2, diag_3 into **9 disease categories**:

| Category | ICD-9 Code Range |
|---|---|
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Diabetes | 250.xx |
| Injury | 800–999 |
| Musculoskeletal | 710–739 |
| Genitourinary | 580–629, 788 |
| Neoplasms | 140–239 |
| Other | Everything else |

**Note:** Codes starting with "E" (external causes) and "V" (supplementary) → "Other".
**Note:** Codes 785, 786, 787, 788 are symptom codes assigned to circulatory, respiratory, digestive, genitourinary respectively.

### Why This Matters
Our prepare.py truncates ICD-9 codes to 3 chars and label-encodes them as ordinal integers. This gives ~700+ unique integer values that GBM must split on individually. Grouping into 9 categories:
1. Massively reduces noise from rare ICD-9 codes
2. Captures medically meaningful disease relationships
3. Is the standard preprocessing used by papers achieving AUROC ≥ 0.70

### Implementation in train.py
Since prepare.py is READ-ONLY, we must do ICD-9 grouping inside `train_and_predict()`:
1. Load the raw CSV to get original ICD-9 string codes
2. Map each code to one of 9 disease categories (integer 0-8)
3. Replace the label-encoded diag_1/diag_2/diag_3 columns (indices 13, 14, 15) with grouped versions
4. Optionally add both grouped AND original features

## Key Paper: Emi-Johnson & Nkrumah (2025) — CatBoost AUROC 0.70
**Citation:** Emi-Johnson, O.G. & Nkrumah, K.J. "Predicting 30-Day Hospital Readmission in Patients With Diabetes Using ML on EHR Data." *Cureus*, 17(4), e82437. PMC12085305.

### Their Approach
- Used class weighting (NOT SMOTE) for imbalance handling
- One-hot encoding for categoricals, standardization for continuous
- XGBoost achieved 0.667 AUROC; they cited CatBoost 0.70 from related work
- Top SHAP features: **number_inpatient**, **num_medications**, **time_in_hospital**, **insulin use**

## Preprocessing Insights (What Matters Most)
1. **ICD-9 grouping** — biggest potential gain, reduces 700+ codes to 9 categories
2. **Discharge disposition** — codes 11, 13, 14, 19, 20, 21 = patient died/hospice → can't be readmitted
3. **number_inpatient** — consistently #1 predictor across all studies
4. **num_medications × time_in_hospital** — key interaction
5. **Medication change features** — "change" column + individual med changes

## What We've Tried (65 experiments)

### Phase 1: sklearn GBM baseline (experiments 1-54)
- sklearn GBM is consistently best with label-encoded features (AUROC 0.6814)
- XGBoost, LightGBM, CatBoost all worse with label-encoded features
- HistGradientBoosting systematically worse (histogram binning loses info)
- Ensembles provide marginal gains (+0.0002) at high cost
- Raw target encoding and interaction features HURT (added noise)
- SMOTE hurts, class weighting hurts for AUROC optimization
- Early stopping with validation_fraction hurts (reduces training data)
- ICD-9 grouping as EXTRA features + is_dead flag → 0.6818

### Phase 2: CatBoost breakthrough (experiments 55-65)
- **CatBoost with native categorical handling via pandas DataFrame** → 0.6859 (+0.0041!)
  - Key insight: previous CatBoost experiments used numpy float arrays, bypassing CatBoost's ordered target encoding
  - Converting label-encoded columns to string type enables CatBoost's native categorical handling
  - diag_1/2/3 as categoricals ARE valuable despite 700+ cardinality (CatBoost handles this well)
- Adding interaction features (inpatient×meds, inpatient×time, etc.) + medical_specialty → 0.6874
- CatBoost 4000 iters, lr=0.02, l2=1, min_leaf=20 → 0.6877
- CatBoost + GBM ensemble (0.6/0.4 blend) → **0.6879** (current best)
- LightGBM with native categoricals → 0.641 (terrible — LightGBM handles high-cardinality categoricals poorly)
- Lossguide grow policy → 0.658 (massive overfitting)
- Richer features (ratios, age interactions, discharge groups) → 0.6877 (no gain from extra features)
- 5000 iterations got killed by OOM twice; gc.collect() fixed it but didn't improve over 4000 iters

### Key Learnings
1. **CatBoost's native categorical handling is the single biggest improvement** — +0.006 over any other approach
2. **Keep diag_1/2/3 as categoricals** — removing them drops AUROC by 0.002
3. **Interaction features help modestly** (+0.001): number_inpatient × num_medications, × time_in_hospital
4. **medical_specialty from raw data** provides useful signal despite 49% missing
5. **Ensemble of CatBoost + GBM** gives tiny gain (+0.0002) over CatBoost alone
6. **Ratio/age features don't help** — CatBoost can learn these interactions itself
7. **LightGBM is not competitive** on this dataset with these features

## Next Steps (Priority Order)
1. **Stacking meta-learner** — use CatBoost + GBM + XGBoost OOF predictions as features for LR
2. **Feature selection** — remove noisy features that may be hurting CatBoost
3. **CatBoost with symmetric tree + ordered boosting tuning** — try different boosting types
4. **Patient-level deduplication** — some patients appear multiple times; use patient_nbr for grouping
5. **Target: AUROC 0.70** — need +0.012 from current 0.6879
