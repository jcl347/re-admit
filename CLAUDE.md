# CLAUDE.md — Project Context for Autoresearch

## Goal
Reach AUROC 0.70 on the UCI Diabetes 130-US Hospitals 30-day readmission prediction task.
Current best: **AUROC 0.6814** (sklearn GradientBoosting, n_est=2000, lr=0.01, depth=5).

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

## What We've Tried (42 experiments)
- sklearn GBM is consistently best on this preprocessing (AUROC 0.6814)
- XGBoost, LightGBM, CatBoost all worse with current label-encoded features
- HistGradientBoosting systematically worse (histogram binning loses info)
- Ensembles provide marginal gains (+0.0002) at high cost
- Raw target encoding and interaction features HURT (added noise)
- SMOTE hurts, class weighting hurts for AUROC optimization
- Early stopping with validation_fraction hurts (reduces training data)

## Next Steps (Priority Order)
1. **ICD-9 disease grouping** — map diag_1/2/3 to 9 Strack categories in train.py
2. **Dead patient handling** — flag discharge_disposition for died/hospice patients
3. **Combine grouping + GBM** — should unlock the 0.68 → 0.70 gap
4. **Try CatBoost with grouped features** — CatBoost handles categoricals natively
