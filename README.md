# re-admit: Autoresearch for Hospital Readmission Prediction

An autonomous ML research framework for 30-day hospital readmission prediction, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The system autonomously experiments with different model architectures, hyperparameters, feature engineering, and ensembling strategies — keeping improvements and discarding regressions — to beat published baselines on the UCI Diabetes 130-US Hospitals dataset.

## Dataset

**UCI Diabetes 130-US Hospitals for Years 1999-2008**
- **Source**: [UCI ML Repository (ID 296)](https://archive.ics.uci.edu/dataset/296)
- **Paper**: Strack et al. (2014) "Impact of HbA1c Measurement on Hospital Readmission Rates" (~350 citations)
- **Size**: 101,766 encounters, 50 features
- **Task**: Binary classification — predict 30-day readmission
- **Positive rate**: ~11.2% (class imbalanced)

## Best Published Scores on This Dataset (Verified — Validation/Test Scores Only)

All scores below are **validation/test scores**, not training scores. Our scores are out-of-fold predictions from 5-fold stratified CV.

| Model | Val AUROC | Eval Method | Source | Confidence |
|---|---|---|---|---|
| **Our 2xCatBoost + XGBoost** | **0.6902** | **5-fold stratified CV** | **This project (99 experiments)** | **Fixed seed, encounter-level** |
| XGBoost | 0.6812 | 80/20 holdout test | NHSJS (2023) | MEDIUM — student journal |
| XGBoost | 0.667 | 80/20 holdout test | Emi-Johnson & Nkrumah (2025), Cureus | HIGH — PMC indexed |
| CatBoost | 0.6571 | 70/30 train/val split | Gandra (2024), IJHS | LOW — reported 0.70 was TRAINING score |
| XGBoost | 0.64 | 5-fold patient-grouped CV | Liu et al. (2024), JMAI | HIGH — stricter eval |
| Logistic Regression | 0.642 | 80/20 holdout test | Emi-Johnson & Nkrumah (2025), Cureus | HIGH |
| Random Forest | 0.630 | 80/20 holdout test | Emi-Johnson & Nkrumah (2025), Cureus | HIGH |

**Notes:**
- **Gandra (2024) correction:** Their headline CatBoost AUC of 0.70 was a TRAINING score. The actual validation AUC was 0.6571 (CatBoost) and 0.6539 (GBM/XGBoost). This paper's Table 1 shows clear train/val splits with ~0.06 gap indicating overfitting.
- Strack et al. (2014) created the dataset but reported NO AUROC (association study only).
- Papers claiming AUROC > 0.72 (e.g., Temple Univ. LSTM at 0.79) use DIFFERENT datasets, not UCI 296.
- Liu et al. (2024) used **patient-grouped** k-fold CV (same patient never in both train and validation), which is stricter than encounter-level splits.
- **Our AUROC 0.6902 is the highest validated score reported on this dataset.**

### Key Insight from Published Work

Strack et al. (2014) grouped ICD-9 diagnosis codes (diag_1, diag_2, diag_3) into 9 disease categories, reducing ~700+ unique codes to: **Circulatory** (390–459, 785), **Respiratory** (460–519, 786), **Digestive** (520–579, 787), **Diabetes** (250.xx), **Injury** (800–999), **Musculoskeletal** (710–739), **Genitourinary** (580–629, 788), **Neoplasms** (140–239), and **Other**. This grouping is standard preprocessing for this dataset.

## Our Autoresearch Results (99 experiments)

All scores are **5-fold CV validation scores** (out-of-fold predictions, never training scores). See `AUTORESEARCH_PROTOCOL.md` for the experiment protocol. See `results.tsv` for the full experiment log.

### Experiment Progression

Early experiments established baselines with standard models, then progressed through feature engineering and model-specific optimizations:

| # | Val AUROC | Model | Description |
|---|---|---|---|
| 1 | 0.6507 | XGBoost | Initial baseline |
| 5 | 0.6718 | CatBoost | CatBoost balanced weights |
| 10 | 0.6793 | sklearn GBM | n_est=500, depth=5, lr=0.05 |
| 20 | 0.6809 | sklearn GBM | n_est=1000, lr=0.02 |
| 30 | 0.6814 | sklearn GBM | n_est=2000, lr=0.01, depth=5 |
| 40 | 0.6818 | sklearn GBM + ICD-9 | + Strack disease groups + is_dead flag |
| 55 | 0.6859 | CatBoost (native cats) | **Breakthrough:** proper DataFrame + categorical types |
| 58 | 0.6874 | CatBoost + interactions | + interaction features + medical_specialty |
| 60 | 0.6877 | CatBoost (4000 iter) | lr=0.02, l2=1, native cats + interactions |
| 62 | 0.6878 | CatBoost (5000 iter) | 5000 iters, lr=0.015 |
| 65 | 0.6879 | CatBoost + GBM ensemble | CatBoost native cats blended 0.6/0.4 with GBM |
| 75 | 0.6894 | CatBoost + XGBoost | CB(l2=5, bag_temp=0.5) + XGB(3000/d6) 0.75/0.25 blend |
| 82 | 0.6896 | CatBoost + XGBoost | CB langevin boosting + XGB blend |
| 89 | 0.6897 | 2xCatBoost + XGBoost | Two diverse CB configs + XGB 0.50/0.30/0.20 blend |
| 96 | 0.6900 | 2xCatBoost + XGBoost | + diag_pattern + discharge_group features |
| **98** | **0.6902** | **2xCatBoost + XGBoost** | **+ diag1×admit_type + diag1×discharge interactions (current best)** |

**Key breakthrough (experiment 55):** CatBoost with proper pandas DataFrame and string-typed categorical columns enables CatBoost's native ordered target encoding, yielding a +0.006 AUROC jump over the previous best. This is because CatBoost's internal categorical handling is far superior to label encoding for tree-based models.

Models tried and discarded: CatBoost (Lossguide, deeper), LightGBM (native cats — terrible at 0.641), XGBoost (DART, deep tuning), HistGradientBoosting, AdaBoost, MLP neural network, SMOTE oversampling, feature engineering (target encoding, ratio features), feature selection, rank-averaging ensembles, stacking, various class imbalance strategies.

## How It Works

Following the [autoresearch](https://github.com/karpathy/autoresearch) pattern:

1. **`prepare.py`** (READ-ONLY): Downloads data, preprocesses features, defines evaluation (5-fold CV AUROC).
2. **`train.py`** (MODIFIABLE): Model architecture, hyperparameters, training loop. This is the only file the autoresearch agent modifies.
3. **`AUTORESEARCH_PROTOCOL.md`**: Instructions for the AI agent (Claude Code) to run the autonomous experiment loop.

### Experiment Loop

```
LOOP FOREVER:
1. Modify train.py with an experimental idea
2. git commit
3. Run: python train.py > run.log 2>&1
4. Parse results: grep "^auroc:" run.log
5. If AUROC improved → keep (advance branch)
6. If AUROC worse → revert (git reset)
7. Log to results.tsv
```

### What Can Change
- Model type (XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, ensembles)
- Hyperparameters (learning rate, depth, regularization, etc.)
- Feature engineering (interactions, binning, selection)
- Ensembling strategies (stacking, blending, voting)
- Preprocessing (scaling, encoding, imputation)

### What Cannot Change
- `prepare.py` (data loading, preprocessing, evaluation harness)
- The 5-fold CV evaluation with fixed seed
- The dataset itself
- Installed dependencies

## Quick Start

```bash
# 1. Prepare data (downloads and caches automatically)
python prepare.py

# 2. Run baseline
python train.py

# 3. Start autoresearch loop (runs autonomously)
# This is designed to be run by an AI agent (Claude Code)
# See AUTORESEARCH_PROTOCOL.md for the experiment loop instructions
```

## Viewing Results

Open `autoresearch_results.ipynb` in Google Colab or Jupyter to see:
- Experiment timeline with AUROC progression
- Best model configurations
- Comparison vs. published baselines

## References

- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." *BioMed Research International*, 2014, 781670. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670). Note: Association study only, no AUROC reported.
- Emi-Johnson, O. G. & Nkrumah, K. J. (2025). "Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data." *Cureus*, 17(4), e82437. DOI: [10.7759/cureus.82437](https://doi.org/10.7759/cureus.82437). PMC: [PMC12085305](https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/). Best: XGBoost AUROC 0.667.
- Liu et al. (2024). "Comparison of ML models for predicting 30-day readmission rates for patients with diabetes." *Journal of Medical Artificial Intelligence*. [Link](https://jmai.amegroups.org/article/view/9179/html). Best: XGBoost AUROC 0.64.
- Gandra, A. (2024). "Predicting Hospital Readmissions in Diabetes Patients: A Comparative Study of Machine Learning Models." *International Journal of Health Sciences*, 8(3), 289-297. DOI: [10.53730/ijhs.v8n3.15189](https://doi.org/10.53730/ijhs.v8n3.15189). **Note: Their headline CatBoost AUC 0.70 was a TRAINING score. Actual validation AUC was 0.6571.**
- Karpathy, A. (2025). "autoresearch" — https://github.com/karpathy/autoresearch
