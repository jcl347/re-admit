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

## Best Published Scores on This Dataset (Verified)

| Model | AUROC | Source | Confidence |
|---|---|---|---|
| CatBoost | 0.70 | Gandra (2024), IJHS | LOW — low-tier journal |
| XGBoost | 0.6812 | NHSJS (2023) | MEDIUM — student journal |
| XGBoost | 0.667 | Emi-Johnson & Nkrumah (2025), Cureus | HIGH — PMC indexed |
| XGBoost | 0.64 | Liu et al. (2024), JMAI | HIGH — used SMOTE+GWO |
| Logistic Regression | 0.642 | Emi-Johnson & Nkrumah (2025), Cureus | HIGH |
| Random Forest | 0.630 | Emi-Johnson & Nkrumah (2025), Cureus | HIGH |

**Notes:**
- Strack et al. (2014) created the dataset but reported NO AUROC (association study only).
- Papers claiming AUROC > 0.72 (e.g., Temple Univ. LSTM at 0.79) use DIFFERENT datasets, not UCI 296.
- The only CatBoost 0.70 claim comes from Gandra (2024) in a low-impact journal with sparse methodology.
- The practical ceiling on this dataset appears to be AUROC ~0.69-0.70 with standard tabular ML.

### Key Insight from Published Work

Strack et al. (2014) grouped ICD-9 diagnosis codes (diag_1, diag_2, diag_3) into 9 disease categories, reducing ~700+ unique codes to: **Circulatory** (390–459, 785), **Respiratory** (460–519, 786), **Digestive** (520–579, 787), **Diabetes** (250.xx), **Injury** (800–999), **Musculoskeletal** (710–739), **Genitourinary** (580–629, 788), **Neoplasms** (140–239), and **Other**. This grouping is standard preprocessing for this dataset.

## Our Autoresearch Results (65 experiments)

See `results.tsv` for the full experiment log. Summary of best kept results:

| AUROC | Model | Description |
|---|---|---|
| **0.6879** | CatBoost + GBM ensemble | CatBoost native cats + interactions + medical_specialty, blended 0.6/0.4 with GBM (best) |
| 0.6878 | CatBoost (5000 iter) | 5000 iters, lr=0.015, native cats + interactions + medical_specialty |
| 0.6877 | CatBoost (4000 iter) | 4000 iters, lr=0.02, l2=1, native cats + interactions + medical_specialty |
| 0.6877 | CatBoost + rich features | + ratio features, age interactions, discharge groups (no improvement) |
| 0.6874 | CatBoost + interactions | 3000 iters + interaction features + medical_specialty |
| 0.6861 | CatBoost (tuned) | 4000 iters, lr=0.02, native categorical handling |
| 0.6859 | CatBoost (native cats) | First CatBoost with proper DataFrame + categorical types |
| 0.6818 | sklearn GBM + ICD-9 | n_est=2000, lr=0.01, depth=5 + Strack disease groups + is_dead flag |
| 0.6814 | sklearn GradientBoosting | n_est=2000, lr=0.01, depth=5 |
| 0.6793 | sklearn GradientBoosting | n_est=500, lr=0.05 |

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
- Gandra, A. (2024). "Predicting Hospital Readmission for Diabetic Patients Using CatBoost." *International Journal of Health Sciences*, 8(3), 289-297. Reports CatBoost AUROC 0.70 (low-tier journal, treat with caution).
- Karpathy, A. (2025). "autoresearch" — https://github.com/karpathy/autoresearch
