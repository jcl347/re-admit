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

## Best Published Scores on This Dataset

| Model | AUROC | Source |
|---|---|---|
| CATBoost (tuned) | 0.700 | Emi-Johnson & Nkrumah (2025), Cureus |
| XGBoost | 0.667 | Emi-Johnson & Nkrumah (2025), Cureus |
| XGBoost | 0.667 | Liu et al., JMAI |
| Logistic Regression | 0.642 | Liu et al., JMAI |
| Random Forest | 0.630 | Liu et al., JMAI |
| SVM | ~0.640 | Scitepress (2023) |
| LACE Index (clinical baseline) | ~0.660 | Various |

**The practical ceiling on this dataset appears to be AUROC ~0.70 with standard tabular ML. Scores above 0.75 would be noteworthy.**

### Key Insight from Published Work

Strack et al. (2014) grouped ICD-9 diagnosis codes (diag_1, diag_2, diag_3) into 9 disease categories, reducing ~700+ unique codes to: **Circulatory** (390–459, 785), **Respiratory** (460–519, 786), **Digestive** (520–579, 787), **Diabetes** (250.xx), **Injury** (800–999), **Musculoskeletal** (710–739), **Genitourinary** (580–629, 788), **Neoplasms** (140–239), and **Other**. This grouping is standard preprocessing for this dataset and is used by most published studies achieving AUROC ≥ 0.70.

## Our Autoresearch Results (32 experiments)

See `results.tsv` for the full experiment log. Summary of best kept results:

| AUROC | Model | Description |
|---|---|---|
| **0.6814** | sklearn GradientBoosting | n_est=2000, lr=0.01, depth=5 (best) |
| 0.6809 | sklearn GradientBoosting | n_est=1000, lr=0.02 |
| 0.6793 | sklearn GradientBoosting | n_est=500, lr=0.05 |
| 0.6776 | 5-model ensemble | CB+LGB+XGB+GBM+ET optimized weights |
| 0.6720 | 3-model ensemble | CB+LGB+XGB optimized weights |
| 0.6719 | Stacking ensemble | CB+LGB+XGB with LR meta-learner |
| 0.6718 | CatBoost | iter=1000, lr=0.05, auto balanced |
| 0.6531 | LightGBM | n_est=1000, lr=0.05, depth=7 |
| 0.6507 | XGBoost (baseline) | n_est=500, lr=0.1, depth=6 |

Models tried and discarded: CatBoost (deeper/varied), LightGBM (tuned), XGBoost (DART, deep tuning), HistGradientBoosting, AdaBoost, MLP neural network, SMOTE oversampling, feature engineering (interactions), feature selection, target encoding, rank-averaging ensembles, stacking, various class imbalance strategies.

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

- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." *BioMed Research International*, 2014, 781670. DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670)
- Emi-Johnson, O. G. & Nkrumah, K. J. (2025). "Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data." *Cureus*, 17(4), e82437. DOI: [10.7759/cureus.82437](https://doi.org/10.7759/cureus.82437). PMC: [PMC12085305](https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/)
- Liu et al. "Comparison of ML models for predicting 30-day readmission rates for patients with diabetes." *Journal of Medical Artificial Intelligence*. [Link](https://jmai.amegroups.org/article/view/9179/html)
- Karpathy, A. (2025). "autoresearch" — https://github.com/karpathy/autoresearch
