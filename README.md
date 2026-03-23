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
| CATBoost (tuned) | 0.700 | PMC 12085305 (2025) |
| XGBoost | 0.667 | Liu et al., JMAI |
| Logistic Regression | 0.642 | Liu et al., JMAI |
| Random Forest | 0.630 | Liu et al., JMAI |
| SVM | ~0.640 | Scitepress (2023) |
| LACE Index (clinical baseline) | ~0.660 | Various |

**The practical ceiling on this dataset appears to be AUROC ~0.70 with standard tabular ML. Scores above 0.75 would be noteworthy.**

## Our Autoresearch Results

See `results.tsv` for the full experiment log. Summary of best results:

| Experiment | AUROC | Model | Description |
|---|---|---|---|
| *baseline* | *TBD* | XGBoost | Initial baseline run |

*(Updated automatically by the autoresearch loop)*

## How It Works

Following the [autoresearch](https://github.com/karpathy/autoresearch) pattern:

1. **`prepare.py`** (READ-ONLY): Downloads data, preprocesses features, defines evaluation (5-fold CV AUROC).
2. **`train.py`** (MODIFIABLE): Model architecture, hyperparameters, training loop. This is the only file the autoresearch agent modifies.
3. **`run_autoresearch.py`**: The autonomous experiment loop. Modifies `train.py`, runs experiments, logs results, keeps improvements, reverts failures.

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

# 3. Start autoresearch loop (runs autonomously for 5 hours)
# This is designed to be run by an AI agent (Claude Code)
# See run_autoresearch.py for the experiment loop protocol
```

## Viewing Results

Open `autoresearch_results.ipynb` in Google Colab or Jupyter to see:
- Experiment timeline with AUROC progression
- Best model configurations
- Comparison vs. published baselines

## References

- Strack, B. et al. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates." *BioMed Research International*, 781670.
- Liu et al. "Comparison of ML models for diabetes readmission." *JMAI*.
- Karpathy, A. (2025). "autoresearch" — https://github.com/karpathy/autoresearch
