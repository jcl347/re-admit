# Autoresearch Protocol for Readmission Prediction

This document defines the autonomous experiment loop for the readmission prediction task. It is designed to be executed by an AI agent (Claude Code).

## Setup

1. **Branch**: Work on `autoresearch/<tag>` branch (e.g., `autoresearch/mar23`)
2. **Read files**: `README.md`, `prepare.py` (read-only), `train.py` (modifiable)
3. **Verify data**: Run `python prepare.py` to download and cache data
4. **Initialize**: Create `results.tsv` with header row
5. **Baseline**: Run `python train.py > run.log 2>&1` to establish baseline

## Experiment Loop

```
LOOP FOREVER (target: 5 hours, ~60 experiments at 5 min each):

1. Look at git state and current best AUROC
2. Modify train.py with an experimental idea
3. git commit -m "experiment: <description>"
4. Run: python train.py > run.log 2>&1
5. Parse: grep "^auroc:" run.log
6. If grep empty → crash. Run: tail -n 50 run.log for traceback
7. Log results to results.tsv
8. If AUROC improved → keep (advance branch)
9. If AUROC worse → git reset --hard HEAD~1
```

## What You CAN Modify (in train.py)

- **Model type**: XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, GradientBoosting
- **Ensembling**: Stacking, blending, voting, weighted averaging
- **Hyperparameters**: learning_rate, max_depth, n_estimators, regularization, etc.
- **Feature engineering**: Interactions, polynomial features, binning, PCA, feature selection
- **Preprocessing**: Different scaling, encoding, imputation strategies
- **Threshold tuning**: Optimize classification threshold for F1

## What You CANNOT Modify

- `prepare.py` — data loading, preprocessing, evaluation harness (5-fold CV)
- Installed packages (only use what's in pyproject.toml)
- The evaluation metric (AUROC is primary)
- The random seed or fold structure

## Key Metric

**Primary**: `auroc` (mean 5-fold cross-validation AUROC)
**Secondary**: `f1`, `accuracy`, `auprc`

## Experiment Ideas (Priority Order)

### Tier 1: Quick Wins
1. Try LightGBM (often faster + better than XGBoost)
2. Try CatBoost (handles categoricals natively)
3. Tune XGBoost hyperparameters (max_depth, learning_rate, n_estimators)
4. Adjust scale_pos_weight for class imbalance

### Tier 2: Feature Engineering
5. Create interaction features (age × num_medications, etc.)
6. Bin continuous features (age groups, medication count buckets)
7. Feature selection (remove low-importance features)
8. Add polynomial features for top predictors

### Tier 3: Ensembling
9. Voting ensemble (XGBoost + LightGBM + CatBoost)
10. Stacking with logistic regression meta-learner
11. Blending with optimized weights
12. Multi-layer stacking

### Tier 4: Advanced
13. Bayesian hyperparameter optimization
14. Target encoding for high-cardinality categoricals
15. SMOTE or other oversampling for class imbalance
16. Custom loss functions for class imbalance

### Tier 5: Architecture
17. Neural network (MLP with sklearn)
18. TabNet-style attention
19. Gradient boosting + neural network ensemble

## Output Format

The training script prints parseable results:
```
auroc: 0.670000
auroc_std: 0.005000
f1: 0.300000
accuracy: 0.880000
training_seconds: 45.2
peak_memory_mb: 512.3
```

## results.tsv Format

Tab-separated, 7 columns:
```
commit	auroc	f1	accuracy	memory_mb	training_seconds	status	description
a1b2c3d	0.670000	0.300000	0.880000	512.3	45.2	keep	baseline
```

## Simplicity Criterion

All else being equal, simpler is better:
- A 0.001 AUROC improvement with 20 lines of complexity? Probably not worth it.
- A 0.001 AUROC improvement from deleting code? Definitely keep.
- Equal AUROC but much simpler? Keep.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human. Run autonomously until manually stopped or the 5-hour budget is exhausted.
