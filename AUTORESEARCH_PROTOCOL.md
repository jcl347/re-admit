# autoresearch — readmission prediction

This is an experiment to have the LLM do its own research on hospital readmission prediction.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   * `README.md` — repository context, published baselines, dataset info.
   * `prepare.py` — fixed constants, data prep, preprocessing, 5-fold CV evaluation. **Do not modify.**
   * `train.py` — the file you modify. Model architecture, hyperparameters, training loop, ensembling.
4. **Verify data exists**: Check that `~/.cache/re-admit/` contains `processed_data.npz` and `metadata.json`. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Dataset

**UCI Diabetes 130-US Hospitals** (Strack et al., 2014)
- 101,766 encounters, 42 features after preprocessing
- Binary target: 30-day readmission (11.2% positive rate)
- 5-fold stratified cross-validation with fixed seed
- Primary metric: **AUROC** (area under ROC curve)

## Experimentation

Each experiment runs on CPU. The training script runs a fixed 5-fold CV evaluation. You launch it simply as: `python train.py`.

**What you CAN do:**

* Modify `train.py` — this is the **only file you edit**. Everything is fair game: model type, ensembling, hyperparameters, feature engineering, preprocessing, training loop, class imbalance handling, random seed, etc.
* Override the model random seed via `MODEL_SEED` in `train.py`. The CV folds are fixed by `RANDOM_SEED` in `prepare.py`, but you can set `MODEL_SEED` to any value in `train.py` to experiment with different model initialization seeds. This is useful for seed averaging (training the same model with multiple seeds and averaging predictions).

**What you CANNOT do:**

* Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, preprocessing, and training constants (CV folds, sequence of features, random seed, etc).
* Install new packages or add dependencies beyond what's in `pyproject.toml` (numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, joblib, scipy, imbalanced-learn).
* Modify the evaluation harness. The `evaluate_cv` function in `prepare.py` is the ground truth metric.

**The goal is to reach AUROC 0.70.** Everything is fair game: change the model type, the ensembling strategy, the hyperparameters, the feature engineering, the class imbalance handling. The only constraint is that the code runs without crashing.

**No speed penalty:** Do NOT discard experiments just because they are slower. Training time is irrelevant — only AUROC matters. Keep any experiment that improves AUROC, regardless of how long it takes. The 1.5 hour timeout still applies to prevent infinite runs, but within that budget, slower methods are perfectly fine.

**The first run:** Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
======================================================================
EXPERIMENT RESULTS
======================================================================
auroc: 0.681373
auroc_std: 0.005175
auroc_pooled: 0.681200
auprc: 0.228000
f1: 0.036343
accuracy: 0.888578
precision: 0.560000
recall: 0.018000
training_seconds: 1662.3
peak_memory_mb: 75.7
num_params: 0
======================================================================
```

You can extract the key metric from the log file:

```
grep "^auroc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	auroc	f1	accuracy	memory_mb	training_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. auroc achieved (e.g. 0.681373) — use 0.000000 for crashes
3. f1 score — use 0.000000 for crashes
4. accuracy — use 0.000000 for crashes
5. peak memory in MB (e.g. 75.7) — use 0.0 for crashes
6. training_seconds — use 0.0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	auroc	f1	accuracy	memory_mb	training_seconds	status	description
274302f	0.650677	0.262961	0.732897	70.7	10.3	keep	baseline XGBoost
bf74848	0.671775	0.278363	0.692068	71.1	42.5	keep	CatBoost balanced
68b615c	0.681373	0.036343	0.888578	75.7	1662.3	keep	GBM n_est=2000 lr=0.01
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar23`).

```
LOOP FOREVER:
1. Look at the git state: the current branch/commit we're on
2. Tune train.py with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: python train.py > run.log 2>&1 (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: grep "^auroc:\|^f1:\|^accuracy:" run.log
6. If the grep output is empty, the run crashed. Run tail -n 50 run.log to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If AUROC improved (higher), you "advance" the branch, keeping the git commit
9. If AUROC is equal or worse, you git reset back to where you started
```

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout:** Each experiment should take a few minutes total. If a run exceeds 1.5 hours, kill it and treat it as a failure (discard and revert).

**Crashes:** If a run crashes (OOM, or a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP:** Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from the computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, try different ensembling strategies. The loop runs until the human interrupts you, period.

## Experiment ideas (in rough priority order)

### Model types
- XGBoost, LightGBM, CatBoost (gradient boosting family)
- sklearn GradientBoosting, HistGradientBoosting
- RandomForest, ExtraTrees
- LogisticRegression (with feature engineering)
- MLP neural network (sklearn MLPClassifier)

### Ensembling strategies
- Weighted averaging of predictions from multiple models
- Stacking with a meta-learner (LR on out-of-fold predictions)
- Voting (hard/soft)
- Blending with inner-CV optimized weights

### Hyperparameter tuning
- learning_rate, max_depth, n_estimators, regularization
- subsample, colsample_bytree, min_child_weight
- class imbalance: scale_pos_weight, sample_weight, auto_class_weights

### Feature engineering
- Interaction features (e.g., num_medications × time_in_hospital)
- Feature selection (remove low-importance features)
- Binning continuous features
- Target encoding for high-cardinality categoricals

### Seed strategies
- Override `MODEL_SEED` in `train.py` to use a different seed than `RANDOM_SEED`
- Seed averaging: train the same model N times with different seeds, average predictions
- Use different seeds per model in ensembles for diversity

### Class imbalance handling
- SMOTE / ADASYN oversampling
- Class weights (balanced)
- Threshold optimization for F1
- Cost-sensitive learning

### Published baselines to beat
| Model | AUROC | Source |
|---|---|---|
| CATBoost (tuned) | 0.700 | PMC 12085305 (2025) |
| XGBoost | 0.667 | Liu et al., JMAI |
| LACE Index | 0.660 | Clinical baseline |
