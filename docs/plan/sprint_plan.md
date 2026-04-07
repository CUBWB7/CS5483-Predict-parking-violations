# Sprint Plan: Final 2-3 Day Push

## Context

**Current best**: v7 Platform 0.5636, Leaderboard top 0.5992, gap 0.036.
**Steps 8-14 all failed** — TE correction, regularization, DART, NN all failed to beat v7.
**Key findings**: M1-5 TE actually hurts (-0.013), gap 0.079 not caused by TE shift or overfitting.

**Strategy**: Abandon "shrink the gap" approach (proven ineffective). Pivot to "new models/methods to improve quality".

### Complete Results Summary

| Version | Change | OOF | Platform | Gap | Conclusion |
|---------|--------|-----|----------|-----|------------|
| v3 | Tier2 + Optuna | 0.6408 | 0.5620 | 0.079 | baseline |
| v6 | Spearman ES + feature pruning | 0.6377 | 0.5618 | 0.076 | ES not viable |
| v7 | +log1p(tc) weighting | **0.6429** | **0.5636** | 0.079 | **current best** |
| v8a | v7 + M1-5 TE (test only) | 0.6429 | 0.5507 | 0.092 | M1-5 TE harmful |
| v9 | Strong regularization Optuna | 0.6326 | pending | — | OOF -0.010 |
| v9a | v9 + M1-5 TE | 0.6326 | 0.5477 | 0.085 | double penalty |
| v10 | DART Boosting | 0.6406 | not submitted | — | LGB OOF -0.019 |
| v11 | +NN (weight=0) | 0.6429 | not submitted | — | NN OOF 0.42 |

---

## Day 1 (4/08) — Submit 2 + Start New Experiments

### Submission 1: `ensemble_v9.csv` (already generated)

Pure regularization (no M1-5 TE), isolates the regularization effect.

### Experiment A: M1-5 Weight Optimization + Fine-Grained Search (Local, ~5min)

`notebooks/06_sprint.ipynb` — Section A

**Rationale**: v7 ensemble weights (LGB=0.25, XGB=0.45, CB=0.30) were optimized on full 12-month OOF (step=0.05), but test set only contains M1-5. Re-searching with finer granularity (step=0.01) on M1-5 OOF subset may find better weights for the test distribution.

**Implementation**:
1. Load v7 LGB/XGB OOF + test predictions
2. Filter to M1-5 rows (`month_of_year in [1,2,3,4,5]`)
3. Fine-grained grid search (step=0.01) on M1-5 OOF Spearman
4. Generate `submissions/ensemble_v12.csv`

### Submission 2: `ensemble_v12.csv` (from Experiment A)

### Experiment B: Stacking Meta-Learner (Local/GPU)

`notebooks/06_sprint.ipynb` — Section B

**Rationale**: Simple weighted average assumes linear additivity, but meta-learners (Ridge, LightGBM) can capture non-linear interactions between model predictions. Classic Kaggle stacking technique.

**Implementation**:
1. Build meta-features: LGB v7 OOF, XGB v7 OOF, (optional) v9 LGB/XGB OOF, NN v1 OOF
2. Layer 1: 5-fold OOF meta-features (already available)
3. Layer 2 meta-learner options (all with 5-fold CV):
   - **Ridge Regression** (linear, fast baseline)
   - **LightGBM meta** (learns non-linear combinations, 100-500 trees)
   - **Logistic Regression** (more natural for 0-1 targets)
4. Evaluate meta OOF Spearman vs simple weighted average

**Key**: meta-features can include predictions from multiple versions (v3, v7, v9) — even if individual versions are weaker, their combination may provide complementary signal.

---

## Day 2 (4/09) — New Models + New Methods

### Experiment C: Rank-Based Target Training (GPU, ~1h)

`notebooks/06_sprint.ipynb` — Section C

**Rationale**: Spearman only cares about rank ordering. Current models train MSE(pred, y), but y has a bimodal distribution (many 0s and 1s), so MSE spends too much effort on the extremes. Training on `rank(y)/N` (uniform distribution) makes MSE directly optimize ranking accuracy, better aligned with Spearman.

**Implementation**:
1. `y_rank = rankdata(y, method='average') / len(y)`
2. Train LGB + XGB with v7 params (target = y_rank)
3. Evaluate OOF using original y's Spearman
4. Can stack with original v7 predictions

### Experiment D: Adversarial Validation + Sample Re-weighting (GPU, ~30min)

`notebooks/06_sprint.ipynb` — Section D

**Rationale**: Train a classifier to distinguish train vs test data. Training samples with high classification probability = "more test-like samples". Giving these higher weight can narrow the train/test distribution gap. Directly addresses the gap at the data level.

**Implementation**:
1. Merge train + test, label=0/1
2. Train LGB classifier (5-fold CV, evaluate AUC)
3. AUC > 0.5 means train/test are distinguishable → use predicted probabilities as sample_weight
4. Retrain v7 models with adjusted weights

**Decision threshold**: AUC ≈ 0.5 → train/test indistinguishable, method is useless; AUC > 0.6 → significant distribution difference, worth trying

### Experiment E: TabNet (GPU, ~2h)

`notebooks/06_sprint.ipynb` — Section E

**Rationale**: TabNet is a Google Research deep learning model specifically designed for tabular data. It uses attention mechanisms to select important features at each step (similar to GBDT feature selection), and has shown strong performance in Kaggle tabular competitions. Much more suitable than the vanilla ResNet from Step 14.

**Implementation**:
1. `pip install pytorch-tabnet`
2. Use TabNetRegressor, 5-fold CV
3. Key params: `n_d=64, n_a=64, n_steps=5, gamma=1.5`
4. sample_weight = log1p(total_count)
5. Evaluate OOF Spearman and correlation with GBDT

**Expected**: TabNet OOF may reach 0.55-0.62 (significantly above ResNet's 0.42), with lower correlation to GBDT.

### Day 2 submissions: Pick best 2 from Experiments B-E

---

## Day 3 (4/10) — Combine Best + Transition to Report

### Experiment F: Final Ensemble Combination

`notebooks/06_sprint.ipynb` — Section F

**Rationale**: Combine all models/methods with positive signal:
- Layer 1: LGB v7, XGB v7, Rank-target LGB/XGB, TabNet (if effective)
- Layer 2: Stacking meta-learner or Optuna weight search
- Optional: adversarial weights

### Experiment G: Pseudo-Labeling (if time permits)

`notebooks/06_sprint.ipynb` — Section G

**Rationale**: Use "high confidence" test predictions from v7 ensemble (close to 0 or 1) as pseudo-labels, add to training set for retraining. Increases M1-5 training data volume, may improve generalization.

**Implementation**:
1. Select v7 test predictions where < 0.05 or > 0.95
2. Add these pseudo-labeled samples to training set (lower weight, e.g., 0.5)
3. Retrain LGB + XGB
4. Evaluate OOF (on original train only, excluding pseudo-labels)

### Day 3 afternoon: Transition to video production

---

## Key Files

### Files to Create
| File | Purpose |
|------|---------|
| `notebooks/06_sprint.ipynb` | Sprint experiment notebook (Sections A-G) |

### Files to Read
| File | Purpose |
|------|---------|
| `data/train_features_tier2.parquet` | Training data |
| `data/test_features_tier2.parquet` | Test data |
| `models/lgb_oof_v7.npy`, `lgb_test_v7.npy` | v7 LGB predictions |
| `models/xgb_oof_v7.npy`, `xgb_test_v7.npy` | v7 XGB predictions |
| `models/lgb_oof_v9.npy`, `xgb_oof_v9.npy` | v9 predictions (for stacking) |
| `models/nn_oof_v1.npy`, `nn_test_v1.npy` | NN v1 predictions (for stacking) |
| `scripts/step10_gpu.py` | GBDT training template |

---

## Experiment Priority

| Experiment | Method Category | Expected Benefit | Cost | Priority |
|------------|----------------|-----------------|------|----------|
| A: M1-5 Weight Optimization | Ensemble | Low-Med | Very low (local, ~5min) | P0 |
| B: Stacking Meta-Learner | ML/Ensemble | Med | Low (local/GPU 30min) | P0 |
| D: Adversarial Validation | ML/Distribution Alignment | Med | Low (GPU 30min) | P0 |
| C: Rank-Based Target | ML/Loss Optimization | Med-High | Med (GPU ~1h) | P1 |
| E: TabNet | DL/Tabular-Specific | Med | Med (GPU ~2h) | P1 |
| F: Final Ensemble | ML/Combination | Depends on above | Low | P1 |
| G: Pseudo-Labeling | ML/Semi-Supervised | Low-Med | Med (GPU ~1h) | P2 |

---

## Verification

Unified verification flow for each experiment:
1. OOF Spearman vs v7 baseline (0.6429)
2. M1-5 OOF Spearman vs v7 M1-5 (0.6515)
3. Inter-model correlation matrix (diversity check)
4. Submission standard validation (no NaN, 2028750 rows, range [0,1])
5. Record platform score after submission, update progress.md
