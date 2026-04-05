# Score Improvement Plan: Platform 0.5222 → Target 0.59+

## Context

Current best: OOF Spearman 0.5880, Platform 0.5222. Platform leaderboard top: 0.5992.
Two directions for improvement:
1. **Reduce OOF-Platform gap** (currently 0.066) — closing to 0.03 alone reaches ~0.558
2. **Improve OOF itself** — tuning + more models + new features

Plan: create a new notebook `05_improvement.ipynb`, without modifying verified 01-04 notebooks.

---

## Step 1: Improve Target Encoding (Reduce Gap — HIGHEST Priority)

**Expected gain: Platform +0.015~0.025 | Time: 30 min**

Current issue: 5-Fold TE uses 80% data for train encoding, 100% for test, causing distribution shift (KS stat ~0.01).

Changes:
- TE folds from 5 → **10** (train encoding uses 90% data, closer to test's 100%)
- `grid_te` smoothing from 30 → **100**, `grid_period_te` from 50 → **150**
- Unseen test grids (21, 3.6%): use **KDTree nearest-neighbor** instead of global mean fallback

Key file: `notebooks/02_feature_engineering.ipynb` (reference implementation), new code in `05_improvement.ipynb`

---

## Step 2: Increase n_estimators + Lower Learning Rate (Quick OOF Boost)

**Expected gain: OOF +0.003~0.008 | Time: extra training time only**

Current LGB/XGB both hit 3000 rounds without early stopping triggering — models are undertrained.

Changes:
- `n_estimators`: 3000 → **5000~8000**
- `learning_rate`: 0.05 → **0.03**
- `early_stopping_rounds`: 50 → **100**

---

## Step 3: Optuna Hyperparameter Tuning (Improve OOF)

**Expected gain: OOF +0.010~0.020 | Time: 2-3 hours (mostly machine time)**

Both models use untuned manual defaults.

Implementation:
- **Subsample for speed**: 1M rows from 6M for Optuna trials
- **Objective**: 3-Fold CV Spearman
- **Trials**: 50-100

LightGBM search space:
```
num_leaves: [15, 127]
max_depth: [-1, 12]  
learning_rate: [0.01, 0.1]
min_child_samples: [20, 200]
reg_lambda: [0.1, 10.0]
reg_alpha: [0.0, 5.0]
feature_fraction: [0.5, 1.0]
bagging_fraction: [0.5, 1.0]
```

XGBoost search space:
```
max_depth: [4, 10]
learning_rate: [0.01, 0.1]
min_child_weight: [10, 200]
reg_lambda: [0.1, 10.0]
reg_alpha: [0.0, 5.0]
subsample: [0.5, 1.0]
colsample_bytree: [0.5, 1.0]
```

After finding optimal params, train final models on full 6M data with 5-Fold CV.

---

## Step 4: Add CatBoost as Third Model (Improve Ensemble Diversity)

**Expected gain: Ensemble +0.005~0.015 | Time: 45 min**

LGB-XGB OOF correlation 0.9778, minimal ensemble benefit. CatBoost uses symmetric trees + ordered boosting — architecturally different.

Key points:
- CatBoost can use `grid_id` and `grid_period` as **native categorical features** without TE → naturally eliminates TE distribution shift
- Or use the same improved TE features for consistency
- Same 5-Fold CV + OOF predictions structure

```python
cb_params = {
    'iterations': 5000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'early_stopping_rounds': 100,
}
```

---

## Step 4b: CatBoost Optuna Hyperparameter Tuning (NEW — not in original plan)

**Expected gain: CB OOF 0.5728 → 0.62+, Ensemble +0.002~0.008 | Time: 5-7 hours (machine time)**

CatBoost v3 used untuned defaults (depth=6, lr=0.05, l2_leaf_reg=3.0) and scored OOF 0.5728 — far below tuned LGB (0.6322) and XGB (0.6379). It received 0 weight in the ensemble. Optuna tuning should close this gap.

### Core Challenge: Native Categorical Features are Extremely Slow

CatBoost v3 with `grid_id` (742 categories) + `grid_period` (~3114 categories) as native categoricals took 378 min (6.3h) for 5-fold × 8000 iterations. The ordered target statistics computation scales superlinearly with category count.

**Key Decision: Drop `grid_period` from native categoricals, keep only `grid_id`.**
- `grid_period` (~3114 values) causes ~60% of categorical compute overhead
- `grid_period_te` (K-Fold target encoding) is already in the 26 base features
- Estimated speedup: ~2.5x
- Full retrain also uses grid_id only for consistency with Optuna configuration

### Phase 1: Optuna Tuning (40 trials)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Subsample | 500K rows | CatBoost slower than LGB/XGB, so half of the 1M used for LGB/XGB. Covers 602/742 grid_id (81%) |
| CV | 3-Fold | Same as LGB/XGB Optuna |
| Max iterations | 2000 | CatBoost ~3-4x slower per iteration than XGB |
| Early stopping | 50 rounds | Kill bad configs early |
| Trials | 40 | TPE converges in 30-40 trials |
| Pruning | None | CatBoost 1.2.7 callback bug prevents Optuna pruning |
| Metric | Spearman (manual) | Same as LGB/XGB |

Features: 26 base FEATURES + `grid_id` = 27 features, `grid_id` as sole native categorical.

Search space:
```
depth: [4, 10]                    # symmetric tree depth
learning_rate: [0.01, 0.1] log   # same range as LGB/XGB
l2_leaf_reg: [1.0, 10.0] log     # L2 regularization
random_strength: [0.1, 10.0] log # CatBoost-specific: split score perturbation
bagging_temperature: [0.0, 1.0]  # CatBoost-specific: Bayesian bootstrap
border_count: [32, 255]          # numeric feature split candidates
min_data_in_leaf: [1, 100]       # minimum leaf samples
```

Not tuned: `one_hot_max_size` (742 categories too large), `grow_policy` (keep Symmetric for diversity), `boosting_type` (keep Ordered).

Estimated time: **~110 min (1.8h)**, worst case ~170 min (2.8h).

### Phase 2: Full Retrain

Optuna best params → full 6M data, 5-Fold CV, 10000 iterations, early_stopping=150.
Same 27 features (grid_id only as native categorical).

Estimated time: **~200-250 min (3.5-4.2h)**.

### Phase 3: Rebuild Ensemble v4

- Recompute inter-model correlations with new CB v4 OOF
- 3-model weight grid search (step 0.05)
- Generate `submissions/ensemble_v4.csv`
- Save `models/cb_oof_v4.npy`, `models/cb_test_v4.npy`

### Expected Outcome

| Scenario | CB v4 OOF | Ensemble v4 OOF | vs v3 |
|----------|-----------|-----------------|-------|
| Pessimistic | 0.60-0.61 | 0.6415-0.6425 | +0.001~0.002 |
| **Expected** | **0.62-0.63** | **0.6430-0.6460** | **+0.002~0.005** |
| Optimistic | 0.63+ | 0.6460-0.6500 | +0.005~0.009 |

### Risk Mitigation

| Risk | Fallback |
|------|----------|
| Optuna too slow (>4h) | Reduce to 300K subsample + 30 trials |
| CB v4 OOF < 0.60 | Try `grow_policy='Lossguide'`; or abandon CB, move to Step 5 |
| Params don't transfer to full data | Check transfer ratio; refine top-5 params on 1M if needed |
| Retrain too slow (>8h) | Reduce to 8000 iterations + ES=100 |

---

## Step 5: Stacking Meta-Learner (Replace Simple Weighting)

**Expected gain: +0.005~0.010 | Time: 30 min**

Use LGB + XGB + CatBoost OOF predictions as features for a Ridge regression meta-learner.
Optionally add `total_count`, `grid_te` and other top features for conditional weighting.

---

## Step 6: Rank Normalization Post-Processing (Free Improvement)

**Expected gain: +0.002~0.005 | Time: 10 min**

Spearman only cares about ranking, not absolute values. Apply rank normalization to final predictions:

```python
from scipy.stats import rankdata
test_ranks = rankdata(ensemble_test) / len(ensemble_test)
train_sorted = np.sort(y_train.values)
normalized = np.interp(test_ranks, np.linspace(0, 1, len(train_sorted)), train_sorted)
```

Monotonic transform — cannot hurt Spearman, can only help.

---

## Step 7: Tier 3 Feature Engineering (Optional)

**Expected gain: OOF +0.003~0.008 | Time: 1 hour**

Only implement highest-ROI features:
- **Grid × Day-of-Week TE**: `grid_dow = grid_id * 10 + day_of_week`, K-Fold TE with smooth=150
- **Temperature discretization**: simple 4-bin categorization

Skip: KMeans clustering (742 grids already sufficient), 6-hour weather window (requires temporal ordering, high complexity)

---

## Execution Order

| Order | Steps | Est. Time | Cumulative Expected Platform |
|-------|-------|-----------|------------------------------|
| 1 ✅ | Step 1 (TE improvement) + Step 2 (more rounds) + Step 6 (rank normalization) | 2 hours | ~0.545-0.555 |
| 2 ✅ | Step 3 (Optuna LGB+XGB) + Step 4 (CatBoost untuned) | 2-3 hours + 6.3h | Ensemble OOF 0.6408 |
| 3 ✅ | Step 4b (CatBoost Optuna tuning + retrain + ensemble v4) | 33.7 min (GPU) | Ensemble OOF 0.6408 (CB weight=0) |
| ~~4~~ | ~~Step 5 (Stacking)~~ | ~~30 min~~ | ~~Abandoned: CB weight=0, no benefit~~ |
| **4 ← NEXT** | **Step 7 (Tier 3: Grid×DOW TE)** | **1 hour** | **OOF +0.003~0.008** |

---

## Implementation

Create new notebook **`notebooks/05_improvement.ipynb`**, pipeline:
1. Load raw data + tier2 parquet
2. Regenerate improved TE (10-fold, high smoothing, KDTree fallback)
3. Add Tier 3 features (if doing)
4. Optuna subsample tuning
5. Full-data train LGB + XGB + CatBoost (tuned params, 5000+ rounds)
6. Stacking meta-learner
7. Rank normalization
8. Generate submission files

Key files:
- `notebooks/02_feature_engineering.ipynb` — reference TE implementation
- `notebooks/03_modeling.ipynb` — reference model training pipeline
- `data/train_features_tier2.parquet` / `test_features_tier2.parquet` — base feature data
- `data/encoding_maps_tier2.pkl` — encoding maps

---

## Verification

1. After each improvement, compute OOF Spearman and compare to baseline 0.5880
2. Check TE distribution shift: `scipy.stats.ks_2samp(train_te, test_te)`, target KS stat < 0.005
3. Check inter-model correlation, target < 0.97 (current LGB-XGB is 0.9778)
4. Submit to platform for actual score validation
5. Confirm test predictions have no NaN, row count = 2,028,750
