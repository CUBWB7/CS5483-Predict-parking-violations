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
| 1 | Step 1 (TE improvement) + Step 2 (more rounds) + Step 6 (rank normalization) | 2 hours | ~0.545-0.555 |
| 2 | Step 3 (Optuna tuning) — can run overnight | 2-3 hours machine time | ~0.555-0.570 |
| 3 | Step 4 (CatBoost) — prepare while Optuna runs | 45 min | ~0.565-0.580 |
| 4 | Step 5 (Stacking) | 30 min | ~0.570-0.585 |
| 5 | Step 7 (Tier 3 features) — if time permits | 1 hour | ~0.575-0.595 |

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
