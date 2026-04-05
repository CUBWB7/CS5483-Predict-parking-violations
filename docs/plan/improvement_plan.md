# Score Improvement Plan

## Context

**Current best**: OOF Spearman **0.6408**, Platform **0.5620**. OOF-Platform gap: **0.0788**.
Platform leaderboard top: 0.5992.

All improvement work in `notebooks/05_improvement.ipynb`, without modifying verified 01-04 notebooks.

---

## Step 1: Improve Target Encoding ✅ (Limited Effect)

**Expected: Platform +0.015~0.025 | Actual: KS stat improved by only 0.001 | Time: 30 min**

Current issue: 5-Fold TE uses 80% data for train encoding, 100% for test, causing distribution shift (KS stat ~0.01).

Changes:
- TE folds from 5 → **10** (train encoding uses 90% data, closer to test's 100%)
- `grid_te` smoothing from 30 → **100**, `grid_period_te` from 50 → **150**
- Unseen test grids (21, 3.6%): use **KDTree nearest-neighbor** instead of global mean fallback

Key file: `notebooks/02_feature_engineering.ipynb` (reference implementation), new code in `05_improvement.ipynb`

---

## Step 2: Increase n_estimators + Lower Learning Rate ✅ (Main v2 Contributor)

**Expected: OOF +0.003~0.008 | Actual: OOF +0.012~0.014 | Time: extra training time only**

Current LGB/XGB both hit 3000 rounds without early stopping triggering — models are undertrained.

Changes:
- `n_estimators`: 3000 → **5000~8000**
- `learning_rate`: 0.05 → **0.03**
- `early_stopping_rounds`: 50 → **100**

---

## Step 3: Optuna Hyperparameter Tuning ✅ (Biggest Contributor)

**Expected: OOF +0.010~0.020 | Actual: OOF +0.036 | Time: ~5 hours**

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

## Step 4: Add CatBoost as Third Model ✅ (No Ensemble Contribution)

**Expected: Ensemble +0.005~0.015 | Actual: CB weight=0, Ensemble unchanged | Time: 6.3h**

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

## Step 4b: CatBoost Optuna Hyperparameter Tuning ✅ (CB Improved, Ensemble Unchanged)

**Expected: CB OOF 0.62+, Ensemble +0.002~0.005 | Actual: CB 0.6175, CB weight still 0 | Time: 33.7 min (GPU)**

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

## Step 5: Stacking Meta-Learner ❌ Abandoned

**Expected: +0.005~0.010 | Actual: Not executed**

Original plan: use LGB + XGB + CatBoost OOF predictions as features for a Ridge regression meta-learner.

**Why abandoned:**
- CB weight = 0 in ensemble → meta-learner only has LGB + XGB (2 inputs)
- LGB-XGB correlation = 0.964 → Ridge on 2 near-identical signals ≈ weighted average
- Even adding auxiliary features (total_count, grid_te) won't help: base models already consumed them
- Risk of overfitting OOF noise → could widen the already large OOF-Platform gap (0.079)

---

## Step 6: Rank Normalization Post-Processing ❌ Harmful

**Expected: +0.002~0.005 | Actual: Platform -0.007 (0.5266 vs 0.5338)**

Rank normalization was tested in v2. Despite being theoretically monotonic, it **hurt** platform score.
Root cause: Spearman(before, after) = 0.988 ≠ 1.0 — floating-point ties introduced noise.

---

## Step 7: Tier 3 Feature Engineering ❌ No Improvement

**Expected: OOF +0.003~0.008 | Actual: OOF +0.0001 (flat) | Time: ~3.5 hours**

### Feature Selection (Data-Driven)

Compared 4 candidate cross-TE features:

| Feature | Spearman (leaky) | Corr w/ grid_te | Corr w/ grid_period_te | Verdict |
|---------|------------------|-----------------|-----------------------|---------|
| grid_period_te (existing) | 0.311 | 0.9801 | — | Baseline |
| grid_dow_te (original plan) | 0.309 | 0.9933 | 0.9717 | Rejected: near-redundant |
| **grid_month_te (selected)** | **0.326** | **0.9456** | 0.9255 | Best candidate |
| grid_hour_te | 0.315 | 0.9753 | 0.9863 | Rejected: overlaps grid_period_te |

### Result: grid_month_te Failed

| Metric | Value | Issue |
|--------|-------|-------|
| KS stat (train/test shift) | **0.123** | Far above 0.02 target — 26.4% of groups have <50 samples |
| LGB v5 OOF | 0.6315 | **-0.0007** vs v3 (noise from unstable TE) |
| XGB v5 OOF | 0.6382 | +0.0003 (within random variation) |
| Ensemble v5 OOF | 0.6408 | +0.0001 (flat) |

**Root cause**: fine-grained cross-TE features (6,561 groups) have too many small groups for stable encoding. All 4 candidates have >0.94 correlation with existing TE features — the feature space is saturated.

### All Tier 3 Candidates Exhausted

- ~~Grid × Month TE~~: tested, KS=0.123, no gain
- ~~Grid × Day-of-Week TE~~: corr 0.9933 with grid_te, would be even worse
- ~~Grid × Hour TE~~: corr 0.9863 with grid_period_te, redundant
- ~~Temperature discretization~~: weather features ρ < 0.03, won't help
- ~~KMeans clustering~~: 742 grids already sufficient
- ~~6-hour weather window~~: high complexity, low expected gain

---

## Execution Order (Final)

| Order | Steps | Time | Result |
|-------|-------|------|--------|
| 1 ✅ | Step 1 (TE) + Step 2 (more rounds) + Step 6 (rank norm) | 2h | Platform 0.5338, rank norm harmful |
| 2 ✅ | Step 3 (Optuna) + Step 4 (CatBoost) | ~11h | **Ensemble OOF 0.6408, Platform 0.5620** |
| 3 ✅ | Step 4b (CB Optuna, GPU) | 34 min | CB 0.6175, still weight=0 |
| 4 ❌ | Step 5 (Stacking) | — | Abandoned |
| 5 ❌ | Step 7 (Grid×Month TE) | 3.5h | No improvement (KS=0.123) |
| — ❌ | Step 6 (Rank Norm) | 10 min | Harmful (-0.007) |

**Net result**: Steps 1-3 drove all gains. Steps 4-7 yielded nothing additional.

---

## Retrospective & Bottleneck Analysis

### What worked
| Step | OOF Gain | Key insight |
|------|----------|-------------|
| Step 2 (more rounds) | +0.013 | Models were severely undertrained at 3000 rounds |
| Step 3 (Optuna) | +0.036 | By far the largest gain; default params were far from optimal |

### What didn't work and why
| Step | Why it failed |
|------|---------------|
| Step 1 (TE improvement) | TE shift root cause is grid granularity, not fold count |
| Step 4/4b (CatBoost) | OOF 0.015+ below LGB/XGB; tuning raised corr to 0.97 (lost diversity) |
| Step 5 (Stacking) | Only 2 useful base models with corr 0.964 → equivalent to averaging |
| Step 6 (Rank Norm) | Floating-point ties break monotonicity assumption |
| Step 7 (Tier 3 TE) | All cross-TE features >0.94 corr with existing; feature space saturated |

### The core bottleneck: OOF-Platform gap (0.079)

The gap accounts for ~0.08 Spearman — closing it to 0.04 would reach Platform **0.60+**.
The gap likely comes from:
1. **Target Encoding distribution shift** — TE features (grid_te, grid_period_te) are the strongest features but have inherent train/test mismatch
2. **Possible temporal shift** — train/test may cover different time periods with different violation patterns
3. **Overfitting to training distribution** — 10000 rounds of GBDT on 6M rows may memorize training-specific patterns

### Potential new directions (not yet attempted)

| Idea | Expected Gain | Effort | Rationale |
|------|---------------|--------|-----------|
| **A. Reduce n_estimators for LGB** | Platform +0.005~0.015 | 2h | LGB never triggers ES at 10000; may be overfitting. Try 6000-8000 rounds to reduce gap |
| **B. Stronger regularization** | Platform +0.003~0.010 | 3h | Re-Optuna with tighter reg_lambda/alpha ranges, lower num_leaves |
| **C. Drop weak features** | OOF +0.001~0.003 | 1h | Remove is_raining, has_snow, periodic encodings (ablation showed some hurt) |
| **D. TE with leave-one-out** | Gap -0.01~0.02 | 2h | LOO encoding instead of K-Fold may reduce train/test TE shift |
| **E. Neural network (TabNet/FT-Transformer)** | OOF +0.005~0.020 | 4h+ | Architecturally different from GBDT → better ensemble diversity. Needs GPU |
| **F. LGB with dart boosting** | OOF +0.002~0.008 | 3h | DART (dropout trees) reduces overfitting, may close gap |

---

## Key Files

- `notebooks/05_improvement.ipynb` — all improvement code (54 cells)
- `notebooks/02_feature_engineering.ipynb` — reference TE implementation
- `data/train_features_tier2.parquet` / `test_features_tier2.parquet` — base feature data
- `models/` — all OOF and test prediction .npy files (v1–v5)
- `submissions/` — all submission CSV files

## Current Best Submission

**`submissions/ensemble_v3.csv`** — Platform **0.5620** (OOF 0.6408)
