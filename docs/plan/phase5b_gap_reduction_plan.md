# Phase 5b: Closing the OOF-Platform Gap

## Context

**Current best**: Ensemble v7, OOF **0.6429**, Platform **0.5636**. Leaderboard top: 0.5992.

The OOF-Platform gap (0.079) is the main bottleneck. Raising OOF without closing the gap wastes effort. This plan focuses on **reducing overfitting and distribution shift** to close the gap first, then raising OOF.

---

## Key Discovery: Test Set = Months 1-5 Only

| Month Group | Test Rows | Train Rows | Mean invalid_ratio | Ensemble OOF Spearman |
|-------------|-----------|------------|--------------------|-----------------------|
| M1-5 | 2,028,750 | 2,470,429 | 0.535 | **0.6492** |
| M6-12 | 0 | 3,606,117 | 0.481 | 0.6297 |

The model is already BETTER at M1-5 (OOF 0.6492), but the real gap is **0.087** (0.6492 → 0.5620). The gap is NOT from monthly pattern mismatch — TE means differ by only 0.001. The gap is from **overfitting + TE encoding shift**.

### Per-Month OOF Spearman (Ensemble v3)

| Month | Spearman | n |
|-------|----------|---|
| 1 | 0.6455 | 505,486 |
| 2 | 0.6399 | 478,240 |
| 3 | 0.6569 | 517,284 |
| 4 | 0.6538 | 519,078 |
| 5 | 0.6412 | 450,341 |
| 6 | 0.6406 | 484,050 |
| 7 | 0.6303 | 529,927 |
| 8 | 0.6298 | 531,972 |
| 9 | 0.6262 | 514,097 |
| 10 | 0.6270 | 538,948 |
| 11 | 0.6234 | 437,770 |
| 12 | 0.6180 | 569,353 |

---

## Root Causes of Gap (Updated After Steps 8-10)

1. ~~**LGB overfitting**~~: **RULED OUT** — Hard cap 6000 rounds gave Platform 0.5618 vs v3's 0.5620. Round count is not the gap source.
2. **TE encoding shift**: K-fold OOF TE vs full-data test TE (KS=0.01-0.013, small but systematic) — **primary suspect, next target (Step 11)**
3. ~~**Noisy labels**~~: **PARTIALLY ADDRESSED** — Sample weighting (v7) improved OOF +0.0021 and Platform +0.0016, but gap unchanged (0.079). Noisy labels hurt model quality, not distribution shift.
4. ~~**Weak features add noise**~~: **RULED OUT** — Pruning 8 features was neutral (XGB -0.0005)

---

## Implementation Steps

All code in `scripts/` (GPU server scripts) or `notebooks/05_improvement.ipynb`.

### Step 8: LGB Overfitting Fix ❌ Failed

**Priority: HIGH | Time: 1.5h | Actual: Platform -0.0002**

**Spearman ES (v6):** Subsample noise caused premature stopping at ~2000 rounds → LGB OOF 0.6098 (-0.0223). Full eval never stops (Spearman and l2 improve in lockstep).

**Hard cap 6000 rounds (v6b):** LGB OOF 0.6263 (-0.0058), Ensemble OOF 0.6392 (-0.0016), **Platform 0.5618** (-0.0002).

**Conclusion:** LGB round count is NOT the gap source. Revert to 10000 rounds for future steps.

| Version | LGB OOF | XGB OOF | Ensemble OOF | M1-5 OOF | Platform |
|---------|---------|---------|--------------|----------|----------|
| v3 (baseline) | 0.6322 | 0.6379 | 0.6408 | 0.6492 | **0.5620** |
| v6 (Spearman ES) | 0.6098 | 0.6375 | 0.6377 | 0.6457 | — |
| v6b (6000 cap) | 0.6263 | 0.6375 | 0.6392 | 0.6476 | **0.5618** |

---

### Step 9: Feature Pruning ✅ Neutral

**Actual: XGB -0.0005 (within noise), Platform no improvement**

Removed 8 features (6 periodic sin/cos + is_raining + has_snow). Harmless but no platform gain. Future steps revert to 26 features since pruning didn't help platform.

---

### Step 10: Sample Weighting by total_count ✅ New Best Platform

**Priority: HIGH | Time: ~45 min (GPU) | Actual: OOF +0.0021, Platform +0.0016**

Used v3 base (10000 rounds, 26 features, Optuna v3 params). Only change: `sample_weight = np.log1p(total_count)`.

| Version | LGB OOF | XGB OOF | Ensemble OOF | M1-5 OOF | Platform |
|---------|---------|---------|--------------|----------|----------|
| v3 (baseline) | 0.6322 | 0.6379 | 0.6408 | 0.6492 | 0.5620 |
| **v7 (weighted)** | 0.6336 | 0.6403 | **0.6429** | **0.6515** | **0.5636** |
| Delta | +0.0015 | +0.0024 | +0.0021 | +0.0022 | +0.0016 |

**Conclusion:** OOF gain transferred ~1:1 to platform (+0.0016), but **gap did NOT shrink** (0.0793 vs 0.0788). Sample weighting improved model quality but did not address distribution shift.

---

### Step 11: M1-5 Focused TE + Training ← NEXT

**Priority: HIGH | Time: ~1h (GPU) | Expected: Platform +0.005~0.020**

Since test = M1-5 only, and the gap (0.079) persists across all model-quality interventions, the remaining hypothesis is: **TE encoding shift is amplified by M6-12 data** irrelevant to the test set.

Current TE uses ALL 12 months for test encoding. M6-12 violation patterns differ (M1-5 OOF=0.6515 vs M6-12=0.6297), and including them may add noise to test TE values.

**Two variants:**

#### 11A: M1-5 TE for Test Only (Low Risk, ~15 min)

Recompute test `grid_te` and `grid_period_te` using only M1-5 training rows. Training is unchanged (same v7 models, same OOF). Only test predictions use the new TE.

```python
# Recompute test TE from M1-5 training data only
train_m1_5 = train_df[train_df['month_of_year'].isin([1, 2, 3, 4, 5])]
full_stats = train_m1_5.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
global_mean = train_m1_5['invalid_ratio'].mean()
smoothed = (full_stats['count'] * full_stats['mean'] + 100 * global_mean) / (full_stats['count'] + 100)
test_grid_te_m1_5 = test_df['grid_id'].map(smoothed).fillna(global_mean)
# Same for grid_period_te (smooth=150), fallback to grid_te for unseen

# Replace test TE columns, predict with existing v7 models
```

Since training is unchanged, OOF stays the same — only platform submission can evaluate.

#### 11B: Train on M1-5 Only (Medium Risk, ~45 min)

Both TE computation and model training restricted to M1-5 (2.47M rows). New K-fold CV on M1-5 data. Risk: 60% data loss may hurt generalization.

```python
# K-fold TE on M1-5 only (train encoding)
# Full M1-5 stats for test encoding
# Retrain LGB v8b + XGB v8b on 2.47M rows
# Same Optuna v3 params + sample weighting
```

**Implementation**: `scripts/step11_gpu.py`

**What to watch:**

| Signal | Interpretation |
|--------|---------------|
| KS stat decreases (e.g., 0.010 → 0.005) | M1-5 TE reduces shift — 11A likely helps |
| KS stat unchanged | TE shift NOT from month mismatch |
| 11A Platform > 0.5636 | M1-5 TE closer to test reality |
| 11B M1-5 OOF ≥ 0.640 | M1-5 data sufficient |
| 11B M1-5 OOF < 0.635 | Data loss hurts too much |

**After Step 11:** Submit both variants. If Platform ≥ 0.575 → done or try Step 12. If both < 0.565 → Step 12.

---

### Step 12: Stronger Regularization via Constrained Re-Optuna (If Steps 10-11 Insufficient)

**Priority: LOW | Time: 3h | Expected: Platform +0.003~0.010**

Re-run Optuna with constrained search space favoring regularization:

| Param | Current | New Range | Rationale |
|-------|---------|-----------|-----------|
| num_leaves | 100 | [15, 63] | Reduce tree complexity |
| reg_lambda | 0.452 | [1.0, 20.0] | Increase L2 |
| reg_alpha | 1.243 | [2.0, 10.0] | Increase L1 |
| feature_fraction | 0.844 | [0.5, 0.8] | Reduce per-tree features |

30 trials on M1-5 subsample, 3-fold CV, Spearman metric.

---

### Step 13: DART Boosting (If Time Permits)

**Priority: LOW | Time: 3h | Expected: Platform +0.003~0.010**

DART (Dropouts meet Multiple Additive Regression Trees) randomly drops trees → regularization effect. Since LGB overfits, DART may help.

**Implementation:**
- `boosting_type='dart'`, `drop_rate=0.1`, `max_drop=50`, `skip_drop=0.5`
- Reduce `n_estimators` to 3000-5000 (DART is slower and converges differently)
- Note: DART training ~2-3x slower than GBDT

---

## Execution Order & Decision Points

```
Done:
  Step 8 (Spearman ES) → v6 ❌ Spearman ES failed
  Step 9 (Feature Pruning) → neutral
  Step 8 revised (6000 cap) → v6b, Platform 0.5618 ❌ No improvement
  Step 10 (Sample Weighting) → v7, Platform 0.5636 ✅ New best

Next (~1h):
  Step 11A (M1-5 TE for test) → v8a, submit to platform
  Step 11B (M1-5 train only) → v8b, submit to platform

If needed (3h):
  Step 12 or 13 → if platform still < 0.58
```

**Stop criteria**: Platform ≥ 0.59 OR all steps exhausted.

---

## Expected Outcome

| Scenario | Platform Score | Gap to Leaderboard Top (0.5992) |
|----------|---------------|---------------------------------|
| Current best (v7) | 0.5636 | 0.036 |
| Step 11 works | **0.575-0.585** | 0.014-0.024 |
| Steps 11+12 work | **0.585-0.600** | 0.000-0.014 |

---

## Verification Protocol

After each step:
1. Compare full OOF Spearman (should not drop dramatically)
2. Compare M1-5 OOF Spearman (primary metric, proxy for platform)
3. Print KS stat for TE distribution shift (should decrease for Step 11)
4. Submit to platform to verify gap reduction
5. Record all results in `docs/logs/progress.md`

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/step11_gpu.py` | Step 11 script (to be created) |
| `scripts/step10_gpu.py` | Step 10 script (completed) |
| `notebooks/05_improvement.ipynb` | Main improvement notebook |
| `data/train_features_tier2.parquet` | 26 features + grid_id/grid_period (6.08M rows) |
| `data/test_features_tier2.parquet` | Test data, months 1-5 only (2.03M rows) |
| `models/lgb_oof_v7.npy` / `xgb_oof_v7.npy` | Current best OOF (v7 baseline) |
| `models/cb_oof_v4.npy` / `cb_test_v4.npy` | CB v4 (reuse in ensemble) |
| `docs/plan/improvement_plan.md` | Master improvement plan |
| `docs/logs/progress.md` | Progress log |
