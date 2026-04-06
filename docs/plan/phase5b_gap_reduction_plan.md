# Phase 5b: Closing the OOF-Platform Gap

## Context

**Current best**: Ensemble v3, OOF 0.6408, Platform **0.5620**. Leaderboard top: 0.5992.

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

## Root Causes of Gap (Updated After Step 8/9)

1. ~~**LGB overfitting**~~: **RULED OUT** — Hard cap 6000 rounds gave Platform 0.5618 vs v3's 0.5620. Round count is not the gap source.
2. **TE encoding shift**: K-fold OOF TE vs full-data test TE (KS=0.01-0.013, small but systematic) — still suspected
3. **Noisy labels**: 25.2% of data has total_count=1 (binary {0,1} only, Spearman=0.41 vs 0.73 for count≥50) — **next target (Step 10)**
4. ~~**Weak features add noise**~~: **RULED OUT** — Pruning 8 features was neutral (XGB -0.0005)

---

## Implementation Steps

All code in `notebooks/05_improvement.ipynb` (append new cells).

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

### Step 10: Sample Weighting by total_count ← NEXT

**Priority: HIGH | Time: ~2.5h | Expected: OOF +0.002~0.005**

25.2% of samples have total_count=1 → pure binary noise (Spearman=0.41 vs 0.73 for count≥50). Noisy labels distort the loss landscape and may contribute to the gap.

**Configuration: Revert to v3 base** (10000 rounds, 26 features, Optuna v3 params). Only change: add sample_weight.

**Implementation (3 new cells: 59-61):**

Cell 59 [markdown]: Step 10 header

Cell 60 [code]: LGB v7 + XGB v7 with sample weighting
1. Compute weights: `sample_weight = np.log1p(train_df['total_count'].values)`
   - total_count=1 → weight 0.69, total_count=10 → 2.40, total_count=100 → 4.62
2. LGB v7: Optuna v3 params, 10000 rounds, 26 features, `sample_weight` in `.fit()`
   ```python
   model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=w_tr, ...)
   ```
   Note: sample_weight applies to training only; eval set is unweighted (default).
3. XGB v7: same approach
4. Print OOF Spearman, M1-5 Spearman, best_iter for both

Cell 61 [code]: Ensemble v7
1. 3-model weight grid search (LGB v7, XGB v7, CB v4)
2. Save OOF/test npy, generate submission CSV
3. Pre-submission validation + improvement history table

**What to watch:**
- OOF increases → weighting helped focus on cleaner labels
- OOF decreases → weighting too aggressive, try `sqrt(total_count)` variant
- OOF same but platform improves → weighting reduced overfitting to noise

**After Step 10:** Submit to platform. If Platform ≥ 0.575 → Step 11. If < 0.565 → try sqrt variant or move to Step 11.

---

### Step 11: M1-5 Focused Training (Experimental)

**Priority: MEDIUM | Time: 2h | Expected: Platform +0.005~0.020**

Since test = M1-5 only, training on M1-5 data (2.47M rows) may reduce noise from irrelevant summer patterns.

**Two variants:**
- **11A**: Train on ALL data, but recompute test TE from M1-5 only (grid_period_te KS improves 0.0102→0.0090)
- **11B**: Train on M1-5 only (2.47M rows, still large enough)

**Implementation:**
1. Implement variant 11A first (lower risk)
2. If 11A improves, try 11B
3. Evaluate M1-5 OOF Spearman as platform proxy

---

### Step 12: Stronger Regularization via Constrained Re-Optuna (If Steps 8-10 Insufficient)

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
  Conclusion: LGB rounds and weak features are NOT the gap source

Next (~2.5h):
  Step 10 (Sample Weighting) → v3 base + log1p weighting as v7
  → Submit to platform

Then (2h):
  Step 11 (M1-5 Training) → compare 11A and 11B
  → Submit best

If needed (3h):
  Step 12 or 13 → if platform still < 0.58
```

**Stop criteria**: Platform ≥ 0.59 OR all steps exhausted.

---

## Expected Outcome

| Scenario | Platform Score | Gap to Leaderboard Top (0.5992) |
|----------|---------------|---------------------------------|
| Pessimistic (only Step 8 works) | 0.565-0.570 | 0.029-0.034 |
| **Expected (Steps 8+9+10 work)** | **0.575-0.590** | **0.009-0.024** |
| Optimistic (all steps work) | 0.590-0.610 | close to or above top |

---

## Verification Protocol

After each step:
1. Compare full OOF Spearman (should not drop dramatically)
2. Compare M1-5 OOF Spearman (primary metric, proxy for platform)
3. Check LGB best_iteration (should be < 8000 after Step 8)
4. Submit to platform to verify gap reduction
5. Record all results in `docs/logs/progress.md`

---

## Key Files

| File | Purpose |
|------|---------|
| `notebooks/05_improvement.ipynb` | All new code (append cells) |
| `data/train_features_tier2.parquet` | 26 features + grid_id/grid_period (6.08M rows) |
| `data/test_features_tier2.parquet` | Test data, months 1-5 only (2.03M rows) |
| `models/lgb_oof_v3.npy` | Current best LGB OOF (baseline) |
| `models/xgb_oof_v3.npy` | Current best XGB OOF (baseline) |
| `docs/plan/improvement_plan.md` | Master improvement plan (update with new steps) |
| `docs/logs/progress.md` | Progress log (update with results) |
