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

### Step 11: M1-5 Focused TE + Training ⏳ Awaiting Platform Submission

**Priority: HIGH | Time: ~70 min (GPU) | Actual: OOF unchanged (11A) / M1-5 OOF -0.006 (11B) | Platform: pending**

Since test = M1-5 only, and the gap (0.079) persists across all model-quality interventions, the remaining hypothesis is: **TE encoding shift is amplified by M6-12 data** irrelevant to the test set.

**Implementation**: `scripts/step11_gpu.py`

#### KS Distribution Diagnostic

| Feature | KS stat (orig TE vs M1-5 TE) | Orig mean | M1-5 mean |
|---------|-------------------------------|-----------|-----------|
| grid_te | **0.1305** | 0.5017 | 0.5337 |
| grid_period_te | **0.1330** | 0.4996 | 0.5297 |

KS > 0.13 confirms M1-5 TE differs significantly from full-data TE. M1-5 mean ~0.03 higher (M1-5 months have higher violation rates).

#### 11A: M1-5 TE for Test Only (v8a — Low Risk)

Training identical to v7 (all 6M rows). Only test `grid_te` / `grid_period_te` replaced with M1-5 stats (smooth=100/150).

| Version | LGB OOF | XGB OOF | Ensemble OOF | M1-5 OOF | Platform |
|---------|---------|---------|--------------|----------|----------|
| v7 (baseline) | 0.6336 | 0.6403 | 0.6429 | 0.6515 | 0.5636 |
| **v8a (11A)** | 0.6336 | 0.6403 | **0.6429** | **0.6515** | **pending** |

OOF identical (expected — training unchanged). Weights: LGB=0.35, XGB=0.65, CB=0.00. Runtime: LGB 35min + XGB 10min.

#### 11B: Train on M1-5 Only (v8b — Medium Risk)

Both TE and training restricted to M1-5 (2.47M rows). K-fold TE within M1-5 (smooth=30/50). Test TE from full M1-5 stats.

| Version | LGB M1-5 OOF | XGB M1-5 OOF | Ensemble M1-5 OOF | Platform |
|---------|-------------|-------------|-------------------|----------|
| v7 M1-5 | 0.6428 | 0.6482 | 0.6515 | 0.5636 |
| **v8b (11B)** | 0.6384 (-0.0044) | 0.6417 (-0.0065) | **0.6455 (-0.0060)** | **pending** |

Weights: LGB=0.35, XGB=0.50, **CB=0.15** (CB first time non-zero — weaker LGB/XGB allows CB to contribute). XGB best_iter ~4500 (v7: ~7500), earlier convergence with less data.

#### Interpretation

| Signal from plan | Actual | Interpretation |
|-----------------|--------|----------------|
| KS stat decreases? | N/A — different comparison | KS 0.13 measures full vs M1-5 TE gap, not train-test shift |
| 11A Platform > 0.5636? | **pending** | Only platform can tell if M1-5 TE helps |
| 11B M1-5 OOF ≥ 0.640? | **0.6455 ✅** | M1-5 data sufficient (above 0.640 threshold) |
| 11B M1-5 OOF < 0.635? | 0.6455 > 0.635 | Data loss hurts but is tolerable |

**After Step 11:** Submit v8a + v8b to platform (next day — daily limit reached). If Platform ≥ 0.575 → done or try Step 12. If both < 0.565 → Step 12.

---

### Step 12: Stronger Regularization via Constrained Re-Optuna ← NEXT

**Priority: MEDIUM | Time: 2-3h (GPU) | Expected: Platform +0.003~0.010**

**Rationale**: Step 12 is **orthogonal to Step 11** (TE encoding). Step 11 fixes what the model sees at test time; Step 12 fixes how the model learns. They can stack: final submission = best TE × best regularization. No need to wait for v8a platform score.

#### Constrained Search Space

**LGB parameters:**

| Param | Step 3 Range | Step 3 Best (v7) | Step 12 Range | Direction |
|-------|-------------|-------------------|---------------|-----------|
| num_leaves | [15, 127] | **100** | **[15, 63]** | ↓ Reduce tree complexity |
| learning_rate | [0.01, 0.1] | 0.0564 | [0.01, 0.1] | unchanged |
| min_child_samples | [20, 200] | 69 | [50, 300] | ↑ slightly higher |
| reg_lambda | [0.1, 10.0] | **0.452** | **[1.0, 20.0]** | ↑↑ stronger L2 |
| reg_alpha | [0.0, 5.0] | **1.243** | **[2.0, 10.0]** | ↑↑ stronger L1 |
| feature_fraction | [0.5, 1.0] | **0.844** | **[0.5, 0.8]** | ↓ reduce per-tree features |
| bagging_fraction | [0.5, 1.0] | 0.972 | [0.5, 0.9] | ↓ slightly lower |

**XGB parameters:**

| Param | Step 3 Best (v7) | Step 12 Range | Direction |
|-------|-------------------|---------------|-----------|
| max_depth | **10** | **[4, 8]** | ↓ reduce depth |
| learning_rate | 0.0362 | [0.01, 0.1] | unchanged |
| min_child_weight | 11 | [10, 200] | unchanged |
| reg_lambda | **1.561** | **[2.0, 20.0]** | ↑ stronger L2 |
| reg_alpha | **1.239** | **[2.0, 10.0]** | ↑ stronger L1 |
| colsample_bytree | **0.951** | **[0.5, 0.8]** | ↓↓ much lower |
| subsample | 0.948 | [0.5, 0.9] | ↓ slightly lower |

#### Implementation: `scripts/step12_gpu.py`

1. **Optuna search** (M1-5 subsample 1M rows, 3-fold CV, Spearman metric):
   - LGB: 40 trials with constrained ranges
   - XGB: 40 trials with constrained ranges
   - Est. time: ~1.5h
2. **Full retrain** (all 6.08M rows, `sample_weight=np.log1p(total_count)`, 10000 rounds, ES=150):
   - LGB v9: 5-fold CV with best constrained params
   - XGB v9: 5-fold CV with best constrained params
   - Est. time: ~1h
3. **Ensemble weight search** (grid search: LGB_v9 + XGB_v9 + CB_v4, step=0.05)
4. **Generate 2 submissions** (to decide TE approach after v8a result):
   - `ensemble_v9.csv` — full-data TE (if v8a fails)
   - `ensemble_v9a.csv` — M1-5 TE (if v8a succeeds, reuse Step 11 M1-5 TE logic)
5. **Save** models, OOF predictions, test predictions to `models/`

#### What to Watch

| Signal | Interpretation |
|--------|---------------|
| Optuna best params: num_leaves < 63, reg_lambda > 2.0 | Constraints are binding — search found useful regularization |
| OOF Spearman: 0.638-0.643 | Mild OOF drop is expected and OK (regularization trades train fit for generalization) |
| OOF Spearman < 0.635 | Over-regularized — constraints too tight, consider relaxing |
| M1-5 OOF tracks full OOF | Regularization affects both equally — good |
| Platform > v7 (0.5636) | Gap reduced — regularization helps generalization |

#### Output Files

| File | Description |
|------|-------------|
| `scripts/step12_gpu.py` | Step 12 GPU script |
| `models/lgb_[oof\|test]_v9.npy` | LGB v9 predictions |
| `models/xgb_[oof\|test]_v9.npy` | XGB v9 predictions |
| `submissions/ensemble_v9.csv` | v9 with full-data TE |
| `submissions/ensemble_v9a.csv` | v9 with M1-5 TE |

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
  Step 11A (M1-5 TE for test) → v8a ✅ trained, ⏳ awaiting platform submission
  Step 11B (M1-5 train only) → v8b ✅ trained, ❌ M1-5 OOF -0.006, not recommended

In parallel (independent axes):
  → Submit v8a to platform (2026-04-07) — determines TE approach
  → Step 12: Constrained Optuna (can start now) — determines regularization

After both complete:
  → Combine: best TE (v7 or M1-5) × best regularization (v7 or v9 params)
  → If still < 0.58 → Step 13 (DART)
```

**Stop criteria**: Platform ≥ 0.59 OR all steps exhausted OR deadline pressure (video 2026-04-15).

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
| `scripts/step12_gpu.py` | Step 12 script (**to create**) |
| `scripts/step11_gpu.py` | Step 11 script (completed) |
| `scripts/step10_gpu.py` | Step 10 script (completed) |
| `notebooks/05_improvement.ipynb` | Main improvement notebook |
| `data/train_features_tier2.parquet` | 26 features + grid_id/grid_period (6.08M rows) |
| `data/test_features_tier2.parquet` | Test data, months 1-5 only (2.03M rows) |
| `models/lgb_oof_v7.npy` / `xgb_oof_v7.npy` | Current best OOF (v7 baseline) |
| `models/cb_oof_v4.npy` / `cb_test_v4.npy` | CB v4 (reuse in ensemble) |
| `docs/plan/improvement_plan.md` | Master improvement plan |
| `docs/logs/progress.md` | Progress log |
