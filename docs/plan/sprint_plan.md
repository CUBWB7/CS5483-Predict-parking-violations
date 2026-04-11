# Sprint Plan: Final 2-3 Day Push (v2 — Revised)

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
| Exp C | Rank-target LGB+XGB | **0.6464** | **0.5698** 🎉 | 0.077 | **NEW BEST**, +0.0062 vs v7 platform |
| Exp E | TabM DL (ICLR 2025) | 0.4445 | not submitted | — | ❌ OOF too low, weight=0 |
| Exp H(a) | Remove noise (tc=1) | **0.6442** | 0.5613 ❌ | 0.083 | OOF +0.0013 but platform -0.0023 |
| Exp G L1 | Pseudo-label (199 rows) | 0.6463 | — | — | Null result — threshold/range mismatch |
| **Exp I-A** | **LGB n_iter 20K + fine ensemble** | **0.6478** | **0.5705** 🎉 | 0.077 | **NEW BEST** (+0.0007 vs Exp C, ranked #5) |
| Exp I-B | Optuna re-tune for rank-target | 0.6474 | — | — | Optuna no gain vs v7 params |

### Lessons from Steps 8-14

| Attempt | Why it failed | Takeaway |
|---------|--------------|----------|
| M1-5 TE (v8a) | Full-data TE is actually better for test | Don't assume temporal TE is better |
| Strong regularization (v9) | Over-regularized, OOF -0.010 | Gap is NOT caused by overfitting |
| DART (v10) | Precision loss (-0.019) > diversity gain | Dropout hurts on 6M noisy data |
| ResNet NN (v11) | OOF 0.42, too weak for ensemble | Simple NN architecture insufficient |

### New Literature Support

6 papers downloaded to `research_parking_violations/papers/AI/`:
1. Adversarial Validation (Ivanescu 2021) — sample re-weighting for distribution shift
2. Temporal Shift Limits (2025) — minimizing time lag improves generalization
3. Differentiable Sorting (Blondel 2020, ICML) — torchsort for Spearman loss
4. Pseudo-Labeling for Tabular (2023) — curriculum pseudo-labeling for GBDT
5. GBDT Label Noise (2024) — handling noisy labels in GBDT
6. TabM (Gorishniy 2024, ICLR 2025) — SOTA tabular deep learning

---

## Day 1 (4/08) — Quick Wins + Diagnostics

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

### Experiment D: Adversarial Validation + Temporal CV (Local/GPU, ~30min)

`notebooks/06_sprint.ipynb` — Section D

**Rationale**: Train a classifier to distinguish train vs test data. Samples with high "test-like" probability should receive higher training weight. Additionally, switch CV to Temporal CV (validate on M1-5 only) for more realistic platform score estimation.

**Reference**: Paper 1 (Ivanescu 2021), Paper 2 (Temporal Shift 2025)

**Implementation**:

Part 1 — Adversarial Validation:
1. Merge train + test features, label = 0 (train) / 1 (test)
2. Train LGB classifier (5-fold CV, evaluate AUC)
3. **Decision threshold**: AUC ~ 0.5 = indistinguishable (skip); AUC > 0.6 = significant shift (proceed)
4. If AUC > 0.6: use predicted probabilities as additional sample_weight multiplier
   ```python
   final_weight = log1p_weight * av_probability  # combine with existing weighting
   ```

Part 2 — Temporal CV (diagnostic only):
1. Build CV split: train on M1-4, validate on M5 (or similar temporal split)
2. Compare Temporal CV Spearman vs Random 5-fold OOF Spearman
3. If Temporal CV << Random CV, it confirms distribution shift is real and quantifies the expected gap
4. Use Temporal CV as the primary evaluation metric for subsequent experiments

---

## Day 2 (4/09) — New Models + New Methods

### Experiment C: Rank-Based Target Training (GPU, ~1h) ✅ COMPLETED

`notebooks/06_sprint.ipynb` — Section C | `scripts/step_c_gpu.py`

**Result**: Stage 1 ✓ PASS. Rank-only ensemble OOF **0.6464** (+0.0035 vs v7).
- LGB rank OOF: 0.6373 (+0.0037), XGB rank OOF: 0.6430 (+0.0027)
- v7 models get ~0 weight in 4-model ensemble → rank-target fully replaces v7
- Correlation with v7: ~0.99 → stacking unlikely to help
- Stage 2 (torchsort): viable but low priority given deadline pressure
- **Pending**: submit to platform for actual score

### Experiment H: GBDT Label Noise Handling (GPU, ~1h) ✅ COMPLETED

`notebooks/06_sprint.ipynb` — Section H | `scripts/step_h_gpu.py`

**Result**: ✓ PASS. Best strategy (a) Remove: OOF **0.6442** (+0.0013 vs v7), M1-5 **0.6526**.
- 36,508 noise candidates identified (0.60% of train), mostly pred>0.85 & y=0
- All three strategies beat v7; (a) Remove ≈ (b) Down-weight > (c) Label smooth
- Correlation with v7: >0.99 → low diversity, stacking unlikely to help
- **Pending**: submit `ensemble_ha.csv` to platform for actual score

### Experiment E: TabM Deep Learning Model (GPU, ~2h) ❌ COMPLETED — OOF too low

`notebooks/06_sprint.ipynb` — Section E | `scripts/step_e_gpu.py`

**Result**: OOF **0.4445** — FAIL (target ≥ 0.55). Diversity PASS (corr 0.7457 < 0.85), but accuracy too low to contribute to ensemble. TabM optimal weight = 0 in grid search.
- TabM standalone: OOF 0.4445, M1-5 0.4601
- Best ensemble (rank_LGB + rank_XGB + TabM): OOF 0.6464 (TabM weight = 0, no improvement over Exp C)
- Two DL attempts (ResNet 0.42, TabM 0.44) confirm DL ceiling on this dataset ~0.44-0.45
- **Conclusion**: DL not viable for this dataset. Do not invest further in DL direction.

### Day 2 submissions: Pick best 2 from Experiments C, D, H, E

---

## Day 3 (4/10) — Combine Best + Transition to Report

### Experiment G: Pseudo-Labeling with Curriculum Strategy ✅ COMPLETED — Null Result

`notebooks/06_sprint.ipynb` — Section G | `scripts/step_g_gpu.py`

**Result**: Null result. Layer 1 OOF **0.6463** (-0.0001 vs Exp C), Layer 2 OOF **0.6462** (-0.0002). Safety ✓ PASS but no improvement.

**Root cause of failure**: rank-target training compresses prediction range to [0.079, 0.867]. Test predictions cannot reach the < 0.02 or > 0.98 thresholds structurally. Only **1 row** selected for Layer 1 (0.00%), 199 rows total — statistically zero impact on 6.07M training rows.

**Key finding**: Standard pseudo-labeling thresholds designed for raw [0,1] predictions are **incompatible with rank-target models**. The threshold should have been adapted to the actual prediction range (e.g., < 0.15 or > 0.85). This is a design-stage mismatch, not a training issue.

**Decision**: Exp G models excluded from Exp F ensemble (OOF degradation, not improvement).

### Experiment B: Stacking Meta-Learner ← SKIP

**Decision**: Skip. All effective models (rank LGB, rank XGB) correlate >0.99 with each other and with v7. Stacking cannot extract diversity that doesn't exist. Exp H models hurt on platform despite OOF gain — including them in a meta-learner would propagate the damage.

### Experiment F: Final Ensemble Combination

`notebooks/06_sprint.ipynb` — Section F

**Rationale**: Lock in the best submission using models with proven platform gains.

**Model inclusion decision**:
| Model | OOF | Platform | Include? | Reason |
|-------|-----|----------|---------|--------|
| rank LGB (Exp C) | 0.6373 | — | ✅ | Core model |
| rank XGB (Exp C) | 0.6430 | — | ✅ | Core model |
| pseudo-label LGB/XGB (Exp G) | 0.6463 | — | ❌ | OOF -0.0001, null result, excluded |
| v7 LGB/XGB | 0.6336/0.6403 | — | ❌ | Dominated by rank-target |
| H models (noise removal) | 0.6442 | 0.5613 | ❌ | Platform hurt despite OOF gain |
| TabM | 0.4445 | — | ❌ | Weight=0 in all grid searches |

**Result**: Strategy 1 (Exp C rank-only) selected. LGB=0.39, XGB=0.61, OOF=0.6464, M1-5=0.6527.
Exp G excluded (OOF 0.6463 < threshold 0.6464 by 0.0001). Final file: `submissions/ensemble_final.csv`.

### Day 3 afternoon: Transition to video production

---

## Experiment Priority (Revised)

| Priority | Experiment | Method | Result | Status |
|----------|------------|--------|--------|--------|
| ✅ | C: Rank-Based Target | Loss Optimization | OOF **0.6464**, Platform **0.5698** 🎉 | Done — NEW BEST |
| ✅ | D: Adversarial Validation | Distribution Shift Diag. | AUC≈1.0, AV harmful, Temporal CV useful | Done |
| ✅ | H: Label Noise Handling | Data Quality | OOF +0.0013, Platform **0.5613** ❌ | Done — excluded from final |
| ✅ | E: TabM DL | SOTA Tabular DL | OOF **0.4445**, weight=0 | Done — DL ceiling ~0.44 |
| ✅ | G: Pseudo-Labeling | Semi-Supervised | OOF 0.6463 (-0.0001) — **null result** | Done — excluded (threshold/range mismatch) |
| ✅ | F: Final Ensemble | Combination | OOF **0.6464**, LGB=0.39 XGB=0.61 — Exp C rank-only | Done — `ensemble_final.csv` generated |
| ✅ | I: GBDT Re-tuning | Hyperparameter Opt. | OOF **0.6478** (+0.0014), LGB n_iter 10K→20K | Done — Part A best, Optuna no gain |
| ~~P2~~ | ~~B: Stacking~~ | ~~Meta-Learner~~ | — | ~~SKIP~~ (corr >0.99, no diversity) |
| ~~P0~~ | ~~A: M1-5 Weight Opt.~~ | ~~Ensemble~~ | — | ~~SKIP~~ (superseded by Exp C) |
| ~~P2~~ | ~~J: TabM v2 (rank-target)~~ | ~~DL + rank-target~~ | — | ~~SKIP~~ (DL ceiling, deadline pressure) |

---

## Key Files

### Files to Create
| File | Purpose |
|------|---------|
| `notebooks/06_sprint.ipynb` | Sprint experiment notebook (Sections A-H) |

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

### Reference Papers (in `research_parking_violations/papers/AI/`)
| File | Topic |
|------|-------|
| `adversarial_validation_2021.pdf` | Experiment D methodology |
| `temporal_shift_2025.pdf` | Experiment D theory |
| `differentiable_sorting_2020.pdf` | Experiment C Stage 2 |
| `pseudo_labeling_tabular_2023.pdf` | Experiment G curriculum strategy |
| `gbdt_label_noise_2024.pdf` | Experiment H methodology |
| `tabm_2024.pdf` | Experiment E model architecture |

---

## Verification

Unified verification flow for each experiment:
1. OOF Spearman vs v7 baseline (0.6429)
2. M1-5 OOF Spearman vs v7 M1-5 (0.6515)
3. Inter-model correlation matrix (diversity check)
4. Submission standard validation (no NaN, 2028750 rows, range [0,1])
5. Record platform score after submission, update progress.md
