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

### Experiment G: Pseudo-Labeling with Curriculum Strategy (GPU, ~1h)

`notebooks/06_sprint.ipynb` — Section G

**Rationale**: Use confident test predictions as pseudo-labels to augment training. Since test is M1-5 only, this adds more M1-5 data for training — directly addressing the distribution imbalance. Curriculum approach (high confidence first, then medium) reduces confirmation bias.

**Reference**: Paper 4 (Revisiting Self-Training 2023)

**Implementation** (curriculum, not simple thresholding):
1. Layer 1 — High confidence pseudo-labels:
   - Select v7 test predictions where < 0.02 or > 0.98
   - Add to training set with weight = 0.7
   - Retrain LGB + XGB (v7 params)
2. Layer 2 — Medium confidence (optional):
   - Use Layer 1 model to re-predict test
   - Add predictions where < 0.10 or > 0.90 (excluding Layer 1 samples)
   - Weight = 0.3
   - Retrain again
3. Evaluate OOF on **original train only** (exclude pseudo-labels)
4. **Safety check**: if OOF drops > 0.003 after adding pseudo-labels, abort immediately

**Success criteria**: OOF >= 0.6429 with pseudo-labels (no degradation)

### Experiment B: Stacking Meta-Learner (Local, ~30min)

`notebooks/06_sprint.ipynb` — Section B

**Rationale**: Only worth doing if Experiments C, E, or H produced new effective models. Simple weighted average assumes linear additivity, but meta-learners can capture non-linear interactions.

**Precondition**: At least 3 models with non-zero ensemble weight available.

**Implementation**:
1. Build meta-features from all models with OOF > 0.55:
   - LGB v7, XGB v7
   - Rank-target LGB/XGB (if C succeeded)
   - ~~TabM~~ (E failed, weight=0)
   - Noise-handled LGB/XGB (H succeeded)
2. Layer 2 meta-learner options (5-fold CV):
   - **Ridge Regression** (linear, fast baseline)
   - **LightGBM meta** (100-500 trees, learns non-linear combinations)
3. Compare meta OOF Spearman vs simple weighted average
4. If meta-learner > weighted average by >= 0.001, use meta-learner for submission

### Experiment F: Final Ensemble Combination

`notebooks/06_sprint.ipynb` — Section F

**Rationale**: Combine all models/methods with positive signal:
- Layer 1: LGB v7, XGB v7, Rank-target LGB/XGB, Noise-handled LGB/XGB (from H)
- Layer 2: Stacking meta-learner or Optuna weight search
- Note: TabM excluded (weight=0), AV weights excluded (harmful)

### Day 3 afternoon: Transition to video production

---

## Experiment Priority (Revised)

| Priority | Experiment | Method | Expected Benefit | Cost | Change from v1 |
|----------|------------|--------|-----------------|------|-----------------|
| P0 | A: M1-5 Weight Optimization | Ensemble | Low-Med | 5 min local | unchanged |
| P0 | D: Adversarial Validation | Distribution Alignment | Med | 30 min GPU | + Temporal CV |
| ✅ | C: Rank-Based Target | Loss Optimization | **+0.0035 OOF** | 1h GPU | Stage 1 done, rank replaces v7 |
| ✅ | H: Label Noise Handling | Data Quality | **+0.0013 OOF** | 2.5h GPU | (a) Remove best, high corr w/ v7 |
| ❌ | E: TabM (was TabNet) | DL/Tabular SOTA | OOF 0.4445, weight=0 | 2h GPU | DL ceiling ~0.44 |
| P1 | G: Pseudo-Labeling | Semi-Supervised | Low-Med | 1h GPU | + curriculum strategy |
| P2 | B: Stacking Meta-Learner | Ensemble | Conditional | 30 min | depends on C/E/H |
| P2 | F: Final Ensemble | Combination | Depends | Low | unchanged |

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
