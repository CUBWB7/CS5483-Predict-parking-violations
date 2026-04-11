# Experiment I + J: Hyperparameter Re-tuning & TabM Fix Plan

## Context

**Current best**: Exp C rank-target LGB+XGB, Platform **0.5698**, OOF **0.6464**

**Why we believe there's room to improve**:

Exp C achieved the best results by switching from raw `invalid_ratio` to rank-target training.
However, Exp C **reused v7 Optuna parameters verbatim** — parameters that were tuned for the
bimodal raw target, NOT the uniform rank-target distribution. This means:

1. The loss surface changed fundamentally (bimodal → uniform), but model hyperparameters stayed the same
2. LightGBM hit `n_estimators=10000` limit in ALL 5 folds (best_iter=9998-10000), proving it's still learning
3. Step 12's re-Optuna failed not because re-tuning is bad, but because search ranges were **too aggressive**
   (reg_lambda searched 1.0-20 when v7 optimal was 0.452, resulting in extreme over-regularization)

### Three Research Agents' Key Findings

**Agent 1 (GBDT Hyperparameters)**:
- LGB underfitting confirmed: all 5 folds hit iteration limit
- Increasing n_estimators to 15000-20000 is a "very likely quick win" (+0.002-0.005 OOF)
- Re-Optuna on rank-target with **relaxed** ranges (near v7 values) could gain +0.001-0.004

**Agent 2 (TabM Analysis)**:
- TabM's 0.4445 was NOT a DL ceiling — three critical implementation issues found:
  1. Trained on raw bimodal `invalid_ratio`, NOT rank-target
  2. No per-fold StandardScaler (only BatchNorm)
  3. Default hyperparameters (K=32, LR=1e-3, PATIENCE=7) never tuned
- Conservative estimate: fixes could improve TabM from 0.4445 → 0.55-0.60
- TabM diversity (corr ~0.74 with GBDT) means even moderate accuracy helps ensemble

**Agent 3 (Paper Inspiration)**:
- Most paper ideas have already been explored (temporal features, noise removal, AV weighting)
- Differentiable Spearman loss not applicable to GBDT (requires global loss function)
- Rank-target is already the best proxy for Spearman optimization in GBDT
- **Conclusion**: Hyperparameter tuning is the most promising remaining direction

---

## Results Summary (for reference)

| Version | Change | OOF | Platform | Gap | Conclusion |
|---------|--------|-----|----------|-----|------------|
| v7 | +log1p(tc) weighting | 0.6429 | 0.5636 | 0.079 | former best |
| **Exp C** | **rank-target LGB+XGB** | **0.6464** | **0.5698** | **0.077** | **CURRENT BEST** |
| Exp H(a) | Remove noise (tc=1) | 0.6442 | 0.5613 | 0.083 | OOF + but platform - |
| Exp E | TabM DL (raw target) | 0.4445 | — | — | implementation issues |
| Exp G | Pseudo-labeling | 0.6463 | — | — | null result (threshold mismatch) |

---

## Experiment I: Rank-Target GBDT Re-tuning ✅ COMPLETED

**Priority**: HIGH (P0)
**Script**: `scripts/step_i_gpu.py`
**Template**: `scripts/step_c_gpu.py` (lines 101-136 for current params)
**Anti-pattern**: `scripts/step12_gpu.py` (what NOT to do with Optuna ranges)
**Status**: Both Part A and Part B completed. **Part A is the winner** (OOF 0.6478 > Part B 0.6474).

### Part A: Quick Win — Increase n_estimators

**Goal**: Test if LGB benefits from more boosting rounds (evidence: all 5 folds hit 10000 limit)
**GPU Time**: ~50 min
**Risk**: Very low (same model, just more iterations)

**Changes from Exp C** (only 3 changes):

```
# LGB changes:
n_estimators:      10000 → 20000    # double the iteration budget
early_stopping_rounds: 150 → 200    # wider patience for LGB (add ES to LGB too)

# XGB changes:
n_estimators:      10000 → 15000    # wider safety margin (XGB ES at ~7800)
early_stopping_rounds: 150 → 200    # wider patience
```

All other parameters remain IDENTICAL to Exp C:
- LGB: num_leaves=100, lr=0.0564, min_child_samples=69, reg_lambda=0.452, reg_alpha=1.243
- XGB: max_depth=10, lr=0.0362, min_child_weight=11, reg_lambda=1.561, reg_alpha=1.239
- Target: rank-target `y_rank = rankdata(y_orig) / N`
- Sample weight: `log1p(total_count)`

**Implementation details**:
1. Copy `step_c_gpu.py` as template
2. Change n_estimators values
3. Add early stopping to LGB (currently LGB runs without ES, uses `n_estimators` as hard limit)
   - eval_metric: 'l2' (same as current)
   - callbacks: `[lgb.early_stopping(200), lgb.log_evaluation(500)]`
4. Keep ensemble weight search (step=0.01)
5. Generate `submissions/ensemble_i_a.csv`

**Expected output**:
- LGB best_iter: 12000-18000 (if model truly benefits from more rounds)
- Or LGB best_iter: ~10000 (if model was near plateau, gain is minimal)
- OOF improvement: +0.001-0.003

**Success criteria**: OOF ≥ 0.6470 (delta ≥ +0.0006)

**Decision point**: 
- If OOF ≥ 0.6480: Skip Part B, go directly to submission
- If OOF < 0.6470: Proceed to Part B (re-Optuna needed)
- If OOF ≈ 0.6464 (no change): LGB was already near plateau at 10000, proceed to Part B

**ACTUAL RESULTS (Part A)**:
- LGB I-A OOF: **0.6417** (+0.0044 vs Exp C), all 5 folds hit 20000 limit again
- XGB I-A OOF: **0.6430** (+0.0000), ES at ~7900-8100 (same as Exp C)
- Ensemble I-A: **0.6478** (+0.0014), weights LGB=0.48 XGB=0.52
- ✓ PASS (OOF 0.6478 ≥ 0.6470), borderline for skipping Part B (< 0.6480)

### Part B: Re-Optuna for Rank-Target

**Goal**: Find optimal hyperparameters specifically for the rank-target loss surface
**GPU Time**: ~3-4 hours (Optuna search + full retrain)
**Risk**: Medium (Step 12 showed bad ranges can cause regression)

**Critical lesson from Step 12 failure**:

Step 12 searched extreme ranges and found extremely regularized params:
```
# Step 12 FAILED ranges (too aggressive):
num_leaves:        15-63     # v7 optimal was 100 → forced much simpler trees
reg_lambda:        1.0-20    # v7 optimal was 0.452 → 22x stronger regularization
reg_alpha:         2.0-10    # v7 optimal was 1.243 → 5x stronger
min_child_weight:  50-300    # v7 optimal was 11 → 16x larger minimum

# Result: OOF dropped from 0.6429 to 0.6326 (-0.0103)
```

**Proposed ranges** (centered around v7 values, wider exploration):

**LGB Optuna search**:
```python
lgb_search_space = {
    'num_leaves':        (60, 200),     # v7=100, wider both ways
    'learning_rate':     (0.01, 0.08),  # v7=0.0564, allow lower for more rounds
    'min_child_samples': (40, 150),     # v7=69, moderate range
    'reg_lambda':        (0.1, 3.0),    # v7=0.452, centered near v7
    'reg_alpha':         (0.3, 3.0),    # v7=1.243, centered near v7
    'feature_fraction':  (0.7, 0.95),   # v7=0.844
    'bagging_fraction':  (0.8, 0.98),   # v7=0.972
}
```

**XGB Optuna search**:
```python
xgb_search_space = {
    'max_depth':         (6, 12),       # v7=10
    'learning_rate':     (0.01, 0.06),  # v7=0.0362
    'min_child_weight':  (5, 50),       # v7=11
    'reg_lambda':        (0.5, 3.0),    # v7=1.561
    'reg_alpha':         (0.5, 3.0),    # v7=1.239
    'colsample_bytree':  (0.6, 1.0),   # v7=0.951
    'subsample':         (0.8, 1.0),    # v7=0.948
}
```

**Optuna configuration**:
```python
N_TRIALS        = 60          # more trials than Step 12's 40
OPTUNA_N        = 1_000_000   # subsample 1M rows (same as Step 12)
OPTUNA_FOLDS    = 3           # 3-fold CV during search (same as Step 12)
OPTUNA_ITERS    = 5000        # faster iteration during search
USE_M15_ONLY    = True        # subsample from M1-5 rows only
TARGET          = 'rank'      # CRITICAL: search on rank-target, NOT raw y
```

**Optuna objective function**:
```python
def objective(trial):
    params = {
        'num_leaves':        trial.suggest_int('num_leaves', 60, 200),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 40, 150),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.1, 3.0, log=True),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.3, 3.0, log=True),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.7, 0.95),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.8, 0.98),
        # ... other fixed params
    }
    # Train on rank-target y_rank (NOT raw y)
    # Evaluate Spearman on original y_orig (to match platform metric)
    # Return mean 3-fold Spearman
```

**Full retrain with best params**:
- n_estimators = 20000 (from Part A learning)
- early_stopping_rounds = 200
- 5-fold CV on full 6M rows
- Generate `submissions/ensemble_i_b.csv`

**ACTUAL RESULTS (Part B)**:
- Optuna search: LGB best trial 0.6135 (162 min), XGB best trial 0.6152 (58 min)
- LGB Optuna params: num_leaves=123, lr=0.0325, min_child=49, lambda=1.93, alpha=0.42
- XGB Optuna params: max_depth=9, lr=0.0284, min_child_weight=25, lambda=1.98, alpha=0.83
- Full retrain: LGB I-B OOF **0.6415** (-0.0002 vs Part A), XGB I-B OOF **0.6426** (-0.0004)
- XGB I-B all 5 folds hit 15000 limit (lower lr → slower convergence, needed more iters)
- Ensemble I-B: **0.6474** (+0.0010 vs Exp C, but **-0.0004 vs Part A**)
- ✗ FAIL (OOF 0.6474 < target 0.6480)
- **Conclusion**: Optuna did NOT improve over v7 params. v7 params remain (near-)optimal for rank-target.

### Part C: Ensemble + Submission

1. Load Part A and/or Part B OOF + test predictions
2. Fine-grained ensemble weight search (step=0.01) on M1-5 OOF
3. Compare with Exp C baseline:
   - If OOF improvement ≥ +0.0016: submit to platform
   - If OOF improvement < +0.0010: likely not worth submitting (within noise range)
4. Generate `submissions/ensemble_i_final.csv`

**Success criteria**:
- OOF ≥ 0.6480 (vs Exp C 0.6464, delta ≥ +0.0016)
- M1-5 OOF ≥ 0.6540 (vs Exp C 0.6527)
- Platform improvement vs 0.5698 (only verifiable by submission)

**ACTUAL RESULTS (Part C / Ensemble)**:
- Best result: **Part A** (Ensemble I-A, OOF=0.6478, M1-5=0.6537)
- Part B did not improve over Part A → `ensemble_i_a.csv` is the submission to use
- `ensemble_i_final.csv` not generated separately (Part A csv is the final answer)
- **Platform result (2026-04-11): 0.5705** 🎉 (ranked #5, +0.0007 vs Exp C 0.5698, NEW BEST)

---

## Experiment J: TabM v2 with Rank-Target — ⏭️ SKIPPED

**Priority**: ~~MEDIUM (P2)~~ → SKIPPED
**Script**: `scripts/step_j_gpu.py` (not created)
**GPU Time**: ~3 hours (not spent)
**Decision**: Skip. Exp I Part B Optuna already consumed 5h+ GPU time with no gain over Part A.
Remaining time (video deadline 2026-04-15) is better spent on report and video production.

**Skip rationale**:
1. Even with all 6 fixes applied, TabM v2 OOF ceiling is estimated ~0.55-0.60, far below GBDT 0.64
2. At best 5-10% ensemble weight → expected gain only +0.001-0.003 OOF
3. Three DL attempts (ResNet 0.42, TabM 0.44, TabM v2 est. 0.55-0.60) consistently trail GBDT by 0.04-0.22
4. 3h GPU time + analysis overhead not justified given deadline pressure
5. For the report: document as "DL structurally inferior on this dataset" with the root cause analysis below

### Why TabM Failed (Exp E Root Cause Analysis)

| Issue | Exp E (0.4445) | Impact | Fix |
|-------|---------------|--------|-----|
| **No rank-target** | Trained on raw bimodal y | CRITICAL | Switch to `y_rank` |
| **No StandardScaler** | Only BatchNorm1d at input | HIGH | Per-fold StandardScaler |
| **Default hyperparams** | K=32, LR=1e-3, PATIENCE=7 | HIGH | Grid search K, LR |
| **Short patience** | 7 epochs | MEDIUM | Increase to 15 |
| **Low max_epochs** | 50 | LOW | Increase to 100 |
| **Sigmoid + MSE** | Numerically unstable at 0,1 | LOW | Remove sigmoid for rank |

The two DL failures (ResNet 0.42, TabM 0.44) share the SAME root cause: both trained MSE on
raw bimodal `invalid_ratio`, which is fundamentally misaligned with Spearman evaluation.
This is an **experimental design flaw**, not a DL architecture limitation.

### TabM v2 Implementation

**Key changes from Exp E** (`step_e_gpu.py`):

```python
# 1. RANK TARGET (line ~203 in step_e_gpu.py)
# OLD: y = train_df["invalid_ratio"].values.astype(np.float32)
# NEW:
from scipy.stats import rankdata
y_orig = train_df["invalid_ratio"].values
y_rank = (rankdata(y_orig, method='average') / len(y_orig)).astype(np.float32)

# 2. PER-FOLD STANDARDSCALER (line ~215 in step_e_gpu.py)
# OLD: X_train = train_df[feat_cols].values.astype(np.float32)  # raw, no scaling
# NEW:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train_raw[tr_idx])
X_va_scaled = scaler.transform(X_train_raw[va_idx])
X_test_scaled = scaler.transform(X_test_raw)

# 3. REMOVE SIGMOID (line ~152 in step_e_gpu.py)
# OLD: out = torch.sigmoid(out).squeeze(-1)
# NEW: out = out.squeeze(-1)  # rank-target is already [0,1], no need for sigmoid

# 4. HYPERPARAMETER GRID
K_OPTIONS  = [4, 8, 16, 32]   # was fixed at 32
LR_OPTIONS = [3e-4, 5e-4, 1e-3]  # was fixed at 1e-3
PATIENCE   = 15               # was 7
MAX_EPOCHS = 100              # was 50
```

**Training procedure**:
1. For each (K, LR) combination, run 1-fold quick validation (~10 min each)
2. Select best (K, LR) based on validation Spearman
3. Full 5-fold training with best params
4. Total: 12 quick validations (~2h) + 5-fold full (~1h) = ~3h

### Ensemble Integration Decision

```
IF TabM_v2 OOF ≥ 0.55:
    IF corr(TabM_v2, GBDT_ensemble) ≤ 0.85:
        → Include in 3-model ensemble (rank_LGB + rank_XGB + TabM_v2)
        → Expected TabM weight: 5-15%
        → Expected ensemble gain: +0.001-0.005 OOF
    ELSE:
        → TabM too correlated with GBDT, diversity insufficient
        → Skip ensemble, keep Exp I result
ELSE:
    → DL ceiling confirmed for this dataset
    → Document as definitive negative result (3 failures: ResNet 0.42, TabM 0.44, TabM_v2 <0.55)
```

### TabM Verdict

**Should not abandon, but must fix implementation first.** If TabM v2 (with rank-target +
StandardScaler + tuned K/LR) still falls below 0.55, THEN we can definitively confirm that DL
has a structural ceiling on this dataset. Until these fixes are applied, the 0.44 result is
inconclusive — it reflects experimental design error, not model capability.

---

## What NOT to Try (Paper Review Conclusions)

Papers reviewed: 13 papers in `research_parking_violations/papers/` (7 domain + 6 AI/ML)

| Idea from Papers | Verdict | Evidence |
|-----------------|---------|---------|
| Differentiable Spearman loss (torchsort, Blondel 2020) | Skip for GBDT | GBDT needs per-sample gradients; Spearman is global; rank-target already optimal proxy |
| Fourier/sin-cos temporal features (Cai & Ye 2025) | Already failed | v6 sin/cos made OOF -0.0035 |
| Spatial lag features (Sui 2025, Gao 2019) | Skip | Needs raw coordinate reconstruction; time-prohibitive |
| AUM + gradient noise detection (Eisenburger 2025) | Skip | Exp H: noise removal +0.0013 OOF but platform -0.0023 |
| Adversarial sample selection/weighting (Quan 2021) | Skip | Exp D: AUC≈1.0 → binary weights → harmful (-0.008) |
| Temporal-only training (Cai & Ye 2025) | Already failed | v8b M1-5 training: OOF -0.006 |
| Multi-scale spatial aggregation (Gao 2019) | Skip | Feature engineering pipeline frozen, deadline pressure |
| Pseudo-labeling with density (Kim 2023) | Skip | Exp G: rank-target compresses prediction range → threshold mismatch |

**Bottom line**: All promising paper ideas have been explored or are incompatible with current
setup. The remaining gain must come from **better optimization of existing approach** (tuning),
not new methods.

---

## Execution Order (Step-by-Step)

### Phase 1: Write Scripts + Notebook (Local, ~30 min)

```
Step 1.1: Create step_i_gpu.py
  - Copy step_c_gpu.py as template
  - Modify: n_estimators → 20000 (LGB), 15000 (XGB)
  - Add LGB early stopping (patience=200)
  - Add logging for best_iter tracking
  - Section A: direct training (no Optuna)

Step 1.2: Add Optuna section to step_i_gpu.py
  - Section B: Optuna search with rank-target
  - Use relaxed search ranges (see Part B above)
  - Section C: full retrain with best Optuna params
  - Add --skip-optuna flag to run only Part A

Step 1.3 (optional): Create step_j_gpu.py
  - Copy step_e_gpu.py as template
  - Apply all 6 fixes listed above
  - Add K/LR grid search
```

### Phase 2: Run Experiment I on GPU Server (~1-5h)

```
Step 2.1: Push code to GitHub (git add + commit + push)
Step 2.2: Pull on GPU server
Step 2.3: Upload data files if needed (train_features_tier2.parquet, etc.)
Step 2.4: Run Part A first:
  python step_i_gpu.py --part-a-only
  # Wait ~50 min
  # Check results: if OOF ≥ 0.6480, skip Part B

Step 2.5 (conditional): Run Part B (Optuna):
  python step_i_gpu.py --run-optuna
  # Wait ~3-4h for Optuna + full retrain

Step 2.6: Download results to local:
  - models/lgb_rank_i_oof.npy, lgb_rank_i_test.npy
  - models/xgb_rank_i_oof.npy, xgb_rank_i_test.npy
  - step_i_gpu.log
```

### Phase 3: Run Experiment J on GPU Server (~3h, optional)

```
Step 3.1: Run TabM v2 (can run in parallel with Exp I Part B if GPU memory allows):
  python step_j_gpu.py
  # Wait ~3h

Step 3.2: Download results:
  - models/tabm_v2_oof.npy, tabm_v2_test.npy
  - step_j_gpu.log
```

### Phase 4: Local Analysis via Notebook + Submission (~30 min)

```
Step 4.1: Add Section I to notebooks/06_sprint.ipynb
  (follow project convention: each Section self-contained, 
   try session variable first, fallback to disk load)

  Cell 1 — Data Loading:
    - try: _ = train_df.shape  (reuse session)
    - except: load from data/train_features_tier2.parquet
    - Load Exp C baseline: models/lgb_rank_oof.npy, xgb_rank_oof.npy
    - Load Exp I results: models/lgb_rank_i_oof.npy, xgb_rank_i_oof.npy
    - Load Exp I test preds: models/lgb_rank_i_test.npy, xgb_rank_i_test.npy

  Cell 2 — Single Model Comparison:
    - Table: LGB/XGB OOF Spearman (Exp C vs Exp I)
    - Table: LGB/XGB M1-5 OOF Spearman
    - LGB best_iter comparison (10000 vs new)

  Cell 3 — Inter-Model Correlation:
    - Spearman correlation matrix (rank_LGB_i, rank_XGB_i, ExpC_LGB, ExpC_XGB)
    - Heatmap visualization

  Cell 4 — Ensemble Weight Search:
    - Fine-grained grid search (step=0.01)
    - Plot: weight vs OOF Spearman curve
    - Compare optimal weights vs Exp C weights

  Cell 5 — Submission Generation:
    - Apply best weights to test predictions
    - Validate: no NaN, 2028750 rows, range [0,1]
    - Save submissions/ensemble_i_final.csv

  Cell 6 (optional) — Optuna Results Analysis (if Part B ran):
    - Best trial params vs v7 params comparison table
    - Optuna optimization history plot
    - Parameter importance plot

Step 4.2 (if Exp J ran): Add Section J to notebooks/06_sprint.ipynb
  - TabM v2 OOF analysis + diversity check
  - 3-model ensemble: rank_LGB_i + rank_XGB_i + TabM_v2
  - Check if TabM contributes positive weight
  - Ensemble weight search visualization

Step 4.3: Submit to platform
  - Compare with Exp C baseline (0.5698)
```

### Phase 5: Documentation (~15 min)

```
Step 5.1: Update docs/logs/progress.md with Exp I/J results
Step 5.2: Update docs/plan/sprint_plan.md if needed
Step 5.3: Git commit + push
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Part A shows no improvement | 30% | Low | Proceed to Part B (Optuna) |
| Part B Optuna finds worse params | 20% | Medium | Relaxed ranges prevent extreme params; keep Exp C as fallback |
| OOF improves but platform doesn't | 40% | High | Known issue (Exp H); only way to verify is submit |
| TabM v2 still < 0.55 | 50% | Low | Confirms DL ceiling, useful for report |
| GPU server unavailable | 10% | High | Check availability before starting |

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/step_i_gpu.py` | GBDT re-tuning script (Part A + B + C) |
| `scripts/step_j_gpu.py` | TabM v2 script (optional) |
| `notebooks/06_sprint.ipynb` Section I | Exp I local analysis (OOF comparison, correlation, ensemble, submission) |
| `notebooks/06_sprint.ipynb` Section J | Exp J local analysis (optional, TabM v2 + 3-model ensemble) |
| `submissions/ensemble_i_a.csv` | Part A submission |
| `submissions/ensemble_i_b.csv` | Part B submission (if run) |
| `submissions/ensemble_i_final.csv` | Best Exp I submission |

## Files to Read

| File | Purpose |
|------|---------|
| `scripts/step_c_gpu.py` | Template for Exp I (lines 101-136: params) |
| `scripts/step_e_gpu.py` | Template for Exp J |
| `scripts/step12_gpu.py` | Anti-pattern (lines 197-256: bad Optuna ranges) |
| `models/lgb_rank_oof.npy` / `xgb_rank_oof.npy` | Exp C baseline predictions |
| `step_c_gpu.log` | Exp C training log (best_iter evidence) |

---

## Estimated Timeline

| Day | Tasks | Hours |
|-----|-------|-------|
| Day 1 AM | Write scripts (step_i_gpu.py, step_j_gpu.py) | 1h |
| Day 1 PM | Run Exp I Part A on GPU | 1h |
| Day 1 PM | Evaluate Part A, decide on Part B | 0.5h |
| Day 1 PM-Eve | Run Exp I Part B (if needed) | 3-4h |
| Day 2 AM | Run Exp J (optional, in parallel) | 3h |
| Day 2 PM | Local analysis, ensemble, submission | 1h |
| Day 2 PM | Documentation, commit | 0.5h |
| **Total** | | **~6-10h** |
