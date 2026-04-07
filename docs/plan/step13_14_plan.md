# Implementation Plan: Step 13 (DART) + Step 14 (Neural Network)

## Context

**Current state**: Ensemble v7 OOF 0.6429, Platform 0.5636, gap 0.079. v8a/v9 awaiting submission tomorrow.
**Goal**: Two orthogonal experiments while waiting — DART regularization + NN ensemble diversity.
**Core problem**: LGB-XGB correlation 0.968, ensemble lacks diversity; TE distribution shift causes gap.

### Paper Analysis Summary

Reviewed all 7 project papers + FunSearch/AlphaEvolve. Key findings:
- Most useful techniques from papers already implemented (sine encoding, K-fold TE, grid spatial discretization, sample weighting)
- Remaining paper techniques either require external data (POI, mobility, crime — unavailable in competition) or are too complex for timeline
- FunSearch/AlphaEvolve not applicable (designed for algorithm discovery, not ML pipeline optimization, zero precedent in competitions)
- **Best remaining ideas**: DART boosting (regularization) + Neural Network (ensemble diversity)

---

## Step 13: DART Boosting — `scripts/step13_gpu.py` ❌ COMPLETED (Negative Result)

### Rationale
DART (Dropouts meet Multiple Additive Regression Trees) randomly drops existing trees each boosting round, forcing new trees to learn independently and reducing overfitting. LGB never triggers early stopping at 10000 rounds — DART addresses this directly.

### Implementation

**Based on step10_gpu.py template**, only changing LGB's boosting_type:

```python
# LGB DART parameters — final tuned version (2nd run)
lgb_params_dart = {
    **lgb_v7_params,              # reuse v7 Optuna params
    'boosting_type': 'dart',      # core change
    'drop_rate':     0.05,        # drop 5% of trees (0.1 was too aggressive)
    'max_drop':      30,          # max 30 trees dropped per round (was 50)
    'skip_drop':     0.5,         # 50% chance to skip drop (speedup)
    'n_estimators':  7000,        # more rounds to compensate for dropout
    # Note: DART does not support standard early_stopping (loss is non-monotonic)
}
```

**XGB unchanged**: keep v7 GBDT params (XGB DART is less stable).

### Tuning History
1. **1st run** (drop_rate=0.1, max_drop=50, n_estimators=5000): Fold 0 OOF **0.6054** — far too low, interrupted
2. **2nd run** (drop_rate=0.05, max_drop=30, n_estimators=7000): 5-Fold OOF **0.6147** — still below target

### Results (2nd run, 5-Fold, GPU, 182 min)

| Metric | v7 | v10 (DART) | Delta |
|--------|-----|------------|-------|
| LGB OOF | 0.6336 | 0.6147 | **-0.0190** |
| LGB M1-5 OOF | 0.6428 | 0.6266 | -0.0162 |
| LGB-XGB correlation | 0.9681 | **0.9476** | -0.0205 (improved) |
| Ensemble OOF | 0.6429 | 0.6406 | -0.0023 |
| Ensemble M1-5 | 0.6515 | 0.6490 | -0.0025 |

Ensemble v10 weights: LGB=0.10, XGB=0.85, CB=0.05 (DART LGB nearly ignored)

### Success Criteria Check

| Criterion | Target | Actual | Result |
|-----------|--------|--------|--------|
| DART LGB OOF ≥ 0.625 | 0.625 | 0.6147 | ❌ FAIL |
| LGB-XGB correlation < 0.965 | < 0.965 | 0.9476 | ✅ PASS |
| Ensemble OOF ≥ 0.640 | 0.640 | 0.6406 | ✅ PASS (barely) |

### Conclusion
**Meaningful negative result**: DART achieves diversity goal (correlation 0.968→0.948) but accuracy cost is too high (-0.019). Not submitted to platform. Retained as experimental data for report.

### Output Files
- `models/lgb_oof_v10.npy`, `models/lgb_test_v10.npy` (DART LGB)
- `submissions/ensemble_v10.csv` (full-data TE, not submitted)
- `submissions/ensemble_v10a.csv` (M1-5 TE, not submitted)

### Version: v10 (DART LGB + v7 XGB ensemble)

### Actual Time: ~3h on GPU (including 1st failed run)

---

## Step 14: Neural Network (MLP/ResNet) — `scripts/step14_gpu.py`

### Rationale
Reference Vo 2025's 6-layer residual network (same THESi data source). Train a model with fundamentally different prediction patterns from GBDT to increase ensemble diversity. Current LGB-XGB correlation is 0.968 — almost no diversity benefit.

### Architecture Design (Reference: Vo 2025)

```
Input (26 features)
  → BatchNorm
  → Linear(26, 256) → ReLU → Dropout(0.3)
  → Linear(256, 128) → ReLU → Dropout(0.3)
  → Linear(128, 64) → ReLU → Dropout(0.2)
  ──────────────────────────── skip connection ─┐
  → Linear(64, 64) → ReLU → Dropout(0.2)       │
  → Linear(64, 64) → ReLU + skip ←─────────────┘
  → Linear(64, 32) → ReLU → Dropout(0.1)
  → Linear(32, 1) → Sigmoid
```

**Key Design Decisions**:
1. **Sigmoid output**: target is [0,1] ratio, Sigmoid naturally constrains range
2. **Skip connection**: following Vo 2025, residual connection in middle layers
3. **Decreasing Dropout**: 256→128→64→64→64→32→1, Dropout decreases from 0.3 to 0.1
4. **BatchNorm at input**: replaces manual feature standardization (more convenient)

### Training Setup

```python
BATCH_SIZE    = 4096        # large batch for GPU utilization
LR            = 1e-3        # Adam default
WEIGHT_DECAY  = 1e-4        # L2 regularization
EPOCHS        = 30          # with early stopping
ES_PATIENCE   = 5           # stop after 5 epochs no improvement
SCHEDULER     = 'cosine'    # CosineAnnealingLR

# Loss: MSE (consistent with GBDT)
# Eval metric: Spearman correlation (on validation set)
```

### Data Processing
1. **Feature standardization**: StandardScaler fit on train fold, transform on val/test
2. **Sample weight**: via WeightedRandomSampler or weighted loss
3. **DataLoader**: `num_workers=4`, `pin_memory=True`

### 5-Fold CV Structure
```python
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[tr_idx])
    X_va_scaled = scaler.transform(X_train[va_idx])
    X_test_scaled = scaler.transform(X_test)
    
    model = ParkingResNet(n_features=26)
    # ... train loop with early stopping on val Spearman ...
    
    nn_oof[va_idx] = predict(model, X_va_scaled)
    nn_test += predict(model, X_test_scaled) / N_FOLDS
```

### Output Files
- `models/nn_oof_v1.npy`, `models/nn_test_v1.npy`
- Model weights: `models/nn_fold{0-4}.pt`

### Version: nn_v1

### Ensemble Integration

After training NN, run 4-model ensemble weight search:
```
LGB_v7 + XGB_v7 + CB_v4 + NN_v1
```

Even if NN OOF is only 0.60-0.62, low correlation with GBDT (< 0.90) can still boost ensemble.

### Estimated Time: ~2-3h on GPU (including debugging)

---

## Execution Order

```
1. Step 14 (NN) first — ~3h
   Reason: NN needs more debugging time and has higher value (ensemble diversity)
   
2. Step 13 (DART) second — ~1.5h
   Reason: Simple implementation, can finish quickly after NN

3. Final Ensemble — ~30min
   Combine all available models: LGB_v7 + XGB_v7 + DART_LGB + NN
   If v8a/v9 results are good, include those too
```

### Total Estimated Time: ~5h (GPU server)

---

## Key Files

### Files to Read
| File | Purpose |
|------|---------|
| `data/train_features_tier2.parquet` | Training data (6.08M rows, 26 features) |
| `data/test_features_tier2.parquet` | Test data (2.03M rows) |
| `models/lgb_oof_v7.npy` | v7 LGB OOF baseline |
| `models/xgb_oof_v7.npy` | v7 XGB OOF baseline |
| `models/cb_oof_v4.npy` | CB v4 OOF (if available) |
| `scripts/step10_gpu.py` | Template: data loading, CV, ensemble, save patterns |

### Files to Create
| File | Purpose |
|------|---------|
| `scripts/step13_gpu.py` | DART boosting script |
| `scripts/step14_gpu.py` | Neural network script |

---

## Success Criteria

### Step 13 (DART)
- DART LGB OOF ≥ 0.625 (mild drop acceptable)
- DART LGB correlation with v7 XGB < 0.965 (slight diversity increase)
- Ensemble (DART + v7 XGB) OOF ≥ 0.640

### Step 14 (NN)
- NN OOF ≥ 0.58 (significantly lower than GBDT is OK, diversity matters)
- **NN correlation with v7 Ensemble < 0.90** (this is the core metric)
- 4-model Ensemble OOF > v7 Ensemble 0.6429

### If Both Fail
- Abandon new models, focus on v8a/v9 results and report/video preparation
- Negative results from Step 13/14 are still valuable experimental data for the report

---

## Risks & Mitigations

| Risk | Probability | Mitigation |
|------|------------|------------|
| NN OOF too low (< 0.55) | Medium | Try deeper network / tune lr / increase epochs |
| DART training too slow | Low | Reduce to 3000 rounds |
| NN-GBDT correlation still high (> 0.95) | Medium | Switch to pure MLP (no residual) or change loss function |
| GPU memory insufficient | Low | Reduce batch_size or use gradient accumulation |
