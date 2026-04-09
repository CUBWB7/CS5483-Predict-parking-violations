"""
Experiment C: Rank-Based Target Training — LGB + XGB with y_rank target
========================================================================
Run from the project root:
    conda activate parking
    python scripts/step_c_gpu.py

Key idea: Instead of training on raw invalid_ratio values (bimodal: many
0s and 1s due to total_count=1 noise), train on rank(y)/N — a uniform
[0, 1] target. MSE on rank targets directly optimises ranking accuracy,
which is what Spearman correlation measures.

Model params: same Optuna v3 params as v7, + log1p(total_count) weighting.

GPU acceleration:
    - XGBoost: device='cuda'  (XGBoost >= 2.0; change to 'cpu' if unavailable)
    - LightGBM: CPU (n_jobs=-1, all cores)

Files produced:
    models/lgb_rank_oof.npy / lgb_rank_test.npy
    models/xgb_rank_oof.npy / xgb_rank_test.npy
    models/lgb_rank_fold{n}_oof.npy  (fold checkpoints, for resuming)
    models/xgb_rank_fold{n}_oof.npy
    submissions/ensemble_c_rank.csv        (rank-target ensemble only)
    submissions/ensemble_c_combined.csv    (rank + v7, if v7 files present)

Expected runtime on GPU server: ~60-90 min (LGB ~45 min, XGB ~25 min).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')

# ── Tee stdout to log file ────────────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open('step_c_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Experiment C: Rank-Based Target Training')
print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]

y_orig = train_df['invalid_ratio'].values   # original target for evaluation
# Rank target: maps y to [1/N, 1] uniformly, eliminates bimodal noise structure.
# MSE on this target directly minimises rank errors (what Spearman measures).
y_rank = rankdata(y_orig, method='average') / len(y_orig)

sample_weight = np.log1p(train_df['total_count'].values)  # same as v7
m15_mask      = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

print(f'  Train: {train_df.shape}  Test: {test_df.shape}')
print(f'  Features ({len(FEATURES)}): {FEATURES}')
print(f'\n  y_orig  — min: {y_orig.min():.4f}  max: {y_orig.max():.4f}  '
      f'mean: {y_orig.mean():.4f}')
print(f'  y_rank  — min: {y_rank.min():.6f}  max: {y_rank.max():.6f}  '
      f'mean: {y_rank.mean():.4f}')
print(f'  Sample weight — mean: {sample_weight.mean():.3f}  '
      f'max: {sample_weight.max():.3f}')
print(f'  M1-5 rows: {m15_mask.sum():,} / {len(train_df):,}')

# ── Model params (Optuna v3, same as v7) ─────────────────────────────────────
lgb_params = {
    'num_leaves':        100,
    'learning_rate':     0.0564,
    'min_child_samples': 69,
    'reg_lambda':        0.452,
    'reg_alpha':         1.243,
    'feature_fraction':  0.844,
    'bagging_fraction':  0.972,
    'bagging_freq':      5,
    'objective':         'regression',
    'metric':            'l2',
    'boosting_type':     'gbdt',
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      SEED,
    'n_estimators':      10000,
}

xgb_params = {
    'max_depth':            10,
    'learning_rate':        0.0362,
    'min_child_weight':     11,
    'reg_lambda':           1.561,
    'reg_alpha':            1.239,
    'colsample_bytree':     0.951,
    'subsample':            0.948,
    'objective':            'reg:squarederror',
    'eval_metric':          'rmse',
    'tree_method':          'hist',
    'device':               'cuda',   # change to 'cpu' if no GPU
    'n_estimators':         10000,
    'verbosity':            0,
    'random_state':         SEED,
    'n_jobs':               -1,
    'early_stopping_rounds': 150,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ════════════════════════════════════════════════════════════════════════════
# LGB — Rank Target
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'LGB — Rank Target (5-Fold)')
print(f'{"="*60}')

lgb_rank_oof   = np.zeros(len(train_df))
lgb_rank_test  = np.zeros(len(test_df))
lgb_rank_scores     = []
lgb_rank_best_iters = []
t_lgb = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_rank)):
    oof_ckpt  = f'{MODEL_DIR}lgb_rank_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}lgb_rank_fold{fold}_test.npy'

    # Resume from fold checkpoint if available (in case of server restart)
    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        lgb_rank_oof[va_idx]  = np.load(oof_ckpt)
        lgb_rank_test        += np.load(test_ckpt)   # already divided
        fold_rho = spearmanr(y_orig[va_idx], lgb_rank_oof[va_idx])[0]
        lgb_rank_scores.append(fold_rho)
        lgb_rank_best_iters.append(-1)
        print(f'  Fold {fold}: RESUMED from checkpoint  Spearman={fold_rho:.4f}')
        continue

    X_tr = X_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_tr = y_rank[tr_idx]     # rank target for training
    y_va = y_rank[va_idx]     # rank target for early stopping (l2 on ranks)
    w_tr = sample_weight[tr_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=2000),
        ]
    )

    lgb_rank_oof[va_idx] = model.predict(X_va)
    fold_test_pred = model.predict(X_test) / N_FOLDS
    lgb_rank_test += fold_test_pred

    # Evaluate vs ORIGINAL y (Spearman on ranks of predicted ranks vs true labels)
    fold_rho = spearmanr(y_orig[va_idx], lgb_rank_oof[va_idx])[0]
    lgb_rank_scores.append(fold_rho)
    lgb_rank_best_iters.append(model.best_iteration_)

    # Save fold checkpoint (oof only stores the val slice)
    np.save(oof_ckpt, lgb_rank_oof[va_idx])
    np.save(test_ckpt, fold_test_pred)

    elapsed = (time.time() - t_lgb) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
          f'best_iter={model.best_iteration_}  [{elapsed:.1f} min]')

lgb_rho_full = spearmanr(y_orig, lgb_rank_oof)[0]
lgb_rho_m15  = spearmanr(y_orig[m15_mask], lgb_rank_oof[m15_mask])[0]
lgb_total_min = (time.time() - t_lgb) / 60

print(f'\n  LGB Rank — OOF Spearman:   {lgb_rho_full:.4f}  (v7: 0.6336)')
print(f'  LGB Rank — M1-5 Spearman:  {lgb_rho_m15:.4f}  (v7: 0.6428)')
print(f'  Fold scores: {[f"{s:.4f}" for s in lgb_rank_scores]}')
print(f'  Best iters:  {lgb_rank_best_iters}')
print(f'  Total time:  {lgb_total_min:.1f} min')
print(f'  Success criterion (OOF >= 0.635): '
      f'{"✓ PASS" if lgb_rho_full >= 0.635 else "✗ FAIL"}')

np.save(f'{MODEL_DIR}lgb_rank_oof.npy',  lgb_rank_oof)
np.save(f'{MODEL_DIR}lgb_rank_test.npy', lgb_rank_test)
print(f'  Saved: lgb_rank_oof.npy, lgb_rank_test.npy')

# ════════════════════════════════════════════════════════════════════════════
# XGB — Rank Target
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'XGB — Rank Target (5-Fold)')
print(f'{"="*60}')

xgb_rank_oof   = np.zeros(len(train_df))
xgb_rank_test  = np.zeros(len(test_df))
xgb_rank_scores     = []
xgb_rank_best_iters = []
t_xgb = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_rank)):
    oof_ckpt  = f'{MODEL_DIR}xgb_rank_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}xgb_rank_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        xgb_rank_oof[va_idx]  = np.load(oof_ckpt)
        xgb_rank_test        += np.load(test_ckpt)
        fold_rho = spearmanr(y_orig[va_idx], xgb_rank_oof[va_idx])[0]
        xgb_rank_scores.append(fold_rho)
        xgb_rank_best_iters.append(-1)
        print(f'  Fold {fold}: RESUMED from checkpoint  Spearman={fold_rho:.4f}')
        continue

    X_tr = X_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_tr = y_rank[tr_idx]
    y_va = y_rank[va_idx]
    w_tr = sample_weight[tr_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        verbose=2000,
    )

    xgb_rank_oof[va_idx] = model.predict(X_va)
    fold_test_pred = model.predict(X_test) / N_FOLDS
    xgb_rank_test += fold_test_pred

    fold_rho = spearmanr(y_orig[va_idx], xgb_rank_oof[va_idx])[0]
    xgb_rank_scores.append(fold_rho)
    xgb_rank_best_iters.append(model.best_iteration)

    np.save(oof_ckpt, xgb_rank_oof[va_idx])
    np.save(test_ckpt, fold_test_pred)

    elapsed = (time.time() - t_xgb) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
          f'best_iter={model.best_iteration}  [{elapsed:.1f} min]')

xgb_rho_full = spearmanr(y_orig, xgb_rank_oof)[0]
xgb_rho_m15  = spearmanr(y_orig[m15_mask], xgb_rank_oof[m15_mask])[0]
xgb_total_min = (time.time() - t_xgb) / 60

print(f'\n  XGB Rank — OOF Spearman:   {xgb_rho_full:.4f}  (v7: 0.6403)')
print(f'  XGB Rank — M1-5 Spearman:  {xgb_rho_m15:.4f}  (v7: 0.6482)')
print(f'  Fold scores: {[f"{s:.4f}" for s in xgb_rank_scores]}')
print(f'  Best iters:  {xgb_rank_best_iters}')
print(f'  Total time:  {xgb_total_min:.1f} min')
print(f'  Success criterion (OOF >= 0.635): '
      f'{"✓ PASS" if xgb_rho_full >= 0.635 else "✗ FAIL"}')

np.save(f'{MODEL_DIR}xgb_rank_oof.npy',  xgb_rank_oof)
np.save(f'{MODEL_DIR}xgb_rank_test.npy', xgb_rank_test)
print(f'  Saved: xgb_rank_oof.npy, xgb_rank_test.npy')

# ════════════════════════════════════════════════════════════════════════════
# Inter-model correlations
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'Inter-Model Correlations')
print(f'{"="*60}')

corr_lr = spearmanr(lgb_rank_oof, xgb_rank_oof)[0]
print(f'  rank LGB — rank XGB: {corr_lr:.4f}')

# Try to load v7 OOF (fallback to v8a which has identical OOF)
def _load_v7(name_v7, name_v8a):
    for name in [name_v7, name_v8a]:
        path = f'{MODEL_DIR}{name}'
        if os.path.exists(path):
            print(f'  Loaded {name}')
            return np.load(path)
    return None

v7_lgb_oof = _load_v7('lgb_oof_v7.npy', 'lgb_oof_v8a.npy')
v7_xgb_oof = _load_v7('xgb_oof_v7.npy', 'xgb_oof_v8a.npy')

if v7_lgb_oof is not None:
    print(f'  v7 LGB  — rank LGB: {spearmanr(v7_lgb_oof, lgb_rank_oof)[0]:.4f}')
    print(f'  v7 XGB  — rank XGB: {spearmanr(v7_xgb_oof, xgb_rank_oof)[0]:.4f}')
    print(f'  v7 LGB  — v7 XGB:   {spearmanr(v7_lgb_oof, v7_xgb_oof)[0]:.4f}  (ref)')
else:
    print('  (v7 OOF files not found; skipping v7 correlation)')

# ════════════════════════════════════════════════════════════════════════════
# Ensemble A — Rank-Target Only (2 models)
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'Ensemble A: Rank-Target Only (LGB + XGB)')
print(f'{"="*60}')

best_a_rho, best_a_lgb_w = -1, 0.5
for lgb_w in np.arange(0.0, 1.01, 0.05):
    xgb_w = 1.0 - lgb_w
    pred  = lgb_w * lgb_rank_oof + xgb_w * xgb_rank_oof
    rho   = spearmanr(y_orig, pred)[0]
    if rho > best_a_rho:
        best_a_rho   = rho
        best_a_lgb_w = lgb_w

best_a_xgb_w = 1.0 - best_a_lgb_w
ens_a_oof    = best_a_lgb_w * lgb_rank_oof + best_a_xgb_w * xgb_rank_oof
ens_a_m15    = spearmanr(y_orig[m15_mask], ens_a_oof[m15_mask])[0]

print(f'  Best weights: LGB={best_a_lgb_w:.2f}  XGB={best_a_xgb_w:.2f}')
print(f'  OOF Spearman: {best_a_rho:.4f}  (v7: 0.6429)')
print(f'  M1-5 Spearman: {ens_a_m15:.4f}  (v7: 0.6515)')

# Generate submission (rank-target only)
ens_a_test = best_a_lgb_w * lgb_rank_test + best_a_xgb_w * xgb_rank_test
ens_a_test = np.clip(ens_a_test, 0, 1)
sub_a = pd.DataFrame({'id': test_df.index, 'invalid_ratio': ens_a_test})
sub_a.to_csv(f'{SUBMIT_DIR}ensemble_c_rank.csv', index=False)
print(f'  Saved: submissions/ensemble_c_rank.csv  '
      f'({len(ens_a_test):,} rows, range [{ens_a_test.min():.4f}, {ens_a_test.max():.4f}])')

# ════════════════════════════════════════════════════════════════════════════
# Ensemble B — Combined: v7 + Rank-Target (4 models)
# Only possible if v7 predictions are available
# ════════════════════════════════════════════════════════════════════════════
v7_lgb_test_path = next((f'{MODEL_DIR}{n}' for n in ['lgb_test_v7.npy', 'lgb_test_v8a.npy']
                         if os.path.exists(f'{MODEL_DIR}{n}')), None)
v7_xgb_test_path = next((f'{MODEL_DIR}{n}' for n in ['xgb_test_v7.npy', 'xgb_test_v8a.npy']
                         if os.path.exists(f'{MODEL_DIR}{n}')), None)

if (v7_lgb_oof is not None
        and v7_lgb_test_path is not None
        and v7_xgb_test_path is not None):

    print(f'\n{"="*60}')
    print(f'Ensemble B: Combined v7 + Rank-Target (4 models)')
    print(f'{"="*60}')

    v7_lgb_test = np.load(v7_lgb_test_path)
    v7_xgb_test = np.load(v7_xgb_test_path)

    # Grid search over 3 free weights (4th = residual)
    best_b_rho = -1
    best_b_w   = (0.175, 0.325, 0.175, 0.325)  # equal-ish default
    ws = np.arange(0.0, 1.01, 0.05)

    for w1 in ws:           # v7 LGB
        for w2 in ws:       # v7 XGB
            for w3 in ws:   # rank LGB
                w4 = round(1.0 - w1 - w2 - w3, 4)
                if w4 < 0 or w4 > 1:
                    continue
                pred = (w1 * v7_lgb_oof + w2 * v7_xgb_oof
                      + w3 * lgb_rank_oof + w4 * xgb_rank_oof)
                rho  = spearmanr(y_orig, pred)[0]
                if rho > best_b_rho:
                    best_b_rho = rho
                    best_b_w   = (w1, w2, w3, w4)

    w1, w2, w3, w4 = best_b_w
    ens_b_oof = (w1 * v7_lgb_oof + w2 * v7_xgb_oof
               + w3 * lgb_rank_oof + w4 * xgb_rank_oof)
    ens_b_m15 = spearmanr(y_orig[m15_mask], ens_b_oof[m15_mask])[0]

    print(f'  Best weights: v7LGB={w1:.2f}  v7XGB={w2:.2f}  '
          f'rankLGB={w3:.2f}  rankXGB={w4:.2f}')
    print(f'  OOF Spearman: {best_b_rho:.4f}  (v7: 0.6429)')
    print(f'  M1-5 Spearman: {ens_b_m15:.4f}  (v7: 0.6515)')

    ens_b_test = (w1 * v7_lgb_test + w2 * v7_xgb_test
                + w3 * lgb_rank_test + w4 * xgb_rank_test)
    ens_b_test = np.clip(ens_b_test, 0, 1)
    sub_b = pd.DataFrame({'id': test_df.index, 'invalid_ratio': ens_b_test})
    sub_b.to_csv(f'{SUBMIT_DIR}ensemble_c_combined.csv', index=False)
    print(f'  Saved: submissions/ensemble_c_combined.csv  '
          f'({len(ens_b_test):,} rows, '
          f'range [{ens_b_test.min():.4f}, {ens_b_test.max():.4f}])')
else:
    print('\n  (v7 test predictions not found — skipping 4-model ensemble)')
    print('  Download lgb_test_v7.npy + xgb_test_v7.npy if you want the combined submission.')

# ════════════════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'EXPERIMENT C COMPLETE — {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}')
print(f'\n  {"Model":<25} {"OOF":>8} {"M1-5":>8} {"vs v7 OOF":>10}')
print(f'  {"-"*55}')
print(f'  {"LGB (rank target)":<25} {lgb_rho_full:>8.4f} {lgb_rho_m15:>8.4f} {lgb_rho_full-0.6336:>+10.4f}')
print(f'  {"XGB (rank target)":<25} {xgb_rho_full:>8.4f} {xgb_rho_m15:>8.4f} {xgb_rho_full-0.6403:>+10.4f}')
print(f'  {"Ensemble (rank only)":<25} {best_a_rho:>8.4f} {ens_a_m15:>8.4f} {best_a_rho-0.6429:>+10.4f}')
print(f'  {"v7 baseline (ref)":<25} {"0.6429":>8} {"0.6515":>8} {"—":>10}')
print(f'\n  Rank LGB-XGB correlation: {corr_lr:.4f}')
print(f'\n  Stage 1 success (OOF >= 0.635): '
      f'{"✓ PASS (proceed to Stage 2 torchsort if desired)"  if max(lgb_rho_full, xgb_rho_full) >= 0.635 else "✗ FAIL (rank target did not improve; skip Stage 2)"}')
print(f'\n  Files to download for local analysis:')
print(f'    models/lgb_rank_oof.npy  lgb_rank_test.npy')
print(f'    models/xgb_rank_oof.npy  xgb_rank_test.npy')
print(f'    submissions/ensemble_c_rank.csv')
if v7_lgb_oof is not None and os.path.exists(v7_lgb_test_path):
    print(f'    submissions/ensemble_c_combined.csv')
print(f'    step_c_gpu.log')
