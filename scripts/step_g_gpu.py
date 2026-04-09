"""
Experiment G: Pseudo-Labeling with Curriculum Strategy
=======================================================
Run from the project root:
    conda activate parking
    python scripts/step_g_gpu.py

Key idea: The test set is M1-5 only, but the model was trained on the full year.
Adding high-confidence test pseudo-labels to the training set gives the model more
M1-5 examples — directly addressing the distribution shift that causes the
OOF-platform gap.

Baseline: Exp C rank-target LGB + XGB (OOF 0.6464, Platform 0.5698).

Strategy (curriculum):
  Layer 1 — High confidence:  rank_test_avg < 0.02 or > 0.98  (weight=0.7)
  Layer 2 — Medium confidence: 0.10 > or < 0.90, excluding Layer 1  (weight=0.3)
             Only runs if Layer 1 OOF >= 0.6461 (no degradation vs Exp C)

Safety:  if Layer 1 OOF drops > 0.003 vs Exp C (0.6464), abort.

CRITICAL: pseudo-label rows are excluded from LGB/XGB eval_set (valid_sets).
OOF is evaluated ONLY on original train rows.

GPU acceleration:
    - XGBoost: device='cuda'  (change to 'cpu' if no GPU)
    - LightGBM: CPU (n_jobs=-1)

Files produced (Layer 1):
    models/lgb_g1_oof.npy / lgb_g1_test.npy
    models/xgb_g1_oof.npy / xgb_g1_test.npy
    submissions/ensemble_g1.csv

Files produced (Layer 2, if OOF passes):
    models/lgb_g2_oof.npy / lgb_g2_test.npy
    models/xgb_g2_oof.npy / xgb_g2_test.npy
    submissions/ensemble_g2.csv

Expected runtime: ~90-120 min (LGB ~55 min × 2, XGB ~30 min × 2).
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

log_file = open('step_g_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

# Safety threshold: abort if OOF drops below this vs Exp C baseline (0.6464)
OOF_SAFETY_THRESHOLD = 0.6464 - 0.003   # = 0.6434

# Pseudo-label confidence thresholds
LAYER1_HIGH = 0.02   # < 0.02 or > 0.98
LAYER2_MED  = 0.10   # < 0.10 or > 0.90 (excluding Layer 1)
LAYER1_WEIGHT = 0.7
LAYER2_WEIGHT = 0.3

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Experiment G: Pseudo-Labeling with Curriculum Strategy')
print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]

y_orig        = train_df['invalid_ratio'].values
y_rank        = rankdata(y_orig, method='average') / len(y_orig)
sample_weight = np.log1p(train_df['total_count'].values)
m15_mask      = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

print(f'  Train: {train_df.shape}  Test: {test_df.shape}')
print(f'  Features ({len(FEATURES)}): {FEATURES[:5]}...')

# ── Load Exp C rank-target predictions ───────────────────────────────────────
print('\nLoading Exp C rank predictions...')
lgb_rank_oof  = np.load(f'{MODEL_DIR}lgb_rank_oof.npy')
xgb_rank_oof  = np.load(f'{MODEL_DIR}xgb_rank_oof.npy')
lgb_rank_test = np.load(f'{MODEL_DIR}lgb_rank_test.npy')
xgb_rank_test = np.load(f'{MODEL_DIR}xgb_rank_test.npy')

exp_c_rho = spearmanr(y_orig, 0.3 * lgb_rank_oof + 0.7 * xgb_rank_oof)[0]
print(f'  Exp C baseline OOF (LGB=0.3, XGB=0.7): {exp_c_rho:.4f}  (target: 0.6464)')

# ── Model params (identical to Exp C / v7 Optuna) ────────────────────────────
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


def train_with_pseudo(X_train_orig, y_rank_orig, y_orig, sample_weight_orig,
                      X_test, X_pseudo, y_pseudo, w_pseudo,
                      layer_name, ckpt_prefix):
    """
    Train LGB + XGB with pseudo-labeled rows appended.

    Pseudo-label rows are EXCLUDED from eval_set (valid_sets) — OOF is
    measured on original train fold only, matching Exp C evaluation.

    Returns: lgb_oof, lgb_test, xgb_oof, xgb_test
    """
    n_train   = len(X_train_orig)
    n_pseudo  = len(X_pseudo)
    print(f'\n  Pseudo-label set: {n_pseudo:,} rows  '
          f'({n_pseudo/len(X_test)*100:.2f}% of test)')

    # Concatenate pseudo-label features once (reused every fold)
    X_pseudo_reset = X_pseudo.reset_index(drop=True)

    # ── LGB ──────────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'LGB — {layer_name} (5-Fold)')
    print(f'{"="*60}')

    lgb_oof   = np.zeros(n_train)
    lgb_test  = np.zeros(len(X_test))
    lgb_scores     = []
    lgb_best_iters = []
    t_lgb = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig, y_rank_orig)):
        oof_ckpt  = f'{MODEL_DIR}{ckpt_prefix}_lgb_fold{fold}_oof.npy'
        test_ckpt = f'{MODEL_DIR}{ckpt_prefix}_lgb_fold{fold}_test.npy'

        if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
            lgb_oof[va_idx]  = np.load(oof_ckpt)
            lgb_test        += np.load(test_ckpt)
            fold_rho = spearmanr(y_orig[va_idx], lgb_oof[va_idx])[0]
            lgb_scores.append(fold_rho)
            lgb_best_iters.append(-1)
            print(f'  Fold {fold}: RESUMED  Spearman={fold_rho:.4f}')
            continue

        # Augmented training: original fold + pseudo-labels
        X_tr_orig = X_train_orig.iloc[tr_idx]
        y_tr_orig = y_rank_orig[tr_idx]
        w_tr_orig = sample_weight_orig[tr_idx]

        X_tr_aug = pd.concat([X_tr_orig, X_pseudo_reset], ignore_index=True)
        y_tr_aug = np.concatenate([y_tr_orig, y_pseudo])
        w_tr_aug = np.concatenate([w_tr_orig, w_pseudo])

        # Validation: ORIGINAL train fold only (no pseudo-labels in eval_set)
        X_va = X_train_orig.iloc[va_idx]
        y_va = y_rank_orig[va_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr_aug, y_tr_aug,
            eval_set=[(X_va, y_va)],
            sample_weight=w_tr_aug,
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(period=2000),
            ]
        )

        lgb_oof[va_idx]    = model.predict(X_va)
        fold_test_pred      = model.predict(X_test) / N_FOLDS
        lgb_test           += fold_test_pred

        fold_rho = spearmanr(y_orig[va_idx], lgb_oof[va_idx])[0]
        lgb_scores.append(fold_rho)
        lgb_best_iters.append(model.best_iteration_)

        np.save(oof_ckpt, lgb_oof[va_idx])
        np.save(test_ckpt, fold_test_pred)

        elapsed = (time.time() - t_lgb) / 60
        print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
              f'best_iter={model.best_iteration_}  [{elapsed:.1f} min]')

    lgb_rho_full = spearmanr(y_orig, lgb_oof)[0]
    lgb_rho_m15  = spearmanr(y_orig[m15_mask], lgb_oof[m15_mask])[0]
    print(f'\n  LGB {layer_name} — OOF:  {lgb_rho_full:.4f}  (Exp C LGB: 0.6373)')
    print(f'  LGB {layer_name} — M1-5: {lgb_rho_m15:.4f}')
    print(f'  Best iters: {lgb_best_iters}')
    print(f'  Time: {(time.time()-t_lgb)/60:.1f} min')

    np.save(f'{MODEL_DIR}{ckpt_prefix}_lgb_oof.npy',  lgb_oof)
    np.save(f'{MODEL_DIR}{ckpt_prefix}_lgb_test.npy', lgb_test)

    # ── XGB ──────────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'XGB — {layer_name} (5-Fold)')
    print(f'{"="*60}')

    xgb_oof   = np.zeros(n_train)
    xgb_test  = np.zeros(len(X_test))
    xgb_scores     = []
    xgb_best_iters = []
    t_xgb = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig, y_rank_orig)):
        oof_ckpt  = f'{MODEL_DIR}{ckpt_prefix}_xgb_fold{fold}_oof.npy'
        test_ckpt = f'{MODEL_DIR}{ckpt_prefix}_xgb_fold{fold}_test.npy'

        if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
            xgb_oof[va_idx]  = np.load(oof_ckpt)
            xgb_test        += np.load(test_ckpt)
            fold_rho = spearmanr(y_orig[va_idx], xgb_oof[va_idx])[0]
            xgb_scores.append(fold_rho)
            xgb_best_iters.append(-1)
            print(f'  Fold {fold}: RESUMED  Spearman={fold_rho:.4f}')
            continue

        X_tr_orig = X_train_orig.iloc[tr_idx]
        y_tr_orig = y_rank_orig[tr_idx]
        w_tr_orig = sample_weight_orig[tr_idx]

        X_tr_aug = pd.concat([X_tr_orig, X_pseudo_reset], ignore_index=True)
        y_tr_aug = np.concatenate([y_tr_orig, y_pseudo])
        w_tr_aug = np.concatenate([w_tr_orig, w_pseudo])

        X_va = X_train_orig.iloc[va_idx]
        y_va = y_rank_orig[va_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr_aug, y_tr_aug,
            eval_set=[(X_va, y_va)],
            sample_weight=w_tr_aug,
            verbose=2000,
        )

        xgb_oof[va_idx]    = model.predict(X_va)
        fold_test_pred      = model.predict(X_test) / N_FOLDS
        xgb_test           += fold_test_pred

        fold_rho = spearmanr(y_orig[va_idx], xgb_oof[va_idx])[0]
        xgb_scores.append(fold_rho)
        xgb_best_iters.append(model.best_iteration)

        np.save(oof_ckpt, xgb_oof[va_idx])
        np.save(test_ckpt, fold_test_pred)

        elapsed = (time.time() - t_xgb) / 60
        print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
              f'best_iter={model.best_iteration}  [{elapsed:.1f} min]')

    xgb_rho_full = spearmanr(y_orig, xgb_oof)[0]
    xgb_rho_m15  = spearmanr(y_orig[m15_mask], xgb_oof[m15_mask])[0]
    print(f'\n  XGB {layer_name} — OOF:  {xgb_rho_full:.4f}  (Exp C XGB: 0.6430)')
    print(f'  XGB {layer_name} — M1-5: {xgb_rho_m15:.4f}')
    print(f'  Best iters: {xgb_best_iters}')
    print(f'  Time: {(time.time()-t_xgb)/60:.1f} min')

    np.save(f'{MODEL_DIR}{ckpt_prefix}_xgb_oof.npy',  xgb_oof)
    np.save(f'{MODEL_DIR}{ckpt_prefix}_xgb_test.npy', xgb_test)

    # Ensemble OOF (quick check)
    best_rho, best_w = -1, 0.3
    for w in np.arange(0.0, 1.01, 0.05):
        r = spearmanr(y_orig, w * lgb_oof + (1-w) * xgb_oof)[0]
        if r > best_rho:
            best_rho, best_w = r, w
    ens_oof  = best_w * lgb_oof + (1-best_w) * xgb_oof
    ens_m15  = spearmanr(y_orig[m15_mask], ens_oof[m15_mask])[0]

    print(f'\n  Ensemble OOF (LGB={best_w:.2f}, XGB={1-best_w:.2f}): {best_rho:.4f}  (Exp C: 0.6464)')
    print(f'  Ensemble M1-5: {ens_m15:.4f}')

    return lgb_oof, lgb_test, xgb_oof, xgb_test, best_rho, best_w


# ════════════════════════════════════════════════════════════════════════════
# Layer 1 — High confidence pseudo-labels (< 0.02 or > 0.98)
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('Layer 1: High-confidence pseudo-labels')
print(f'{"="*60}')

rank_test_avg = (lgb_rank_test + xgb_rank_test) / 2
layer1_mask   = (rank_test_avg < LAYER1_HIGH) | (rank_test_avg > (1 - LAYER1_HIGH))
layer1_idx    = np.where(layer1_mask)[0]

print(f'  rank_test_avg range: [{rank_test_avg.min():.4f}, {rank_test_avg.max():.4f}]')
print(f'  Layer 1 threshold: < {LAYER1_HIGH} or > {1-LAYER1_HIGH}')
print(f'  Layer 1 pseudo-labels: {len(layer1_idx):,} / {len(test_df):,} '
      f'({len(layer1_idx)/len(test_df)*100:.2f}%)')
print(f'    Low-confidence (%<0.02): {(rank_test_avg < LAYER1_HIGH).sum():,}')
print(f'    High-confidence (>0.98): {(rank_test_avg > 1-LAYER1_HIGH).sum():,}')

X_pseudo_l1 = X_test.iloc[layer1_idx]
y_pseudo_l1 = rank_test_avg[layer1_idx]
w_pseudo_l1 = np.full(len(layer1_idx), LAYER1_WEIGHT)

lgb_g1_oof, lgb_g1_test, xgb_g1_oof, xgb_g1_test, g1_oof_rho, g1_lgb_w = \
    train_with_pseudo(
        X_train, y_rank, y_orig, sample_weight,
        X_test, X_pseudo_l1, y_pseudo_l1, w_pseudo_l1,
        layer_name='Layer 1',
        ckpt_prefix='g1',
    )

# ── Safety check ──────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('Safety Check')
print(f'{"="*60}')
print(f'  Layer 1 OOF: {g1_oof_rho:.4f}')
print(f'  Exp C OOF:   0.6464')
print(f'  Delta:       {g1_oof_rho - 0.6464:+.4f}')
print(f'  Threshold:   >= {OOF_SAFETY_THRESHOLD:.4f}')
layer1_pass = g1_oof_rho >= OOF_SAFETY_THRESHOLD
print(f'  Status:      {"✓ PASS" if layer1_pass else "✗ FAIL — aborting Layer 2"}')

# Generate Layer 1 submission
g1_test = g1_lgb_w * lgb_g1_test + (1-g1_lgb_w) * xgb_g1_test
g1_test = np.clip(g1_test, 0, 1)
sub_g1  = pd.DataFrame({'id': test_df.index, 'invalid_ratio': g1_test})
sub_g1.to_csv(f'{SUBMIT_DIR}ensemble_g1.csv', index=False)
print(f'\n  Saved: submissions/ensemble_g1.csv  '
      f'({len(g1_test):,} rows, range [{g1_test.min():.4f}, {g1_test.max():.4f}])')


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — Medium confidence (only if Layer 1 passes)
# ════════════════════════════════════════════════════════════════════════════
if not layer1_pass:
    print('\nLayer 2 skipped (Layer 1 failed safety check).')
else:
    print(f'\n{"="*60}')
    print('Layer 2: Medium-confidence pseudo-labels')
    print(f'{"="*60}')

    # Re-predict test using Layer 1 models
    # Use the already-trained ensemble to select medium-confidence rows
    layer1_set = set(layer1_idx.tolist())
    layer2_mask = (
        ((rank_test_avg < LAYER2_MED) | (rank_test_avg > (1 - LAYER2_MED)))
        & ~layer1_mask
    )
    layer2_idx = np.where(layer2_mask)[0]

    print(f'  Layer 2 threshold: < {LAYER2_MED} or > {1-LAYER2_MED}  (excluding Layer 1)')
    print(f'  Layer 2 pseudo-labels: {len(layer2_idx):,} / {len(test_df):,} '
          f'({len(layer2_idx)/len(test_df)*100:.2f}%)')

    # Combined pseudo-labels: Layer 1 (w=0.7) + Layer 2 (w=0.3)
    all_pseudo_idx = np.concatenate([layer1_idx, layer2_idx])
    all_pseudo_y   = rank_test_avg[all_pseudo_idx]
    all_pseudo_w   = np.concatenate([
        np.full(len(layer1_idx), LAYER1_WEIGHT),
        np.full(len(layer2_idx), LAYER2_WEIGHT),
    ])
    X_pseudo_all = X_test.iloc[all_pseudo_idx]

    lgb_g2_oof, lgb_g2_test, xgb_g2_oof, xgb_g2_test, g2_oof_rho, g2_lgb_w = \
        train_with_pseudo(
            X_train, y_rank, y_orig, sample_weight,
            X_test, X_pseudo_all, all_pseudo_y, all_pseudo_w,
            layer_name='Layer 2',
            ckpt_prefix='g2',
        )

    print(f'\n  Layer 2 OOF: {g2_oof_rho:.4f}  (Layer 1: {g1_oof_rho:.4f})')
    print(f'  Delta L2 vs L1: {g2_oof_rho - g1_oof_rho:+.4f}')

    g2_test = g2_lgb_w * lgb_g2_test + (1-g2_lgb_w) * xgb_g2_test
    g2_test = np.clip(g2_test, 0, 1)
    sub_g2  = pd.DataFrame({'id': test_df.index, 'invalid_ratio': g2_test})
    sub_g2.to_csv(f'{SUBMIT_DIR}ensemble_g2.csv', index=False)
    print(f'  Saved: submissions/ensemble_g2.csv  '
          f'({len(g2_test):,} rows, range [{g2_test.min():.4f}, {g2_test.max():.4f}])')


# ════════════════════════════════════════════════════════════════════════════
# Inter-model correlations
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('Inter-Model Correlations')
print(f'{"="*60}')

corr_g1 = spearmanr(lgb_g1_oof, xgb_g1_oof)[0]
print(f'  Layer 1 LGB — Layer 1 XGB: {corr_g1:.4f}')
corr_g1_c = spearmanr(lgb_g1_oof, lgb_rank_oof)[0]
print(f'  Layer 1 LGB — Exp C LGB:   {corr_g1_c:.4f}')
corr_xg1_c = spearmanr(xgb_g1_oof, xgb_rank_oof)[0]
print(f'  Layer 1 XGB — Exp C XGB:   {corr_xg1_c:.4f}')


# ════════════════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'EXPERIMENT G COMPLETE — {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}')

lgb_g1_rho  = spearmanr(y_orig, lgb_g1_oof)[0]
xgb_g1_rho  = spearmanr(y_orig, xgb_g1_oof)[0]
lgb_g1_m15  = spearmanr(y_orig[m15_mask], lgb_g1_oof[m15_mask])[0]
xgb_g1_m15  = spearmanr(y_orig[m15_mask], xgb_g1_oof[m15_mask])[0]

print(f'\n  {"Model":<30} {"OOF":>8} {"M1-5":>8} {"vs Exp C":>10}')
print(f'  {"-"*60}')
print(f'  {"Exp C LGB (baseline)":<30} {"0.6373":>8} {"—":>8} {"—":>10}')
print(f'  {"Exp C XGB (baseline)":<30} {"0.6430":>8} {"—":>8} {"—":>10}')
print(f'  {"Exp C Ensemble":<30} {"0.6464":>8} {"—":>8} {"—":>10}')
print(f'  {"-"*60}')
print(f'  {"Layer 1 LGB":<30} {lgb_g1_rho:>8.4f} {lgb_g1_m15:>8.4f} '
      f'{lgb_g1_rho-0.6373:>+10.4f}')
print(f'  {"Layer 1 XGB":<30} {xgb_g1_rho:>8.4f} {xgb_g1_m15:>8.4f} '
      f'{xgb_g1_rho-0.6430:>+10.4f}')
print(f'  {"Layer 1 Ensemble":<30} {g1_oof_rho:>8.4f} {"—":>8} '
      f'{g1_oof_rho-0.6464:>+10.4f}')

if layer1_pass and 'g2_oof_rho' in dir():
    lgb_g2_rho = spearmanr(y_orig, lgb_g2_oof)[0]
    xgb_g2_rho = spearmanr(y_orig, xgb_g2_oof)[0]
    lgb_g2_m15 = spearmanr(y_orig[m15_mask], lgb_g2_oof[m15_mask])[0]
    xgb_g2_m15 = spearmanr(y_orig[m15_mask], xgb_g2_oof[m15_mask])[0]
    print(f'  {"Layer 2 LGB":<30} {lgb_g2_rho:>8.4f} {lgb_g2_m15:>8.4f} '
          f'{lgb_g2_rho-0.6373:>+10.4f}')
    print(f'  {"Layer 2 XGB":<30} {xgb_g2_rho:>8.4f} {xgb_g2_m15:>8.4f} '
          f'{xgb_g2_rho-0.6430:>+10.4f}')
    print(f'  {"Layer 2 Ensemble":<30} {g2_oof_rho:>8.4f} {"—":>8} '
          f'{g2_oof_rho-0.6464:>+10.4f}')

print(f'\n  Layer 1 safety check: {"✓ PASS" if layer1_pass else "✗ FAIL"}')
print(f'\n  Files to download for local notebook analysis:')
print(f'    models/lgb_g1_oof.npy  lgb_g1_test.npy')
print(f'    models/xgb_g1_oof.npy  xgb_g1_test.npy')
print(f'    submissions/ensemble_g1.csv')
if layer1_pass and 'g2_oof_rho' in dir():
    print(f'    models/lgb_g2_oof.npy  lgb_g2_test.npy')
    print(f'    models/xgb_g2_oof.npy  xgb_g2_test.npy')
    print(f'    submissions/ensemble_g2.csv')
print(f'    step_g_gpu.log')
