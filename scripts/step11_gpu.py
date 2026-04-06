"""
Step 11: M1-5 Focused TE + Training — v8a (11A) + v8b (11B)
============================================================
Run from the project root:
    conda activate parking
    python scripts/step11_gpu.py

Two variants, both targeting the OOF–Platform gap (currently 0.079):

  11A (v8a): Train on ALL data (identical to v7), save model objects,
             then re-predict test using TE computed from M1-5 rows only.
             OOF is unchanged (same as v7). Only platform can evaluate.

  11B (v8b): Train on M1-5 rows only (2.47M). K-fold TE also restricted
             to M1-5. Risk: 60% data loss may hurt generalization.

Progress is printed to stdout and step11_gpu.log.
Model objects saved to models/{lgb,xgb}_v8a_fold{n}.{txt,json}.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, ks_2samp
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

log_file = open('step11_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Step 11 started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

TARGET       = 'invalid_ratio'
EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]
y            = train_df[TARGET]

print(f'  Train: {train_df.shape}, Test: {test_df.shape}')
print(f'  Features (26): {len(FEATURES)}')

m1_5_mask = train_df['month_of_year'].isin([1, 2, 3, 4, 5])
print(f'  M1-5 rows: {m1_5_mask.sum():,}  |  M6-12 rows: {(~m1_5_mask).sum():,}')

# ── Load v7 baseline ──────────────────────────────────────────────────────────
print('\nLoading v7 baseline...')
lgb_oof_v7  = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7  = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7 = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7 = np.load(f'{MODEL_DIR}xgb_test_v7.npy')
cb_oof_v4   = np.load(f'{MODEL_DIR}cb_oof_v4.npy')
cb_test_v4  = np.load(f'{MODEL_DIR}cb_test_v4.npy')

lgb_oof_v7_rho = spearmanr(y, lgb_oof_v7)[0]
xgb_oof_v7_rho = spearmanr(y, xgb_oof_v7)[0]

# Best ensemble v7 weights (reproduce grid search from step10)
best_rho_v7 = 0.0
best_w_v7   = (0.0, 1.0, 0.0)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        blend = w1*lgb_oof_v7 + w2*xgb_oof_v7 + w3*cb_oof_v4
        rho = spearmanr(y, blend)[0]
        if rho > best_rho_v7:
            best_rho_v7 = rho
            best_w_v7   = (round(w1,2), round(w2,2), round(w3,2))

w1_v7, w2_v7, w3_v7 = best_w_v7
ens_v7_oof     = w1_v7*lgb_oof_v7 + w2_v7*xgb_oof_v7 + w3_v7*cb_oof_v4
ens_v7_m1_5    = spearmanr(y[m1_5_mask], ens_v7_oof[m1_5_mask])[0]

print(f'  LGB v7 OOF:      {lgb_oof_v7_rho:.4f}')
print(f'  XGB v7 OOF:      {xgb_oof_v7_rho:.4f}')
print(f'  Ensemble v7 OOF: {best_rho_v7:.4f}  weights={best_w_v7}')
print(f'  Ensemble v7 M1-5 OOF: {ens_v7_m1_5:.4f}  (platform proxy)')

# ── Recompute M1-5 TE for test (diagnostic + 11A input) ──────────────────────
#
# Original: test TE computed from all 12 months (full_stats, smooth=30/50).
# 11A:      recompute from M1-5 rows only, same smooth params.
#           The plan uses smooth=100 / smooth=150 for the M1-5 subset to
#           maintain stability with fewer rows (~2.47M vs ~6.08M total).
#
print('\n=== KS Distribution Diagnostic ===\n')

train_m1_5 = train_df[m1_5_mask].copy()
global_mean_full = train_df['invalid_ratio'].mean()
global_mean_m1_5 = train_m1_5['invalid_ratio'].mean()

# --- grid_te (smooth=100 for M1-5 test TE; original used smooth=30) ----------
full_stats_grid = train_df.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
full_stats_grid_m1_5 = train_m1_5.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])

smooth_grid_orig = 30
smooth_grid_m1_5 = 100

# Original test TE encoding
enc_grid_orig = (
    full_stats_grid['count'] * full_stats_grid['mean']
    + smooth_grid_orig * global_mean_full
) / (full_stats_grid['count'] + smooth_grid_orig)
test_grid_te_orig = test_df['grid_id'].map(enc_grid_orig).fillna(global_mean_full)

# M1-5 test TE encoding
enc_grid_m1_5 = (
    full_stats_grid_m1_5['count'] * full_stats_grid_m1_5['mean']
    + smooth_grid_m1_5 * global_mean_m1_5
) / (full_stats_grid_m1_5['count'] + smooth_grid_m1_5)
test_grid_te_m1_5 = test_df['grid_id'].map(enc_grid_m1_5).fillna(global_mean_m1_5)

ks_grid = ks_2samp(test_grid_te_orig.values, test_grid_te_m1_5.values)
print(f'  grid_te  KS stat (orig vs M1-5): {ks_grid.statistic:.4f}  p={ks_grid.pvalue:.4f}')
print(f'    orig  mean={test_grid_te_orig.mean():.4f}  std={test_grid_te_orig.std():.4f}')
print(f'    M1-5  mean={test_grid_te_m1_5.mean():.4f}  std={test_grid_te_m1_5.std():.4f}')

# --- grid_period_te (smooth=150 for M1-5; original used smooth=50) -----------
full_stats_gp = train_df.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
full_stats_gp_m1_5 = train_m1_5.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])

smooth_gp_orig = 50
smooth_gp_m1_5 = 150

enc_gp_orig = (
    full_stats_gp['count'] * full_stats_gp['mean']
    + smooth_gp_orig * global_mean_full
) / (full_stats_gp['count'] + smooth_gp_orig)
test_gp_te_orig = test_df['grid_period'].map(enc_gp_orig).fillna(global_mean_full)

enc_gp_m1_5 = (
    full_stats_gp_m1_5['count'] * full_stats_gp_m1_5['mean']
    + smooth_gp_m1_5 * global_mean_m1_5
) / (full_stats_gp_m1_5['count'] + smooth_gp_m1_5)
# For unseen grid_periods in M1-5 stats, fall back to grid_te M1-5
test_gp_te_m1_5 = test_df['grid_period'].map(enc_gp_m1_5)
missing_mask = test_gp_te_m1_5.isna()
test_gp_te_m1_5[missing_mask] = test_grid_te_m1_5[missing_mask]  # fallback
test_gp_te_m1_5 = test_gp_te_m1_5.fillna(global_mean_m1_5)

ks_gp = ks_2samp(test_gp_te_orig.values, test_gp_te_m1_5.values)
print(f'\n  grid_period_te KS stat (orig vs M1-5): {ks_gp.statistic:.4f}  p={ks_gp.pvalue:.4f}')
print(f'    orig  mean={test_gp_te_orig.mean():.4f}  std={test_gp_te_orig.std():.4f}')
print(f'    M1-5  mean={test_gp_te_m1_5.mean():.4f}  std={test_gp_te_m1_5.std():.4f}')
print(f'    (M1-5 fallback to grid_te for {missing_mask.sum():,} rows)')

if ks_grid.statistic < 0.005 and ks_gp.statistic < 0.005:
    print('\n  KS near 0 — TE shift is NOT from month mismatch. 11A unlikely to help.')
else:
    print('\n  KS > 0 — M1-5 TE differs from full-data TE. 11A may reduce test shift.')

# ── Shared hyperparameters (identical to v7) ──────────────────────────────────
lgb_params = {
    'num_leaves':        100,
    'learning_rate':     0.0564,
    'min_child_samples': 69,
    'reg_lambda':        0.452,
    'reg_alpha':         1.243,
    'feature_fraction':  0.844,
    'bagging_fraction':  0.972,
    'objective':         'regression',
    'metric':            'l2',
    'boosting_type':     'gbdt',
    'bagging_freq':      5,
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      SEED,
    'n_estimators':      10000,
}

xgb_params = {
    'max_depth':             10,
    'learning_rate':         0.0362,
    'min_child_weight':      11,
    'reg_lambda':            1.561,
    'reg_alpha':             1.239,
    'colsample_bytree':      0.951,
    'subsample':             0.948,
    'objective':             'reg:squarederror',
    'tree_method':           'hist',
    'device':                'cuda',   # change to 'cpu' if no GPU
    'random_state':          SEED,
    'n_jobs':                -1,
    'verbosity':             0,
    'n_estimators':          10000,
    'early_stopping_rounds': 150,
}

# ── Helper: build a test feature matrix with replaced TE columns ──────────────
def build_test_with_te(test_df, features, new_grid_te, new_gp_te):
    """Return a copy of test features with grid_te / grid_period_te replaced."""
    X_test = test_df[features].copy()
    if 'grid_te' in X_test.columns:
        X_test['grid_te'] = new_grid_te.values
    if 'grid_period_te' in X_test.columns:
        X_test['grid_period_te'] = new_gp_te.values
    return X_test

# ──────────────────────────────────────────────────────────────────────────────
# VARIANT 11A — Train on all data, M1-5 TE for test predictions only
# ──────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('11A: Full-data training + M1-5 TE for test (v8a)')
print(f'{"="*60}\n')

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
sample_weight_all = np.log1p(train_df['total_count'].values)

X_all      = train_df[FEATURES]
X_test_all = test_df[FEATURES]
X_test_m1_5_te = build_test_with_te(test_df, FEATURES,
                                     test_grid_te_m1_5,
                                     test_gp_te_m1_5)

# --- LGB v8a -----------------------------------------------------------------
lgb_oof_v8a  = np.zeros(len(train_df))
lgb_test_v8a = np.zeros(len(test_df))  # predictions on M1-5 TE test
lgb_scores_v8a = []
lgb_best_iters_v8a = []

print(f'=== LGB v8a: full data + M1-5 TE test ({N_FOLDS}-Fold) ===\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
    oof_ckpt   = f'{MODEL_DIR}lgb_v8a_fold{fold}_oof.npy'
    test_ckpt  = f'{MODEL_DIR}lgb_v8a_fold{fold}_test.npy'
    model_path = f'{MODEL_DIR}lgb_v8a_fold{fold}.txt'

    # Resume from checkpoint if both OOF and model object are saved
    if (os.path.exists(oof_ckpt) and os.path.exists(test_ckpt)
            and os.path.exists(model_path)):
        lgb_oof_v8a[va_idx] = np.load(oof_ckpt)
        lgb_test_v8a += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], lgb_oof_v8a[va_idx])[0]
        lgb_scores_v8a.append(fold_rho)
        lgb_best_iters_v8a.append(-1)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_all.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_all.iloc[va_idx], y.iloc[va_idx]
    w_tr       = sample_weight_all[tr_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=2000),
        ],
    )

    va_pred = model.predict(X_va)
    # Predict on test using M1-5 TE features
    test_pred = model.predict(X_test_m1_5_te)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, test_pred)
    # Save model object so we can re-predict without retraining
    model.booster_.save_model(model_path)

    lgb_oof_v8a[va_idx] = va_pred
    lgb_test_v8a       += test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    lgb_scores_v8a.append(fold_rho)
    lgb_best_iters_v8a.append(model.best_iteration_)
    es_flag = 'ES' if model.best_iteration_ < 9850 else 'full'
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration_} ({es_flag}), '
          f'elapsed={elapsed_fold:.1f}min')

lgb_v8a_rho  = spearmanr(y, lgb_oof_v8a)[0]
lgb_v8a_m1_5 = spearmanr(y[m1_5_mask], lgb_oof_v8a[m1_5_mask])[0]
print(f'\nLGB v8a OOF:  {lgb_v8a_rho:.4f}  (v7: {lgb_oof_v7_rho:.4f},'
      f' delta: {lgb_v8a_rho - lgb_oof_v7_rho:+.4f})')
print(f'LGB v8a M1-5: {lgb_v8a_m1_5:.4f}')
print(f'Best iters:   {lgb_best_iters_v8a}')
print(f'LGB v8a time: {(time.time()-t0)/60:.1f} min')

# --- XGB v8a -----------------------------------------------------------------
xgb_oof_v8a  = np.zeros(len(train_df))
xgb_test_v8a = np.zeros(len(test_df))
xgb_scores_v8a = []
xgb_best_iters_v8a = []

print(f'\n=== XGB v8a: full data + M1-5 TE test ({N_FOLDS}-Fold, GPU) ===\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
    oof_ckpt   = f'{MODEL_DIR}xgb_v8a_fold{fold}_oof.npy'
    test_ckpt  = f'{MODEL_DIR}xgb_v8a_fold{fold}_test.npy'
    model_path = f'{MODEL_DIR}xgb_v8a_fold{fold}.json'

    if (os.path.exists(oof_ckpt) and os.path.exists(test_ckpt)
            and os.path.exists(model_path)):
        xgb_oof_v8a[va_idx] = np.load(oof_ckpt)
        xgb_test_v8a += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], xgb_oof_v8a[va_idx])[0]
        xgb_scores_v8a.append(fold_rho)
        xgb_best_iters_v8a.append(-1)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_all.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_all.iloc[va_idx], y.iloc[va_idx]
    w_tr       = sample_weight_all[tr_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        verbose=2000,
    )

    va_pred   = model.predict(X_va)
    test_pred = model.predict(X_test_m1_5_te)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, test_pred)
    model.save_model(model_path)

    xgb_oof_v8a[va_idx] = va_pred
    xgb_test_v8a       += test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    xgb_scores_v8a.append(fold_rho)
    xgb_best_iters_v8a.append(model.best_iteration)
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration}, elapsed={elapsed_fold:.1f}min')

xgb_v8a_rho  = spearmanr(y, xgb_oof_v8a)[0]
xgb_v8a_m1_5 = spearmanr(y[m1_5_mask], xgb_oof_v8a[m1_5_mask])[0]
print(f'\nXGB v8a OOF:  {xgb_v8a_rho:.4f}  (v7: {xgb_oof_v7_rho:.4f},'
      f' delta: {xgb_v8a_rho - xgb_oof_v7_rho:+.4f})')
print(f'XGB v8a M1-5: {xgb_v8a_m1_5:.4f}')
print(f'Best iters:   {xgb_best_iters_v8a}')
print(f'XGB v8a time: {(time.time()-t0)/60:.1f} min')

# --- Save + Ensemble v8a -----------------------------------------------------
np.save(f'{MODEL_DIR}lgb_oof_v8a.npy',  lgb_oof_v8a)
np.save(f'{MODEL_DIR}lgb_test_v8a.npy', lgb_test_v8a)
np.save(f'{MODEL_DIR}xgb_oof_v8a.npy',  xgb_oof_v8a)
np.save(f'{MODEL_DIR}xgb_test_v8a.npy', xgb_test_v8a)
print(f'\nSaved: lgb/xgb _oof/test_v8a.npy')

best_rho_v8a = 0.0
best_w_v8a   = (0.0, 1.0, 0.0)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        blend = w1*lgb_oof_v8a + w2*xgb_oof_v8a + w3*cb_oof_v4
        rho = spearmanr(y, blend)[0]
        if rho > best_rho_v8a:
            best_rho_v8a = rho
            best_w_v8a   = (round(w1,2), round(w2,2), round(w3,2))

w1_8a, w2_8a, w3_8a = best_w_v8a
ens_oof_v8a  = w1_8a*lgb_oof_v8a  + w2_8a*xgb_oof_v8a  + w3_8a*cb_oof_v4
ens_test_v8a = w1_8a*lgb_test_v8a + w2_8a*xgb_test_v8a + w3_8a*cb_test_v4
ens_test_v8a = np.clip(ens_test_v8a, 0, 1)

ens_v8a_m1_5 = spearmanr(y[m1_5_mask], ens_oof_v8a[m1_5_mask])[0]

print(f'\n=== Ensemble v8a (11A) ===')
print(f'  Weights: LGB={w1_8a}, XGB={w2_8a}, CB={w3_8a}')
print(f'  OOF:  {best_rho_v8a:.4f}  (v7: {best_rho_v7:.4f},'
      f' delta: {best_rho_v8a - best_rho_v7:+.4f})')
print(f'  M1-5: {ens_v8a_m1_5:.4f}  (v7: {ens_v7_m1_5:.4f},'
      f' delta: {ens_v8a_m1_5 - ens_v7_m1_5:+.4f})')

sub_v8a = pd.DataFrame({'invalid_ratio': ens_test_v8a})
sub_v8a.to_csv(f'{SUBMIT_DIR}ensemble_v8a.csv', index=True, index_label='')
print(f'\nSaved: {SUBMIT_DIR}ensemble_v8a.csv')

# Pre-submit checks
print(f'  NaN: {np.isnan(ens_test_v8a).sum()}  '
      f'Rows: {len(ens_test_v8a)}  '
      f'Range: [{ens_test_v8a.min():.4f}, {ens_test_v8a.max():.4f}]')

# ──────────────────────────────────────────────────────────────────────────────
# VARIANT 11B — Train on M1-5 only with M1-5 K-fold TE
# ──────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('11B: M1-5 only training + M1-5 K-fold TE (v8b)')
print(f'{"="*60}\n')

# Filter to M1-5 rows
train_m1_5_df = train_df[m1_5_mask].copy().reset_index(drop=True)
y_m1_5        = train_m1_5_df[TARGET]
sample_weight_m1_5 = np.log1p(train_m1_5_df['total_count'].values)

print(f'  M1-5 training set: {train_m1_5_df.shape}')

# --- K-fold TE for M1-5 training data ----------------------------------------
# Must recompute TE within M1-5-only folds to prevent leakage.
# Use same smooth params as original feature engineering (30 / 50).
print('  Computing M1-5 K-fold TE...')

global_mean_m1_5 = y_m1_5.mean()

kf_te = KFold(n_splits=5, shuffle=True, random_state=SEED)
grid_te_m1_5_oof      = np.zeros(len(train_m1_5_df))
grid_period_te_m1_5_oof = np.zeros(len(train_m1_5_df))

for fold_idx, (te_tr_idx, te_va_idx) in enumerate(kf_te.split(train_m1_5_df)):
    # grid_te (smooth=30)
    fold_train = train_m1_5_df.iloc[te_tr_idx]
    stats_g = fold_train.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
    enc_g = (stats_g['count'] * stats_g['mean'] + 30 * global_mean_m1_5) / (stats_g['count'] + 30)
    grid_te_m1_5_oof[te_va_idx] = (
        train_m1_5_df.iloc[te_va_idx]['grid_id'].map(enc_g).fillna(global_mean_m1_5).values
    )

    # grid_period_te (smooth=50); fallback to grid_te for unseen grid_periods
    stats_gp = fold_train.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
    enc_gp = (stats_gp['count'] * stats_gp['mean'] + 50 * global_mean_m1_5) / (stats_gp['count'] + 50)
    gp_vals = train_m1_5_df.iloc[te_va_idx]['grid_period'].map(enc_gp)
    missing = gp_vals.isna()
    gp_vals[missing] = grid_te_m1_5_oof[te_va_idx][missing]  # fallback
    grid_period_te_m1_5_oof[te_va_idx] = gp_vals.fillna(global_mean_m1_5).values

# Inject new TE into training features (replace existing tier2 TE columns)
train_m1_5_df = train_m1_5_df.copy()
train_m1_5_df['grid_te']        = grid_te_m1_5_oof
train_m1_5_df['grid_period_te'] = grid_period_te_m1_5_oof

print(f'  grid_te range:        [{grid_te_m1_5_oof.min():.4f}, {grid_te_m1_5_oof.max():.4f}]')
print(f'  grid_period_te range: [{grid_period_te_m1_5_oof.min():.4f}, {grid_period_te_m1_5_oof.max():.4f}]')

# Test TE: full M1-5 statistics (same smooth as orig: 30/50)
full_g_m1_5 = train_m1_5_df.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
enc_g_full = (full_g_m1_5['count'] * full_g_m1_5['mean'] + 30 * global_mean_m1_5) / (full_g_m1_5['count'] + 30)
test_grid_te_11b = test_df['grid_id'].map(enc_g_full).fillna(global_mean_m1_5)

full_gp_m1_5 = train_m1_5_df.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
enc_gp_full = (full_gp_m1_5['count'] * full_gp_m1_5['mean'] + 50 * global_mean_m1_5) / (full_gp_m1_5['count'] + 50)
test_gp_te_11b = test_df['grid_period'].map(enc_gp_full)
missing_11b = test_gp_te_11b.isna()
test_gp_te_11b[missing_11b] = test_grid_te_11b[missing_11b]
test_gp_te_11b = test_gp_te_11b.fillna(global_mean_m1_5)

X_m1_5     = train_m1_5_df[FEATURES]
X_test_11b = build_test_with_te(test_df, FEATURES, test_grid_te_11b, test_gp_te_11b)

# --- LGB v8b -----------------------------------------------------------------
kf_11b = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
lgb_oof_v8b  = np.zeros(len(train_m1_5_df))
lgb_test_v8b = np.zeros(len(test_df))
lgb_scores_v8b = []

print(f'\n=== LGB v8b: M1-5 only + M1-5 TE ({N_FOLDS}-Fold) ===\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf_11b.split(X_m1_5)):
    oof_ckpt  = f'{MODEL_DIR}lgb_v8b_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}lgb_v8b_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        lgb_oof_v8b[va_idx] = np.load(oof_ckpt)
        lgb_test_v8b += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y_m1_5.iloc[va_idx], lgb_oof_v8b[va_idx])[0]
        lgb_scores_v8b.append(fold_rho)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_m1_5.iloc[tr_idx], y_m1_5.iloc[tr_idx]
    X_va, y_va = X_m1_5.iloc[va_idx], y_m1_5.iloc[va_idx]
    w_tr       = sample_weight_m1_5[tr_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=2000),
        ],
    )

    va_pred   = model.predict(X_va)
    test_pred = model.predict(X_test_11b)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, test_pred)

    lgb_oof_v8b[va_idx] = va_pred
    lgb_test_v8b       += test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    lgb_scores_v8b.append(fold_rho)
    es_flag = 'ES' if model.best_iteration_ < 9850 else 'full'
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration_} ({es_flag}), '
          f'elapsed={elapsed_fold:.1f}min')

lgb_v8b_rho  = spearmanr(y_m1_5, lgb_oof_v8b)[0]
print(f'\nLGB v8b M1-5 OOF Spearman: {lgb_v8b_rho:.4f}  '
      f'(v7 M1-5 OOF: {spearmanr(y[m1_5_mask], lgb_oof_v7[m1_5_mask])[0]:.4f})')
print(f'LGB v8b time: {(time.time()-t0)/60:.1f} min')

# --- XGB v8b -----------------------------------------------------------------
xgb_oof_v8b  = np.zeros(len(train_m1_5_df))
xgb_test_v8b = np.zeros(len(test_df))
xgb_scores_v8b = []

print(f'\n=== XGB v8b: M1-5 only + M1-5 TE ({N_FOLDS}-Fold, GPU) ===\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf_11b.split(X_m1_5)):
    oof_ckpt  = f'{MODEL_DIR}xgb_v8b_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}xgb_v8b_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        xgb_oof_v8b[va_idx] = np.load(oof_ckpt)
        xgb_test_v8b += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y_m1_5.iloc[va_idx], xgb_oof_v8b[va_idx])[0]
        xgb_scores_v8b.append(fold_rho)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_m1_5.iloc[tr_idx], y_m1_5.iloc[tr_idx]
    X_va, y_va = X_m1_5.iloc[va_idx], y_m1_5.iloc[va_idx]
    w_tr       = sample_weight_m1_5[tr_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        verbose=2000,
    )

    va_pred   = model.predict(X_va)
    test_pred = model.predict(X_test_11b)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, test_pred)

    xgb_oof_v8b[va_idx] = va_pred
    xgb_test_v8b       += test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    xgb_scores_v8b.append(fold_rho)
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration}, elapsed={elapsed_fold:.1f}min')

xgb_v8b_rho  = spearmanr(y_m1_5, xgb_oof_v8b)[0]
print(f'\nXGB v8b M1-5 OOF Spearman: {xgb_v8b_rho:.4f}  '
      f'(v7 M1-5 OOF: {spearmanr(y[m1_5_mask], xgb_oof_v7[m1_5_mask])[0]:.4f})')
print(f'XGB v8b time: {(time.time()-t0)/60:.1f} min')

# --- Save + Ensemble v8b (M1-5 CB reuse — still cb_test_v4 for test) ---------
np.save(f'{MODEL_DIR}lgb_oof_v8b.npy',  lgb_oof_v8b)
np.save(f'{MODEL_DIR}lgb_test_v8b.npy', lgb_test_v8b)
np.save(f'{MODEL_DIR}xgb_oof_v8b.npy',  xgb_oof_v8b)
np.save(f'{MODEL_DIR}xgb_test_v8b.npy', xgb_test_v8b)
print(f'\nSaved: lgb/xgb _oof/test_v8b.npy')

# For v8b OOF ensemble, compare only on M1-5 rows (both OOF are over M1-5)
# CB v4 OOF covers all rows; subset to M1-5 for weight search
cb_oof_m1_5 = cb_oof_v4[m1_5_mask.values]

best_rho_v8b = 0.0
best_w_v8b   = (0.0, 1.0, 0.0)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        blend = w1*lgb_oof_v8b + w2*xgb_oof_v8b + w3*cb_oof_m1_5
        rho = spearmanr(y_m1_5, blend)[0]
        if rho > best_rho_v8b:
            best_rho_v8b = rho
            best_w_v8b   = (round(w1,2), round(w2,2), round(w3,2))

w1_8b, w2_8b, w3_8b = best_w_v8b
ens_test_v8b = w1_8b*lgb_test_v8b + w2_8b*xgb_test_v8b + w3_8b*cb_test_v4
ens_test_v8b = np.clip(ens_test_v8b, 0, 1)

ens_oof_v8b_m1_5 = w1_8b*lgb_oof_v8b + w2_8b*xgb_oof_v8b + w3_8b*cb_oof_m1_5

print(f'\n=== Ensemble v8b (11B) ===')
print(f'  Weights: LGB={w1_8b}, XGB={w2_8b}, CB={w3_8b}')
print(f'  M1-5 OOF: {best_rho_v8b:.4f}  (v7 M1-5 OOF: {ens_v7_m1_5:.4f})')

sub_v8b = pd.DataFrame({'invalid_ratio': ens_test_v8b})
sub_v8b.to_csv(f'{SUBMIT_DIR}ensemble_v8b.csv', index=True, index_label='')
print(f'\nSaved: {SUBMIT_DIR}ensemble_v8b.csv')

print(f'  NaN: {np.isnan(ens_test_v8b).sum()}  '
      f'Rows: {len(ens_test_v8b)}  '
      f'Range: [{ens_test_v8b.min():.4f}, {ens_test_v8b.max():.4f}]')

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('Step 11 Summary')
print(f'{"="*60}\n')

print(f'  KS stat — grid_te:        {ks_grid.statistic:.4f}')
print(f'  KS stat — grid_period_te: {ks_gp.statistic:.4f}')

print(f'\n  {"Version":<22} {"OOF":>8} {"M1-5 OOF":>10} {"Platform":>10}')
print(f'  {"-"*54}')
print(f'  {"v7 (baseline)":<22} {best_rho_v7:>8.4f} {ens_v7_m1_5:>10.4f} {"0.5636":>10}')
print(f'  {"v8a (11A full+M1-5TE)":<22} {best_rho_v8a:>8.4f} {ens_v8a_m1_5:>10.4f} {"submit →":>10}')
print(f'  {"v8b (11B M1-5 only)":<22} {"(M1-5 only)":>8} {best_rho_v8b:>10.4f} {"submit →":>10}')

print(f'\nNext steps:')
print(f'  1. Submit {SUBMIT_DIR}ensemble_v8a.csv → platform')
print(f'  2. Submit {SUBMIT_DIR}ensemble_v8b.csv → platform')
print(f'  3. If both < 0.565 → proceed to Step 12 (constrained re-Optuna)')
print(f'  4. If either >= 0.575 → done or try Step 13 (DART)')

print(f'\nStep 11 finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log_file.close()
