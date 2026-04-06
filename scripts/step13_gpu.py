"""
Step 13: DART Boosting — LGB v10 + XGB v7 Ensemble
====================================================
Run from the project root:
    conda activate parking
    python scripts/step13_gpu.py

Strategy:
  - Only LGB changes: boosting_type='dart' (replaces 'gbdt')
  - XGB unchanged: reuse v7 predictions (no retraining)
  - CB unchanged: reuse v4 predictions
  - Same v7 Optuna params + log1p(total_count) sample weights
  - Fixed 5000 rounds — DART loss is non-monotonic, no early stopping

Two submissions generated:
  - ensemble_v10.csv  : full-data TE (standard)
  - ensemble_v10a.csv : M1-5 TE (reuse Step 11 TE logic on DART fold models)

Expected runtime: ~1.5h on GPU server.
Fold checkpoints: models/lgb_v10_fold{n}_{oof,test}.npy + lgb_v10_fold{n}.txt
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
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

log_file = open('step13_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Step 13 (DART LGB v10) started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
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

# ── Load v7 and CB v4 baselines ───────────────────────────────────────────────
print('\nLoading v7 / CB v4 baselines...')
lgb_oof_v7  = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7  = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7 = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7 = np.load(f'{MODEL_DIR}xgb_test_v7.npy')
cb_oof_v4   = np.load(f'{MODEL_DIR}cb_oof_v4.npy')
cb_test_v4  = np.load(f'{MODEL_DIR}cb_test_v4.npy')

# Also load v8a XGB test (M1-5 TE) for ensemble_v10a
xgb_test_v8a_path = f'{MODEL_DIR}xgb_test_v8a.npy'
has_xgb_v8a = os.path.exists(xgb_test_v8a_path)
if has_xgb_v8a:
    xgb_test_v8a = np.load(xgb_test_v8a_path)
    xgb_oof_v8a  = np.load(f'{MODEL_DIR}xgb_oof_v8a.npy')
    print('  Loaded XGB v8a predictions (M1-5 TE) for ensemble_v10a.')
else:
    print('  WARNING: xgb_test_v8a.npy not found — ensemble_v10a will reuse v7 XGB test.')
    xgb_test_v8a = xgb_test_v7
    xgb_oof_v8a  = xgb_oof_v7

lgb_oof_v7_rho = spearmanr(y, lgb_oof_v7)[0]
xgb_oof_v7_rho = spearmanr(y, xgb_oof_v7)[0]
cb_oof_v4_rho  = spearmanr(y, cb_oof_v4)[0]

# Reproduce v7 ensemble weights for reference
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

ens_v7_m1_5 = spearmanr(
    y[m1_5_mask],
    best_w_v7[0]*lgb_oof_v7[m1_5_mask] +
    best_w_v7[1]*xgb_oof_v7[m1_5_mask] +
    best_w_v7[2]*cb_oof_v4[m1_5_mask]
)[0]

print(f'\n  LGB v7 OOF:      {lgb_oof_v7_rho:.4f}')
print(f'  XGB v7 OOF:      {xgb_oof_v7_rho:.4f}')
print(f'  CB  v4 OOF:      {cb_oof_v4_rho:.4f}')
print(f'  Ensemble v7 OOF: {best_rho_v7:.4f}  weights={best_w_v7}')
print(f'  Ensemble v7 M1-5 OOF: {ens_v7_m1_5:.4f}')

# ── Build M1-5 TE for test (same logic as step11, for v10a submission) ────────
print('\n=== Building M1-5 TE for Test (v10a) ===\n')

train_m1_5   = train_df[m1_5_mask].copy()
global_mean_full = train_df['invalid_ratio'].mean()
global_mean_m1_5 = train_m1_5['invalid_ratio'].mean()

# grid_te: smooth=100 for M1-5 subset (vs original smooth=30)
full_stats_grid      = train_df.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
full_stats_grid_m1_5 = train_m1_5.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])

smooth_grid_orig = 30
smooth_grid_m1_5 = 100

enc_grid_orig  = (full_stats_grid['count'] * full_stats_grid['mean']
                  + smooth_grid_orig * global_mean_full
                  ) / (full_stats_grid['count'] + smooth_grid_orig)
enc_grid_m1_5  = (full_stats_grid_m1_5['count'] * full_stats_grid_m1_5['mean']
                  + smooth_grid_m1_5 * global_mean_m1_5
                  ) / (full_stats_grid_m1_5['count'] + smooth_grid_m1_5)

test_grid_te_orig  = test_df['grid_id'].map(enc_grid_orig).fillna(global_mean_full)
test_grid_te_m1_5  = test_df['grid_id'].map(enc_grid_m1_5).fillna(global_mean_m1_5)

# grid_period_te: smooth=150 for M1-5 subset (vs original smooth=50)
full_stats_gp      = train_df.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
full_stats_gp_m1_5 = train_m1_5.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])

smooth_gp_orig = 50
smooth_gp_m1_5 = 150

enc_gp_orig  = (full_stats_gp['count'] * full_stats_gp['mean']
                + smooth_gp_orig * global_mean_full
                ) / (full_stats_gp['count'] + smooth_gp_orig)
enc_gp_m1_5  = (full_stats_gp_m1_5['count'] * full_stats_gp_m1_5['mean']
                + smooth_gp_m1_5 * global_mean_m1_5
                ) / (full_stats_gp_m1_5['count'] + smooth_gp_m1_5)

test_gp_te_orig  = test_df['grid_period'].map(enc_gp_orig).fillna(global_mean_full)
test_gp_te_m1_5  = test_df['grid_period'].map(enc_gp_m1_5)
missing_mask = test_gp_te_m1_5.isna()
test_gp_te_m1_5[missing_mask] = test_grid_te_m1_5[missing_mask]   # fallback
test_gp_te_m1_5 = test_gp_te_m1_5.fillna(global_mean_m1_5)

ks_grid = ks_2samp(test_grid_te_orig.values, test_grid_te_m1_5.values)
ks_gp   = ks_2samp(test_gp_te_orig.values,   test_gp_te_m1_5.values)
print(f'  grid_te        KS (orig vs M1-5): {ks_grid.statistic:.4f}  p={ks_grid.pvalue:.4f}')
print(f'  grid_period_te KS (orig vs M1-5): {ks_gp.statistic:.4f}  p={ks_gp.pvalue:.4f}')


def build_test_m1_5(test_df, features, new_grid_te, new_gp_te):
    """Return test feature matrix with grid_te / grid_period_te replaced by M1-5 values."""
    X_test = test_df[features].copy()
    if 'grid_te' in X_test.columns:
        X_test['grid_te'] = new_grid_te.values
    if 'grid_period_te' in X_test.columns:
        X_test['grid_period_te'] = new_gp_te.values
    return X_test


X_test_full = test_df[FEATURES]
X_test_m1_5 = build_test_m1_5(test_df, FEATURES, test_grid_te_m1_5, test_gp_te_m1_5)

# ── Sample weights (same as v7) ───────────────────────────────────────────────
sample_weight_all = np.log1p(train_df['total_count'].values)

# ── DART LGB params (v7 Optuna params + dart settings) ───────────────────────
# DART randomly drops trees each round → forces new trees to learn independently.
# Non-monotonic loss means standard early_stopping is unreliable → use fixed rounds.
lgb_params_dart = {
    'num_leaves':        100,       # v7 Optuna best
    'learning_rate':     0.0564,    # v7 Optuna best
    'min_child_samples': 69,        # v7 Optuna best
    'reg_lambda':        0.452,     # v7 Optuna best
    'reg_alpha':         1.243,     # v7 Optuna best
    'feature_fraction':  0.844,     # v7 Optuna best
    'bagging_fraction':  0.972,     # v7 Optuna best
    'objective':         'regression',
    'metric':            'l2',
    'boosting_type':     'dart',    # key change from v7 'gbdt'
    'drop_rate':         0.1,       # drop 10% of trees each round
    'max_drop':          50,        # cap at 50 trees dropped per round
    'skip_drop':         0.5,       # 50% chance to skip dropout (speedup)
    'bagging_freq':      5,
    'verbose':           -1,
    'n_jobs':            -1,
    'random_state':      SEED,
    'n_estimators':      5000,      # DART converges faster; no early stopping
}

# ── LGB v10 (DART) — 5-Fold CV ───────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_oof_v10   = np.zeros(len(train_df))
lgb_test_v10  = np.zeros(len(test_df))   # full-data TE predictions
lgb_test_v10a = np.zeros(len(test_df))   # M1-5 TE predictions (for v10a submission)
lgb_scores_v10 = []

print(f'\n=== LGB v10 (DART): v7 params + dart + 5000 rounds ({N_FOLDS}-Fold) ===\n')
print(f'  DART settings: drop_rate=0.1, max_drop=50, skip_drop=0.5')
print(f'  No early stopping (DART loss is non-monotonic)\n')
t0 = time.time()

X_train = train_df[FEATURES]

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    oof_ckpt       = f'{MODEL_DIR}lgb_v10_fold{fold}_oof.npy'
    test_ckpt      = f'{MODEL_DIR}lgb_v10_fold{fold}_test.npy'
    test_m1_5_ckpt = f'{MODEL_DIR}lgb_v10_fold{fold}_test_m1_5.npy'
    model_path     = f'{MODEL_DIR}lgb_v10_fold{fold}.txt'

    # Resume from checkpoint if all outputs exist
    if (os.path.exists(oof_ckpt) and os.path.exists(test_ckpt)
            and os.path.exists(test_m1_5_ckpt)):
        lgb_oof_v10[va_idx]  = np.load(oof_ckpt)
        lgb_test_v10        += np.load(test_ckpt) / N_FOLDS
        lgb_test_v10a       += np.load(test_m1_5_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], lgb_oof_v10[va_idx])[0]
        lgb_scores_v10.append(fold_rho)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
    w_tr       = sample_weight_all[tr_idx]

    model = lgb.LGBMRegressor(**lgb_params_dart)
    # No callbacks: DART does not support early stopping reliably
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        callbacks=[lgb.log_evaluation(period=1000)],
    )

    va_pred        = model.predict(X_va)
    test_pred_full = model.predict(X_test_full)
    test_pred_m1_5 = model.predict(X_test_m1_5)

    np.save(oof_ckpt,       va_pred)
    np.save(test_ckpt,      test_pred_full)
    np.save(test_m1_5_ckpt, test_pred_m1_5)
    model.booster_.save_model(model_path)

    lgb_oof_v10[va_idx]  = va_pred
    lgb_test_v10        += test_pred_full / N_FOLDS
    lgb_test_v10a       += test_pred_m1_5 / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    lgb_scores_v10.append(fold_rho)
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, elapsed={elapsed_fold:.1f}min')

lgb_v10_rho  = spearmanr(y, lgb_oof_v10)[0]
lgb_v10_m1_5 = spearmanr(y[m1_5_mask], lgb_oof_v10[m1_5_mask])[0]
lgb_corr_xgb = np.corrcoef(lgb_oof_v10, xgb_oof_v7)[0, 1]
elapsed = time.time() - t0

print(f'\nLGB v10 (DART) OOF:  {lgb_v10_rho:.4f}'
      f'  (v7: {lgb_oof_v7_rho:.4f}, delta: {lgb_v10_rho - lgb_oof_v7_rho:+.4f})')
print(f'LGB v10 (DART) M1-5: {lgb_v10_m1_5:.4f}'
      f'  (v7: {spearmanr(y[m1_5_mask], lgb_oof_v7[m1_5_mask])[0]:.4f})')
print(f'LGB v10 - XGB v7 correlation: {lgb_corr_xgb:.4f}'
      f'  (v7 LGB-XGB: 0.9681 — lower is better for diversity)')
print(f'LGB v10 total time: {elapsed/60:.1f} min')

# Success threshold check
if lgb_v10_rho >= 0.625:
    print(f'  OOF >= 0.625 threshold: PASS')
elif lgb_v10_rho >= 0.610:
    print(f'  OOF in 0.610-0.625 range: marginal — proceed with caution')
else:
    print(f'  WARNING: OOF < 0.610 — DART may be over-regularizing. '
          f'Consider reducing drop_rate to 0.05 or increasing n_estimators to 7000.')

# ── Save final predictions ────────────────────────────────────────────────────
np.save(f'{MODEL_DIR}lgb_oof_v10.npy',  lgb_oof_v10)
np.save(f'{MODEL_DIR}lgb_test_v10.npy', lgb_test_v10)
print(f'\nSaved: lgb_oof_v10.npy, lgb_test_v10.npy')

# ── Inter-model correlations ──────────────────────────────────────────────────
corr_lgb_xgb = np.corrcoef(lgb_oof_v10, xgb_oof_v7)[0, 1]
corr_lgb_cb  = np.corrcoef(lgb_oof_v10, cb_oof_v4)[0, 1]
corr_xgb_cb  = np.corrcoef(xgb_oof_v7,  cb_oof_v4)[0, 1]

print(f'\n=== Inter-Model Correlations ===')
print(f'  DART LGB v10 - XGB v7: {corr_lgb_xgb:.4f}  (v7 LGB-XGB: 0.9681)')
print(f'  DART LGB v10 - CB  v4: {corr_lgb_cb:.4f}  (v7 LGB-CB: ~0.966)')
print(f'  XGB v7       - CB  v4: {corr_xgb_cb:.4f}  (v7 XGB-CB: ~0.962)')

# ── Ensemble v10 (full-data TE) weight search ─────────────────────────────────
print(f'\n=== Ensemble v10: DART LGB + XGB v7 + CB v4 (full-data TE) ===\n')

best_rho_v10 = 0.0
best_w_v10   = (0.0, 1.0, 0.0)

for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        blend = w1*lgb_oof_v10 + w2*xgb_oof_v7 + w3*cb_oof_v4
        rho = spearmanr(y, blend)[0]
        if rho > best_rho_v10:
            best_rho_v10 = rho
            best_w_v10   = (round(w1,2), round(w2,2), round(w3,2))

w1_v10, w2_v10, w3_v10 = best_w_v10
ens_oof_v10  = w1_v10*lgb_oof_v10  + w2_v10*xgb_oof_v7  + w3_v10*cb_oof_v4
ens_test_v10 = w1_v10*lgb_test_v10 + w2_v10*xgb_test_v7 + w3_v10*cb_test_v4
ens_test_v10 = np.clip(ens_test_v10, 0, 1)

ens_v10_m1_5 = spearmanr(y[m1_5_mask], ens_oof_v10[m1_5_mask])[0]

print(f'  Best weights: LGB={w1_v10}, XGB={w2_v10}, CB={w3_v10}')
print(f'  Ensemble v10 OOF:  {best_rho_v10:.4f}'
      f'  (v7: {best_rho_v7:.4f}, delta: {best_rho_v10 - best_rho_v7:+.4f})')
print(f'  Ensemble v10 M1-5: {ens_v10_m1_5:.4f}'
      f'  (v7: {ens_v7_m1_5:.4f}, delta: {ens_v10_m1_5 - ens_v7_m1_5:+.4f})')

# Generate ensemble_v10.csv
sub_v10 = pd.DataFrame({'invalid_ratio': ens_test_v10})
sub_v10.to_csv(f'{SUBMIT_DIR}ensemble_v10.csv', index=True, index_label='')
print(f'\nSaved: {SUBMIT_DIR}ensemble_v10.csv')

# ── Ensemble v10a (M1-5 TE) weight search ─────────────────────────────────────
# LGB v10a: DART LGB re-predicted on M1-5 TE test features (lgb_test_v10a)
# XGB v10a: reuse xgb_test_v8a (XGB v7 model, M1-5 TE test)
# CB v10a:  reuse cb_test_v4 (CB not TE-dependent in the same way)
print(f'\n=== Ensemble v10a: DART LGB (M1-5 TE) + XGB v8a + CB v4 ===\n')

# For OOF weight search we use full OOF (same as v10 — training was identical)
best_rho_v10a = 0.0
best_w_v10a   = (0.0, 1.0, 0.0)

for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        # OOF uses same predictions as v10 (training data uses full TE)
        blend = w1*lgb_oof_v10 + w2*xgb_oof_v8a + w3*cb_oof_v4
        rho = spearmanr(y, blend)[0]
        if rho > best_rho_v10a:
            best_rho_v10a = rho
            best_w_v10a   = (round(w1,2), round(w2,2), round(w3,2))

w1_v10a, w2_v10a, w3_v10a = best_w_v10a
ens_oof_v10a  = w1_v10a*lgb_oof_v10  + w2_v10a*xgb_oof_v8a  + w3_v10a*cb_oof_v4
ens_test_v10a = w1_v10a*lgb_test_v10a + w2_v10a*xgb_test_v8a + w3_v10a*cb_test_v4
ens_test_v10a = np.clip(ens_test_v10a, 0, 1)

ens_v10a_m1_5 = spearmanr(y[m1_5_mask], ens_oof_v10a[m1_5_mask])[0]

print(f'  Best weights: LGB={w1_v10a}, XGB={w2_v10a}, CB={w3_v10a}')
print(f'  Ensemble v10a OOF:  {best_rho_v10a:.4f}  (v10: {best_rho_v10:.4f})')
print(f'  Ensemble v10a M1-5: {ens_v10a_m1_5:.4f}  (v10: {ens_v10_m1_5:.4f})')

# Generate ensemble_v10a.csv
sub_v10a = pd.DataFrame({'invalid_ratio': ens_test_v10a})
sub_v10a.to_csv(f'{SUBMIT_DIR}ensemble_v10a.csv', index=True, index_label='')
print(f'\nSaved: {SUBMIT_DIR}ensemble_v10a.csv')

# ── Pre-submission validation ─────────────────────────────────────────────────
print(f'\n=== Pre-Submission Validation ===')
for name, arr in [('ensemble_v10', ens_test_v10), ('ensemble_v10a', ens_test_v10a)]:
    checks = [
        ('NaN count',  str(np.isnan(arr).sum()),     np.isnan(arr).sum() == 0),
        ('Row count',  str(len(arr)),                len(arr) == 2028750),
        ('Range',      f'[{arr.min():.4f}, {arr.max():.4f}]',
                       arr.min() >= 0 and arr.max() <= 1),
    ]
    print(f'\n  {name}:')
    for label, val, ok in checks:
        print(f'    {label:<12s} {val:<30s} {"PASS" if ok else "FAIL"}')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('=== Complete Results Summary ===')
print(f'{"="*60}')
print(f'\n{"Version":<20s} {"LGB":>8s} {"XGB":>8s} {"CB":>8s} {"Ensemble":>10s} {"M1-5":>8s}')
print('-' * 68)
print(f'{"v7 (baseline)":<20s} {lgb_oof_v7_rho:>8.4f} {xgb_oof_v7_rho:>8.4f}'
      f' {cb_oof_v4_rho:>8.4f} {best_rho_v7:>10.4f} {ens_v7_m1_5:>8.4f}')
print(f'{"v10 (DART LGB)":<20s} {lgb_v10_rho:>8.4f} {"(reuse)":>8s}'
      f' {"(reuse)":>8s} {best_rho_v10:>10.4f} {ens_v10_m1_5:>8.4f}')
print(f'{"v10a (DART+M1-5TE)":<20s} {lgb_v10_rho:>8.4f} {"(v8a)":>8s}'
      f' {"(reuse)":>8s} {best_rho_v10a:>10.4f} {ens_v10a_m1_5:>8.4f}')

print(f'\nDART diversity check:')
print(f'  LGB v10 - XGB v7 correlation: {corr_lgb_xgb:.4f}'
      f'  (target: < 0.965)')
if corr_lgb_xgb < 0.965:
    print(f'  Diversity IMPROVED — DART reduces LGB-XGB correlation.')
else:
    print(f'  Diversity unchanged — DART did not reduce correlation.')

print(f'\nNext steps:')
print(f'  1. Submit ensemble_v10.csv  (DART + full-data TE)')
print(f'  2. Submit ensemble_v10a.csv (DART + M1-5 TE)')
print(f'  3. If Platform >= 0.575 → done. If < 0.564 → Step 14 (NN).')
print(f'\nStep 13 finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log_file.close()
