"""
Step 10: Sample Weighting by total_count — LGB v7 + XGB v7 + Ensemble v7
=========================================================================
Run from the project root:
    conda activate parking
    python scripts/step10_gpu.py

GPU acceleration:
    - XGBoost: device='cuda' (XGBoost >= 2.0); change to 'cpu' if unavailable
    - LightGBM: CPU with n_jobs=-1 (LGB GPU requires special build)

Progress is printed to stdout and also written to step10_gpu.log.
Fold checkpoints are saved to models/{lgb,xgb}_v7_fold{n}_{oof,test}.npy
— if the server restarts, re-run and completed folds will be skipped.

Expected runtime on GPU server: ~60-90 min total (LGB ~45 min, XGB ~30 min).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
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

log_file = open('step10_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Step 10 started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
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
print(f'  Features (v3 base, 26): {len(FEATURES)}')

# ── Load v3/v4 baselines ──────────────────────────────────────────────────────
print('\nLoading v3/v4 baseline predictions...')
lgb_oof_v3  = np.load(f'{MODEL_DIR}lgb_oof_v3.npy')
xgb_oof_v3  = np.load(f'{MODEL_DIR}xgb_oof_v3.npy')
cb_oof_v4   = np.load(f'{MODEL_DIR}cb_oof_v4.npy')
cb_test_v4  = np.load(f'{MODEL_DIR}cb_test_v4.npy')

lgb_oof_v3_rho = spearmanr(y, lgb_oof_v3)[0]
xgb_oof_v3_rho = spearmanr(y, xgb_oof_v3)[0]
cb_oof_v4_rho  = spearmanr(y, cb_oof_v4)[0]

best_rho_v4 = 0.0
best_w_v4   = (0.35, 0.65, 0.00)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        blend = w1*lgb_oof_v3 + w2*xgb_oof_v3 + w3*cb_oof_v4
        rho = spearmanr(y, blend)[0]
        if rho > best_rho_v4:
            best_rho_v4 = rho
            best_w_v4   = (round(w1,2), round(w2,2), round(w3,2))

m1_5_mask = train_df['month_of_year'].isin([1, 2, 3, 4, 5])
ens_v4_oof = (best_w_v4[0]*lgb_oof_v3
            + best_w_v4[1]*xgb_oof_v3
            + best_w_v4[2]*cb_oof_v4)
ens_v4_m1_5_rho = spearmanr(y[m1_5_mask], ens_v4_oof[m1_5_mask])[0]

print(f'  LGB v3 OOF:       {lgb_oof_v3_rho:.4f}')
print(f'  XGB v3 OOF:       {xgb_oof_v3_rho:.4f}')
print(f'  CB  v4 OOF:       {cb_oof_v4_rho:.4f}')
print(f'  Ensemble v4 OOF:  {best_rho_v4:.4f}  weights={best_w_v4}')
print(f'  Ensemble v4 M1-5: {ens_v4_m1_5_rho:.4f}')

# ── Sample weights ────────────────────────────────────────────────────────────
# total_count=1 (25.2% of data, Spearman=0.41) → weight 0.693
# total_count=10 → weight 2.40 | total_count=100 → weight 4.62
sample_weight_all = np.log1p(train_df['total_count'].values)

print(f'\nSample weight stats:')
print(f'  total_count=1  → weight {np.log1p(1):.3f}  '
      f'({(train_df["total_count"]==1).mean()*100:.1f}% of samples)')
print(f'  total_count=10 → weight {np.log1p(10):.3f}')
print(f'  total_count=50 → weight {np.log1p(50):.3f}')
print(f'  mean={sample_weight_all.mean():.3f}  '
      f'max={sample_weight_all.max():.3f}')

X_v7      = train_df[FEATURES]
X_test_v7 = test_df[FEATURES]

# ── LGB v7 ────────────────────────────────────────────────────────────────────
# Optuna v3 params + n_estimators=10000 + log1p(total_count) sample weights.
# CPU only (LGB GPU requires a special build); use all available cores.
lgb_params_v7 = {
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
    'n_jobs':            -1,       # use all CPU cores on the server
    'random_state':      SEED,
    'n_estimators':      10000,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
lgb_oof_v7  = np.zeros(len(train_df))
lgb_test_v7 = np.zeros(len(test_df))
lgb_scores_v7     = []
lgb_best_iters_v7 = []

print(f'\n=== LGB v7: Optuna v3 + log1p sample weighting ({N_FOLDS}-Fold) ===\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_v7)):
    oof_ckpt  = f'{MODEL_DIR}lgb_v7_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}lgb_v7_fold{fold}_test.npy'

    # Resume from checkpoint if available
    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        lgb_oof_v7[va_idx] = np.load(oof_ckpt)
        lgb_test_v7 += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], lgb_oof_v7[va_idx])[0]
        lgb_scores_v7.append(fold_rho)
        lgb_best_iters_v7.append(-1)   # unknown from checkpoint
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_v7.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_v7.iloc[va_idx], y.iloc[va_idx]
    w_tr = sample_weight_all[tr_idx]   # weights for training fold only

    model = lgb.LGBMRegressor(**lgb_params_v7)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=2000),
        ]
    )

    va_pred        = model.predict(X_va)
    fold_test_pred = model.predict(X_test_v7)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, fold_test_pred)

    lgb_oof_v7[va_idx]  = va_pred
    lgb_test_v7 += fold_test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    lgb_scores_v7.append(fold_rho)
    lgb_best_iters_v7.append(model.best_iteration_)
    es_flag = 'ES triggered' if model.best_iteration_ < 9850 else 'ran to limit'
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration_} ({es_flag}), elapsed={elapsed_fold:.1f}min')

lgb_oof_v7_rho  = spearmanr(y, lgb_oof_v7)[0]
lgb_oof_v7_m1_5 = spearmanr(y[m1_5_mask], lgb_oof_v7[m1_5_mask])[0]
lgb_oof_v3_m1_5 = spearmanr(y[m1_5_mask], lgb_oof_v3[m1_5_mask])[0]
elapsed = time.time() - t0

print(f'\nLGB v7 OOF Spearman:  {lgb_oof_v7_rho:.4f}'
      f'  (v3: {lgb_oof_v3_rho:.4f}, delta: {lgb_oof_v7_rho - lgb_oof_v3_rho:+.4f})')
print(f'LGB v7 M1-5 Spearman: {lgb_oof_v7_m1_5:.4f}'
      f'  (v3: {lgb_oof_v3_m1_5:.4f}, delta: {lgb_oof_v7_m1_5 - lgb_oof_v3_m1_5:+.4f})')
print(f'Best iters: {lgb_best_iters_v7}')
print(f'LGB v7 total time: {elapsed/60:.1f} min')

# ── XGB v7 ────────────────────────────────────────────────────────────────────
# Optuna v3 params + log1p(total_count) sample weights.
# Uses XGBoost GPU (device='cuda'); change to device='cpu' if unavailable.
xgb_params_v7 = {
    'max_depth':             10,
    'learning_rate':         0.0362,
    'min_child_weight':      11,
    'reg_lambda':            1.561,
    'reg_alpha':             1.239,
    'colsample_bytree':      0.951,
    'subsample':             0.948,
    'objective':             'reg:squarederror',
    'tree_method':           'hist',
    'device':                'cuda',   # change to 'cpu' if no GPU available
    'random_state':          SEED,
    'n_jobs':                -1,
    'verbosity':             0,
    'n_estimators':          10000,
    'early_stopping_rounds': 150,
}

xgb_oof_v7  = np.zeros(len(train_df))
xgb_test_v7 = np.zeros(len(test_df))
xgb_scores_v7     = []
xgb_best_iters_v7 = []

print(f'\n=== XGB v7: Optuna v3 + log1p sample weighting ({N_FOLDS}-Fold, GPU) ===\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_v7)):
    oof_ckpt  = f'{MODEL_DIR}xgb_v7_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}xgb_v7_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        xgb_oof_v7[va_idx] = np.load(oof_ckpt)
        xgb_test_v7 += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], xgb_oof_v7[va_idx])[0]
        xgb_scores_v7.append(fold_rho)
        xgb_best_iters_v7.append(-1)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_v7.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_v7.iloc[va_idx], y.iloc[va_idx]
    w_tr = sample_weight_all[tr_idx]

    model = xgb.XGBRegressor(**xgb_params_v7)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        verbose=2000,
    )

    va_pred        = model.predict(X_va)
    fold_test_pred = model.predict(X_test_v7)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, fold_test_pred)

    xgb_oof_v7[va_idx]  = va_pred
    xgb_test_v7 += fold_test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    xgb_scores_v7.append(fold_rho)
    xgb_best_iters_v7.append(model.best_iteration)
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration}, elapsed={elapsed_fold:.1f}min')

xgb_oof_v7_rho  = spearmanr(y, xgb_oof_v7)[0]
xgb_oof_v7_m1_5 = spearmanr(y[m1_5_mask], xgb_oof_v7[m1_5_mask])[0]
xgb_oof_v3_m1_5 = spearmanr(y[m1_5_mask], xgb_oof_v3[m1_5_mask])[0]
elapsed = time.time() - t0

print(f'\nXGB v7 OOF Spearman:  {xgb_oof_v7_rho:.4f}'
      f'  (v3: {xgb_oof_v3_rho:.4f}, delta: {xgb_oof_v7_rho - xgb_oof_v3_rho:+.4f})')
print(f'XGB v7 M1-5 Spearman: {xgb_oof_v7_m1_5:.4f}'
      f'  (v3: {xgb_oof_v3_m1_5:.4f}, delta: {xgb_oof_v7_m1_5 - xgb_oof_v3_m1_5:+.4f})')
print(f'Best iters: {xgb_best_iters_v7}')
print(f'XGB v7 total time: {elapsed/60:.1f} min')

# ── Save final OOF / test predictions ────────────────────────────────────────
np.save(f'{MODEL_DIR}lgb_oof_v7.npy',  lgb_oof_v7)
np.save(f'{MODEL_DIR}lgb_test_v7.npy', lgb_test_v7)
np.save(f'{MODEL_DIR}xgb_oof_v7.npy',  xgb_oof_v7)
np.save(f'{MODEL_DIR}xgb_test_v7.npy', xgb_test_v7)
print(f'\nSaved: lgb/xgb _oof/test_v7.npy')

# ── Ensemble v7 ───────────────────────────────────────────────────────────────
corr_lgb_xgb_v7 = np.corrcoef(lgb_oof_v7, xgb_oof_v7)[0, 1]
corr_lgb_cb_v7  = np.corrcoef(lgb_oof_v7, cb_oof_v4)[0, 1]
corr_xgb_cb_v7  = np.corrcoef(xgb_oof_v7, cb_oof_v4)[0, 1]

print(f'\n=== Inter-Model Correlations ===')
print(f'  LGB-XGB: {corr_lgb_xgb_v7:.4f}  (v4: 0.9652)')
print(f'  LGB-CB:  {corr_lgb_cb_v7:.4f}  (v4: 0.9661)')
print(f'  XGB-CB:  {corr_xgb_cb_v7:.4f}  (v4: 0.9615)')

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
ensemble_oof_v7  = w1_v7*lgb_oof_v7  + w2_v7*xgb_oof_v7  + w3_v7*cb_oof_v4
ensemble_test_v7 = w1_v7*lgb_test_v7 + w2_v7*xgb_test_v7 + w3_v7*cb_test_v4
ensemble_test_v7 = np.clip(ensemble_test_v7, 0, 1)

ens_v7_m1_5_rho = spearmanr(y[m1_5_mask], ensemble_oof_v7[m1_5_mask])[0]

print(f'\n=== Ensemble v7 Results ===')
print(f'  Best weights: LGB={w1_v7}, XGB={w2_v7}, CB={w3_v7}')
print(f'  Ensemble v7 OOF:  {best_rho_v7:.4f}'
      f'  (v4: {best_rho_v4:.4f}, delta: {best_rho_v7 - best_rho_v4:+.4f})')
print(f'  Ensemble v7 M1-5: {ens_v7_m1_5_rho:.4f}'
      f'  (v4: {ens_v4_m1_5_rho:.4f}, delta: {ens_v7_m1_5_rho - ens_v4_m1_5_rho:+.4f})')

# Generate submission
sub_v7 = pd.read_csv(f'{SUBMIT_DIR}ensemble_v3.csv')
sub_v7['invalid_ratio'] = ensemble_test_v7
sub_v7.to_csv(f'{SUBMIT_DIR}ensemble_v7.csv', index=False)
print(f'\nSaved: {SUBMIT_DIR}ensemble_v7.csv')

# Pre-submission validation
print(f'\n=== Pre-Submission Validation ===')
checks = [
    ('LGB v7 OOF',   f'{lgb_oof_v7_rho:.4f}',
                     lgb_oof_v7_rho > lgb_oof_v3_rho - 0.003),
    ('XGB v7 OOF',   f'{xgb_oof_v7_rho:.4f}',
                     xgb_oof_v7_rho > xgb_oof_v3_rho - 0.003),
    ('Ensemble v7',  f'{best_rho_v7:.4f}',
                     best_rho_v7 >= best_rho_v4 - 0.002),
    ('NaN count',    str(np.isnan(ensemble_test_v7).sum()),
                     np.isnan(ensemble_test_v7).sum() == 0),
    ('Row count',    str(len(ensemble_test_v7)),
                     len(ensemble_test_v7) == 2028750),
    ('Range',        f'[{ensemble_test_v7.min():.4f}, {ensemble_test_v7.max():.4f}]',
                     ensemble_test_v7.min() >= 0 and ensemble_test_v7.max() <= 1),
]
for name, val, ok in checks:
    print(f'  {name:<18s} {val:<30s} {"PASS" if ok else "FAIL"}')

# Improvement history
print(f'\n=== Complete Improvement History ===')
print(f'{"Version":<20s} {"LGB":>8s} {"XGB":>8s} {"CB":>8s} {"Ensemble":>10s} {"Platform":>10s}')
print('-' * 72)
print(f'{"v1 (baseline)":<20s} {"0.5815":>8s} {"0.5870":>8s} {"-":>8s} {"0.5880":>10s} {"0.5222":>10s}')
print(f'{"v2 (Step1+2)":<20s} {"0.5959":>8s} {"0.5994":>8s} {"-":>8s} {"0.6012":>10s} {"0.5338":>10s}')
print(f'{"v3 (Optuna)":<20s} {lgb_oof_v3_rho:>8.4f} {xgb_oof_v3_rho:>8.4f} {"0.5728":>8s} {best_rho_v4:>10.4f} {"0.5620":>10s}')
print(f'{"v4 (CB Optuna)":<20s} {lgb_oof_v3_rho:>8.4f} {xgb_oof_v3_rho:>8.4f} {cb_oof_v4_rho:>8.4f} {best_rho_v4:>10.4f} {"—":>10s}')
print(f'{"v6b (6k cap)":<20s} {"0.6263":>8s} {"0.6375":>8s} {cb_oof_v4_rho:>8.4f} {"0.6392":>10s} {"0.5618":>10s}')
print(f'{"v7 (wt log1p)":<20s} {lgb_oof_v7_rho:>8.4f} {xgb_oof_v7_rho:>8.4f} {cb_oof_v4_rho:>8.4f} {best_rho_v7:>10.4f} {"submit →":>10s}')

print(f'\nNext: submit {SUBMIT_DIR}ensemble_v7.csv to platform.')
print('If Platform >= 0.575 → proceed to Step 11 (M1-5 Focused Training).')
print('If Platform < 0.565 → try sqrt(total_count) weighting or Step 11.')
print(f'\nStep 10 finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log_file.close()
