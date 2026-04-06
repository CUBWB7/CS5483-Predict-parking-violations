"""
Step 12: Stronger Regularization via Constrained Re-Optuna — v9
================================================================
Run from the project root:
    conda activate parking
    python scripts/step12_gpu.py

Goal: reduce OOF-Platform gap (currently ~0.079) by finding more regularized
hyperparameters that trade slight OOF drop for better generalization.

Pipeline:
  1. Optuna LGB  — 40 trials, constrained ranges, M1-5 subsample 1M, 3-fold CV
  2. Optuna XGB  — 40 trials, constrained ranges, M1-5 subsample 1M, 3-fold CV
  3. LGB v9 — full 5-fold retrain with best constrained params + log1p weighting
  4. XGB v9 — full 5-fold retrain with best constrained params + log1p weighting
  5. Ensemble grid search (LGB_v9 + XGB_v9 + CB_v4, step=0.05)
  6. ensemble_v9.csv  — full-data TE (safe baseline)
     ensemble_v9a.csv — M1-5 TE    (to pair with v8a result)

Progress → stdout + step12_gpu.log.
Fold checkpoints → models/{lgb,xgb}_v9_fold{n}_{oof,test}.npy (resume-safe).

Expected runtime on GPU server: ~2-3h total.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, ks_2samp
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

log_file = open('step12_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
N_TRIALS   = 40
OPTUNA_N   = 1_000_000   # rows sampled from M1-5 for Optuna trials
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Step 12 started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

TARGET       = 'invalid_ratio'
EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]
y            = train_df[TARGET]

m1_5_mask = train_df['month_of_year'].isin([1, 2, 3, 4, 5])

print(f'  Train: {train_df.shape},  Test: {test_df.shape}')
print(f'  Features (26): {len(FEATURES)}')
print(f'  M1-5 rows: {m1_5_mask.sum():,}  |  M6-12 rows: {(~m1_5_mask).sum():,}')

# ── Load v7 / CB-v4 baselines ─────────────────────────────────────────────────
print('\nLoading v7 / CB-v4 baseline predictions...')
lgb_oof_v7  = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7  = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7 = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7 = np.load(f'{MODEL_DIR}xgb_test_v7.npy')
cb_oof_v4   = np.load(f'{MODEL_DIR}cb_oof_v4.npy')
cb_test_v4  = np.load(f'{MODEL_DIR}cb_test_v4.npy')

lgb_oof_v7_rho = spearmanr(y, lgb_oof_v7)[0]
xgb_oof_v7_rho = spearmanr(y, xgb_oof_v7)[0]
cb_oof_v4_rho  = spearmanr(y, cb_oof_v4)[0]

# Reproduce v7 ensemble weights
best_rho_v7 = 0.0
best_w_v7   = (0.0, 1.0, 0.0)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        rho = spearmanr(y, w1*lgb_oof_v7 + w2*xgb_oof_v7 + w3*cb_oof_v4)[0]
        if rho > best_rho_v7:
            best_rho_v7 = rho
            best_w_v7   = (round(w1,2), round(w2,2), round(w3,2))

ens_v7_oof   = (best_w_v7[0]*lgb_oof_v7
              + best_w_v7[1]*xgb_oof_v7
              + best_w_v7[2]*cb_oof_v4)
ens_v7_m1_5  = spearmanr(y[m1_5_mask], ens_v7_oof[m1_5_mask])[0]

print(f'  LGB v7 OOF:      {lgb_oof_v7_rho:.4f}')
print(f'  XGB v7 OOF:      {xgb_oof_v7_rho:.4f}')
print(f'  CB  v4 OOF:      {cb_oof_v4_rho:.4f}')
print(f'  Ensemble v7 OOF: {best_rho_v7:.4f}  weights={best_w_v7}')
print(f'  Ensemble v7 M1-5:{ens_v7_m1_5:.4f}')

# ── Recompute M1-5 TE for test (same logic as step11_gpu.py) ─────────────────
# Used later to generate ensemble_v9a.csv (M1-5 TE variant).
print('\n=== Recomputing M1-5 TE for test set ===\n')

train_m1_5       = train_df[m1_5_mask].copy()
global_mean_full = train_df['invalid_ratio'].mean()
global_mean_m1_5 = train_m1_5['invalid_ratio'].mean()

# grid_te: smooth=100 for M1-5 subset (original smooth=30 on full data)
full_stats_grid       = train_df.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
full_stats_grid_m1_5  = train_m1_5.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
smooth_grid_m1_5      = 100

enc_grid_orig = (
    full_stats_grid['count'] * full_stats_grid['mean'] + 30 * global_mean_full
) / (full_stats_grid['count'] + 30)
test_grid_te_orig  = test_df['grid_id'].map(enc_grid_orig).fillna(global_mean_full)

enc_grid_m1_5 = (
    full_stats_grid_m1_5['count'] * full_stats_grid_m1_5['mean']
    + smooth_grid_m1_5 * global_mean_m1_5
) / (full_stats_grid_m1_5['count'] + smooth_grid_m1_5)
test_grid_te_m1_5 = test_df['grid_id'].map(enc_grid_m1_5).fillna(global_mean_m1_5)

# grid_period_te: smooth=150 for M1-5 subset (original smooth=50)
full_stats_gp      = train_df.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
full_stats_gp_m1_5 = train_m1_5.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
smooth_gp_m1_5     = 150

enc_gp_orig = (
    full_stats_gp['count'] * full_stats_gp['mean'] + 50 * global_mean_full
) / (full_stats_gp['count'] + 50)
test_gp_te_orig = test_df['grid_period'].map(enc_gp_orig).fillna(global_mean_full)

enc_gp_m1_5 = (
    full_stats_gp_m1_5['count'] * full_stats_gp_m1_5['mean']
    + smooth_gp_m1_5 * global_mean_m1_5
) / (full_stats_gp_m1_5['count'] + smooth_gp_m1_5)
test_gp_te_m1_5 = test_df['grid_period'].map(enc_gp_m1_5)
missing_mask = test_gp_te_m1_5.isna()
test_gp_te_m1_5[missing_mask] = test_grid_te_m1_5[missing_mask]
test_gp_te_m1_5 = test_gp_te_m1_5.fillna(global_mean_m1_5)

ks_grid = ks_2samp(test_grid_te_orig.values, test_grid_te_m1_5.values)
ks_gp   = ks_2samp(test_gp_te_orig.values,   test_gp_te_m1_5.values)
print(f'  grid_te       KS (orig vs M1-5): {ks_grid.statistic:.4f}')
print(f'  grid_period_te KS (orig vs M1-5): {ks_gp.statistic:.4f}')
print(f'  (M1-5 fallback applied to {missing_mask.sum():,} rows)')


def build_test_with_m1_5_te(test_df, features, new_grid_te, new_gp_te):
    """Return test feature matrix with grid_te / grid_period_te replaced by M1-5 values."""
    X_test = test_df[features].copy()
    if 'grid_te' in X_test.columns:
        X_test['grid_te'] = new_grid_te.values
    if 'grid_period_te' in X_test.columns:
        X_test['grid_period_te'] = new_gp_te.values
    return X_test


# ── Sample weights (identical to v7) ─────────────────────────────────────────
# total_count=1 (25.2% of data) → weight 0.693
sample_weight_all = np.log1p(train_df['total_count'].values)

# ── Build Optuna subsample (M1-5 rows, capped at 1M) ─────────────────────────
# Using M1-5 rows makes the search signal closer to the test distribution.
m1_5_idx = np.where(m1_5_mask.values)[0]
if len(m1_5_idx) > OPTUNA_N:
    rng = np.random.RandomState(SEED)
    sub_idx = rng.choice(m1_5_idx, size=OPTUNA_N, replace=False)
else:
    sub_idx = m1_5_idx

sub_df = train_df.iloc[sub_idx].reset_index(drop=True)
X_sub  = sub_df[FEATURES]
y_sub  = sub_df[TARGET]
w_sub  = np.log1p(sub_df['total_count'].values)  # weights for Optuna trials

print(f'\nOptuna subsample (M1-5): {len(sub_df):,} rows')

# ── Optuna LGB (Step 12 constrained ranges) ───────────────────────────────────
# Constrained vs Step 3:
#   num_leaves:        [15, 127] → [15, 63]  (smaller trees)
#   min_child_samples: [20, 200] → [50, 300] (higher leaf threshold)
#   reg_lambda:        [0.1, 10] → [1.0, 20] (stronger L2)
#   reg_alpha:         [0.0,  5] → [2.0, 10] (stronger L1)
#   feature_fraction:  [0.5,  1] → [0.5, 0.8](fewer features per tree)
#   bagging_fraction:  [0.5,  1] → [0.5, 0.9](slightly more aggressive)
kf_optuna = KFold(n_splits=3, shuffle=True, random_state=SEED)

def lgb_objective(trial):
    params = {
        'num_leaves':        trial.suggest_int(   'num_leaves',        15,  63),
        'learning_rate':     trial.suggest_float(  'learning_rate',    0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int(   'min_child_samples', 50, 300),
        'reg_lambda':        trial.suggest_float(  'reg_lambda',       1.0, 20.0, log=True),
        'reg_alpha':         trial.suggest_float(  'reg_alpha',        2.0, 10.0),
        'feature_fraction':  trial.suggest_float(  'feature_fraction', 0.5,  0.8),
        'bagging_fraction':  trial.suggest_float(  'bagging_fraction', 0.5,  0.9),
        'objective':         'regression',
        'metric':            'l2',
        'boosting_type':     'gbdt',
        'bagging_freq':      5,
        'verbose':           -1,
        'n_jobs':            -1,
        'random_state':      SEED,
        'n_estimators':      3000,   # fixed rounds for fast search
    }
    fold_scores = []
    for tr_idx, va_idx in kf_optuna.split(X_sub):
        X_tr, y_tr = X_sub.iloc[tr_idx], y_sub.iloc[tr_idx]
        X_va, y_va = X_sub.iloc[va_idx], y_sub.iloc[va_idx]
        w_tr = w_sub[tr_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            sample_weight=w_tr,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        fold_scores.append(spearmanr(y_va, model.predict(X_va))[0])
    return float(np.mean(fold_scores))

print(f'\n=== Optuna LGB ({N_TRIALS} trials, constrained ranges) ===\n')
t0 = time.time()
lgb_study = optuna.create_study(direction='maximize', study_name='lgb_step12',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'Best LGB Spearman (subsample): {lgb_study.best_value:.4f}')
print(f'Best LGB params: {lgb_study.best_params}')
print(f'Optuna LGB time: {(time.time()-t0)/60:.1f} min')

lgb_best_params = lgb_study.best_params


# ── Optuna XGB (Step 12 constrained ranges) ───────────────────────────────────
# Constrained vs Step 3:
#   max_depth:       [4, 10]  → [4, 8]    (shallower trees)
#   reg_lambda:      [0.1,10] → [2.0, 20] (stronger L2)
#   reg_alpha:       [0.0, 5] → [2.0, 10] (stronger L1)
#   colsample_bytree:[0.5, 1] → [0.5, 0.8](fewer features per tree)
#   subsample:       [0.5, 1] → [0.5, 0.9](slightly more aggressive)

def xgb_objective(trial):
    params = {
        'max_depth':         trial.suggest_int(   'max_depth',         4,   8),
        'learning_rate':     trial.suggest_float(  'learning_rate',    0.01, 0.1, log=True),
        'min_child_weight':  trial.suggest_int(   'min_child_weight',  10, 200),
        'reg_lambda':        trial.suggest_float(  'reg_lambda',       2.0, 20.0, log=True),
        'reg_alpha':         trial.suggest_float(  'reg_alpha',        2.0, 10.0),
        'colsample_bytree':  trial.suggest_float(  'colsample_bytree', 0.5,  0.8),
        'subsample':         trial.suggest_float(  'subsample',        0.5,  0.9),
        'objective':         'reg:squarederror',
        'tree_method':       'hist',
        'device':            'cuda',   # change to 'cpu' if no GPU
        'random_state':      SEED,
        'n_jobs':            -1,
        'verbosity':         0,
        'n_estimators':      3000,     # fixed rounds for fast search
    }
    fold_scores = []
    for tr_idx, va_idx in kf_optuna.split(X_sub):
        X_tr, y_tr = X_sub.iloc[tr_idx], y_sub.iloc[tr_idx]
        X_va, y_va = X_sub.iloc[va_idx], y_sub.iloc[va_idx]
        w_tr = w_sub[tr_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            sample_weight=w_tr,
            verbose=False,
        )
        fold_scores.append(spearmanr(y_va, model.predict(X_va))[0])
    return float(np.mean(fold_scores))

print(f'\n=== Optuna XGB ({N_TRIALS} trials, constrained ranges) ===\n')
t0 = time.time()
xgb_study = optuna.create_study(direction='maximize', study_name='xgb_step12',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f'Best XGB Spearman (subsample): {xgb_study.best_value:.4f}')
print(f'Best XGB params: {xgb_study.best_params}')
print(f'Optuna XGB time: {(time.time()-t0)/60:.1f} min')

xgb_best_params = xgb_study.best_params

# ── LGB v9 — full 5-fold retrain ──────────────────────────────────────────────
# Best constrained params + 10000 rounds + ES=150 + log1p(total_count) weights.
lgb_params_v9 = {
    **lgb_best_params,
    'objective':     'regression',
    'metric':        'l2',
    'boosting_type': 'gbdt',
    'bagging_freq':  5,
    'verbose':       -1,
    'n_jobs':        -1,
    'random_state':  SEED,
    'n_estimators':  10000,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
X_train      = train_df[FEATURES]
X_test_full  = test_df[FEATURES]                          # full-data TE
X_test_m1_5  = build_test_with_m1_5_te(test_df, FEATURES,
                                        test_grid_te_m1_5,
                                        test_gp_te_m1_5)  # M1-5 TE

lgb_oof_v9   = np.zeros(len(train_df))
lgb_test_v9  = np.zeros(len(test_df))   # full-data TE test preds
lgb_test_v9a = np.zeros(len(test_df))   # M1-5 TE test preds
lgb_scores_v9     = []
lgb_best_iters_v9 = []

print(f'\n=== LGB v9: constrained Optuna + log1p weighting ({N_FOLDS}-Fold) ===')
print(f'  Params: {lgb_params_v9}\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    oof_ckpt  = f'{MODEL_DIR}lgb_v9_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}lgb_v9_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        lgb_oof_v9[va_idx] = np.load(oof_ckpt)
        lgb_test_v9  += np.load(test_ckpt) / N_FOLDS
        # No separate M1-5 TE checkpoint — re-use full TE for now
        fold_rho = spearmanr(y.iloc[va_idx], lgb_oof_v9[va_idx])[0]
        lgb_scores_v9.append(fold_rho)
        lgb_best_iters_v9.append(-1)
        print(f'  Fold {fold}: loaded checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
    w_tr = sample_weight_all[tr_idx]

    model = lgb.LGBMRegressor(**lgb_params_v9)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=2000),
        ]
    )

    va_pred         = model.predict(X_va)
    test_pred_full  = model.predict(X_test_full)
    test_pred_m1_5  = model.predict(X_test_m1_5)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, test_pred_full)

    lgb_oof_v9[va_idx]  = va_pred
    lgb_test_v9  += test_pred_full  / N_FOLDS
    lgb_test_v9a += test_pred_m1_5  / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    lgb_scores_v9.append(fold_rho)
    lgb_best_iters_v9.append(model.best_iteration_)
    es_flag = 'ES triggered' if model.best_iteration_ < 9850 else 'ran to limit'
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration_} ({es_flag}), '
          f'elapsed={( time.time()-t0)/60:.1f}min')

lgb_oof_v9_rho  = spearmanr(y,              lgb_oof_v9)[0]
lgb_oof_v9_m1_5 = spearmanr(y[m1_5_mask],   lgb_oof_v9[m1_5_mask])[0]
elapsed = time.time() - t0

print(f'\nLGB v9 OOF Spearman:  {lgb_oof_v9_rho:.4f}'
      f'  (v7: {lgb_oof_v7_rho:.4f}, delta: {lgb_oof_v9_rho - lgb_oof_v7_rho:+.4f})')
print(f'LGB v9 M1-5 Spearman: {lgb_oof_v9_m1_5:.4f}')
print(f'Best iters: {lgb_best_iters_v9}')
print(f'LGB v9 total time: {elapsed/60:.1f} min')

# ── XGB v9 — full 5-fold retrain ──────────────────────────────────────────────
xgb_params_v9 = {
    **xgb_best_params,
    'objective':             'reg:squarederror',
    'tree_method':           'hist',
    'device':                'cuda',
    'random_state':          SEED,
    'n_jobs':                -1,
    'verbosity':             0,
    'n_estimators':          10000,
    'early_stopping_rounds': 150,
}

xgb_oof_v9   = np.zeros(len(train_df))
xgb_test_v9  = np.zeros(len(test_df))   # full-data TE
xgb_test_v9a = np.zeros(len(test_df))   # M1-5 TE
xgb_scores_v9     = []
xgb_best_iters_v9 = []

print(f'\n=== XGB v9: constrained Optuna + log1p weighting ({N_FOLDS}-Fold, GPU) ===')
print(f'  Params: {xgb_params_v9}\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    oof_ckpt  = f'{MODEL_DIR}xgb_v9_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}xgb_v9_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        xgb_oof_v9[va_idx] = np.load(oof_ckpt)
        xgb_test_v9  += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], xgb_oof_v9[va_idx])[0]
        xgb_scores_v9.append(fold_rho)
        xgb_best_iters_v9.append(-1)
        print(f'  Fold {fold}: loaded checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
    w_tr = sample_weight_all[tr_idx]

    model = xgb.XGBRegressor(**xgb_params_v9)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        verbose=2000,
    )

    va_pred         = model.predict(X_va)
    test_pred_full  = model.predict(X_test_full)
    test_pred_m1_5  = model.predict(X_test_m1_5)

    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, test_pred_full)

    xgb_oof_v9[va_idx]  = va_pred
    xgb_test_v9  += test_pred_full  / N_FOLDS
    xgb_test_v9a += test_pred_m1_5  / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    xgb_scores_v9.append(fold_rho)
    xgb_best_iters_v9.append(model.best_iteration)
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.best_iteration}, elapsed={elapsed_fold:.1f}min')

xgb_oof_v9_rho  = spearmanr(y,            xgb_oof_v9)[0]
xgb_oof_v9_m1_5 = spearmanr(y[m1_5_mask], xgb_oof_v9[m1_5_mask])[0]
elapsed = time.time() - t0

print(f'\nXGB v9 OOF Spearman:  {xgb_oof_v9_rho:.4f}'
      f'  (v7: {xgb_oof_v7_rho:.4f}, delta: {xgb_oof_v9_rho - xgb_oof_v7_rho:+.4f})')
print(f'XGB v9 M1-5 Spearman: {xgb_oof_v9_m1_5:.4f}')
print(f'Best iters: {xgb_best_iters_v9}')
print(f'XGB v9 total time: {elapsed/60:.1f} min')

# ── Save OOF / test predictions ───────────────────────────────────────────────
np.save(f'{MODEL_DIR}lgb_oof_v9.npy',  lgb_oof_v9)
np.save(f'{MODEL_DIR}lgb_test_v9.npy', lgb_test_v9)
np.save(f'{MODEL_DIR}xgb_oof_v9.npy',  xgb_oof_v9)
np.save(f'{MODEL_DIR}xgb_test_v9.npy', xgb_test_v9)
print(f'\nSaved: lgb/xgb _oof/test_v9.npy')

# ── Inter-Model Correlations ──────────────────────────────────────────────────
corr_lgb_xgb_v9 = np.corrcoef(lgb_oof_v9, xgb_oof_v9)[0, 1]
corr_lgb_cb_v9  = np.corrcoef(lgb_oof_v9, cb_oof_v4)[0, 1]
corr_xgb_cb_v9  = np.corrcoef(xgb_oof_v9, cb_oof_v4)[0, 1]

print(f'\n=== Inter-Model Correlations (v9) ===')
print(f'  LGB-XGB: {corr_lgb_xgb_v9:.4f}  (v7: 0.9647)')
print(f'  LGB-CB:  {corr_lgb_cb_v9:.4f}  (v7: 0.9657)')
print(f'  XGB-CB:  {corr_xgb_cb_v9:.4f}  (v7: 0.9563)')

# ── Ensemble weight search (v9) ───────────────────────────────────────────────
best_rho_v9 = 0.0
best_w_v9   = (0.0, 1.0, 0.0)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        rho = spearmanr(y, w1*lgb_oof_v9 + w2*xgb_oof_v9 + w3*cb_oof_v4)[0]
        if rho > best_rho_v9:
            best_rho_v9 = rho
            best_w_v9   = (round(w1,2), round(w2,2), round(w3,2))

w1_v9, w2_v9, w3_v9 = best_w_v9
ens_v9_oof   = w1_v9*lgb_oof_v9 + w2_v9*xgb_oof_v9 + w3_v9*cb_oof_v4
ens_v9_m1_5  = spearmanr(y[m1_5_mask], ens_v9_oof[m1_5_mask])[0]

# Full-data TE test predictions (ensemble_v9.csv)
ens_v9_test  = np.clip(w1_v9*lgb_test_v9 + w2_v9*xgb_test_v9 + w3_v9*cb_test_v4, 0, 1)

# M1-5 TE test predictions (ensemble_v9a.csv)
ens_v9a_test = np.clip(w1_v9*lgb_test_v9a + w2_v9*xgb_test_v9a + w3_v9*cb_test_v4, 0, 1)

print(f'\n=== Ensemble v9 Results ===')
print(f'  Best weights: LGB={w1_v9}, XGB={w2_v9}, CB={w3_v9}')
print(f'  Ensemble v9 OOF:  {best_rho_v9:.4f}'
      f'  (v7: {best_rho_v7:.4f}, delta: {best_rho_v9 - best_rho_v7:+.4f})')
print(f'  Ensemble v9 M1-5: {ens_v9_m1_5:.4f}'
      f'  (v7: {ens_v7_m1_5:.4f}, delta: {ens_v9_m1_5 - ens_v7_m1_5:+.4f})')

# Interpretation guidance
if best_rho_v9 >= best_rho_v7 - 0.001:
    print('  OOF roughly maintained — regularization is not too aggressive.')
elif best_rho_v9 >= best_rho_v7 - 0.005:
    print('  OOF slightly reduced (expected trade-off for better generalization).')
else:
    print('  OOF dropped significantly — may be over-regularized; check platform score.')

# ── Generate submissions ──────────────────────────────────────────────────────
sub_v9  = pd.DataFrame({'invalid_ratio': ens_v9_test})
sub_v9a = pd.DataFrame({'invalid_ratio': ens_v9a_test})

sub_v9.to_csv( f'{SUBMIT_DIR}ensemble_v9.csv',  index=True, index_label='')
sub_v9a.to_csv(f'{SUBMIT_DIR}ensemble_v9a.csv', index=True, index_label='')

print(f'\nSaved: {SUBMIT_DIR}ensemble_v9.csv  (full-data TE)')
print(f'Saved: {SUBMIT_DIR}ensemble_v9a.csv (M1-5 TE)')

# ── Pre-Submission Validation ─────────────────────────────────────────────────
print(f'\n=== Pre-Submission Validation ===')
for label, preds in [('v9 (full TE)', ens_v9_test), ('v9a (M1-5 TE)', ens_v9a_test)]:
    checks = [
        ('NaN count',  str(np.isnan(preds).sum()),      np.isnan(preds).sum() == 0),
        ('Row count',  str(len(preds)),                  len(preds) == 2028750),
        ('Range',      f'[{preds.min():.4f},{preds.max():.4f}]',
                       preds.min() >= 0 and preds.max() <= 1),
    ]
    print(f'\n  {label}:')
    for name, val, ok in checks:
        print(f'    {name:<12s} {val:<25s} {"PASS" if ok else "FAIL"}')

# ── Improvement History ───────────────────────────────────────────────────────
print(f'\n=== Complete Improvement History ===')
print(f'{"Version":<22s} {"LGB":>8s} {"XGB":>8s} {"CB":>8s} {"Ensemble":>10s} {"M1-5":>8s} {"Platform":>10s}')
print('-' * 80)
rows = [
    ('v1 (baseline)',     '0.5815', '0.5870', '—',                       '0.5880',           '—',                    '0.5222'),
    ('v2 (Step1+2)',      '0.5959', '0.5994', '—',                       '0.6012',           '—',                    '0.5338'),
    ('v3 (Optuna)',       '0.6322', '0.6379', f'{cb_oof_v4_rho:.4f}',   '0.6408',           '0.6492',               '0.5620'),
    ('v7 (weighted)',     f'{lgb_oof_v7_rho:.4f}', f'{xgb_oof_v7_rho:.4f}',
                          f'{cb_oof_v4_rho:.4f}',  f'{best_rho_v7:.4f}', f'{ens_v7_m1_5:.4f}', '0.5636'),
    ('v9 (Step12)',       f'{lgb_oof_v9_rho:.4f}', f'{xgb_oof_v9_rho:.4f}',
                          f'{cb_oof_v4_rho:.4f}',  f'{best_rho_v9:.4f}', f'{ens_v9_m1_5:.4f}', 'submit→'),
]
for r in rows:
    print(f'  {r[0]:<20s} {r[1]:>8s} {r[2]:>8s} {r[3]:>8s} {r[4]:>10s} {r[5]:>8s} {r[6]:>10s}')

print(f'\nNext steps:')
print(f'  1. Submit ensemble_v9.csv (full-data TE) to platform.')
print(f'  2. If v8a platform > 0.5636: also submit ensemble_v9a.csv (M1-5 TE variant).')
print(f'  3. Decision thresholds: >= 0.575 = success; ~0.564 = neutral; < 0.560 = revert.')
print(f'\nStep 12 finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')

sys.stdout = sys.__stdout__
log_file.close()
