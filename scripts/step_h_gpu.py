"""
Step H: GBDT Label Noise Handling — LGB + XGB with 3 noise strategies
=======================================================================
Run from the project root:
    conda activate parking
    python scripts/step_h_gpu.py

Background
----------
25% of training samples have total_count=1, so invalid_ratio is either exactly
0 or exactly 1 — pure binary noise.  v7 already reduces their influence via
log1p(total_count) weighting (weight 0.693 vs 2.40 for tc=10), but the labels
themselves are unchanged.

We use the v7 ensemble OOF predictions to identify the most "confidently wrong"
tc=1 samples as noise candidates, then test three handling strategies:

    (a) Remove     — drop noise candidates from training entirely
    (b) Down-weight — set sample_weight = 0.1 (vs 0.693 baseline)
    (c) Label smooth — y=1 noise → 0.8,  y=0 noise → 0.2

Noise criteria (applied only to tc=1 samples):
    pred < 0.15  but  y = 1   →  predicted low, actually 1  (noisy)
    pred > 0.85  but  y = 0   →  predicted high, actually 0  (noisy)

All three strategies use IDENTICAL 5-fold splits (same random_state=42 as v7),
so OOF Spearman is comparable across strategies.

Reference: Paper 5 (GBDT Label Noise 2024)

Expected runtime on GPU server:
    ~1h per strategy (LGB ~40-45 min + XGB ~15-20 min)
    ~3h total for all three strategies

Fold checkpoints:  models/{lgb,xgb}_h{a,b,c}_fold{n}_{oof,test}.npy
Final outputs:     models/{lgb,xgb}_h{a,b,c}_{oof,test}.npy
Submissions:       submissions/ensemble_h{a,b,c}.csv

Usage:
    # Run all three strategies
    python scripts/step_h_gpu.py

    # Run only strategy b (edit STRATEGIES below)
    # STRATEGIES = ['b']
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

# ── Tee stdout to log file ─────────────────────────────────────────────────────
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

log_file = open('step_h_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ─────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

# Which strategies to run (change to subset for partial runs)
STRATEGIES = ['a', 'b', 'c']

# Noise identification thresholds
NOISE_LOW_THRESH  = 0.15   # pred < threshold but y=1 → noisy
NOISE_HIGH_THRESH = 0.85   # pred > threshold but y=0 → noisy

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Step H: GBDT Label Noise Handling')
print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'Strategies to run: {STRATEGIES}')
print(f'{"="*60}\n')

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

TARGET       = 'invalid_ratio'
EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]
y            = train_df[TARGET].values.astype(np.float32)
tc           = train_df['total_count'].values

print(f'  Train: {train_df.shape},  Test: {test_df.shape}')
print(f'  Features: {len(FEATURES)}')

X      = train_df[FEATURES]
X_test = test_df[FEATURES]
m1_5_mask = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values

# ── Load v7 OOF predictions for noise identification ───────────────────────────
print('\nLoading v7 OOF predictions...')
lgb_oof_v7 = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7 = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7 = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7 = np.load(f'{MODEL_DIR}xgb_test_v7.npy')

# v7 ensemble (established weights LGB=0.35, XGB=0.65)
ens_oof_v7 = 0.35 * lgb_oof_v7 + 0.65 * xgb_oof_v7

lgb_v7_rho = spearmanr(y, lgb_oof_v7)[0]
xgb_v7_rho = spearmanr(y, xgb_oof_v7)[0]
ens_v7_rho = spearmanr(y, ens_oof_v7)[0]

print(f'  LGB v7 OOF:           {lgb_v7_rho:.4f}')
print(f'  XGB v7 OOF:           {xgb_v7_rho:.4f}')
print(f'  Ensemble v7 OOF:      {ens_v7_rho:.4f}')
print(f'  Ensemble v7 M1-5 OOF: {spearmanr(y[m1_5_mask], ens_oof_v7[m1_5_mask])[0]:.4f}')

# ── Identify noise candidates ──────────────────────────────────────────────────
print('\n=== Noise Candidate Identification ===')
tc1_mask   = (tc == 1)
noise_low  = tc1_mask & (ens_oof_v7 < NOISE_LOW_THRESH)  & (y == 1)
noise_high = tc1_mask & (ens_oof_v7 > NOISE_HIGH_THRESH) & (y == 0)
noise_mask = noise_low | noise_high

print(f'  tc=1 samples:             {tc1_mask.sum():>8,}  ({tc1_mask.mean()*100:.1f}% of train)')
print(f'  noise: pred<{NOISE_LOW_THRESH} & y=1:  {noise_low.sum():>8,}')
print(f'  noise: pred>{NOISE_HIGH_THRESH} & y=0:  {noise_high.sum():>8,}')
print(f'  Total noise candidates:   {noise_mask.sum():>8,}  ({noise_mask.mean()*100:.2f}% of train)')

# ── Base sample weights (v7: log1p weighting) ──────────────────────────────────
base_weight = np.log1p(tc).astype(np.float32)
print(f'\nBase weight (log1p):  tc=1→{np.log1p(1):.3f}, tc=10→{np.log1p(10):.3f}')
print(f'Noise weight in (b):  0.100  (was {np.log1p(1):.3f})')

# ── v7 model hyperparameters (Optuna v3 params) ────────────────────────────────
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

# ── KFold splits — same seed as v7 for comparable OOF evaluation ───────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(kf.split(X))   # fixed splits used across ALL strategies


# ── Training function (one strategy, one model type) ──────────────────────────
def train_one_model(model_type, strategy_tag, y_modified, sample_weight):
    """
    Train one GBDT model (LGB or XGB) using a given noise-handling strategy.

    Parameters
    ----------
    model_type    : 'lgb' or 'xgb'
    strategy_tag  : 'a', 'b', or 'c'
    y_modified    : target array (same length as train_df; may have smoothed values)
    sample_weight : weight array (same length as train_df; may have reduced noise weights)

    For strategy (a), noise candidates are excluded from the training fold but
    kept in the validation fold for fair OOF evaluation.

    Returns
    -------
    oof  : np.ndarray shape (n_train,)  — OOF predictions on original validation sets
    test : np.ndarray shape (n_test,)   — averaged test predictions
    """
    tag   = f'{model_type}_h{strategy_tag}'
    oof   = np.zeros(len(train_df))
    test  = np.zeros(len(test_df))
    scores = []
    t0 = time.time()

    print(f'\n--- {tag} ({N_FOLDS}-Fold) ---')

    for fold, (tr_idx, va_idx) in enumerate(fold_splits):
        oof_ckpt  = f'{MODEL_DIR}{tag}_fold{fold}_oof.npy'
        test_ckpt = f'{MODEL_DIR}{tag}_fold{fold}_test.npy'

        if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
            oof[va_idx] = np.load(oof_ckpt)
            test       += np.load(test_ckpt) / N_FOLDS
            fold_rho    = spearmanr(y[va_idx], oof[va_idx])[0]
            scores.append(fold_rho)
            print(f'  Fold {fold}: loaded checkpoint  Spearman={fold_rho:.4f}')
            continue

        # --- Build training set for this fold + strategy ---------------------
        if strategy_tag == 'a':
            # Exclude noise candidates from the training portion only
            keep_tr = ~noise_mask[tr_idx]
            tr_used = tr_idx[keep_tr]
        else:
            tr_used = tr_idx

        X_tr = X.iloc[tr_used]
        y_tr = y_modified[tr_used]
        w_tr = sample_weight[tr_used]

        X_va = X.iloc[va_idx]
        y_va = y[va_idx]   # always original y for fair OOF evaluation

        # --- Train ----------------------------------------------------------
        if model_type == 'lgb':
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
            best_iter = model.best_iteration_
        else:
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                sample_weight=w_tr,
                verbose=2000,
            )
            best_iter = model.best_iteration

        va_pred        = model.predict(X_va)
        fold_test_pred = model.predict(X_test)

        np.save(oof_ckpt,  va_pred)
        np.save(test_ckpt, fold_test_pred)

        oof[va_idx] = va_pred
        test       += fold_test_pred / N_FOLDS

        fold_rho = spearmanr(y_va, va_pred)[0]
        scores.append(fold_rho)
        elapsed = (time.time() - t0) / 60
        es_flag  = 'ES triggered' if best_iter < 9850 else 'ran to limit'
        print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
              f'best_iter={best_iter} ({es_flag})  elapsed={elapsed:.1f}min')

    oof_rho = spearmanr(y, oof)[0]
    m15_rho = spearmanr(y[m1_5_mask], oof[m1_5_mask])[0]
    elapsed = (time.time() - t0) / 60

    v7_base = {'lgb': 0.6336, 'xgb': 0.6403}[model_type]
    v7_m15  = {'lgb': 0.6428, 'xgb': 0.6482}[model_type]
    print(f'\n{tag} OOF Spearman: {oof_rho:.4f}  (v7: {v7_base:.4f}, delta: {oof_rho-v7_base:+.4f})')
    print(f'{tag} M1-5:         {m15_rho:.4f}  (v7: {v7_m15:.4f}, delta: {m15_rho-v7_m15:+.4f})')
    print(f'{tag} total time:   {elapsed:.1f} min')

    np.save(f'{MODEL_DIR}{tag}_oof.npy',  oof)
    np.save(f'{MODEL_DIR}{tag}_test.npy', test)
    print(f'Saved {MODEL_DIR}{tag}_oof.npy and {MODEL_DIR}{tag}_test.npy')

    return oof, test


# ── Strategy definitions ───────────────────────────────────────────────────────
strategy_configs = {
    'a': {
        'name':   '(a) Remove noise candidates',
        'y':      y.copy(),                         # original labels
        'weight': base_weight.copy(),               # original weights
        # noise samples excluded from tr_idx inside train_one_model
    },
    'b': {
        'name':   '(b) Down-weight noise candidates (0.1)',
        'y':      y.copy(),
        'weight': base_weight.copy(),               # will modify below
    },
    'c': {
        'name':   '(c) Label smoothing for noise candidates',
        'y':      y.copy().astype(np.float32),      # will modify below
        'weight': base_weight.copy(),
    },
}

# Modify weights / labels for b and c
strategy_configs['b']['weight'][noise_mask] = 0.1

strategy_configs['c']['y'][noise_low]  = 0.8
strategy_configs['c']['y'][noise_high] = 0.2


# ── Run training ───────────────────────────────────────────────────────────────
results = {}   # {strategy_tag: (lgb_oof, lgb_test, xgb_oof, xgb_test)}

for strat in STRATEGIES:
    cfg = strategy_configs[strat]
    print(f'\n\n{"#"*60}')
    print(f'# STRATEGY {strat.upper()}: {cfg["name"]}')
    print(f'{"#"*60}')
    if strat == 'a':
        n_removed = noise_mask.sum()
        print(f'  Noise candidates removed from each training fold: ~{n_removed:,} samples')
    elif strat == 'b':
        print(f'  Noise weight: 0.100  (vs log1p(1)={np.log1p(1):.3f} baseline)')
    else:
        print(f'  Noise y=1 → 0.8,  y=0 → 0.2')

    lgb_oof, lgb_test = train_one_model('lgb', strat, cfg['y'], cfg['weight'])
    xgb_oof, xgb_test = train_one_model('xgb', strat, cfg['y'], cfg['weight'])
    results[strat] = (lgb_oof, lgb_test, xgb_oof, xgb_test)


# ── Load v7 CB predictions for ensemble search ─────────────────────────────────
cb_oof_path = f'{MODEL_DIR}cb_oof_v4.npy'
if os.path.exists(cb_oof_path):
    cb_oof_v4  = np.load(cb_oof_path)
    cb_test_v4 = np.load(f'{MODEL_DIR}cb_test_v4.npy')
    print(f'\nLoaded cb_oof_v4 for ensemble search.')
else:
    cb_oof_v4  = None
    cb_test_v4 = None
    print('\ncb_oof_v4.npy not found; ensemble search will use LGB+XGB only.')


# ── Ensemble weight search + submission generation ─────────────────────────────
def best_ensemble(lgb_oof, xgb_oof, lgb_test, xgb_test):
    """Grid-search optimal LGB/XGB blend weights (step=0.01)."""
    best_rho, best_w = -1, (0.35, 0.65)
    for lw in np.arange(0.0, 1.01, 0.01):
        xw = round(1.0 - lw, 2)
        rho = spearmanr(y, lw * lgb_oof + xw * xgb_oof)[0]
        if rho > best_rho:
            best_rho, best_w = rho, (round(lw, 2), round(xw, 2))
    oof_blend  = best_w[0] * lgb_oof  + best_w[1] * xgb_oof
    test_blend = best_w[0] * lgb_test + best_w[1] * xgb_test
    return best_rho, best_w, oof_blend, test_blend


print(f'\n\n{"="*60}')
print('STEP H — FINAL SUMMARY')
print(f'{"="*60}')
print(f'\n{"Strategy":<28} {"LGB OOF":>8} {"XGB OOF":>8} {"Ens OOF":>8} {"LGB wt":>7} {"XGB wt":>7}')
print('-' * 70)
print(f'{"v7 baseline":<28} {"0.6336":>8} {"0.6403":>8} {"0.6429":>8} {"0.35":>7} {"0.65":>7}')

for strat in STRATEGIES:
    lgb_oof, lgb_test, xgb_oof, xgb_test = results[strat]
    lgb_rho = spearmanr(y, lgb_oof)[0]
    xgb_rho = spearmanr(y, xgb_oof)[0]
    ens_rho, best_w, ens_oof, ens_test = best_ensemble(lgb_oof, xgb_oof, lgb_test, xgb_test)
    ens_test = np.clip(ens_test, 0, 1)

    name = strategy_configs[strat]['name']
    print(f'{name:<28} {lgb_rho:>8.4f} {xgb_rho:>8.4f} {ens_rho:>8.4f} {best_w[0]:>7.2f} {best_w[1]:>7.2f}')

    # Save submission
    sub = pd.DataFrame({'invalid_ratio': ens_test})
    sub_path = f'{SUBMIT_DIR}ensemble_h{strat}.csv'
    sub.to_csv(sub_path, index=True, index_label='')

    # Validation checks
    assert np.isnan(ens_test).sum() == 0,     f'NaN in {sub_path}'
    assert len(ens_test) == 2028750,          f'Row count mismatch: {len(ens_test)}'
    assert ens_test.min() >= 0,               f'Value below 0 in {sub_path}'
    assert ens_test.max() <= 1,               f'Value above 1 in {sub_path}'
    print(f'  Saved {sub_path}  '
          f'[{ens_test.min():.4f}, {ens_test.max():.4f}]  rows={len(ens_test):,}')

print(f'\nStep H completed at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log_file.close()
