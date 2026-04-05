"""
Step 4b: CatBoost v4 GPU Retrain + Ensemble v4
===============================================
Run from the project root:
    python step4b_gpu.py

Progress is printed to stdout and also written to step4b_gpu.log.
Fold checkpoints are saved to models/cb_v4_fold{n}_oof.npy and
models/cb_v4_fold{n}_test.npy — if the server restarts, re-run
the script and completed folds will be skipped automatically.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
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

log_file = open('step4b_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED      = 42
N_FOLDS   = 5
DATA_DIR  = 'data/'
MODEL_DIR = 'models/'
SUBMIT_DIR = 'submissions/'

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Step 4b started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
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
print(f'  Numeric features: {len(FEATURES)}')

# ── Load saved v3 predictions ─────────────────────────────────────────────────
print('\nLoading v3 predictions...')
lgb_oof_v3  = np.load(f'{MODEL_DIR}lgb_oof_v3.npy')
xgb_oof_v3  = np.load(f'{MODEL_DIR}xgb_oof_v3.npy')
lgb_test_v3 = np.load(f'{MODEL_DIR}lgb_test_v3.npy')
xgb_test_v3 = np.load(f'{MODEL_DIR}xgb_test_v3.npy')
cb_oof_v3   = np.load(f'{MODEL_DIR}cb_oof.npy')   # untuned CB for reference

lgb_oof_v3_rho = spearmanr(y, lgb_oof_v3)[0]
xgb_oof_v3_rho = spearmanr(y, xgb_oof_v3)[0]
print(f'  LGB v3 OOF: {lgb_oof_v3_rho:.4f}')
print(f'  XGB v3 OOF: {xgb_oof_v3_rho:.4f}')

# v3 ensemble reference
best_rho3 = 0
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        rho = spearmanr(y, w1*lgb_oof_v3 + w2*xgb_oof_v3 + w3*cb_oof_v3)[0]
        if rho > best_rho3:
            best_rho3 = rho
print(f'  Ensemble v3 OOF: {best_rho3:.4f}')

# ── CatBoost features ─────────────────────────────────────────────────────────
CB_FEATURES_V4 = FEATURES + ['grid_id']
CAT_IDX_V4     = [CB_FEATURES_V4.index('grid_id')]

# ── Optuna best params (hardcoded from local run) ─────────────────────────────
cb_tuned_v4 = {
    # --- Optuna best params ---
    'depth':               10,
    'learning_rate':       0.08271645692786933,
    'l2_leaf_reg':         4.118330821745591,
    'random_strength':     6.243845636492408,
    'bagging_temperature': 0.44895675071369556,
    'border_count':        235,
    'min_data_in_leaf':    21,
    # --- Fixed params ---
    'iterations':            8000,
    'random_seed':           SEED,
    'task_type':             'GPU',
    'devices':               '0',
    'verbose':               500,
    'eval_metric':           'RMSE',
    'early_stopping_rounds': 100,
    'grow_policy':           'SymmetricTree',
    # boosting_type='Ordered' is CPU-only; GPU always uses Plain
}

# ── Full retrain with fold checkpoints ────────────────────────────────────────
X_cb_v4      = train_df[CB_FEATURES_V4].copy()
X_test_cb_v4 = test_df[CB_FEATURES_V4].copy()
X_cb_v4['grid_id']      = X_cb_v4['grid_id'].astype(str)
X_test_cb_v4['grid_id'] = X_test_cb_v4['grid_id'].astype(str)

kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cb_oof_v4  = np.zeros(len(train_df))
cb_test_v4 = np.zeros(len(test_df))
cb_scores_v4 = []

print(f'\n=== CatBoost v4 Full Retrain ({N_FOLDS}-Fold, {len(CB_FEATURES_V4)} features, GPU) ===')
print(f'Estimated time: ~1.5-2.5h\n')
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb_v4)):
    oof_ckpt  = f'{MODEL_DIR}cb_v4_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}cb_v4_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        cb_oof_v4[va_idx] = np.load(oof_ckpt)
        cb_test_v4 += np.load(test_ckpt) / N_FOLDS
        fold_rho = spearmanr(y.iloc[va_idx], cb_oof_v4[va_idx])[0]
        cb_scores_v4.append(fold_rho)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    X_tr = X_cb_v4.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X_cb_v4.iloc[va_idx]
    y_va = y.iloc[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=CAT_IDX_V4)
    val_pool   = Pool(X_va, y_va, cat_features=CAT_IDX_V4)
    test_pool  = Pool(X_test_cb_v4, cat_features=CAT_IDX_V4)

    model = CatBoostRegressor(**cb_tuned_v4)
    model.fit(train_pool, eval_set=val_pool)

    va_pred        = model.predict(X_va)
    fold_test_pred = model.predict(test_pool)

    # Save checkpoint immediately
    np.save(oof_ckpt,  va_pred)
    np.save(test_ckpt, fold_test_pred)

    cb_oof_v4[va_idx] = va_pred
    cb_test_v4 += fold_test_pred / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred)[0]
    cb_scores_v4.append(fold_rho)
    elapsed_fold = (time.time() - t0) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}, '
          f'best_iter={model.get_best_iteration()}, elapsed={elapsed_fold:.1f}min')

cb_oof_v4_rho = spearmanr(y, cb_oof_v4)[0]
elapsed = time.time() - t0
print(f'\nCatBoost v4 OOF Spearman: {cb_oof_v4_rho:.4f}  (untuned CB: 0.5728)')
print(f'Fold scores: {[f"{s:.4f}" for s in cb_scores_v4]}')
print(f'Time: {elapsed/60:.1f} min')

# ── Ensemble v4 ───────────────────────────────────────────────────────────────
corr_lgb_xgb = np.corrcoef(lgb_oof_v3, xgb_oof_v3)[0, 1]
corr_lgb_cb  = np.corrcoef(lgb_oof_v3, cb_oof_v4)[0, 1]
corr_xgb_cb  = np.corrcoef(xgb_oof_v3, cb_oof_v4)[0, 1]

print(f'\n=== Inter-Model Correlations ===')
print(f'  LGB-XGB:    {corr_lgb_xgb:.4f}')
print(f'  LGB-CB v4:  {corr_lgb_cb:.4f}')
print(f'  XGB-CB v4:  {corr_xgb_cb:.4f}')

best_w1_v4, best_w2_v4, best_rho_v4 = 0, 0, 0
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        blend = w1 * lgb_oof_v3 + w2 * xgb_oof_v3 + w3 * cb_oof_v4
        rho = spearmanr(y, blend)[0]
        if rho > best_rho_v4:
            best_w1_v4, best_w2_v4, best_rho_v4 = w1, w2, rho

best_w3_v4 = round(1.0 - best_w1_v4 - best_w2_v4, 2)

best_w_2m, best_rho_2m = 0, 0
for w in np.arange(0, 1.01, 0.05):
    rho = spearmanr(y, w * lgb_oof_v3 + (1-w) * xgb_oof_v3)[0]
    if rho > best_rho_2m:
        best_w_2m, best_rho_2m = w, rho

print(f'\n=== Ensemble v4 Results ===')
print(f'  2-model (LGB+XGB):    Spearman={best_rho_2m:.4f}')
print(f'  3-model (LGB+XGB+CB): Spearman={best_rho_v4:.4f}  '
      f'(LGB={best_w1_v4:.2f}, XGB={best_w2_v4:.2f}, CB={best_w3_v4:.2f})')
print(f'  CB v4 ensemble benefit: {best_rho_v4 - best_rho_2m:+.4f}')
print(f'  vs Ensemble v3 ({best_rho3:.4f}): {best_rho_v4 - best_rho3:+.4f}')

ensemble_test_v4 = (best_w1_v4 * lgb_test_v3
                    + best_w2_v4 * xgb_test_v3
                    + best_w3_v4 * cb_test_v4)
ensemble_test_v4 = np.clip(ensemble_test_v4, 0, 1)

# ── Save results ──────────────────────────────────────────────────────────────
np.save(f'{MODEL_DIR}cb_oof_v4.npy',  cb_oof_v4)
np.save(f'{MODEL_DIR}cb_test_v4.npy', cb_test_v4)

sub_v4 = pd.DataFrame({'invalid_ratio': ensemble_test_v4})
sub_v4.to_csv(f'{SUBMIT_DIR}ensemble_v4.csv', index=True, index_label='')

print(f'\n=== Pre-Submission Validation ===')
checks = [
    ('CB v4 OOF',   f'{cb_oof_v4_rho:.4f}',  cb_oof_v4_rho > 0.60),
    ('Ensemble v4', f'{best_rho_v4:.4f}',     best_rho_v4 > 0.64),
    ('NaN count',   str(np.isnan(ensemble_test_v4).sum()),
                    np.isnan(ensemble_test_v4).sum() == 0),
    ('Row count',   str(len(ensemble_test_v4)),
                    len(ensemble_test_v4) == 2028750),
]
for name, val, ok in checks:
    print(f'  {name:<18s} {val:<20s} {"PASS" if ok else "FAIL"}')

print(f'\nSaved: {SUBMIT_DIR}ensemble_v4.csv')
print(f'\nStep 4b finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log_file.close()
