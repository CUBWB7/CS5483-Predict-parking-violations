"""
Reconstruct lgb/xgb _oof/test_v7.npy from per-fold checkpoints.
Run from the project root:
    conda activate parking
    python scripts/recover_v7_preds.py

This is needed when step10_gpu.py finished all folds but crashed
before saving the aggregated files.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

SEED      = 42
N_FOLDS   = 5
DATA_DIR  = 'data/'
MODEL_DIR = 'models/'

print('Loading data shapes...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')
y        = train_df['invalid_ratio']
print(f'  Train: {len(train_df):,}  Test: {len(test_df):,}')

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_oof_v7  = np.zeros(len(train_df))
lgb_test_v7 = np.zeros(len(test_df))
xgb_oof_v7  = np.zeros(len(train_df))
xgb_test_v7 = np.zeros(len(test_df))

print('\nReconstructing from fold checkpoints...')
for fold, (_, va_idx) in enumerate(kf.split(train_df)):
    lgb_oof_v7[va_idx] = np.load(f'{MODEL_DIR}lgb_v7_fold{fold}_oof.npy')
    lgb_test_v7       += np.load(f'{MODEL_DIR}lgb_v7_fold{fold}_test.npy') / N_FOLDS
    xgb_oof_v7[va_idx] = np.load(f'{MODEL_DIR}xgb_v7_fold{fold}_oof.npy')
    xgb_test_v7       += np.load(f'{MODEL_DIR}xgb_v7_fold{fold}_test.npy') / N_FOLDS
    print(f'  Fold {fold}: done')

lgb_rho = spearmanr(y, lgb_oof_v7)[0]
xgb_rho = spearmanr(y, xgb_oof_v7)[0]
print(f'\nLGB v7 OOF Spearman: {lgb_rho:.4f}  (expected ~0.6336)')
print(f'XGB v7 OOF Spearman: {xgb_rho:.4f}  (expected ~0.6403)')

np.save(f'{MODEL_DIR}lgb_oof_v7.npy',  lgb_oof_v7)
np.save(f'{MODEL_DIR}lgb_test_v7.npy', lgb_test_v7)
np.save(f'{MODEL_DIR}xgb_oof_v7.npy',  xgb_oof_v7)
np.save(f'{MODEL_DIR}xgb_test_v7.npy', xgb_test_v7)
print(f'\nSaved: lgb/xgb _oof/test_v7.npy')

# Also check cb_v4 (needed by step12_gpu.py)
import os
for f in ['cb_oof_v4.npy', 'cb_test_v4.npy']:
    path = f'{MODEL_DIR}{f}'
    status = f'OK  ({os.path.getsize(path)/1e6:.1f} MB)' if os.path.exists(path) else 'MISSING — check step4b_gpu outputs'
    print(f'  {f}: {status}')
