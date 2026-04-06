"""
Reconstruct aggregated prediction files from per-fold checkpoints.
Run from the project root:
    conda activate parking
    python scripts/recover_v7_preds.py

Recovers:
  - lgb_oof_v7.npy / lgb_test_v7.npy  (from step10_gpu.py fold checkpoints)
  - xgb_oof_v7.npy / xgb_test_v7.npy  (from step10_gpu.py fold checkpoints)
  - cb_oof_v4.npy  / cb_test_v4.npy   (from step4b_gpu.py fold checkpoints)
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
print('Saved: lgb/xgb _oof/test_v7.npy')

# ── Recover cb_v4 from step4b_gpu.py fold checkpoints ────────────────────────
import os

cb_v4_ckpts = [f'{MODEL_DIR}cb_v4_fold{i}_oof.npy' for i in range(N_FOLDS)]
if all(os.path.exists(p) for p in cb_v4_ckpts):
    print('\nReconstructing CB v4 from fold checkpoints...')
    cb_oof_v4  = np.zeros(len(train_df))
    cb_test_v4 = np.zeros(len(test_df))
    for fold, (_, va_idx) in enumerate(kf.split(train_df)):
        cb_oof_v4[va_idx] = np.load(f'{MODEL_DIR}cb_v4_fold{fold}_oof.npy')
        cb_test_v4       += np.load(f'{MODEL_DIR}cb_v4_fold{fold}_test.npy') / N_FOLDS
        print(f'  Fold {fold}: done')
    cb_rho = spearmanr(y, cb_oof_v4)[0]
    print(f'CB v4 OOF Spearman: {cb_rho:.4f}  (expected ~0.6175)')
    np.save(f'{MODEL_DIR}cb_oof_v4.npy',  cb_oof_v4)
    np.save(f'{MODEL_DIR}cb_test_v4.npy', cb_test_v4)
    print('Saved: cb_oof_v4.npy / cb_test_v4.npy')
else:
    missing = [p for p in cb_v4_ckpts if not os.path.exists(p)]
    print(f'\nCB v4 fold checkpoints missing: {missing}')
    print('step4b_gpu.py needs to be re-run to generate CB v4 predictions.')

print('\nAll done. Run check_step12_files.py to verify.')
