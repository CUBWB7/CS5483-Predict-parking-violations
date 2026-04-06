"""
Check that all files required by step12_gpu.py are present on the server.
Run from the project root:
    conda activate parking
    python scripts/check_step12_files.py
"""

import os

DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

REQUIRED = {
    'Data (parquet)': [
        f'{DATA_DIR}train_features_tier2.parquet',
        f'{DATA_DIR}test_features_tier2.parquet',
    ],
    'Baseline predictions (gitignored — must upload manually)': [
        f'{MODEL_DIR}lgb_oof_v7.npy',
        f'{MODEL_DIR}lgb_test_v7.npy',
        f'{MODEL_DIR}xgb_oof_v7.npy',
        f'{MODEL_DIR}xgb_test_v7.npy',
        f'{MODEL_DIR}cb_oof_v4.npy',
        f'{MODEL_DIR}cb_test_v4.npy',
    ],
    'Script': [
        'scripts/step12_gpu.py',
    ],
    'Output dirs (must exist)': [
        MODEL_DIR,
        SUBMIT_DIR,
    ],
}

all_ok = True
for group, paths in REQUIRED.items():
    print(f'\n[{group}]')
    for p in paths:
        exists = os.path.exists(p)
        size   = f'{os.path.getsize(p)/1e6:.1f} MB' if exists else ''
        status = 'OK' if exists else 'MISSING'
        marker = '  ' if exists else '!!'
        print(f'  {marker} {status:<8s}  {p:<55s} {size}')
        if not exists:
            all_ok = False

print()
if all_ok:
    print('All files present. Ready to run step12_gpu.py.')
else:
    print('Some files are missing. Upload them before running step12_gpu.py.')
    print()
    print('Quick upload hint (from local machine, project root):')
    print('  scp models/lgb_oof_v7.npy models/lgb_test_v7.npy \\')
    print('      models/xgb_oof_v7.npy models/xgb_test_v7.npy \\')
    print('      models/cb_oof_v4.npy  models/cb_test_v4.npy  \\')
    print('      <user>@<server>:<project_root>/models/')
