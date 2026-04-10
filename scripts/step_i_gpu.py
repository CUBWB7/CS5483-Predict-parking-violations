"""
Experiment I: Rank-Target GBDT Re-tuning — LGB + XGB with increased iterations
================================================================================
Run from the project root:
    conda activate parking

    # Part A only (quick win: more iterations)
    python scripts/step_i_gpu.py --part-a-only

    # Part A + Part B (Optuna re-tuning for rank-target)
    python scripts/step_i_gpu.py --run-optuna

Rationale:
    Exp C used Optuna v3 hyperparameters tuned for bimodal raw invalid_ratio.
    The rank-target loss surface is fundamentally different (uniform distribution).
    Evidence: All 5 LGB folds in Exp C hit best_iter=9998-10000 (n_estimators limit),
    meaning LGB was still learning and stopped due to budget, not convergence.

Part A — Quick Win:
    LGB: n_estimators 10000 → 20000, early_stopping patience 150 → 200
    XGB: n_estimators 10000 → 15000, early_stopping patience 150 → 200
    Expected gain: +0.001-0.003 OOF if LGB was underfitting

Part B — Re-Optuna (--run-optuna flag):
    Search ranges centered near v7 values (avoid Step 12's extreme over-regularisation).
    Optuna on rank-target, 60 trials, M1-5 subsample 1M rows, 3-fold CV.
    Full retrain with best params at n_estimators=20000.

GPU acceleration:
    - XGBoost: device='cuda'  (change to 'cpu' if unavailable)
    - LightGBM: CPU (n_jobs=-1)

Files produced (Part A):
    models/lgb_rank_i_oof.npy / lgb_rank_i_test.npy
    models/xgb_rank_i_oof.npy / xgb_rank_i_test.npy
    models/lgb_rank_i_fold{n}_oof.npy  (fold checkpoints)
    models/xgb_rank_i_fold{n}_oof.npy
    submissions/ensemble_i_a.csv

Files produced (Part B, additional):
    models/lgb_rank_ib_oof.npy / lgb_rank_ib_test.npy
    models/xgb_rank_ib_oof.npy / xgb_rank_ib_test.npy
    submissions/ensemble_i_b.csv

Expected runtime on GPU server: Part A ~60-90 min, Part B ~4-5h total.
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')

# ── CLI Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Experiment I: Rank-Target GBDT Re-tuning')
group = parser.add_mutually_exclusive_group()
group.add_argument('--part-a-only', action='store_true',
                   help='Run Part A only (increased n_estimators, skip Optuna)')
group.add_argument('--run-optuna', action='store_true',
                   help='Run Part A, then Part B (Optuna re-tuning + full retrain)')
args = parser.parse_args()

# Default: run Part A only if no flag given
RUN_OPTUNA = args.run_optuna

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

log_file = open('step_i_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

# Part B Optuna config
OPTUNA_TRIALS  = 60
OPTUNA_N       = 1_000_000   # subsample size from M1-5 rows
OPTUNA_FOLDS   = 3
OPTUNA_ITERS   = 5000        # faster iteration during search

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Experiment I: Rank-Target GBDT Re-tuning')
print(f'Mode: {"Part A + B (Optuna)" if RUN_OPTUNA else "Part A only (increased iterations)"}')
print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]

y_orig = train_df['invalid_ratio'].values
# Rank target: maps y to uniform [0,1] — MSE on ranks directly minimises
# rank errors (what Spearman measures). Same approach as Exp C.
y_rank = rankdata(y_orig, method='average') / len(y_orig)

sample_weight = np.log1p(train_df['total_count'].values)  # same as v7
m15_mask      = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values

X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]

print(f'  Train: {train_df.shape}  Test: {test_df.shape}')
print(f'  Features ({len(FEATURES)}): {FEATURES}')
print(f'\n  y_orig — min: {y_orig.min():.4f}  max: {y_orig.max():.4f}  '
      f'mean: {y_orig.mean():.4f}')
print(f'  y_rank — min: {y_rank.min():.6f}  max: {y_rank.max():.6f}  '
      f'mean: {y_rank.mean():.4f}')
print(f'  Sample weight — mean: {sample_weight.mean():.3f}  '
      f'max: {sample_weight.max():.3f}')
print(f'  M1-5 rows: {m15_mask.sum():,} / {len(train_df):,}')

# ── Part A Model Params (Optuna v3 params, increased n_estimators) ─────────────
# Key change vs Exp C: n_estimators LGB 10000→20000, XGB 10000→15000
# Rationale: Exp C all 5 LGB folds hit best_iter=9998-10000 (budget limit, not convergence)
lgb_params_a = {
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
    'n_estimators':      20000,   # was 10000 in Exp C
}

xgb_params_a = {
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
    'n_estimators':         15000,    # was 10000 in Exp C
    'verbosity':            0,
    'random_state':         SEED,
    'n_jobs':               -1,
    'early_stopping_rounds': 200,     # was 150 in Exp C
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# ════════════════════════════════════════════════════════════════════════════
# PART A — LGB: Increased n_estimators (20000)
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'PART A — LGB Rank Target (5-Fold, n_estimators=20000)')
print(f'{"="*60}')

lgb_rank_i_oof   = np.zeros(len(train_df))
lgb_rank_i_test  = np.zeros(len(test_df))
lgb_rank_i_scores     = []
lgb_rank_i_best_iters = []
t_lgb = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_rank)):
    oof_ckpt  = f'{MODEL_DIR}lgb_rank_i_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}lgb_rank_i_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        lgb_rank_i_oof[va_idx]  = np.load(oof_ckpt)
        lgb_rank_i_test        += np.load(test_ckpt)
        fold_rho = spearmanr(y_orig[va_idx], lgb_rank_i_oof[va_idx])[0]
        lgb_rank_i_scores.append(fold_rho)
        lgb_rank_i_best_iters.append(-1)
        print(f'  Fold {fold}: RESUMED from checkpoint  Spearman={fold_rho:.4f}')
        continue

    X_tr = X_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_tr = y_rank[tr_idx]
    y_va = y_rank[va_idx]
    w_tr = sample_weight[tr_idx]

    model = lgb.LGBMRegressor(**lgb_params_a)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),  # was 150 in Exp C
            lgb.log_evaluation(period=2000),
        ]
    )

    lgb_rank_i_oof[va_idx] = model.predict(X_va)
    fold_test_pred = model.predict(X_test) / N_FOLDS
    lgb_rank_i_test += fold_test_pred

    fold_rho = spearmanr(y_orig[va_idx], lgb_rank_i_oof[va_idx])[0]
    lgb_rank_i_scores.append(fold_rho)
    lgb_rank_i_best_iters.append(model.best_iteration_)

    np.save(oof_ckpt, lgb_rank_i_oof[va_idx])
    np.save(test_ckpt, fold_test_pred)

    elapsed = (time.time() - t_lgb) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
          f'best_iter={model.best_iteration_}  [{elapsed:.1f} min]')

lgb_rho_full_a = spearmanr(y_orig, lgb_rank_i_oof)[0]
lgb_rho_m15_a  = spearmanr(y_orig[m15_mask], lgb_rank_i_oof[m15_mask])[0]
lgb_total_min  = (time.time() - t_lgb) / 60

print(f'\n  LGB (Exp I-A) — OOF Spearman:  {lgb_rho_full_a:.4f}  (Exp C: 0.6373)')
print(f'  LGB (Exp I-A) — M1-5 Spearman: {lgb_rho_m15_a:.4f}  (Exp C: ~0.6455)')
print(f'  Fold scores: {[f"{s:.4f}" for s in lgb_rank_i_scores]}')
print(f'  Best iters:  {lgb_rank_i_best_iters}')
print(f'  Total time:  {lgb_total_min:.1f} min')

# Diagnosis: did iterations actually help?
still_at_limit = sum(1 for it in lgb_rank_i_best_iters
                     if it > 0 and it >= lgb_params_a['n_estimators'] - 10)
if still_at_limit > 0:
    print(f'  WARNING: {still_at_limit}/5 folds still at iteration limit — '
          f'consider increasing n_estimators further.')
else:
    print(f'  Early stopping triggered: model found true optimum within budget.')

np.save(f'{MODEL_DIR}lgb_rank_i_oof.npy',  lgb_rank_i_oof)
np.save(f'{MODEL_DIR}lgb_rank_i_test.npy', lgb_rank_i_test)
print(f'  Saved: lgb_rank_i_oof.npy, lgb_rank_i_test.npy')


# ════════════════════════════════════════════════════════════════════════════
# PART A — XGB: Increased n_estimators (15000)
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'PART A — XGB Rank Target (5-Fold, n_estimators=15000)')
print(f'{"="*60}')

xgb_rank_i_oof   = np.zeros(len(train_df))
xgb_rank_i_test  = np.zeros(len(test_df))
xgb_rank_i_scores     = []
xgb_rank_i_best_iters = []
t_xgb = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_rank)):
    oof_ckpt  = f'{MODEL_DIR}xgb_rank_i_fold{fold}_oof.npy'
    test_ckpt = f'{MODEL_DIR}xgb_rank_i_fold{fold}_test.npy'

    if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
        xgb_rank_i_oof[va_idx]  = np.load(oof_ckpt)
        xgb_rank_i_test        += np.load(test_ckpt)
        fold_rho = spearmanr(y_orig[va_idx], xgb_rank_i_oof[va_idx])[0]
        xgb_rank_i_scores.append(fold_rho)
        xgb_rank_i_best_iters.append(-1)
        print(f'  Fold {fold}: RESUMED from checkpoint  Spearman={fold_rho:.4f}')
        continue

    X_tr = X_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_tr = y_rank[tr_idx]
    y_va = y_rank[va_idx]
    w_tr = sample_weight[tr_idx]

    model = xgb.XGBRegressor(**xgb_params_a)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        sample_weight=w_tr,
        verbose=2000,
    )

    xgb_rank_i_oof[va_idx] = model.predict(X_va)
    fold_test_pred = model.predict(X_test) / N_FOLDS
    xgb_rank_i_test += fold_test_pred

    fold_rho = spearmanr(y_orig[va_idx], xgb_rank_i_oof[va_idx])[0]
    xgb_rank_i_scores.append(fold_rho)
    xgb_rank_i_best_iters.append(model.best_iteration)

    np.save(oof_ckpt, xgb_rank_i_oof[va_idx])
    np.save(test_ckpt, fold_test_pred)

    elapsed = (time.time() - t_xgb) / 60
    print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
          f'best_iter={model.best_iteration}  [{elapsed:.1f} min]')

xgb_rho_full_a = spearmanr(y_orig, xgb_rank_i_oof)[0]
xgb_rho_m15_a  = spearmanr(y_orig[m15_mask], xgb_rank_i_oof[m15_mask])[0]
xgb_total_min  = (time.time() - t_xgb) / 60

print(f'\n  XGB (Exp I-A) — OOF Spearman:  {xgb_rho_full_a:.4f}  (Exp C: 0.6430)')
print(f'  XGB (Exp I-A) — M1-5 Spearman: {xgb_rho_m15_a:.4f}  (Exp C: ~0.6510)')
print(f'  Fold scores: {[f"{s:.4f}" for s in xgb_rank_i_scores]}')
print(f'  Best iters:  {xgb_rank_i_best_iters}')
print(f'  Total time:  {xgb_total_min:.1f} min')

np.save(f'{MODEL_DIR}xgb_rank_i_oof.npy',  xgb_rank_i_oof)
np.save(f'{MODEL_DIR}xgb_rank_i_test.npy', xgb_rank_i_test)
print(f'  Saved: xgb_rank_i_oof.npy, xgb_rank_i_test.npy')


# ════════════════════════════════════════════════════════════════════════════
# Inter-model correlations (Part A models)
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'Inter-Model Correlations (Part A)')
print(f'{"="*60}')

corr_lr_a = spearmanr(lgb_rank_i_oof, xgb_rank_i_oof)[0]
print(f'  rank LGB_i — rank XGB_i: {corr_lr_a:.4f}')

# Load Exp C baseline for comparison
def _load_npy(path):
    if os.path.exists(path):
        return np.load(path)
    return None

c_lgb_oof = _load_npy(f'{MODEL_DIR}lgb_rank_oof.npy')
c_xgb_oof = _load_npy(f'{MODEL_DIR}xgb_rank_oof.npy')

if c_lgb_oof is not None:
    print(f'  Exp C LGB — Exp I LGB:   {spearmanr(c_lgb_oof, lgb_rank_i_oof)[0]:.4f}')
    print(f'  Exp C XGB — Exp I XGB:   {spearmanr(c_xgb_oof, xgb_rank_i_oof)[0]:.4f}')
    print(f'  Exp C LGB — Exp C XGB:   {spearmanr(c_lgb_oof, c_xgb_oof)[0]:.4f}  (ref)')
else:
    print('  (Exp C OOF files not found; skipping comparison)')


# ════════════════════════════════════════════════════════════════════════════
# Ensemble A — Fine-grained search on M1-5 OOF (step=0.01)
# Test set is M1-5 only, so optimising on M1-5 OOF gives better test weights
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'Ensemble A: Fine-grained Weight Search (step=0.01, M1-5 OOF)')
print(f'{"="*60}')

# M1-5 subset for weight optimisation
lgb_m15 = lgb_rank_i_oof[m15_mask]
xgb_m15 = xgb_rank_i_oof[m15_mask]
y_m15   = y_orig[m15_mask]

best_a_rho_m15, best_a_lgb_w = -1, 0.5
for lgb_w in np.arange(0.0, 1.01, 0.01):
    xgb_w = round(1.0 - lgb_w, 2)
    pred  = lgb_w * lgb_m15 + xgb_w * xgb_m15
    rho   = spearmanr(y_m15, pred)[0]
    if rho > best_a_rho_m15:
        best_a_rho_m15 = rho
        best_a_lgb_w   = lgb_w

best_a_xgb_w = round(1.0 - best_a_lgb_w, 2)
ens_a_oof    = best_a_lgb_w * lgb_rank_i_oof + best_a_xgb_w * xgb_rank_i_oof
ens_a_rho_full = spearmanr(y_orig, ens_a_oof)[0]
ens_a_rho_m15  = spearmanr(y_orig[m15_mask], ens_a_oof[m15_mask])[0]

print(f'  Best weights (M1-5 OOF): LGB={best_a_lgb_w:.2f}  XGB={best_a_xgb_w:.2f}')
print(f'  OOF Spearman (full):  {ens_a_rho_full:.4f}  (Exp C: 0.6464)')
print(f'  OOF Spearman (M1-5):  {ens_a_rho_m15:.4f}  (Exp C: 0.6527)')

# Also report weights optimised on full OOF (for comparison)
best_full_rho, best_full_lgb_w = -1, 0.5
for lgb_w in np.arange(0.0, 1.01, 0.01):
    xgb_w = round(1.0 - lgb_w, 2)
    pred  = lgb_w * lgb_rank_i_oof + xgb_w * xgb_rank_i_oof
    rho   = spearmanr(y_orig, pred)[0]
    if rho > best_full_rho:
        best_full_rho = rho
        best_full_lgb_w = lgb_w
best_full_xgb_w = round(1.0 - best_full_lgb_w, 2)
print(f'  (Full OOF best): LGB={best_full_lgb_w:.2f}  XGB={best_full_xgb_w:.2f}  '
      f'OOF={best_full_rho:.4f}')

# Generate submission using M1-5 optimised weights
ens_a_test = best_a_lgb_w * lgb_rank_i_test + best_a_xgb_w * xgb_rank_i_test
ens_a_test = np.clip(ens_a_test, 0, 1)
sub_a = pd.DataFrame({'id': test_df.index, 'invalid_ratio': ens_a_test})
sub_a.to_csv(f'{SUBMIT_DIR}ensemble_i_a.csv', index=False)
print(f'\n  Saved: submissions/ensemble_i_a.csv  '
      f'({len(ens_a_test):,} rows, '
      f'range [{ens_a_test.min():.4f}, {ens_a_test.max():.4f}])')
print(f'  Success criterion (OOF >= 0.6470): '
      f'{"✓ PASS" if ens_a_rho_full >= 0.6470 else "✗ FAIL"}')

# Decision guidance for Part B
if ens_a_rho_full >= 0.6480:
    print(f'\n  Decision: OOF >= 0.6480 → Strong result; submit ensemble_i_a.csv')
elif ens_a_rho_full >= 0.6470:
    print(f'\n  Decision: OOF 0.6470-0.6480 → Borderline; '
          f'Part B Optuna may still help')
else:
    print(f'\n  Decision: OOF < 0.6470 → Part B Optuna recommended')


# ════════════════════════════════════════════════════════════════════════════
# PART B — Optuna Re-tuning for Rank-Target (conditional)
# ════════════════════════════════════════════════════════════════════════════
if not RUN_OPTUNA:
    print(f'\n{"="*60}')
    print(f'Part B (Optuna) SKIPPED — use --run-optuna to enable')
    print(f'{"="*60}')
else:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print('ERROR: optuna not installed. Run: pip install optuna')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'PART B — Optuna Re-tuning (rank-target, relaxed ranges)')
    print(f'Trials: {OPTUNA_TRIALS}  Subsample: {OPTUNA_N:,} M1-5 rows  '
          f'Folds: {OPTUNA_FOLDS}  Iters: {OPTUNA_ITERS}')
    print(f'{"="*60}')

    # Subsample from M1-5 rows only (test distribution)
    m15_indices = np.where(m15_mask)[0]
    rng = np.random.RandomState(SEED)
    n_sample = min(OPTUNA_N, len(m15_indices))
    sample_idx = rng.choice(m15_indices, size=n_sample, replace=False)
    sample_idx = np.sort(sample_idx)

    X_opt = X_train.iloc[sample_idx].values
    y_opt_orig = y_orig[sample_idx]
    # Recompute rank target on the subsample (consistent with training approach)
    y_opt_rank = rankdata(y_opt_orig, method='average') / len(y_opt_orig)
    w_opt = sample_weight[sample_idx]

    print(f'  Optuna subsample: {len(sample_idx):,} M1-5 rows')
    kf_opt = KFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)

    # ── LGB Optuna Search ──────────────────────────────────────────────────
    print(f'\n  Searching LGB hyperparameters...')
    t_opt_lgb = time.time()

    def lgb_objective(trial):
        # Ranges centred near v7 values — avoid Step 12's over-regularisation
        params = {
            'num_leaves':        trial.suggest_int('num_leaves', 60, 200),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 40, 150),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.1, 3.0, log=True),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.3, 3.0, log=True),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.70, 0.95),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.80, 0.98),
            'bagging_freq':      5,
            'objective':         'regression',
            'metric':            'l2',
            'boosting_type':     'gbdt',
            'verbose':           -1,
            'n_jobs':            -1,
            'random_state':      SEED,
            'n_estimators':      OPTUNA_ITERS,
        }

        fold_scores = []
        for tr_idx, va_idx in kf_opt.split(X_opt):
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_opt[tr_idx], y_opt_rank[tr_idx],
                eval_set=[(X_opt[va_idx], y_opt_rank[va_idx])],
                sample_weight=w_opt[tr_idx],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
            )
            pred = m.predict(X_opt[va_idx])
            # Evaluate on original y (Spearman with true invalid_ratio)
            rho = spearmanr(y_opt_orig[va_idx], pred)[0]
            fold_scores.append(rho)
        return np.mean(fold_scores)

    lgb_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    lgb_study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    lgb_best_params = lgb_study.best_params
    lgb_best_trial  = lgb_study.best_value
    print(f'  LGB best trial Spearman: {lgb_best_trial:.4f}  '
          f'[{(time.time()-t_opt_lgb)/60:.1f} min]')
    print(f'  LGB best params: {lgb_best_params}')

    # ── XGB Optuna Search ──────────────────────────────────────────────────
    print(f'\n  Searching XGB hyperparameters...')
    t_opt_xgb = time.time()

    def xgb_objective(trial):
        params = {
            'max_depth':         trial.suggest_int('max_depth', 6, 12),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.06, log=True),
            'min_child_weight':  trial.suggest_int('min_child_weight', 5, 50),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 3.0, log=True),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.5, 3.0, log=True),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.60, 1.0),
            'subsample':         trial.suggest_float('subsample', 0.80, 1.0),
            'objective':         'reg:squarederror',
            'eval_metric':       'rmse',
            'tree_method':       'hist',
            'device':            'cuda',
            'n_estimators':      OPTUNA_ITERS,
            'verbosity':         0,
            'random_state':      SEED,
            'n_jobs':            -1,
            'early_stopping_rounds': 50,
        }

        fold_scores = []
        for tr_idx, va_idx in kf_opt.split(X_opt):
            m = xgb.XGBRegressor(**params)
            m.fit(
                X_opt[tr_idx], y_opt_rank[tr_idx],
                eval_set=[(X_opt[va_idx], y_opt_rank[va_idx])],
                sample_weight=w_opt[tr_idx],
                verbose=False,
            )
            pred = m.predict(X_opt[va_idx])
            rho = spearmanr(y_opt_orig[va_idx], pred)[0]
            fold_scores.append(rho)
        return np.mean(fold_scores)

    xgb_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    xgb_best_params = xgb_study.best_params
    xgb_best_trial  = xgb_study.best_value
    print(f'  XGB best trial Spearman: {xgb_best_trial:.4f}  '
          f'[{(time.time()-t_opt_xgb)/60:.1f} min]')
    print(f'  XGB best params: {xgb_best_params}')

    # ── Part B Full Retrain with Best Params ───────────────────────────────
    print(f'\n{"="*60}')
    print(f'PART B — Full Retrain with Optuna Best Params (5-Fold)')
    print(f'{"="*60}')

    # Build full param dicts for LGB Part B (merge best params with fixed params)
    lgb_params_b = {
        **lgb_best_params,
        'bagging_freq':    5,
        'objective':       'regression',
        'metric':          'l2',
        'boosting_type':   'gbdt',
        'verbose':         -1,
        'n_jobs':          -1,
        'random_state':    SEED,
        'n_estimators':    20000,   # same budget as Part A
    }
    xgb_params_b = {
        **xgb_best_params,
        'objective':              'reg:squarederror',
        'eval_metric':            'rmse',
        'tree_method':            'hist',
        'device':                 'cuda',
        'n_estimators':           15000,
        'verbosity':              0,
        'random_state':           SEED,
        'n_jobs':                 -1,
        'early_stopping_rounds':  200,
    }

    print(f'\n  LGB Part B params:')
    for k, v in lgb_params_b.items():
        print(f'    {k}: {v}')
    print(f'\n  XGB Part B params:')
    for k, v in xgb_params_b.items():
        print(f'    {k}: {v}')

    # LGB Part B training
    print(f'\n  Training LGB Part B...')
    lgb_rank_ib_oof   = np.zeros(len(train_df))
    lgb_rank_ib_test  = np.zeros(len(test_df))
    lgb_rank_ib_scores     = []
    lgb_rank_ib_best_iters = []
    t_lgb_b = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_rank)):
        oof_ckpt  = f'{MODEL_DIR}lgb_rank_ib_fold{fold}_oof.npy'
        test_ckpt = f'{MODEL_DIR}lgb_rank_ib_fold{fold}_test.npy'

        if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
            lgb_rank_ib_oof[va_idx]  = np.load(oof_ckpt)
            lgb_rank_ib_test        += np.load(test_ckpt)
            fold_rho = spearmanr(y_orig[va_idx], lgb_rank_ib_oof[va_idx])[0]
            lgb_rank_ib_scores.append(fold_rho)
            lgb_rank_ib_best_iters.append(-1)
            print(f'  Fold {fold}: RESUMED  Spearman={fold_rho:.4f}')
            continue

        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_rank[tr_idx]
        y_va = y_rank[va_idx]
        w_tr = sample_weight[tr_idx]

        model = lgb.LGBMRegressor(**lgb_params_b)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            sample_weight=w_tr,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=2000),
            ]
        )

        lgb_rank_ib_oof[va_idx] = model.predict(X_va)
        fold_test_pred = model.predict(X_test) / N_FOLDS
        lgb_rank_ib_test += fold_test_pred

        fold_rho = spearmanr(y_orig[va_idx], lgb_rank_ib_oof[va_idx])[0]
        lgb_rank_ib_scores.append(fold_rho)
        lgb_rank_ib_best_iters.append(model.best_iteration_)

        np.save(oof_ckpt, lgb_rank_ib_oof[va_idx])
        np.save(test_ckpt, fold_test_pred)

        elapsed = (time.time() - t_lgb_b) / 60
        print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
              f'best_iter={model.best_iteration_}  [{elapsed:.1f} min]')

    lgb_rho_full_b = spearmanr(y_orig, lgb_rank_ib_oof)[0]
    lgb_rho_m15_b  = spearmanr(y_orig[m15_mask], lgb_rank_ib_oof[m15_mask])[0]
    print(f'\n  LGB (Exp I-B) — OOF: {lgb_rho_full_b:.4f}  M1-5: {lgb_rho_m15_b:.4f}')
    print(f'  Best iters: {lgb_rank_ib_best_iters}')

    np.save(f'{MODEL_DIR}lgb_rank_ib_oof.npy',  lgb_rank_ib_oof)
    np.save(f'{MODEL_DIR}lgb_rank_ib_test.npy', lgb_rank_ib_test)

    # XGB Part B training
    print(f'\n  Training XGB Part B...')
    xgb_rank_ib_oof   = np.zeros(len(train_df))
    xgb_rank_ib_test  = np.zeros(len(test_df))
    xgb_rank_ib_scores     = []
    xgb_rank_ib_best_iters = []
    t_xgb_b = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_rank)):
        oof_ckpt  = f'{MODEL_DIR}xgb_rank_ib_fold{fold}_oof.npy'
        test_ckpt = f'{MODEL_DIR}xgb_rank_ib_fold{fold}_test.npy'

        if os.path.exists(oof_ckpt) and os.path.exists(test_ckpt):
            xgb_rank_ib_oof[va_idx]  = np.load(oof_ckpt)
            xgb_rank_ib_test        += np.load(test_ckpt)
            fold_rho = spearmanr(y_orig[va_idx], xgb_rank_ib_oof[va_idx])[0]
            xgb_rank_ib_scores.append(fold_rho)
            xgb_rank_ib_best_iters.append(-1)
            print(f'  Fold {fold}: RESUMED  Spearman={fold_rho:.4f}')
            continue

        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_rank[tr_idx]
        y_va = y_rank[va_idx]
        w_tr = sample_weight[tr_idx]

        model = xgb.XGBRegressor(**xgb_params_b)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            sample_weight=w_tr,
            verbose=2000,
        )

        xgb_rank_ib_oof[va_idx] = model.predict(X_va)
        fold_test_pred = model.predict(X_test) / N_FOLDS
        xgb_rank_ib_test += fold_test_pred

        fold_rho = spearmanr(y_orig[va_idx], xgb_rank_ib_oof[va_idx])[0]
        xgb_rank_ib_scores.append(fold_rho)
        xgb_rank_ib_best_iters.append(model.best_iteration)

        np.save(oof_ckpt, xgb_rank_ib_oof[va_idx])
        np.save(test_ckpt, fold_test_pred)

        elapsed = (time.time() - t_xgb_b) / 60
        print(f'  Fold {fold}: Spearman={fold_rho:.4f}  '
              f'best_iter={model.best_iteration}  [{elapsed:.1f} min]')

    xgb_rho_full_b = spearmanr(y_orig, xgb_rank_ib_oof)[0]
    xgb_rho_m15_b  = spearmanr(y_orig[m15_mask], xgb_rank_ib_oof[m15_mask])[0]
    print(f'\n  XGB (Exp I-B) — OOF: {xgb_rho_full_b:.4f}  M1-5: {xgb_rho_m15_b:.4f}')
    print(f'  Best iters: {xgb_rank_ib_best_iters}')

    np.save(f'{MODEL_DIR}xgb_rank_ib_oof.npy',  xgb_rank_ib_oof)
    np.save(f'{MODEL_DIR}xgb_rank_ib_test.npy', xgb_rank_ib_test)

    # ── Ensemble B: Fine-grained search on M1-5 OOF ───────────────────────
    print(f'\n{"="*60}')
    print(f'Ensemble B: Fine-grained Weight Search (step=0.01, M1-5 OOF)')
    print(f'{"="*60}')

    lgb_m15_b = lgb_rank_ib_oof[m15_mask]
    xgb_m15_b = xgb_rank_ib_oof[m15_mask]

    best_b_rho_m15, best_b_lgb_w = -1, 0.5
    for lgb_w in np.arange(0.0, 1.01, 0.01):
        xgb_w = round(1.0 - lgb_w, 2)
        pred  = lgb_w * lgb_m15_b + xgb_w * xgb_m15_b
        rho   = spearmanr(y_m15, pred)[0]
        if rho > best_b_rho_m15:
            best_b_rho_m15 = rho
            best_b_lgb_w   = lgb_w

    best_b_xgb_w = round(1.0 - best_b_lgb_w, 2)
    ens_b_oof    = best_b_lgb_w * lgb_rank_ib_oof + best_b_xgb_w * xgb_rank_ib_oof
    ens_b_rho_full = spearmanr(y_orig, ens_b_oof)[0]
    ens_b_rho_m15  = spearmanr(y_orig[m15_mask], ens_b_oof[m15_mask])[0]

    print(f'  Best weights (M1-5 OOF): LGB={best_b_lgb_w:.2f}  XGB={best_b_xgb_w:.2f}')
    print(f'  OOF Spearman (full):  {ens_b_rho_full:.4f}  (Exp C: 0.6464)')
    print(f'  OOF Spearman (M1-5):  {ens_b_rho_m15:.4f}  (Exp C: 0.6527)')

    ens_b_test = best_b_lgb_w * lgb_rank_ib_test + best_b_xgb_w * xgb_rank_ib_test
    ens_b_test = np.clip(ens_b_test, 0, 1)
    sub_b = pd.DataFrame({'id': test_df.index, 'invalid_ratio': ens_b_test})
    sub_b.to_csv(f'{SUBMIT_DIR}ensemble_i_b.csv', index=False)
    print(f'\n  Saved: submissions/ensemble_i_b.csv  '
          f'({len(ens_b_test):,} rows, '
          f'range [{ens_b_test.min():.4f}, {ens_b_test.max():.4f}])')
    print(f'  Success criterion (OOF >= 0.6480): '
          f'{"✓ PASS" if ens_b_rho_full >= 0.6480 else "✗ FAIL"}')


# ════════════════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'EXPERIMENT I COMPLETE — {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}')
print(f'\n  {"Model":<30} {"OOF":>8} {"M1-5":>8} {"vs Exp C":>10}')
print(f'  {"-"*60}')
print(f'  {"Exp C (baseline)":30} {"0.6464":>8} {"0.6527":>8} {"—":>10}')
print(f'  {"LGB I-A (n_iter=20000)":30} {lgb_rho_full_a:>8.4f} {lgb_rho_m15_a:>8.4f} '
      f'{lgb_rho_full_a-0.6373:>+10.4f}')
print(f'  {"XGB I-A (n_iter=15000)":30} {xgb_rho_full_a:>8.4f} {xgb_rho_m15_a:>8.4f} '
      f'{xgb_rho_full_a-0.6430:>+10.4f}')
print(f'  {"Ensemble I-A":30} {ens_a_rho_full:>8.4f} {ens_a_rho_m15:>8.4f} '
      f'{ens_a_rho_full-0.6464:>+10.4f}')

if RUN_OPTUNA:
    print(f'  {"LGB I-B (Optuna+retrain)":30} {lgb_rho_full_b:>8.4f} {lgb_rho_m15_b:>8.4f} '
          f'{lgb_rho_full_b-0.6373:>+10.4f}')
    print(f'  {"XGB I-B (Optuna+retrain)":30} {xgb_rho_full_b:>8.4f} {xgb_rho_m15_b:>8.4f} '
          f'{xgb_rho_full_b-0.6430:>+10.4f}')
    print(f'  {"Ensemble I-B":30} {ens_b_rho_full:>8.4f} {ens_b_rho_m15:>8.4f} '
          f'{ens_b_rho_full-0.6464:>+10.4f}')

print(f'\n  Files to download for local analysis:')
print(f'    models/lgb_rank_i_oof.npy   lgb_rank_i_test.npy')
print(f'    models/xgb_rank_i_oof.npy   xgb_rank_i_test.npy')
print(f'    submissions/ensemble_i_a.csv')
if RUN_OPTUNA:
    print(f'    models/lgb_rank_ib_oof.npy  lgb_rank_ib_test.npy')
    print(f'    models/xgb_rank_ib_oof.npy  xgb_rank_ib_test.npy')
    print(f'    submissions/ensemble_i_b.csv')
print(f'    step_i_gpu.log')
