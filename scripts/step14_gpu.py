"""
Step 14: Neural Network (MLP/ResNet) — NN v1 + 4-Model Ensemble v11
=====================================================================
Run from the project root:
    conda activate parking
    python scripts/step14_gpu.py

Architecture: 6-layer residual MLP (following Vo 2025)
  Input (26) → BatchNorm → Linear(256) → Linear(128) → Linear(64)
  → [skip] Linear(64)→Linear(64)+skip [/skip]
  → Linear(32) → Linear(1) → Sigmoid

Key design choices:
  - Sigmoid output: target is [0,1] ratio
  - Skip connection in the 64-dim block for deeper gradient flow
  - Weighted MSE loss: consistent with LGB/XGB sample_weight=log1p(total_count)
  - CosineAnnealingLR scheduler
  - Early stopping on val Spearman (patience=5 epochs)
  - Per-fold StandardScaler (fit on train, transform val/test)
  - Fold checkpoints: if server restarts, completed folds are skipped

Ensemble integration:
  - 4-model ensemble: LGB_v7 + XGB_v7 + CB_v4 + NN_v1
  - Two submissions:
      ensemble_v11.csv  — full-data TE
      ensemble_v11a.csv — M1-5 TE (reuse Step 11 logic)

Success criteria:
  - NN OOF >= 0.58
  - NN correlation with v7 Ensemble < 0.90
  - 4-model Ensemble OOF > 0.6429 (v7 baseline)

Expected runtime: ~2-3h on GPU server (5 folds x ~30 epochs x 6M rows).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

log_file = open('step14_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
DATA_DIR   = 'data/'
MODEL_DIR  = 'models/'
SUBMIT_DIR = 'submissions/'

# NN hyperparameters
BATCH_SIZE  = 4096
LR          = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS  = 30
ES_PATIENCE = 5
INFER_BATCH = 8192   # larger batch for fast inference (no gradients)

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f'\n{"="*60}')
print(f'Step 14 (Neural Network v1) started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── Load data ─────────────────────────────────────────────────────────────────
print('\nLoading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')

TARGET       = 'invalid_ratio'
EXCLUDE_COLS = ['invalid_ratio', 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']
FEATURES     = [c for c in train_df.columns if c not in EXCLUDE_COLS]
y            = train_df[TARGET].values

print(f'  Train: {train_df.shape}, Test: {test_df.shape}')
print(f'  Features (26): {len(FEATURES)}')

m1_5_mask = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values
print(f'  M1-5 rows: {m1_5_mask.sum():,}  |  M6-12 rows: {(~m1_5_mask).sum():,}')

# ── Load v7 / CB v4 baselines ─────────────────────────────────────────────────
print('\nLoading v7 / CB v4 baselines...')
lgb_oof_v7  = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7  = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7 = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7 = np.load(f'{MODEL_DIR}xgb_test_v7.npy')

# CB v4 is optional — its ensemble weight is usually 0.00
cb_oof_path  = f'{MODEL_DIR}cb_oof_v4.npy'
cb_test_path = f'{MODEL_DIR}cb_test_v4.npy'
if os.path.exists(cb_oof_path) and os.path.exists(cb_test_path):
    cb_oof_v4  = np.load(cb_oof_path)
    cb_test_v4 = np.load(cb_test_path)
    print('  Loaded CB v4 predictions.')
else:
    cb_oof_v4  = np.zeros(len(train_df))
    cb_test_v4 = np.zeros(len(test_df))
    print('  WARNING: cb_oof_v4.npy not found — using zeros (CB weight will be 0).')

# Also load v8a XGB (M1-5 TE) for ensemble_v11a
xgb_test_v8a_path = f'{MODEL_DIR}xgb_test_v8a.npy'
if os.path.exists(xgb_test_v8a_path):
    xgb_test_v8a = np.load(xgb_test_v8a_path)
    xgb_oof_v8a  = np.load(f'{MODEL_DIR}xgb_oof_v8a.npy')
    print('  Loaded XGB v8a predictions (M1-5 TE) for ensemble_v11a.')
else:
    xgb_test_v8a = xgb_test_v7
    xgb_oof_v8a  = xgb_oof_v7
    print('  WARNING: xgb_test_v8a.npy not found — ensemble_v11a will use v7 XGB.')

lgb_oof_v7_rho = spearmanr(y, lgb_oof_v7)[0]
xgb_oof_v7_rho = spearmanr(y, xgb_oof_v7)[0]
cb_oof_v4_rho  = spearmanr(y, cb_oof_v4)[0] if cb_oof_v4.any() else 0.0

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

ens_v7_oof = best_w_v7[0]*lgb_oof_v7 + best_w_v7[1]*xgb_oof_v7 + best_w_v7[2]*cb_oof_v4
ens_v7_m1_5 = spearmanr(y[m1_5_mask], ens_v7_oof[m1_5_mask])[0]

print(f'\n  LGB v7 OOF:      {lgb_oof_v7_rho:.4f}')
print(f'  XGB v7 OOF:      {xgb_oof_v7_rho:.4f}')
print(f'  CB  v4 OOF:      {cb_oof_v4_rho:.4f}')
print(f'  Ensemble v7 OOF: {best_rho_v7:.4f}  weights={best_w_v7}')
print(f'  Ensemble v7 M1-5 OOF: {ens_v7_m1_5:.4f}')

# ── Build M1-5 TE for test (same logic as step11/step13 for ensemble_v11a) ────
print('\n=== Building M1-5 TE for Test (ensemble_v11a) ===\n')

train_m1_5      = train_df[m1_5_mask].copy()
global_mean_full = train_df['invalid_ratio'].mean()
global_mean_m1_5 = train_m1_5['invalid_ratio'].mean()

# grid_te: smooth=100 for M1-5 subset
full_stats_grid      = train_df.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])
full_stats_grid_m1_5 = train_m1_5.groupby('grid_id')['invalid_ratio'].agg(['mean', 'count'])

smooth_grid_orig = 30
smooth_grid_m1_5 = 100

enc_grid_orig = (full_stats_grid['count'] * full_stats_grid['mean']
                 + smooth_grid_orig * global_mean_full
                 ) / (full_stats_grid['count'] + smooth_grid_orig)
enc_grid_m1_5 = (full_stats_grid_m1_5['count'] * full_stats_grid_m1_5['mean']
                 + smooth_grid_m1_5 * global_mean_m1_5
                 ) / (full_stats_grid_m1_5['count'] + smooth_grid_m1_5)

test_grid_te_orig = test_df['grid_id'].map(enc_grid_orig).fillna(global_mean_full)
test_grid_te_m1_5 = test_df['grid_id'].map(enc_grid_m1_5).fillna(global_mean_m1_5)

# grid_period_te: smooth=150 for M1-5 subset
full_stats_gp      = train_df.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])
full_stats_gp_m1_5 = train_m1_5.groupby('grid_period')['invalid_ratio'].agg(['mean', 'count'])

smooth_gp_orig = 50
smooth_gp_m1_5 = 150

enc_gp_orig = (full_stats_gp['count'] * full_stats_gp['mean']
               + smooth_gp_orig * global_mean_full
               ) / (full_stats_gp['count'] + smooth_gp_orig)
enc_gp_m1_5 = (full_stats_gp_m1_5['count'] * full_stats_gp_m1_5['mean']
               + smooth_gp_m1_5 * global_mean_m1_5
               ) / (full_stats_gp_m1_5['count'] + smooth_gp_m1_5)

test_gp_te_orig = test_df['grid_period'].map(enc_gp_orig).fillna(global_mean_full)
test_gp_te_m1_5 = test_df['grid_period'].map(enc_gp_m1_5)
missing_mask = test_gp_te_m1_5.isna()
test_gp_te_m1_5[missing_mask] = test_grid_te_m1_5[missing_mask]  # fallback
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


X_train_raw = train_df[FEATURES].values.astype(np.float32)
X_test_full = test_df[FEATURES].values.astype(np.float32)
X_test_m1_5 = build_test_m1_5(test_df, FEATURES,
                                test_grid_te_m1_5, test_gp_te_m1_5).values.astype(np.float32)

# ── Sample weights (same as v7/v10) ──────────────────────────────────────────
sample_weight_all = np.log1p(train_df['total_count'].values).astype(np.float32)
print(f'\nSample weight stats: mean={sample_weight_all.mean():.3f}, '
      f'max={sample_weight_all.max():.3f}')

# ── Architecture: Parking ResNet ──────────────────────────────────────────────
# 6-layer MLP with one skip connection in the 64-dim block.
# Input → BN → 256 → 128 → 64 → [64 → 64 + skip] → 32 → 1 → sigmoid
class ParkingResNet(nn.Module):
    def __init__(self, n_features: int = 26):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)

        # Encoder: compress features
        self.fc1   = nn.Linear(n_features, 256)
        self.drop1 = nn.Dropout(0.3)

        self.fc2   = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)

        self.fc3   = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(0.2)

        # Residual block: 64 → 64 → 64 (with skip from input of block)
        self.fc4   = nn.Linear(64, 64)
        self.drop4 = nn.Dropout(0.2)
        self.fc5   = nn.Linear(64, 64)  # skip added after this

        # Decoder: compress to output
        self.fc6   = nn.Linear(64, 32)
        self.drop6 = nn.Dropout(0.1)

        self.fc7   = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_input(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = F.relu(self.fc3(x))
        x = self.drop3(x)

        # Residual block: save input before block, add it back after fc5
        residual = x
        x = F.relu(self.fc4(x))
        x = self.drop4(x)
        x = F.relu(self.fc5(x) + residual)  # skip connection

        x = F.relu(self.fc6(x))
        x = self.drop6(x)

        x = torch.sigmoid(self.fc7(x))
        return x.squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class WeightedRegressionDataset(Dataset):
    """Dataset returning (features, target, weight) triples."""
    def __init__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.w = torch.from_numpy(weights)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


def predict_batched(model: nn.Module, X: np.ndarray, batch_size: int = INFER_BATCH) -> np.ndarray:
    """Run model inference in batches; returns numpy array of predictions."""
    model.eval()
    X_tensor = torch.from_numpy(X)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X_tensor[start:start + batch_size].to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

nn_oof_v1    = np.zeros(len(train_df))
nn_test_v1   = np.zeros(len(test_df))    # full-data TE test predictions
nn_test_v1a  = np.zeros(len(test_df))    # M1-5 TE test predictions
nn_fold_rhos = []

print(f'\n=== NN v1 (ParkingResNet): 5-Fold CV ===\n')
print(f'  Arch: 26→256→128→64→[64→64+skip]→32→1')
print(f'  Batch={BATCH_SIZE}, LR={LR}, WD={WEIGHT_DECAY}, '
      f'MaxEpochs={MAX_EPOCHS}, ES patience={ES_PATIENCE}')
print(f'  Loss: weighted MSE  |  Eval: Spearman')
print(f'  Checkpoints: models/nn_v1_fold{{n}}_*.npy + models/nn_fold{{n}}.pt\n')
t0_total = time.time()

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_raw)):
    oof_ckpt      = f'{MODEL_DIR}nn_v1_fold{fold}_oof.npy'
    test_ckpt     = f'{MODEL_DIR}nn_v1_fold{fold}_test.npy'
    test_m1_5_ckpt = f'{MODEL_DIR}nn_v1_fold{fold}_test_m1_5.npy'
    model_ckpt    = f'{MODEL_DIR}nn_fold{fold}.pt'

    # Resume from checkpoint if all outputs already saved
    if (os.path.exists(oof_ckpt) and os.path.exists(test_ckpt)
            and os.path.exists(test_m1_5_ckpt)):
        nn_oof_v1[va_idx] = np.load(oof_ckpt)
        nn_test_v1       += np.load(test_ckpt) / N_FOLDS
        nn_test_v1a      += np.load(test_m1_5_ckpt) / N_FOLDS
        fold_rho = spearmanr(y[va_idx], nn_oof_v1[va_idx])[0]
        nn_fold_rhos.append(fold_rho)
        print(f'  Fold {fold}: loaded from checkpoint. Spearman={fold_rho:.4f}')
        continue

    t0_fold = time.time()

    # Per-fold StandardScaler (prevents TE leakage across folds)
    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_train_raw[tr_idx])
    X_va_scaled  = scaler.transform(X_train_raw[va_idx])
    X_test_full_scaled = scaler.transform(X_test_full)
    X_test_m1_5_scaled = scaler.transform(X_test_m1_5)

    y_tr = y[tr_idx]
    y_va = y[va_idx]
    w_tr = sample_weight_all[tr_idx]

    # Normalize weights to have mean=1 (keeps loss scale consistent across folds)
    w_tr_norm = w_tr / w_tr.mean()

    # DataLoader for training
    tr_dataset = WeightedRegressionDataset(X_tr_scaled.astype(np.float32),
                                            y_tr, w_tr_norm.astype(np.float32))
    tr_loader  = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=(device.type == 'cuda'))

    # Model, optimizer, scheduler
    model     = ParkingResNet(n_features=len(FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_val_rho   = -np.inf
    patience_count = 0
    best_state     = None

    print(f'  Fold {fold}: {len(tr_idx):,} train / {len(va_idx):,} val rows')

    for epoch in range(1, MAX_EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch, w_batch in tr_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            # Weighted MSE: down-weights noisy low-count samples
            loss = ((pred - y_batch) ** 2 * w_batch).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation phase: compute Spearman on full val set
        va_pred  = predict_batched(model, X_va_scaled.astype(np.float32))
        val_rho  = spearmanr(y_va, va_pred)[0]
        avg_loss = train_loss / n_batches

        print(f'    Epoch {epoch:2d}/{MAX_EPOCHS}: train_loss={avg_loss:.5f}, '
              f'val_Spearman={val_rho:.4f}', end='')

        if val_rho > best_val_rho:
            best_val_rho   = val_rho
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  ← best')
        else:
            patience_count += 1
            print(f'  (patience {patience_count}/{ES_PATIENCE})')

        if patience_count >= ES_PATIENCE:
            print(f'    Early stopping at epoch {epoch}.')
            break

    # Restore best model weights
    model.load_state_dict(best_state)
    torch.save(best_state, model_ckpt)

    # Generate predictions with best model
    va_pred_final   = predict_batched(model, X_va_scaled.astype(np.float32))
    test_pred_full  = predict_batched(model, X_test_full_scaled.astype(np.float32))
    test_pred_m1_5  = predict_batched(model, X_test_m1_5_scaled.astype(np.float32))

    # Save fold checkpoints
    np.save(oof_ckpt,       va_pred_final)
    np.save(test_ckpt,      test_pred_full)
    np.save(test_m1_5_ckpt, test_pred_m1_5)

    nn_oof_v1[va_idx] = va_pred_final
    nn_test_v1       += test_pred_full / N_FOLDS
    nn_test_v1a      += test_pred_m1_5 / N_FOLDS

    fold_rho = spearmanr(y_va, va_pred_final)[0]
    nn_fold_rhos.append(fold_rho)

    elapsed_fold = (time.time() - t0_fold) / 60
    print(f'  Fold {fold}: best val Spearman={best_val_rho:.4f}, '
          f'final Spearman={fold_rho:.4f}, time={elapsed_fold:.1f}min\n')

# ── NN OOF metrics ────────────────────────────────────────────────────────────
nn_oof_rho  = spearmanr(y, nn_oof_v1)[0]
nn_oof_m1_5 = spearmanr(y[m1_5_mask], nn_oof_v1[m1_5_mask])[0]
elapsed_total = (time.time() - t0_total) / 60

print(f'\nNN v1 OOF Spearman:  {nn_oof_rho:.4f}  (v7 Ensemble: {best_rho_v7:.4f})')
print(f'NN v1 M1-5 Spearman: {nn_oof_m1_5:.4f}  (v7 Ensemble M1-5: {ens_v7_m1_5:.4f})')
print(f'Fold scores: {[f"{s:.4f}" for s in nn_fold_rhos]}')
print(f'NN v1 total time: {elapsed_total:.1f} min')

# ── Save final NN predictions ─────────────────────────────────────────────────
np.save(f'{MODEL_DIR}nn_oof_v1.npy',   nn_oof_v1)
np.save(f'{MODEL_DIR}nn_test_v1.npy',  nn_test_v1)
np.save(f'{MODEL_DIR}nn_test_v1a.npy', nn_test_v1a)
print(f'\nSaved: nn_oof_v1.npy, nn_test_v1.npy, nn_test_v1a.npy')

# ── Inter-model correlations ──────────────────────────────────────────────────
corr_nn_lgb = np.corrcoef(nn_oof_v1, lgb_oof_v7)[0, 1]
corr_nn_xgb = np.corrcoef(nn_oof_v1, xgb_oof_v7)[0, 1]
corr_nn_cb  = np.corrcoef(nn_oof_v1, cb_oof_v4)[0, 1]
corr_nn_ens = np.corrcoef(nn_oof_v1, ens_v7_oof)[0, 1]

print(f'\n=== Inter-Model Correlations ===')
print(f'  NN v1 - LGB v7:        {corr_nn_lgb:.4f}  (LGB-XGB v7: 0.9681)')
print(f'  NN v1 - XGB v7:        {corr_nn_xgb:.4f}')
print(f'  NN v1 - CB  v4:        {corr_nn_cb:.4f}')
print(f'  NN v1 - Ensemble v7:   {corr_nn_ens:.4f}  (target: < 0.90)')

if corr_nn_ens < 0.90:
    print(f'  ✅ Diversity goal met — NN provides meaningful orthogonal signal.')
elif corr_nn_ens < 0.95:
    print(f'  ⚠️  Moderate correlation — some diversity benefit, but limited.')
else:
    print(f'  ❌ High correlation — NN adds little diversity over GBDT ensemble.')

# ── Success criteria check ────────────────────────────────────────────────────
print(f'\n=== NN v1 Success Criteria ===')
criteria = [
    ('NN OOF >= 0.58',      nn_oof_rho,   nn_oof_rho >= 0.58),
    ('NN-Ens corr < 0.90',  corr_nn_ens,  corr_nn_ens < 0.90),
]
for name, val, ok in criteria:
    print(f'  {name:<25s} {val:.4f}  {"✅ PASS" if ok else "❌ FAIL"}')

# ── 4-Model Ensemble v11 weight search (LGB_v7 + XGB_v7 + CB_v4 + NN_v1) ────
print(f'\n=== Ensemble v11: LGB_v7 + XGB_v7 + CB_v4 + NN_v1 (full-data TE) ===\n')

best_rho_v11 = 0.0
best_w_v11   = (0.0, 1.0, 0.0, 0.0)

# Grid search over weights with step=0.05 (4 models, ~7770 combos)
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < -0.001:
                continue
            w4 = max(w4, 0.0)
            blend = (w1*lgb_oof_v7 + w2*xgb_oof_v7
                     + w3*cb_oof_v4 + w4*nn_oof_v1)
            rho = spearmanr(y, blend)[0]
            if rho > best_rho_v11:
                best_rho_v11 = rho
                best_w_v11   = (round(w1,2), round(w2,2), round(w3,2), round(w4,2))

w1_v11, w2_v11, w3_v11, w4_v11 = best_w_v11
ens_oof_v11  = (w1_v11*lgb_oof_v7  + w2_v11*xgb_oof_v7
                + w3_v11*cb_oof_v4 + w4_v11*nn_oof_v1)
ens_test_v11 = (w1_v11*lgb_test_v7  + w2_v11*xgb_test_v7
                + w3_v11*cb_test_v4 + w4_v11*nn_test_v1)
ens_test_v11 = np.clip(ens_test_v11, 0, 1)

ens_v11_m1_5 = spearmanr(y[m1_5_mask], ens_oof_v11[m1_5_mask])[0]

print(f'  Best weights: LGB={w1_v11}, XGB={w2_v11}, CB={w3_v11}, NN={w4_v11}')
print(f'  Ensemble v11 OOF:  {best_rho_v11:.4f}'
      f'  (v7: {best_rho_v7:.4f}, delta: {best_rho_v11 - best_rho_v7:+.4f})')
print(f'  Ensemble v11 M1-5: {ens_v11_m1_5:.4f}'
      f'  (v7: {ens_v7_m1_5:.4f}, delta: {ens_v11_m1_5 - ens_v7_m1_5:+.4f})')

if best_rho_v11 > best_rho_v7:
    print(f'  ✅ Ensemble v11 beats v7 baseline!')
else:
    print(f'  ❌ Ensemble v11 does not improve over v7. '
          f'NN weight={w4_v11} — NN adds noise in ensemble.')

# Generate ensemble_v11.csv
sub_v11 = pd.DataFrame({'invalid_ratio': ens_test_v11})
sub_v11.to_csv(f'{SUBMIT_DIR}ensemble_v11.csv', index=True, index_label='')
print(f'\nSaved: {SUBMIT_DIR}ensemble_v11.csv')

# ── 4-Model Ensemble v11a weight search (M1-5 TE) ────────────────────────────
# Uses M1-5 TE test predictions for LGB (nn_test_v1a) and XGB (xgb_test_v8a).
# OOF weights are still searched on full OOF (training always uses full-data TE).
print(f'\n=== Ensemble v11a: LGB_v7 + XGB_v8a + CB_v4 + NN_v1 (M1-5 TE) ===\n')

best_rho_v11a = 0.0
best_w_v11a   = (0.0, 1.0, 0.0, 0.0)

for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < -0.001:
                continue
            w4 = max(w4, 0.0)
            # OOF uses same predictions — only test TE differs for submissions
            blend = (w1*lgb_oof_v7 + w2*xgb_oof_v8a
                     + w3*cb_oof_v4 + w4*nn_oof_v1)
            rho = spearmanr(y, blend)[0]
            if rho > best_rho_v11a:
                best_rho_v11a = rho
                best_w_v11a   = (round(w1,2), round(w2,2), round(w3,2), round(w4,2))

w1_v11a, w2_v11a, w3_v11a, w4_v11a = best_w_v11a
ens_oof_v11a  = (w1_v11a*lgb_oof_v7   + w2_v11a*xgb_oof_v8a
                 + w3_v11a*cb_oof_v4  + w4_v11a*nn_oof_v1)
ens_test_v11a = (w1_v11a*lgb_test_v7  + w2_v11a*xgb_test_v8a
                 + w3_v11a*cb_test_v4 + w4_v11a*nn_test_v1a)
ens_test_v11a = np.clip(ens_test_v11a, 0, 1)

ens_v11a_m1_5 = spearmanr(y[m1_5_mask], ens_oof_v11a[m1_5_mask])[0]

print(f'  Best weights: LGB={w1_v11a}, XGB={w2_v11a}, CB={w3_v11a}, NN={w4_v11a}')
print(f'  Ensemble v11a OOF:  {best_rho_v11a:.4f}  (v11: {best_rho_v11:.4f})')
print(f'  Ensemble v11a M1-5: {ens_v11a_m1_5:.4f}  (v11: {ens_v11_m1_5:.4f})')

# Generate ensemble_v11a.csv
sub_v11a = pd.DataFrame({'invalid_ratio': ens_test_v11a})
sub_v11a.to_csv(f'{SUBMIT_DIR}ensemble_v11a.csv', index=True, index_label='')
print(f'\nSaved: {SUBMIT_DIR}ensemble_v11a.csv')

# ── Pre-submission validation ─────────────────────────────────────────────────
print(f'\n=== Pre-Submission Validation ===')
for name, arr in [('ensemble_v11', ens_test_v11), ('ensemble_v11a', ens_test_v11a)]:
    checks = [
        ('NaN count',  str(np.isnan(arr).sum()),     np.isnan(arr).sum() == 0),
        ('Row count',  str(len(arr)),                len(arr) == 2028750),
        ('Range',      f'[{arr.min():.4f}, {arr.max():.4f}]',
                       arr.min() >= 0 and arr.max() <= 1),
    ]
    print(f'\n  {name}:')
    for label, val, ok in checks:
        print(f'    {label:<12s} {val:<30s} {"PASS" if ok else "FAIL"}')

# ── Full results summary ───────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('=== Complete Results Summary ===')
print(f'{"="*60}')
print(f'\n{"Version":<22s} {"LGB":>8s} {"XGB":>8s} {"CB":>8s} {"NN":>8s} '
      f'{"Ensemble":>10s} {"M1-5":>8s}')
print('-' * 78)
print(f'{"v7 (baseline)":<22s} {lgb_oof_v7_rho:>8.4f} {xgb_oof_v7_rho:>8.4f}'
      f' {cb_oof_v4_rho:>8.4f} {"—":>8s} {best_rho_v7:>10.4f} {ens_v7_m1_5:>8.4f}')
print(f'{"NN v1 (standalone)":<22s} {"—":>8s} {"—":>8s}'
      f' {"—":>8s} {nn_oof_rho:>8.4f} {"—":>10s} {nn_oof_m1_5:>8.4f}')
print(f'{"v11 (4-model ens)":<22s} {"(v7)":>8s} {"(v7)":>8s}'
      f' {"(v4)":>8s} {"(v1)":>8s} {best_rho_v11:>10.4f} {ens_v11_m1_5:>8.4f}')
print(f'{"v11a (4-model+M1-5TE)":<22s} {"(v7)":>8s} {"(v8a)":>8s}'
      f' {"(v4)":>8s} {"(v1)":>8s} {best_rho_v11a:>10.4f} {ens_v11a_m1_5:>8.4f}')

print(f'\nNN diversity (lower = more orthogonal to GBDTs):')
print(f'  NN-LGB corr: {corr_nn_lgb:.4f}')
print(f'  NN-XGB corr: {corr_nn_xgb:.4f}')
print(f'  NN-Ens corr: {corr_nn_ens:.4f}  (target: < 0.90)')

print(f'\nOutput files:')
print(f'  models/nn_oof_v1.npy       — NN OOF predictions')
print(f'  models/nn_test_v1.npy      — NN test predictions (full-data TE)')
print(f'  models/nn_test_v1a.npy     — NN test predictions (M1-5 TE)')
print(f'  models/nn_fold{{0-4}}.pt   — saved model weights per fold')
print(f'  submissions/ensemble_v11.csv   — 4-model ensemble, full-data TE')
print(f'  submissions/ensemble_v11a.csv  — 4-model ensemble, M1-5 TE')

print(f'\nNext steps:')
print(f'  1. Submit ensemble_v11.csv or ensemble_v11a.csv to platform.')
print(f'  2. If NN OOF < 0.58 or corr > 0.90: NN does not improve ensemble;')
print(f'     fallback to best of v7/v8a/v9 for final submission.')
print(f'  3. Deadline: Video 2026-04-15, Report 2026-04-23.')

print(f'\nStep 14 finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'Total runtime: {(time.time() - t0_total) / 60:.1f} min')
log_file.close()
