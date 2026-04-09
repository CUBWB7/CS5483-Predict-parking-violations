"""
Experiment E: TabM Deep Learning Model (GPU)
============================================
TabM (Gorishniy 2024, ICLR 2025) — BatchEnsemble-style MLP ensemble.
Self-contained implementation (no external tabm package required).

Reference: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
           Gorishniy et al., ICLR 2025.

Run on GPU server:
    conda activate parking
    pip install torch  # if not installed
    python scripts/step_e_gpu.py

Output files (in models/):
    tabm_oof.npy          — OOF predictions (6,076,546 rows)
    tabm_test.npy         — Test predictions (2,028,750 rows)
    tabm_fold{0-4}.pt     — Fold model checkpoints
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
MODEL_DIR  = os.path.join(ROOT, "models")
SUBMIT_DIR = os.path.join(ROOT, "submissions")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(SUBMIT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED        = 42
N_FOLDS     = 5
K_ENSEMBLE  = 32      # BatchEnsemble members
N_BLOCKS    = 3       # MLP residual blocks
D_BLOCK     = 256     # Hidden dimension
DROPOUT     = 0.1
BATCH_SIZE  = 4096
LR          = 1e-3
WEIGHT_DECAY= 1e-4
MAX_EPOCHS  = 50
PATIENCE    = 7

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Model Architecture ────────────────────────────────────────────────────────

class BatchEnsembleLinear(nn.Module):
    """
    Linear layer with BatchEnsemble per-member scaling.
    Each ensemble member k has its own input scaling r_k and output scaling s_k.
    Effective weight for member k: diag(s_k) @ W @ diag(r_k)
    This allows K "virtual" models sharing one weight matrix.
    """
    def __init__(self, in_features: int, out_features: int, k: int = 32):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.k = k

        # Shared weight matrix (same for all ensemble members)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Per-member scaling vectors (small noise around 1)
        self.r = nn.Parameter(torch.ones(k, in_features)  + 0.01 * torch.randn(k, in_features))
        self.s = nn.Parameter(torch.ones(k, out_features) + 0.01 * torch.randn(k, out_features))

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, k, in_features]
        x = x * self.r          # per-member input scaling: [batch, k, in]
        x = x @ self.weight     # shared linear projection:  [batch, k, out]
        x = x * self.s          # per-member output scaling: [batch, k, out]
        x = x + self.bias       # shared bias
        return x


class TabM(nn.Module):
    """
    TabM-mini: BatchEnsemble MLP for tabular data.
    Architecture:
        input → BatchNorm → Linear(n_features, d_block) → ReLU
              → k-expand → [BatchEnsembleBlock × n_blocks] → head → Sigmoid → mean over k
    """
    def __init__(self, n_features: int, n_blocks: int = 3,
                 d_block: int = 256, dropout: float = 0.1, k: int = 32):
        super().__init__()
        self.k = k
        self.n_blocks = n_blocks

        # Input normalisation
        self.input_bn = nn.BatchNorm1d(n_features)

        # First shared projection (not BatchEnsemble — all members start the same)
        self.first_linear = nn.Sequential(
            nn.Linear(n_features, d_block),
            nn.ReLU(),
        )

        # BatchEnsemble residual blocks
        self.be_blocks = nn.ModuleList()
        self.block_bns = nn.ModuleList()
        self.block_drops = nn.ModuleList()
        for _ in range(n_blocks):
            self.be_blocks.append(BatchEnsembleLinear(d_block, d_block, k))
            self.block_bns.append(nn.BatchNorm1d(d_block))
            self.block_drops.append(nn.Dropout(dropout))

        # Output head: d_block → 1, then Sigmoid
        self.head = BatchEnsembleLinear(d_block, 1, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features]
        x = self.input_bn(x)                        # [batch, n_features]
        x = self.first_linear(x)                    # [batch, d_block]

        # Expand to k ensemble members
        x = x.unsqueeze(1).expand(-1, self.k, -1)  # [batch, k, d_block]

        for be, bn, drop in zip(self.be_blocks, self.block_bns, self.block_drops):
            residual = x
            b, k, d = x.shape
            # BatchNorm expects [N, C] — flatten batch×k
            x_flat = x.reshape(b * k, d)
            x_flat = bn(x_flat)
            x = x_flat.reshape(b, k, d)
            x = torch.relu(x)
            x = drop(x)
            x = be(x)                               # [batch, k, d_block]
            x = x + residual                        # residual connection

        out = self.head(x)                          # [batch, k, 1]
        out = torch.sigmoid(out).squeeze(-1)        # [batch, k]
        out = out.mean(dim=1)                       # [batch]  — average over ensemble
        return out


# ── Weighted MSE Loss ─────────────────────────────────────────────────────────

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                      weight: torch.Tensor) -> torch.Tensor:
    loss = (pred - target) ** 2 * weight
    return loss.mean()


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch, w_batch in loader:
        X_batch, y_batch, w_batch = (X_batch.to(device), y_batch.to(device),
                                     w_batch.to(device))
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = weighted_mse_loss(pred, y_batch, w_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    for X_batch, *_ in loader:
        X_batch = X_batch.to(device)
        preds.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds)


# ── Main Training ─────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_features_tier2.parquet"))
    test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test_features_tier2.parquet"))
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")

    y = train_df["invalid_ratio"].values.astype(np.float32)
    tc = train_df["total_count"].values

    # Sample weights: log1p(total_count) — same as v7
    weights = np.log1p(tc).astype(np.float32)
    weights /= weights.mean()   # normalise so mean weight ≈ 1

    # Feature columns (same Tier-2 features as GBDT)
    drop_cols = ["invalid_ratio", "total_count"]
    feat_cols = [c for c in train_df.columns if c not in drop_cols]
    print(f"  Features: {len(feat_cols)}")

    X_train = train_df[feat_cols].values.astype(np.float32)
    X_test  = test_df[feat_cols].values.astype(np.float32)

    # M1-5 mask for evaluation
    m15_mask = train_df["month_of_year"].isin([1, 2, 3, 4, 5]).values

    # ── 5-Fold CV ──────────────────────────────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X_train), dtype=np.float32)
    test_preds = np.zeros(len(X_test),  dtype=np.float32)

    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"Fold {fold}  — train: {len(tr_idx):,}  val: {len(val_idx):,}")
        print(f"{'='*60}")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        w_tr        = weights[tr_idx]

        # DataLoaders
        tr_ds  = TensorDataset(torch.from_numpy(X_tr),
                               torch.from_numpy(y_tr),
                               torch.from_numpy(w_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val),
                               torch.from_numpy(y_val),
                               torch.ones(len(X_val)))  # weight unused in eval

        tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                                num_workers=2, pin_memory=True)

        # Model
        model = TabM(n_features=X_train.shape[1],
                     n_blocks=N_BLOCKS, d_block=D_BLOCK,
                     dropout=DROPOUT, k=K_ENSEMBLE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.01)

        best_val_spearman = -1.0
        best_epoch = 0
        patience_ctr = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, tr_loader, optimizer, device)
            val_p = evaluate(model, val_loader, device)
            val_spearman = spearmanr(y_val, val_p).statistic
            scheduler.step()

            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                best_epoch = epoch
                patience_ctr = 0
                torch.save(model.state_dict(),
                           os.path.join(MODEL_DIR, f"tabm_fold{fold}.pt"))
            else:
                patience_ctr += 1

            if epoch % 5 == 0 or patience_ctr == 0:
                print(f"  Epoch {epoch:3d}  loss={train_loss:.5f}  "
                      f"val_spearman={val_spearman:.4f}  "
                      f"best={best_val_spearman:.4f} (ep{best_epoch})")

            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

        # Load best checkpoint and predict
        model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, f"tabm_fold{fold}.pt"),
                       map_location=device))

        oof_preds[val_idx] = evaluate(model, val_loader, device)

        # Test predictions (accumulate, average later)
        test_ds = TensorDataset(torch.from_numpy(X_test),
                                torch.zeros(len(X_test)),
                                torch.ones(len(X_test)))
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                                 num_workers=2, pin_memory=True)
        test_preds += evaluate(model, test_loader, device)

        fold_spearman = spearmanr(y_val, oof_preds[val_idx]).statistic
        fold_scores.append(fold_spearman)
        elapsed = (time.time() - t_fold) / 60
        print(f"\n  Fold {fold} done — OOF Spearman: {fold_spearman:.4f}  ({elapsed:.1f} min)")

    # Average test predictions over folds
    test_preds /= N_FOLDS

    # ── OOF Evaluation ─────────────────────────────────────────────────────────
    oof_all  = spearmanr(y, oof_preds).statistic
    oof_m15  = spearmanr(y[m15_mask], oof_preds[m15_mask]).statistic
    v7_oof_baseline = 0.6429

    print(f"\n{'='*60}")
    print("TabM — Final Results")
    print(f"{'='*60}")
    for fold, sc in enumerate(fold_scores):
        print(f"  Fold {fold}: {sc:.4f}")
    print(f"\n  OOF Spearman (all):  {oof_all:.4f}  (v7 baseline: {v7_oof_baseline})")
    print(f"  M1-5 OOF Spearman:   {oof_m15:.4f}")
    print(f"  Success criterion:   OOF >= 0.55 → {'PASS' if oof_all >= 0.55 else 'FAIL'}")

    # ── Correlation with v7 GBDT ────────────────────────────────────────────────
    v7_oof_path = os.path.join(MODEL_DIR, "lgb_oof_v7.npy")
    if os.path.exists(v7_oof_path):
        # Load ensemble v7 OOF (LGB=0.35, XGB=0.65)
        lgb_oof = np.load(v7_oof_path)
        xgb_oof_path = os.path.join(MODEL_DIR, "xgb_oof_v7.npy")
        if os.path.exists(xgb_oof_path):
            xgb_oof = np.load(xgb_oof_path)
            v7_ens_oof = 0.35 * lgb_oof + 0.65 * xgb_oof
        else:
            v7_ens_oof = lgb_oof
        corr_tabm_v7 = spearmanr(oof_preds, v7_ens_oof).statistic
        print(f"\n  TabM–v7 Ensemble Spearman corr: {corr_tabm_v7:.4f}  (target < 0.85)")
        diversity_ok = corr_tabm_v7 < 0.85
        print(f"  Diversity criterion:  corr < 0.85 → {'PASS' if diversity_ok else 'FAIL'}")

    # ── Save predictions ────────────────────────────────────────────────────────
    np.save(os.path.join(MODEL_DIR, "tabm_oof.npy"),  oof_preds)
    np.save(os.path.join(MODEL_DIR, "tabm_test.npy"), test_preds)
    print(f"\n  Saved: models/tabm_oof.npy  ({oof_preds.shape[0]:,} rows)")
    print(f"  Saved: models/tabm_test.npy ({test_preds.shape[0]:,} rows)")

    # ── Submission validation ───────────────────────────────────────────────────
    assert test_preds.shape[0] == 2028750, f"Wrong test size: {test_preds.shape[0]}"
    assert not np.isnan(test_preds).any(), "NaN in test predictions"
    assert test_preds.min() >= 0.0 and test_preds.max() <= 1.0, "Predictions out of [0,1]"
    print("  Submission validation: PASS (2,028,750 rows, no NaN, range [0,1])")

    # ── Save submission ─────────────────────────────────────────────────────────
    # Generate submission CSV (single-model TabM)
    sub_df = pd.DataFrame({"id": test_df.index, "invalid_ratio": test_preds})
    sub_path = os.path.join(SUBMIT_DIR, "ensemble_tabm.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")

    total_time = (time.time() - t_start) / 60
    print(f"\nTotal time: {total_time:.1f} min")


if __name__ == "__main__":
    main()
