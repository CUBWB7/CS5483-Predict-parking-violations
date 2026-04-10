"""
Experiment F: Final Ensemble Combination
=========================================
Run from project root AFTER step_g_gpu.py completes:
    conda activate parking
    python scripts/step_f_gpu.py

No GPU training needed — purely ensemble weight optimisation (< 1 min).

Required files (from Exp C):
    models/lgb_rank_oof.npy   lgb_rank_test.npy
    models/xgb_rank_oof.npy   xgb_rank_test.npy

Optional files (from Exp G — add as they become available):
    models/lgb_g1_oof.npy     lgb_g1_test.npy
    models/xgb_g1_oof.npy     xgb_g1_test.npy
    models/lgb_g2_oof.npy     lgb_g2_test.npy    (if Layer 2 ran)
    models/xgb_g2_oof.npy     xgb_g2_test.npy    (if Layer 2 ran)

Model inclusion rules (per sprint plan):
    Exp C rank LGB/XGB  — always included (platform 0.5698, new best)
    Exp G pseudo-label  — included only if ensemble OOF >= 0.6464
    v7 LGB/XGB          — excluded (dominated by rank-target)
    Exp H models        — excluded (platform 0.5613 < Exp C 0.5698)
    TabM                — excluded (weight = 0 in all searches)

Files produced:
    submissions/ensemble_final.csv    best OOF combination
    submissions/ensemble_f_expC.csv   Exp C baseline reference
    step_f_gpu.log
"""

import sys, os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ── Tee stdout to log file ────────────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_file = open('step_f_gpu.log', 'a')
sys.stdout = Tee(sys.__stdout__, log_file)

# ── Config ────────────────────────────────────────────────────────────────────
SEED               = 42
DATA_DIR           = 'data/'
MODEL_DIR          = 'models/'
SUBMIT_DIR         = 'submissions/'
G_OOF_THRESHOLD    = 0.6464   # Include Exp G only if its OOF >= this
EXPECTED_TEST_ROWS = 2_028_750

np.random.seed(SEED)

print(f'\n{"="*60}')
print(f'Experiment F: Final Ensemble Combination')
print(f'Started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}\n')

# ── Load train/test data ──────────────────────────────────────────────────────
print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')
y        = train_df['invalid_ratio'].values
m15_mask = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values
y_m15    = y[m15_mask]
print(f'  Train: {train_df.shape}  Test: {test_df.shape}')
assert len(test_df) == EXPECTED_TEST_ROWS, f'Unexpected test size: {len(test_df)}'

# ── Helper: load .npy or return None ─────────────────────────────────────────
def load_npy(name):
    path = f'{MODEL_DIR}{name}'
    if os.path.exists(path):
        arr = np.load(path)
        print(f'  Loaded {name}  shape={arr.shape}')
        return arr
    print(f'  MISSING: {name}')
    return None

# ── Helper: fine-grained 1-D weight search ────────────────────────────────────
def weight_search_1d(oof_a, oof_b, y_full, step=0.01):
    """Return (best_rho, best_w_a) where ensemble = w_a*a + (1-w_a)*b."""
    best_rho, best_w = -1.0, 0.5
    for w in np.arange(0.0, 1.0 + step * 0.5, step):
        r = spearmanr(y_full, w * oof_a + (1.0 - w) * oof_b)[0]
        if r > best_rho:
            best_rho, best_w = r, round(float(w), 2)
    return best_rho, best_w

# ── Load Exp C (required) ─────────────────────────────────────────────────────
print('\nLoading Exp C rank predictions (required)...')
lgb_rank_oof  = load_npy('lgb_rank_oof.npy')
lgb_rank_test = load_npy('lgb_rank_test.npy')
xgb_rank_oof  = load_npy('xgb_rank_oof.npy')
xgb_rank_test = load_npy('xgb_rank_test.npy')

if not all(v is not None for v in [lgb_rank_oof, lgb_rank_test, xgb_rank_oof, xgb_rank_test]):
    print('\nERROR: Exp C predictions missing. Run step_c_gpu.py first.')
    sys.exit(1)

# ── Load Exp G (optional) ─────────────────────────────────────────────────────
print('\nLoading Exp G pseudo-label predictions (optional)...')
lgb_g1_oof  = load_npy('lgb_g1_oof.npy')
lgb_g1_test = load_npy('lgb_g1_test.npy')
xgb_g1_oof  = load_npy('xgb_g1_oof.npy')
xgb_g1_test = load_npy('xgb_g1_test.npy')

lgb_g2_oof  = load_npy('lgb_g2_oof.npy')
lgb_g2_test = load_npy('lgb_g2_test.npy')
xgb_g2_oof  = load_npy('xgb_g2_oof.npy')
xgb_g2_test = load_npy('xgb_g2_test.npy')

_g1_ready = all(v is not None for v in [lgb_g1_oof, lgb_g1_test, xgb_g1_oof, xgb_g1_test])
_g2_ready = all(v is not None for v in [lgb_g2_oof, lgb_g2_test, xgb_g2_oof, xgb_g2_test])

print(f'\nStatus:  Exp C=ready  '
      f'G-Layer1={"ready" if _g1_ready else "missing"}  '
      f'G-Layer2={"ready" if _g2_ready else "missing"}')

# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1: Exp C rank-only (fine-grained weight search)
# ═══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('Strategy 1: Exp C rank-only  (step=0.01)')
print(f'{"="*60}')

lgb_c_rho = spearmanr(y, lgb_rank_oof)[0]
xgb_c_rho = spearmanr(y, xgb_rank_oof)[0]
lgb_c_m15 = spearmanr(y_m15, lgb_rank_oof[m15_mask])[0]
xgb_c_m15 = spearmanr(y_m15, xgb_rank_oof[m15_mask])[0]
print(f'  Exp C LGB: OOF={lgb_c_rho:.4f}  M1-5={lgb_c_m15:.4f}  (was 0.6373)')
print(f'  Exp C XGB: OOF={xgb_c_rho:.4f}  M1-5={xgb_c_m15:.4f}  (was 0.6430)')

c_rho, c_lgb_w = weight_search_1d(lgb_rank_oof, xgb_rank_oof, y)
c_xgb_w    = round(1.0 - c_lgb_w, 2)
c_ens_oof  = c_lgb_w * lgb_rank_oof  + c_xgb_w * xgb_rank_oof
c_ens_test = c_lgb_w * lgb_rank_test + c_xgb_w * xgb_rank_test
c_ens_m15  = spearmanr(y_m15, c_ens_oof[m15_mask])[0]
print(f'  Best weights: LGB={c_lgb_w:.2f}  XGB={c_xgb_w:.2f}')
print(f'  Ensemble OOF={c_rho:.4f}  M1-5={c_ens_m15:.4f}  (Exp C baseline: 0.6464)')

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate Exp G layers and pick the best
# ═══════════════════════════════════════════════════════════════════════════════
best_g_ens_oof  = None
best_g_ens_test = None
best_g_rho      = None
best_g_m15      = None
best_g_lgb_w    = None
g_layer_name    = None

for layer, lgb_oof, xgb_oof, lgb_test, xgb_test, ready in [
    ('Layer 1', lgb_g1_oof, xgb_g1_oof, lgb_g1_test, xgb_g1_test, _g1_ready),
    ('Layer 2', lgb_g2_oof, xgb_g2_oof, lgb_g2_test, xgb_g2_test, _g2_ready),
]:
    if not ready:
        continue
    print(f'\n{"="*60}')
    print(f'Evaluating Exp G {layer}...')
    print(f'{"="*60}')
    lgb_rho = spearmanr(y, lgb_oof)[0]
    xgb_rho = spearmanr(y, xgb_oof)[0]
    print(f'  LGB OOF={lgb_rho:.4f}  XGB OOF={xgb_rho:.4f}')
    g_rho, g_lgb_w = weight_search_1d(lgb_oof, xgb_oof, y)
    g_xgb_w     = round(1.0 - g_lgb_w, 2)
    g_ens_oof   = g_lgb_w * lgb_oof  + g_xgb_w * xgb_oof
    g_ens_test  = g_lgb_w * lgb_test + g_xgb_w * xgb_test
    g_m15       = spearmanr(y_m15, g_ens_oof[m15_mask])[0]
    qualifies   = g_rho >= G_OOF_THRESHOLD
    print(f'  Best weights: LGB={g_lgb_w:.2f}  XGB={g_xgb_w:.2f}')
    print(f'  Ensemble OOF={g_rho:.4f}  M1-5={g_m15:.4f}  '
          f'{"✓ qualifies" if qualifies else "✗ below threshold"}')
    if best_g_rho is None or g_rho > best_g_rho:
        best_g_rho = g_rho; best_g_m15 = g_m15; best_g_lgb_w = g_lgb_w
        best_g_ens_oof = g_ens_oof; best_g_ens_test = g_ens_test
        g_layer_name = layer

_g_qualifies = best_g_rho is not None and best_g_rho >= G_OOF_THRESHOLD

if best_g_rho is not None and not _g_qualifies:
    print(f'\nExp G best OOF {best_g_rho:.4f} < threshold {G_OOF_THRESHOLD} '
          f'— excluded from final ensemble.')

# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2: C + G blend (only if G qualifies)
# ═══════════════════════════════════════════════════════════════════════════════
cg_rho = None; cg_c_w = None; cg_m15 = None
cg_ens_oof = None; cg_ens_test = None

if _g_qualifies:
    print(f'\n{"="*60}')
    print(f'Strategy 2: Exp C + Exp G {g_layer_name} blend  (step=0.01)')
    print(f'{"="*60}')
    cg_rho, cg_c_w = weight_search_1d(c_ens_oof, best_g_ens_oof, y)
    cg_g_w      = round(1.0 - cg_c_w, 2)
    cg_ens_oof  = cg_c_w * c_ens_oof  + cg_g_w * best_g_ens_oof
    cg_ens_test = cg_c_w * c_ens_test + cg_g_w * best_g_ens_test
    cg_m15      = spearmanr(y_m15, cg_ens_oof[m15_mask])[0]
    print(f'  Best blend: C={cg_c_w:.2f}  G({g_layer_name})={cg_g_w:.2f}')
    print(f'  Ensemble OOF={cg_rho:.4f}  M1-5={cg_m15:.4f}  '
          f'vs Exp C: {cg_rho - c_rho:+.4f}')

# ═══════════════════════════════════════════════════════════════════════════════
# Final: pick best and save submissions
# ═══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('Strategy Comparison')
print(f'{"="*60}')
print(f'  {"Strategy":<40} {"OOF":>8} {"M1-5":>8} {"vs C":>8}')
print(f'  {"-"*66}')
print(f'  {"Exp C LGB standalone":<40} {lgb_c_rho:>8.4f} {lgb_c_m15:>8.4f} {"—":>8}')
print(f'  {"Exp C XGB standalone":<40} {xgb_c_rho:>8.4f} {xgb_c_m15:>8.4f} {"—":>8}')
print(f'  {"Exp C rank ensemble (Strategy 1)":<40} {c_rho:>8.4f} {c_ens_m15:>8.4f} {"—":>8}')
if best_g_rho is not None:
    label = f'Exp G {g_layer_name} ensemble'
    print(f'  {label:<40} {best_g_rho:>8.4f} {best_g_m15:>8.4f} {best_g_rho-c_rho:>+8.4f}')
if cg_rho is not None:
    print(f'  {"C + G blend (Strategy 2)":<40} {cg_rho:>8.4f} {cg_m15:>8.4f} {cg_rho-c_rho:>+8.4f}')
print(f'  {"-"*66}')

# Determine final best
if cg_rho is not None and cg_rho >= c_rho:
    final_rho = cg_rho; final_m15 = cg_m15; final_test = cg_ens_test
    final_strategy = f'C+G blend (C={cg_c_w:.2f} G={round(1-cg_c_w,2):.2f})'
elif _g_qualifies and best_g_rho >= c_rho:
    final_rho = best_g_rho; final_m15 = best_g_m15; final_test = best_g_ens_test
    final_strategy = f'Exp G {g_layer_name} (LGB={best_g_lgb_w:.2f} XGB={round(1-best_g_lgb_w,2):.2f})'
else:
    final_rho = c_rho; final_m15 = c_ens_m15; final_test = c_ens_test
    final_strategy = f'Exp C rank-only (LGB={c_lgb_w:.2f} XGB={c_xgb_w:.2f})'

print(f'\n  *** Selected: {final_strategy}')
print(f'      OOF={final_rho:.4f}  M1-5={final_m15:.4f}')

# ── Save submissions ──────────────────────────────────────────────────────────
# Exp C baseline reference (always saved)
c_test_clipped = np.clip(c_ens_test, 0, 1)
sub_c = pd.DataFrame({'id': test_df.index, 'invalid_ratio': c_test_clipped})
sub_c.to_csv(f'{SUBMIT_DIR}ensemble_f_expC.csv', index=False)
print(f'\nSaved: submissions/ensemble_f_expC.csv  '
      f'rows={len(c_test_clipped):,}  OOF={c_rho:.4f}')

# Final best submission
final_test_clipped = np.clip(final_test, 0, 1)
assert len(final_test_clipped) == EXPECTED_TEST_ROWS, \
    f'Size mismatch: {len(final_test_clipped)} vs {EXPECTED_TEST_ROWS}'
assert not np.isnan(final_test_clipped).any(), 'NaN in final predictions'
sub_final = pd.DataFrame({'id': test_df.index, 'invalid_ratio': final_test_clipped})
sub_final.to_csv(f'{SUBMIT_DIR}ensemble_final.csv', index=False)
print(f'Saved: submissions/ensemble_final.csv  '
      f'rows={len(final_test_clipped):,}  '
      f'range=[{final_test_clipped.min():.4f}, {final_test_clipped.max():.4f}]  '
      f'OOF={final_rho:.4f}')

# ── Final summary ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'EXPERIMENT F COMPLETE — {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*60}')
print(f'\n  Best submission: submissions/ensemble_final.csv')
print(f'  OOF={final_rho:.4f}  (Exp C baseline: 0.6464  Platform: 0.5698)')
print(f'  Strategy: {final_strategy}')
if not (_g1_ready or _g2_ready):
    print(f'\n  NOTE: Exp G results not yet available.')
    print(f'  Download from server after step_g_gpu.py finishes:')
    print(f'    models/lgb_g1_oof.npy  lgb_g1_test.npy')
    print(f'    models/xgb_g1_oof.npy  xgb_g1_test.npy')
    print(f'    models/lgb_g2_oof.npy  lgb_g2_test.npy  (if Layer 2 ran)')
    print(f'    models/xgb_g2_oof.npy  xgb_g2_test.npy  (if Layer 2 ran)')
    print(f'  Then re-run this script.')
print(f'\n  Files to download:')
print(f'    submissions/ensemble_final.csv')
print(f'    submissions/ensemble_f_expC.csv')
print(f'    step_f_gpu.log')
