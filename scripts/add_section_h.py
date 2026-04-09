"""Helper script to append Section H cells to notebooks/06_sprint.ipynb."""
import json

with open('notebooks/06_sprint.ipynb', 'r') as f:
    nb = json.load(f)


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell H-0: Markdown header
# ─────────────────────────────────────────────────────────────────────────────
H_HEADER = """\
---
## Section H — GBDT Label Noise Handling

**Rationale**: 25% of training samples have `total_count=1`, making `invalid_ratio`
binary noise (exactly 0 or 1).  v7 already reduces their influence via `log1p(total_count)`
weighting (weight 0.693 vs 2.40 for tc=10), but the labels themselves remain noisy.

We use the v7 ensemble OOF predictions to identify "confidently wrong" tc=1 samples
and test three handling strategies:

| Strategy | Description |
|----------|-------------|
| (a) Remove | Drop noise candidates from training entirely |
| (b) Down-weight | Set sample_weight = 0.1 (vs 0.693 baseline) |
| (c) Label smooth | y=1 noise → 0.8,  y=0 noise → 0.2 |

**Noise criteria** (applied to `total_count=1` samples only):
- `pred < 0.15` but `y = 1` → model confidently predicts low, label says high (noisy)
- `pred > 0.85` but `y = 0` → model confidently predicts high, label says low (noisy)

**GPU training**: Run `python scripts/step_h_gpu.py` on the GPU server (~3h total).
Download the output `.npy` files to `models/` before running the evaluation cells.

**Success criterion**: Ensemble OOF ≥ 0.6424 (matching v7 within −0.0005)

**Reference**: Paper 5 (GBDT Label Noise 2024)\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-1: Self-contained setup
# ─────────────────────────────────────────────────────────────────────────────
H_SETUP = """\
# ── Section H: self-contained setup ──────────────────────────────────────────
# Run this cell first if starting Section H without running prior sections.
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os

SEED = 42
DATA_DIR   = '../data/'
MODEL_DIR  = '../models/'
SUBMIT_DIR = '../submissions/'
np.random.seed(SEED)

# Reuse session variables if already loaded (avoids re-reading 600 MB parquet)
try:
    _ = train_df.shape
    y_h       = train_df['invalid_ratio'].values.astype(np.float32)
    tc_h      = train_df['total_count'].values
    m15_h     = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values
    print(f'Reusing session train_df  shape={train_df.shape}')
except NameError:
    print('Loading data from disk...')
    train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
    test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')
    y_h       = train_df['invalid_ratio'].values.astype(np.float32)
    tc_h      = train_df['total_count'].values
    m15_h     = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values
    print(f'Loaded train_df {train_df.shape},  test_df {test_df.shape}')

# Load v7 OOF + test (used for noise identification and ensemble baseline)
lgb_oof_v7_h  = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7_h  = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7_h = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7_h = np.load(f'{MODEL_DIR}xgb_test_v7.npy')

# v7 ensemble OOF (established weights)
ens_oof_v7_h = 0.35 * lgb_oof_v7_h + 0.65 * xgb_oof_v7_h

print(f'\\nv7 LGB OOF:           {spearmanr(y_h, lgb_oof_v7_h)[0]:.4f}')
print(f'v7 XGB OOF:           {spearmanr(y_h, xgb_oof_v7_h)[0]:.4f}')
print(f'v7 Ensemble OOF:      {spearmanr(y_h, ens_oof_v7_h)[0]:.4f}')
print(f'v7 Ensemble M1-5 OOF: {spearmanr(y_h[m15_h], ens_oof_v7_h[m15_h])[0]:.4f}')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-2: Noise diagnosis + visualisation
# ─────────────────────────────────────────────────────────────────────────────
H_DIAGNOSIS = """\
# ── Section H: Noise Diagnosis ───────────────────────────────────────────────
# Identify noise candidates using v7 OOF predictions on the tc=1 subset.

NOISE_LOW  = 0.15   # pred < threshold but y=1  → noisy (predicted low, label high)
NOISE_HIGH = 0.85   # pred > threshold but y=0  → noisy (predicted high, label low)

tc1_h   = (tc_h == 1)
nlo_h   = tc1_h & (ens_oof_v7_h < NOISE_LOW)  & (y_h == 1)   # low-pred noise
nhi_h   = tc1_h & (ens_oof_v7_h > NOISE_HIGH) & (y_h == 0)   # high-pred noise
noise_h = nlo_h | nhi_h

print('=== tc=1 Subset Analysis ===\\n')
print(f'  Total training samples:          {len(y_h):>10,}')
print(f'  total_count=1 samples:           {tc1_h.sum():>10,}  ({tc1_h.mean()*100:.1f}%)')
print(f'  tc=1 with y=0:                   '
      f'{(tc1_h & (y_h==0)).sum():>10,}  ({(tc1_h & (y_h==0)).mean()*100:.1f}%)')
print(f'  tc=1 with y=1:                   '
      f'{(tc1_h & (y_h==1)).sum():>10,}  ({(tc1_h & (y_h==1)).mean()*100:.1f}%)')
print()
print(f'  Noise: pred<{NOISE_LOW} & y=1 (tc=1):  {nlo_h.sum():>10,}  '
      f'({nlo_h.mean()*100:.2f}% of train)')
print(f'  Noise: pred>{NOISE_HIGH} & y=0 (tc=1):  {nhi_h.sum():>10,}  '
      f'({nhi_h.mean()*100:.2f}% of train)')
print(f'  Total noise candidates:          {noise_h.sum():>10,}  '
      f'({noise_h.mean()*100:.2f}% of train)')
print()

# Spearman on subgroups
clean_tc1 = tc1_h & ~noise_h
print(f'  Spearman  all tc=1:              '
      f'{spearmanr(y_h[tc1_h], ens_oof_v7_h[tc1_h])[0]:.4f}')
print(f'  Spearman  clean tc=1 (non-noise):'
      f' {spearmanr(y_h[clean_tc1], ens_oof_v7_h[clean_tc1])[0]:.4f}')
print(f'  Spearman  tc>=2:                 '
      f'{spearmanr(y_h[tc_h>=2], ens_oof_v7_h[tc_h>=2])[0]:.4f}')

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Section H — Noise Diagnosis: tc=1 Subset', fontsize=13, fontweight='bold')

# Panel 1: prediction distribution for tc=1 by label
ax = axes[0]
bins = np.linspace(0, 1, 51)
ax.hist(ens_oof_v7_h[tc1_h & (y_h == 0)], bins=bins, alpha=0.6,
        color='steelblue', label='y=0', density=True)
ax.hist(ens_oof_v7_h[tc1_h & (y_h == 1)], bins=bins, alpha=0.6,
        color='tomato', label='y=1', density=True)
ax.axvline(NOISE_LOW,  color='steelblue', ls='--', lw=1.5,
           label=f'noise thresh {NOISE_LOW}')
ax.axvline(NOISE_HIGH, color='tomato', ls='--', lw=1.5,
           label=f'noise thresh {NOISE_HIGH}')
ax.set_xlabel('v7 Ensemble Prediction')
ax.set_ylabel('Density')
ax.set_title('tc=1: Prediction Distribution\\n(by actual label)')
ax.legend(fontsize=8)

# Panel 2: noise candidates by month
ax = axes[1]
months = np.arange(1, 13)
mon_col = train_df['month_of_year'].values
noise_pct = [
    100 * (noise_h & (mon_col == m)).sum() / max((tc1_h & (mon_col == m)).sum(), 1)
    for m in months
]
colors_m = ['#d62728' if m <= 5 else '#1f77b4' for m in months]
ax.bar(months, noise_pct, color=colors_m, alpha=0.8)
ax.set_xlabel('Month')
ax.set_ylabel('Noise % of tc=1')
ax.set_title('Noise Rate by Month\\n(red = test months M1-5)')
ax.set_xticks(months)

# Panel 3: sample weights under each strategy
ax = axes[2]
labels_bar = ['Normal\\n(tc=10)', 'tc=1\\n(clean)', 'Noise (a)\\nRemoved',
               'Noise (b)\\n0.100', 'Noise (c)\\nSmoothed']
weights_bar = [np.log1p(10), np.log1p(1), 0, 0.1, np.log1p(1)]
bar_colors  = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd']
ax.bar(labels_bar, weights_bar, color=bar_colors, alpha=0.85,
       edgecolor='k', linewidth=0.5)
ax.axhline(np.log1p(1), color='gray', ls=':', lw=1.2,
           label=f'log1p(1)={np.log1p(1):.3f}')
ax.set_ylabel('Sample Weight')
ax.set_title('Weight per Strategy\\n("Normal" uses tc=10 as example)')
ax.legend(fontsize=8)

plt.tight_layout()
os.makedirs('../docs/figures', exist_ok=True)
plt.savefig('../docs/figures/fig_h_noise_diagnosis.png', dpi=120, bbox_inches='tight')
plt.show()
print('Figure saved to docs/figures/fig_h_noise_diagnosis.png')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-3: GPU note (markdown)
# ─────────────────────────────────────────────────────────────────────────────
H_GPU_NOTE = """\
### GPU Training

Run `python scripts/step_h_gpu.py` on the GPU server.

**Expected runtime**: ~3h total (3 strategies × ~45 min LGB + ~15 min XGB)

Expected output files — download from GPU server and place in `models/`:

| File | Strategy |
|------|----------|
| `lgb_ha_oof.npy`, `lgb_ha_test.npy`, `xgb_ha_oof.npy`, `xgb_ha_test.npy` | (a) Remove |
| `lgb_hb_oof.npy`, `lgb_hb_test.npy`, `xgb_hb_oof.npy`, `xgb_hb_test.npy` | (b) Down-weight |
| `lgb_hc_oof.npy`, `lgb_hc_test.npy`, `xgb_hc_oof.npy`, `xgb_hc_test.npy` | (c) Label smooth |

The GPU script also saves `submissions/ensemble_h{a,b,c}.csv` automatically.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-4: Load GPU results
# ─────────────────────────────────────────────────────────────────────────────
H_LOAD = """\
# ── Section H: Load GPU results ───────────────────────────────────────────────
# Run after downloading .npy files from the GPU server to models/.

def _load_h(name):
    path = f'{MODEL_DIR}{name}'
    if os.path.exists(path):
        arr = np.load(path)
        print(f'  Loaded {name:<32s}  shape={arr.shape}')
        return arr
    print(f'  MISSING: {name}  (run step_h_gpu.py on GPU server)')
    return None

print('Loading Section H GPU results...')
lgb_ha_oof  = _load_h('lgb_ha_oof.npy')
lgb_ha_test = _load_h('lgb_ha_test.npy')
xgb_ha_oof  = _load_h('xgb_ha_oof.npy')
xgb_ha_test = _load_h('xgb_ha_test.npy')

lgb_hb_oof  = _load_h('lgb_hb_oof.npy')
lgb_hb_test = _load_h('lgb_hb_test.npy')
xgb_hb_oof  = _load_h('xgb_hb_oof.npy')
xgb_hb_test = _load_h('xgb_hb_test.npy')

lgb_hc_oof  = _load_h('lgb_hc_oof.npy')
lgb_hc_test = _load_h('lgb_hc_test.npy')
xgb_hc_oof  = _load_h('xgb_hc_oof.npy')
xgb_hc_test = _load_h('xgb_hc_test.npy')

_h_loaded = all(x is not None for x in [
    lgb_ha_oof, lgb_ha_test, xgb_ha_oof, xgb_ha_test,
    lgb_hb_oof, lgb_hb_test, xgb_hb_oof, xgb_hb_test,
    lgb_hc_oof, lgb_hc_test, xgb_hc_oof, xgb_hc_test,
])
print(f'\\nAll files loaded: {_h_loaded}')
if not _h_loaded:
    print('Evaluation cells will not run until all files are present.')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-5: Evaluate all strategies
# ─────────────────────────────────────────────────────────────────────────────
H_EVAL = """\
# ── Section H: Evaluate all strategies ───────────────────────────────────────
assert _h_loaded, "Missing GPU result files — run step_h_gpu.py first."

V7_LGB = 0.6336; V7_XGB = 0.6403; V7_ENS = 0.6429; V7_M15 = 0.6515

def _eval_h(lgb_oof, xgb_oof):
    lgb_rho = spearmanr(y_h, lgb_oof)[0]
    xgb_rho = spearmanr(y_h, xgb_oof)[0]
    best_rho, best_lw = -1.0, 0.35
    for lw in np.arange(0.0, 1.01, 0.05):
        rho = spearmanr(y_h, lw * lgb_oof + (1 - lw) * xgb_oof)[0]
        if rho > best_rho:
            best_rho, best_lw = rho, lw
    ens = best_lw * lgb_oof + (1 - best_lw) * xgb_oof
    m15 = spearmanr(y_h[m15_h], ens[m15_h])[0]
    corr = np.corrcoef(lgb_oof, xgb_oof)[0, 1]
    return {'l': lgb_rho, 'x': xgb_rho, 'e': best_rho, 'm': m15,
            'lw': best_lw, 'xw': round(1 - best_lw, 2), 'c': corr}

h_evals = {
    '(a) Remove':       _eval_h(lgb_ha_oof, xgb_ha_oof),
    '(b) Down-weight':  _eval_h(lgb_hb_oof, xgb_hb_oof),
    '(c) Label smooth': _eval_h(lgb_hc_oof, xgb_hc_oof),
}

print('=== Section H: Strategy Comparison ===\\n')
hdr = f'  {"Strategy":<22} {"LGB OOF":>8} {"XGB OOF":>8} {"Ens OOF":>9} {"M1-5":>8}  {"Corr":>7}  {"LGBw":>5} {"XGBw":>5}'
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
print(f'  {"v7 baseline":<22} {V7_LGB:>8.4f} {V7_XGB:>8.4f} {V7_ENS:>9.4f} {V7_M15:>8.4f}  '
      f'{"0.9647":>7}  {"0.35":>5} {"0.65":>5}')

for tag, r in h_evals.items():
    de = r["e"] - V7_ENS
    dm = r["m"] - V7_M15
    print(f'  {tag:<22} {r["l"]:>8.4f} {r["x"]:>8.4f} {r["e"]:>9.4f} {r["m"]:>8.4f}  '
          f'{r["c"]:>7.4f}  {r["lw"]:>5.2f} {r["xw"]:>5.2f}')
    print(f'    {"delta vs v7":<20} {"":>8} {"":>8} {de:>+9.4f} {dm:>+8.4f}')

best_h_tag = max(h_evals, key=lambda k: h_evals[k]['e'])
best_h_rho = h_evals[best_h_tag]['e']
passed_h   = best_h_rho >= V7_ENS - 0.0005
print(f'\\nBest strategy: {best_h_tag}  OOF={best_h_rho:.4f}  '
      f'(delta vs v7 = {best_h_rho - V7_ENS:+.4f})')
print(f'Success criterion (OOF >= {V7_ENS - 0.0005:.4f}): '
      f'{"PASS" if passed_h else "FAIL"}')

# Correlation with v7 ensemble (for diversity check)
ens_v7_h2 = 0.35 * lgb_oof_v7_h + 0.65 * xgb_oof_v7_h
print('\\nCorrelation with v7 ensemble OOF:')
for tag, (lo, xo) in [('(a) Remove', (lgb_ha_oof, xgb_ha_oof)),
                       ('(b) Down-weight', (lgb_hb_oof, xgb_hb_oof)),
                       ('(c) Label smooth', (lgb_hc_oof, xgb_hc_oof))]:
    r = h_evals[tag]
    ens = r['lw'] * lo + r['xw'] * xo
    corr_v7 = np.corrcoef(ens_v7_h2, ens)[0, 1]
    print(f'  {tag:<22}  corr={corr_v7:.4f}  '
          f'(diversity: {"high" if corr_v7 < 0.985 else "low"})')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-6: Submission validation
# ─────────────────────────────────────────────────────────────────────────────
H_SUBMISSION = """\
# ── Section H: Submission validation ─────────────────────────────────────────
# The GPU script auto-generates submissions/ensemble_h{a,b,c}.csv.
# This cell verifies them and recommends the best one.

print('=== Pre-Submission Validation ===\\n')
for strat, fname in [('a', 'ensemble_ha.csv'),
                     ('b', 'ensemble_hb.csv'),
                     ('c', 'ensemble_hc.csv')]:
    fpath = f'{SUBMIT_DIR}{fname}'
    if not os.path.exists(fpath):
        print(f'  {fname}:  MISSING (run GPU script)')
        continue
    sub  = pd.read_csv(fpath, index_col=0)
    vals = sub['invalid_ratio'].values
    ok   = (len(vals) == 2028750 and
            np.isnan(vals).sum() == 0 and
            vals.min() >= 0 and vals.max() <= 1)
    r    = h_evals[['(a) Remove', '(b) Down-weight', '(c) Label smooth'][ord(strat)-ord('a')]]
    print(f'  {fname}:  rows={len(vals):,}  NaN={np.isnan(vals).sum()}  '
          f'range=[{vals.min():.4f},{vals.max():.4f}]  OOF={r["e"]:.4f}  '
          f'{"PASS" if ok else "FAIL"}')

print()
strat_letter = best_h_tag[1]   # extracts 'a', 'b', or 'c'
print(f'Recommended submission:  submissions/ensemble_h{strat_letter}.csv')
print(f'  Strategy  {best_h_tag}')
print(f'  OOF       {best_h_rho:.4f}  (delta vs v7 = {best_h_rho - V7_ENS:+.4f})')
if not passed_h:
    print()
    print('  Note: No strategy beat v7 baseline.  v7 remains the best submission.')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell H-7: Summary
# ─────────────────────────────────────────────────────────────────────────────
H_SUMMARY = """\
# ── Section H Summary ─────────────────────────────────────────────────────────
print('=' * 70)
print('SECTION H SUMMARY — GBDT Label Noise Handling')
print('=' * 70)

print('\\n--- Noise Identification ---')
print(f'  total_count=1 samples:     {tc1_h.sum():,}  ({tc1_h.mean()*100:.1f}% of train)')
print(f'  Noise candidates:          {noise_h.sum():,}  ({noise_h.mean()*100:.2f}% of train)')
print(f'  Thresholds:  pred<{NOISE_LOW} & y=1  |  pred>{NOISE_HIGH} & y=0')

print('\\n--- Strategy Results ---')
print(f'  {"Strategy":<22}  {"OOF":>8}  {"M1-5":>8}  {"delta OOF":>10}  Status')
print(f'  {"-"*60}')
print(f'  {"v7 baseline":<22}  {V7_ENS:>8.4f}  {V7_M15:>8.4f}  {"baseline":>10}')
for tag, r in h_evals.items():
    de = r["e"] - V7_ENS
    ok = r["e"] >= V7_ENS - 0.0005
    print(f'  {tag:<22}  {r["e"]:>8.4f}  {r["m"]:>8.4f}  {de:>+10.4f}  '
          f'{"PASS" if ok else "FAIL"}')

print('\\n--- Decision ---')
if passed_h:
    print(f'  Best strategy: {best_h_tag}')
    print(f'  Action: Submit submissions/ensemble_h{strat_letter}.csv')
else:
    print('  No strategy beat v7 within tolerance.')
    print('  Action: Keep v7 as best submission.  Proceed to next experiment.')
    print('  Insight: At {:.1f}% of training data, noise candidates are too few'.format(
        noise_h.mean() * 100))
    print('           OR the OOF-based identification is not reliable enough.')

print('\\n--- Key Takeaways ---')
print('  - v7 log1p weighting already partially mitigates tc=1 noise.')
print('  - OOF-based noise identification is an approximation (cross-val leakage possible).')
print('  - Label smoothing (c) is the least disruptive — preferred if OOF is marginal.')
"""

# ─────────────────────────────────────────────────────────────────────────────
# Append all Section H cells
# ─────────────────────────────────────────────────────────────────────────────
new_cells = [
    md_cell(H_HEADER),
    code_cell(H_SETUP),
    code_cell(H_DIAGNOSIS),
    md_cell(H_GPU_NOTE),
    code_cell(H_LOAD),
    code_cell(H_EVAL),
    code_cell(H_SUBMISSION),
    code_cell(H_SUMMARY),
]

nb['cells'].extend(new_cells)

with open('notebooks/06_sprint.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Done.  Notebook now has {len(nb["cells"])} cells.')
print(f'Added {len(new_cells)} Section H cells.')
