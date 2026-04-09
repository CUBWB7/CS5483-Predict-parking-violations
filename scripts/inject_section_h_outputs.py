"""
Execute Section H setup + diagnosis cells locally and inject their outputs
into the notebook JSON (so the notebook has visible output without running
from scratch).
"""
import json, io, os, base64
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/Users/bwb/CS5483_Data_Project2-forCC_2/notebooks')

SEED = 42
DATA_DIR   = '../data/'
MODEL_DIR  = '../models/'
np.random.seed(SEED)

print('Loading data...')
train_df = pd.read_parquet(f'{DATA_DIR}train_features_tier2.parquet')
test_df  = pd.read_parquet(f'{DATA_DIR}test_features_tier2.parquet')
y_h       = train_df['invalid_ratio'].values.astype('float32')
tc_h      = train_df['total_count'].values
m15_h     = train_df['month_of_year'].isin([1, 2, 3, 4, 5]).values

lgb_oof_v7_h  = np.load(f'{MODEL_DIR}lgb_oof_v7.npy')
xgb_oof_v7_h  = np.load(f'{MODEL_DIR}xgb_oof_v7.npy')
lgb_test_v7_h = np.load(f'{MODEL_DIR}lgb_test_v7.npy')
xgb_test_v7_h = np.load(f'{MODEL_DIR}xgb_test_v7.npy')
ens_oof_v7_h  = 0.35 * lgb_oof_v7_h + 0.65 * xgb_oof_v7_h

# ── Build setup cell output text ───────────────────────────────────────────────
setup_text = (
    f'Loaded train_df {train_df.shape},  test_df {test_df.shape}\n'
    f'\nv7 LGB OOF:           {spearmanr(y_h, lgb_oof_v7_h)[0]:.4f}\n'
    f'v7 XGB OOF:           {spearmanr(y_h, xgb_oof_v7_h)[0]:.4f}\n'
    f'v7 Ensemble OOF:      {spearmanr(y_h, ens_oof_v7_h)[0]:.4f}\n'
    f'v7 Ensemble M1-5 OOF: {spearmanr(y_h[m15_h], ens_oof_v7_h[m15_h])[0]:.4f}\n'
)
print("--- Setup output ---")
print(setup_text)

# ── Build diagnosis cell output text + figure ──────────────────────────────────
NOISE_LOW  = 0.15
NOISE_HIGH = 0.85

tc1_h   = (tc_h == 1)
nlo_h   = tc1_h & (ens_oof_v7_h < NOISE_LOW)  & (y_h == 1)
nhi_h   = tc1_h & (ens_oof_v7_h > NOISE_HIGH) & (y_h == 0)
noise_h = nlo_h | nhi_h
clean_tc1 = tc1_h & ~noise_h

diag_text = (
    f'=== tc=1 Subset Analysis ===\n\n'
    f'  Total training samples:          {len(y_h):>10,}\n'
    f'  total_count=1 samples:           {tc1_h.sum():>10,}  ({tc1_h.mean()*100:.1f}%)\n'
    f'  tc=1 with y=0:                   {(tc1_h & (y_h==0)).sum():>10,}  ({(tc1_h & (y_h==0)).mean()*100:.1f}%)\n'
    f'  tc=1 with y=1:                   {(tc1_h & (y_h==1)).sum():>10,}  ({(tc1_h & (y_h==1)).mean()*100:.1f}%)\n'
    f'\n'
    f'  Noise: pred<{NOISE_LOW} & y=1 (tc=1):  {nlo_h.sum():>10,}  ({nlo_h.mean()*100:.2f}% of train)\n'
    f'  Noise: pred>{NOISE_HIGH} & y=0 (tc=1):  {nhi_h.sum():>10,}  ({nhi_h.mean()*100:.2f}% of train)\n'
    f'  Total noise candidates:          {noise_h.sum():>10,}  ({noise_h.mean()*100:.2f}% of train)\n'
    f'\n'
    f'  Spearman  all tc=1:               {spearmanr(y_h[tc1_h], ens_oof_v7_h[tc1_h])[0]:.4f}\n'
    f'  Spearman  clean tc=1 (non-noise): {spearmanr(y_h[clean_tc1], ens_oof_v7_h[clean_tc1])[0]:.4f}\n'
    f'  Spearman  tc>=2:                  {spearmanr(y_h[tc_h>=2], ens_oof_v7_h[tc_h>=2])[0]:.4f}\n'
    f'\nFigure saved to docs/figures/fig_h_noise_diagnosis.png\n'
)
print("--- Diagnosis output ---")
print(diag_text)

# Generate figure
mon_col = train_df['month_of_year'].values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Section H — Noise Diagnosis: tc=1 Subset', fontsize=13, fontweight='bold')

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
ax.set_title('tc=1: Prediction Distribution\n(by actual label)')
ax.legend(fontsize=8)

ax = axes[1]
months = list(range(1, 13))
noise_pct = [
    100 * (noise_h & (mon_col == m)).sum() / max((tc1_h & (mon_col == m)).sum(), 1)
    for m in months
]
colors_m = ['#d62728' if m <= 5 else '#1f77b4' for m in months]
ax.bar(months, noise_pct, color=colors_m, alpha=0.8)
ax.set_xlabel('Month')
ax.set_ylabel('Noise % of tc=1')
ax.set_title('Noise Rate by Month\n(red = test months M1-5)')
ax.set_xticks(months)

ax = axes[2]
labels_bar  = ['Normal\n(tc=10)', 'tc=1\n(clean)', 'Noise (a)\nRemoved',
               'Noise (b)\n0.100', 'Noise (c)\nSmoothed']
weights_bar = [np.log1p(10), np.log1p(1), 0, 0.1, np.log1p(1)]
bar_colors  = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd']
ax.bar(labels_bar, weights_bar, color=bar_colors, alpha=0.85,
       edgecolor='k', linewidth=0.5)
ax.axhline(np.log1p(1), color='gray', ls=':', lw=1.2,
           label=f'log1p(1)={np.log1p(1):.3f}')
ax.set_ylabel('Sample Weight')
ax.set_title('Weight per Strategy\n("Normal" uses tc=10 as example)')
ax.legend(fontsize=8)

plt.tight_layout()
os.makedirs('../docs/figures', exist_ok=True)
fig_png_path = '../docs/figures/fig_h_noise_diagnosis.png'
plt.savefig(fig_png_path, dpi=120, bbox_inches='tight')
print(f'Figure saved to {fig_png_path}')

buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# ── Read notebook and inject outputs ──────────────────────────────────────────
with open('06_sprint.ipynb', 'r') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'\nTotal cells in notebook: {len(cells)}')

# Find Section H setup cell (first code cell after the Section H markdown header)
# The header is a markdown cell containing "## Section H"
h_setup_idx  = None
h_diag_idx   = None
found_h      = False
code_count   = 0

for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown':
        src = ''.join(c['source'])
        if '## Section H' in src:
            found_h = True
            code_count = 0
    if found_h and c['cell_type'] == 'code':
        code_count += 1
        if code_count == 1:
            h_setup_idx = i
        elif code_count == 2:
            h_diag_idx = i
            break

print(f'Section H setup cell:     {h_setup_idx}')
print(f'Section H diagnosis cell: {h_diag_idx}')

assert h_setup_idx is not None, 'Could not find Section H setup cell'
assert h_diag_idx  is not None, 'Could not find Section H diagnosis cell'


def stream_output(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


def image_output(b64_data):
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64_data,
            "text/plain": ["<Figure size 1500x400 with 3 Axes>"],
        },
        "metadata": {"needs_background": "light"},
    }


cells[h_setup_idx]['outputs'] = [stream_output(setup_text)]
cells[h_setup_idx]['execution_count'] = 1

cells[h_diag_idx]['outputs'] = [
    stream_output(diag_text),
    image_output(img_b64),
]
cells[h_diag_idx]['execution_count'] = 2

with open('06_sprint.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('\nOutputs injected successfully.')
print(f'  Cell {h_setup_idx}: setup output ({len(setup_text)} chars)')
print(f'  Cell {h_diag_idx}: diagnosis output ({len(diag_text)} chars) + figure')
