---
theme: neversink
title: Predicting Parking Violation Rates Using Gradient Boosting
info: |
  ChallengeData #163 | CS5483 Data Mining | April 2026
neversink_string: 'CS5483 | ChallengeData #163'
layout: cover
color: sky-light
background: /doc-figures/fig5_spatial_violation.png
---

# Predicting Parking Violation Rates
## Using Gradient Boosting

<br>

**ChallengeData #163** · CS5483 Data Mining · April 2026

<br>

<div class="text-xl font-bold text-black bg-white/70 inline-block px-3 py-1 rounded">
  Platform Spearman: 0.5705 · Rank #5 Globally
</div>

<!--
Open with the real-world context: a smart parking enforcement system in Thessaloniki, Greece.
The heatmap behind us shows actual geographic violation patterns across the city.
-->

---
layout: section
color: sky
---

# Section 1
## Introduction & Problem Setup

---
layout: top-title
color: sky-light
---

:: title ::

# What Are We Predicting?

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-[38%] text-sm leading-relaxed">

- **Dataset**: THESi smart parking system, Thessaloniki, Greece
- **Training set**: 6.07 M observations, 10 features
- **Target**: `invalid_ratio` — fraction of invalid parking events per location-timeslot
- **Evaluation**: Spearman rank correlation ρ
- **Train / Test / Features**: 6.07 M · 1.5 M · 10

> **The metric rewards ranking, not numerical accuracy.**

</div>
<div class="w-[62%]">
<img src="/doc-figures/fig1_target_distribution.png" class="w-full rounded shadow" />
<small class="text-gray-500 text-xs">Distribution of `invalid_ratio` — heavy mass at 0 and 1 from low-count noise</small>
</div>
</div>

<!--
Note the bimodal mass at 0 and 1 — caused by locations with total_count=1,
where the ratio is forced to be binary. This becomes important later.
-->

---
layout: top-title
color: sky-light
---

:: title ::

# Understanding the Evaluation Metric

:: default ::

<div class="grid grid-cols-2 gap-8 -mt-4 text-sm">
<div>

**What is Spearman ρ?**

- Measures **rank agreement** between predictions and ground truth
- Only **relative ordering** matters — not absolute values
- A perfect ranking scores ρ = 1.0 regardless of scale

<br>

| | Objective | Cares About |
|--|-----------|-------------|
| ❌ MSE | Minimize squared error | Absolute values |
| ✅ Spearman ρ | Preserve rank order | Relative ordering |

<br>

> *"Getting the order right matters more than getting the number right."*

</div>
<div class="flex flex-col justify-start pt-2">

**How well did we do?**

<div class="text-5xl font-bold text-center my-4 text-blue-700">0.5705</div>
<div class="text-center text-base font-semibold text-blue-600 mb-4">Platform Spearman · Rank #5 Globally</div>

| Benchmark | Score |
|-----------|-------|
| Official baseline (RF, 10 trees) | 0.197 |
| **Our final result** | **0.5705** |
| Relative improvement | **+190%** |

</div>
</div>

<!--
This metric insight motivates our key innovation: rank-target training.
If we train to minimize MSE but evaluate with Spearman, we're optimizing the wrong objective.
-->

---
layout: section
color: sky
---

# Section 2
## Data Exploration & Feature Engineering

---
layout: top-title
color: sky-light
---

:: title ::

# What the Data Tells Us

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-[35%] text-sm leading-relaxed">

- **Strongest predictor**: `total_count` (ρ = −0.297) — busier locations have fewer violations
- Geographic patterns carry strong spatial signal
- Temporal features show enforcement cycles
- Weather features: minimal predictive power (ρ < 0.03)

<img src="/doc-figures/fig2_totalcount_vs_violation.png" class="w-full rounded shadow mt-3" />

</div>
<div class="w-[65%]">
<img src="/doc-figures/fig4_spearman_correlation.png" class="w-full rounded shadow" />
<small class="text-gray-500 text-xs">Spearman correlations — total_count and location dominate</small>
</div>
</div>

<!--
Most signal comes from where you are and how busy the location is.
Weather and day-of-week are much weaker predictors.
-->

---
layout: top-title
color: sky-light
---

:: title ::

# Challenge: High Noise in Low-Count Observations

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-[38%] text-sm leading-relaxed">

- Locations with `total_count = 1` account for **~25% of training data**
- For these, `invalid_ratio` is exactly 0 or 1 — binary, not continuous
- Creates severe **label noise** that degrades model training

**Solution: Sample Weighting**

```python
sample_weight = np.log1p(total_count)
```

- Downweights noisy tc=1 samples without discarding them
- Preserves the full 6 M training set

> *"Down-weight unreliable samples, don't throw them away."*

</div>
<div class="w-[62%]">
<img src="/doc-figures/fig_h_noise_diagnosis.png" class="w-full rounded shadow" />
<small class="text-gray-500 text-xs">Label noise diagnosis — tc=1 subset shows extreme bimodality</small>
</div>
</div>

---
layout: top-title
color: sky-light
---

:: title ::

# Tier 2 Feature Engineering Pipeline

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-2/5 text-sm leading-relaxed pt-1">

1. **Spatial binning**: Divide map into grids → `grid_x`, `grid_y`, `grid_id`
2. **K-Fold Target Encoding** (k=5): `grid_te`, `period_te` — captures violation rates per zone without leakage
3. **Cyclic encoding**: `sin/cos(hour)`, `sin/cos(month)` — preserves periodicity
4. **Cross features**: `total_count × grid_te` — busyness × location risk
5. **Result**: ~20 engineered features from 10 originals

<br>

> **K-Fold TE prevents data leakage — a critical design choice.**

</div>
<div class="w-3/5 flex justify-center">
<img src="/figures/feature_engineering_pipeline.png" class="h-88 rounded shadow" />
</div>
</div>

---
layout: top-title-two-cols
color: sky-light
---

:: title ::

# What Drives Violation Rates?

:: left ::

<div class="text-sm leading-relaxed">

- `total_count` and `grid_te` are consistently the **top-2 features**
- Geographic (TE) features dominate over weather and raw coordinates
- SHAP: high `total_count` → **lower violation rate**
- Weather: near-zero SHAP contribution

</div>

:: right ::

<div class="flex flex-col gap-2">
<img src="/figures/shap_dep_total_count.png" class="h-48 rounded shadow" />
<div class="flex gap-2">
<img src="/figures/lgbm_feature_importance.png" class="h-44 rounded shadow flex-1 object-contain" />
<img src="/figures/shap_bar.png" class="h-44 rounded shadow flex-1 object-contain" />
</div>
</div>

<!--
SHAP dependence on total_count: as it increases, SHAP becomes more negative.
Busy zones attract compliant behavior or more enforcement attention.
-->

---
layout: section
color: blue
---

# Section 3
## Baseline Development & Gap Analysis

---
layout: top-title-two-cols
color: blue-light
---

:: title ::

# Model: LightGBM + XGBoost Ensemble

:: left ::

<div class="text-sm leading-relaxed">

- **Base models**: LightGBM and XGBoost (gradient boosting)
- **Ensemble**: weighted average, weights optimized on OOF Spearman
- CatBoost tested — final weight → **0**, excluded
- **Cross-validation**: 5-Fold with Spearman early stopping

| Model | OOF ρ |
|-------|-------|
| LightGBM alone | ~0.630 |
| XGBoost alone | ~0.618 |
| **LGB + XGB ensemble** | **0.6429** |
| + CatBoost | no gain |

</div>

:: right ::

<img src="/figures/ablation_study.png" class="h-52 rounded shadow mb-2" />
<img src="/figures/model_comparison.png" class="h-34 rounded shadow" />

---
layout: top-title-two-cols
color: blue-light
---

:: title ::

# Iterative Improvement: v1 → v7

:: left ::

<div class="text-sm leading-relaxed">

| Version | Key Change | Platform ρ | Δ |
|---------|------------|------------|---|
| v1 | Initial LGB + XGB | 0.5222 | — |
| v2 | Increased n_estimators | 0.5338 | +0.0116 |
| v3 | Optuna hyperparameter tuning | 0.5620 | +0.0282 |
| v7 | Sample weighting `log1p(tc)` | **0.5636** | +0.0016 |

- Optuna (v3) was the biggest pre-innovation gain: **+0.0282**
- OOF ≈ 0.643, Platform ≈ 0.564 — consistent gap ~0.079

> Each engineering decision produced **measurable, documented improvement.**

</div>

:: right ::

<img src="/figures/score_progression.png" class="h-80 rounded shadow" />

<small class="text-gray-500 text-xs">Solid = OOF Spearman · Dashed = Platform Spearman</small>

---
layout: top-title-two-cols
color: blue-light
---

:: title ::

# Why Is There a Gap Between OOF and Platform?

:: left ::

<div class="text-sm leading-relaxed">

**Observed**: OOF ~0.643, Platform ~0.564 → gap **~0.079**

| Hypothesis | Test | Result |
|-----------|------|--------|
| Overfitting | Stronger regularization | ❌ Worse scores |
| **Distribution shift** | Adversarial validation | ✅ AUC = 0.9999 |
| | Temporal CV (M1–M4 → M5) | ✅ Gap −0.041 |
| | TE distribution plot | ✅ Clear mismatch |

Train and test come from **different temporal periods** — shift is structural.

> *"Diagnose before you optimize."*

</div>

:: right ::

<img src="/figures/av_probability_distribution.png" class="h-44 rounded shadow mb-2" />
<img src="/figures/oof_platform_gap.png" class="h-40 rounded shadow" />

<small class="text-gray-500 text-xs">Top: near-perfect AV separation. Bottom: stable ~0.077 gap.</small>

---
layout: section
color: teal
---

# Section 4
## Key Innovation: Rank-Target Training

---
layout: statement
color: cyan
---

# Training with MSE ≠ Optimizing Spearman

MSE penalizes large **numerical** deviations.  
Spearman only cares about **relative order**.

*Training in the wrong direction.*

---
layout: top-title
color: cyan-light
---

:: title ::

# Solution: Train to Rank, Not to Regress

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-[38%] text-sm leading-relaxed">

**Transformation:**

$$y_{\text{rank}} = \frac{\text{rankdata}(y)}{N}$$

- Converts target to **uniform [0, 1] distribution** of ranks
- Model learns **relative ordering**, not absolute values
- No other changes to the pipeline

```python
# Replace raw target with rank-normalized target
y_rank = rankdata(train_df['invalid_ratio']) / len(train_df)
# Train exactly as before
lgb_model.fit(X_train, y_rank, sample_weight=weights, ...)
```

> *"One line of code. The biggest single improvement in our entire pipeline."*

</div>
<div class="w-[62%]">
<img src="/figures/rank_target_diagram.png" class="w-full rounded shadow" />
<small class="text-gray-500 text-xs">Left: skewed original target. Right: uniform rank-transformed target.</small>
</div>
</div>

---
layout: top-title-two-cols
color: cyan-light
---

:: title ::

# Impact: Rank-Target Delivers Our Largest Gain

:: left ::

<div class="text-sm leading-relaxed">

| | OOF ρ | Platform ρ | Δ Platform |
|--|-------|------------|------------|
| v7 (baseline) | 0.6429 | 0.5636 | — |
| Exp C (rank-target) | 0.6464 | 0.5698 | **+0.0062** |
| Exp I-A (+iterations) | **0.6478** | **0.5705** | **+0.0069** |

<br>

- Exp C alone: rank-target gives **+0.0062** — more than any FE step
- Exp I-A: LGB iterations 10K → 20K → extra **+0.0007**
- Final: **Platform 0.5705, Rank #5 globally**

<br>

> Metric alignment = the **biggest single-step improvement** of the project.

</div>

:: right ::

<img src="/figures/score_progression.png" class="h-80 rounded shadow" />

<small class="text-gray-500 text-xs">Full journey: v1 (0.5222) → Exp I-A (0.5705)</small>

---
layout: section
color: slate
---

# Section 5
## Experiment Summary & Analysis

---
layout: top-title
color: slate-light
---

:: title ::

# 9 Experiments: What Worked and What Did Not

:: default ::

<img src="/figures/experiment_summary_chart.png" class="h-88 mx-auto rounded shadow" />

<div class="text-xs text-gray-500 text-center mt-1">
  OOF (solid bars) · Platform (hatched bars) ·
  <span class="text-green-700 font-bold">Green = success</span> ·
  <span class="text-red-600 font-bold">Red = failed</span> ·
  <span class="text-yellow-600 font-bold">Yellow = null result</span>
</div>

---
layout: top-title
color: slate-light
---

:: title ::

# Deep Learning Does Not Help Here

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-[38%] text-sm leading-relaxed">

- Tested **TabM** (ICLR 2025 — state-of-the-art tabular DL)
- OOF Spearman: **0.4445** vs GBDT **0.6429** — gap of **0.198**
- Root cause: only 10 input features, no image/text structure → GBDT advantages dominate on compact tabular data

| Model | OOF ρ | Notes |
|-------|-------|-------|
| TabM (ICLR 2025) | 0.4445 | Best DL attempt |
| LGB + XGB (v7) | 0.6429 | GBDT baseline |
| Rank-Target (Exp I-A) | **0.6478** | Our final |

> *"Domain structure matters more than model architecture."*

</div>
<div class="w-[62%]">
<img src="/figures/tabm_correlation.png" class="w-full rounded shadow" />
<small class="text-gray-500 text-xs">TabM predictions poorly correlated with ground truth</small>
</div>
</div>

---
layout: top-title-two-cols
color: slate-light
---

:: title ::

# Understanding the Model with SHAP

:: left ::

<div class="text-sm leading-relaxed">

- `total_count`: strong **negative** SHAP — busy locations comply more
- `grid_te`: captures spatial violation risk per zone
- Weather: near-zero SHAP — no meaningful contribution
- Model decisions are **explainable** and match domain intuition

<br>

**Insight**: the model learns *"where are the risky zones, and how busy are they right now?"*

</div>

:: right ::

<img src="/figures/shap_bar.png" class="w-full max-h-52 object-contain rounded shadow mb-2" />
<img src="/figures/shap_dep_total_count.png" class="w-full max-h-44 object-contain rounded shadow" />

---
layout: section
color: navy-light
---

# Section 6
## Conclusion

---
layout: top-title
color: navy-light
---

:: title ::

# Results: Platform 0.5705, Rank #5

:: default ::

<div class="flex gap-6 h-full items-start">
<div class="w-[40%] text-sm leading-relaxed">

- Official baseline: **0.197** → Our result: **0.5705** (+190%)
- Leaderboard: **Rank #5 globally**

**Key contributions:**

1. **Tier 2 feature engineering** — leakage-free K-Fold TE
2. **Rank-target training** — aligning objective with Spearman
3. **Systematic gap diagnosis** via adversarial validation
4. **Sample weighting** to handle label noise

<img src="/figures/score_progression.png" class="w-full rounded shadow mt-2" />

🔗 [Public Leaderboard](https://challengedata.ens.fr/participants/challenges/163/ranking/public)

</div>
<div class="w-[60%]">
<img src="/subs/challengedata_ranking.png" class="w-full rounded shadow" />
<small class="text-gray-500 text-xs">Rank #5 globally with Platform Spearman 0.5705</small>
</div>
</div>

---
layout: default
color: navy-light
---

# Key Takeaways

<div class="mt-4 text-sm leading-relaxed">

**Lesson 1: Match your training objective to your evaluation metric**

→ Rank-target training was the single biggest improvement in the project (+0.0069 Platform Spearman)

<br>

**Lesson 2: Diagnose before you optimize**

→ The OOF-Platform gap was structural distribution shift — stronger regularization would have made it worse

<br>

**Lesson 3: Systematic iteration with quantitative baselines outperforms guesswork**

→ Every version tracked, every change measured, every null result documented

<br>

**Future directions:** ensemble stacking · richer spatial features (POI density, road type) · cross-city generalization

</div>

<!--
Three lessons that apply beyond this project:
1. Metric alignment is often overlooked but powerful
2. Diagnose systematically before trying fixes
3. Version control your experiments, not just your code
-->
