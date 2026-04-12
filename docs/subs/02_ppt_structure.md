# PPT Structure: Predict Parking Violations (CS5483)

**Target duration:** 11–12 minutes (AI voice-over, ~120 wpm, ~1350–1450 words)  
**Total slides:** ~18 slides  
**Language:** English throughout

---

## Section 1: Introduction & Problem Setup
**Duration:** ~1.5 min | **Slides:** 3

---

### Slide 1 — Title Slide
**Title:** Predicting Parking Violation Rates Using Gradient Boosting  
**Subtitle:** ChallengeData #163 | CS5483 Data Mining | Group [X]  
**Visual:** fig5_spatial_violation.png (background or inset — geographic map of Thessaloniki)

**Speaker notes key point:**  
Open with the real-world context: a smart parking enforcement system in Greece.

---

### Slide 2 — Problem & Data Overview
**Heading:** What Are We Predicting?

**Bullet points:**
- Dataset: THESi street parking system, Thessaloniki, Greece
- Training set: 6.07 million observations, 10 features
- Target variable: `invalid_ratio` — fraction of invalid parking events at a location-time slot
- Evaluation metric: **Spearman rank correlation** (not RMSE or MAE)

**Visual:** fig1_target_distribution.png  
**Key message on slide:** "The metric rewards ranking — not numerical accuracy."

---

### Slide 3 — Why Spearman? (Metric Insight)
**Heading:** Understanding the Evaluation Metric

**Content:**
- Spearman ρ measures how well predictions preserve the **rank order** of true values
- A model that perfectly ranks all locations scores ρ = 1.0, regardless of absolute values
- This is fundamentally different from minimizing mean squared error
- Official baseline (Random Forest, 10 trees): **ρ = 0.197**
- Our final result: **ρ = 0.5705, Rank #5 globally**

**Visual:** Simple diagram or text box contrasting MSE vs Spearman  
**+ Small inset:** Leaderboard screenshot (thumbnail, cropped to show Rank #5 row) — placed bottom-right corner as a teaser  
**Key quote on slide:** *"Getting the order right matters more than getting the number right."*

---

## Section 2: Data Exploration & Feature Engineering
**Duration:** ~2.5 min | **Slides:** 4

---

### Slide 4 — Key EDA Findings
**Heading:** What the Data Tells Us

**Bullet points:**
- Strongest predictor: `total_count` (Spearman ρ = −0.297)
- Geographic location (longitude, latitude) carries spatial violation patterns
- Temporal features (hour, month) show clear enforcement cycles

**Visual:** fig2_totalcount_vs_violation.png + fig4_spearman_correlation.png (side by side)  
**Key message:** "Most signal comes from location and activity volume."

---

### Slide 5 — The Noise Problem
**Heading:** Challenge: High Noise in Low-Count Observations

**Content:**
- Locations with `total_count = 1` account for ~25% of training data
- For these, `invalid_ratio` is either exactly 0 or exactly 1 — binary, not continuous
- This creates severe label noise that degrades model training
- Solution: Apply **sample weights = log1p(total_count)** to downweight noisy samples

**Visual:** fig_h_noise_diagnosis.png  
**Key message on slide:** "We down-weighted unreliable samples rather than discarding them."

---

### Slide 6 — Feature Engineering: Tier 2
**Heading:** Tier 2 Feature Engineering Pipeline

**Content (as numbered steps or flow arrows):**
1. **Spatial binning:** Divide map into grids → `grid_id`
2. **K-Fold Target Encoding (k=5):** Encode grid & time periods → prevents data leakage
3. **Cyclic encoding:** `sin/cos(hour)`, `sin/cos(month)` — preserve cyclical nature
4. **Cross features:** `total_count × grid_te`
5. **Result:** ~20 engineered features from 10 originals

**Visual:** feature_engineering_pipeline.png (new) OR a simple table showing raw → engineered features  
**Key message:** "K-Fold TE prevents leakage — a critical design choice."

---

### Slide 7 — Feature Importance
**Heading:** What Drives Violation Rates?

**Bullet points:**
- `total_count` and `grid_te` are consistently the top two features
- Geographic features dominate over weather and time features
- SHAP analysis confirms: high `total_count` → lower violation rate

**Visual:** shap_summary.png or lgbm_feature_importance.png + shap_dep_total_count.png  
**Key message:** "Location-based Target Encoding captures spatial violation patterns effectively."

---

## Section 3: Baseline Development & Gap Analysis
**Duration:** ~2 min | **Slides:** 3

---

### Slide 8 — Model Architecture
**Heading:** Model: LightGBM + XGBoost Ensemble

**Content:**
- Base models: LightGBM and XGBoost (gradient boosting decision trees)
- Ensemble: weighted average, weights optimized on OOF Spearman
- CatBoost was tested — final weight converged to 0, excluded
- Cross-validation: 5-Fold with Spearman early stopping

**Visual:** model_comparison.png  
**Key message:** "Two-model ensemble outperformed a three-model one."

---

### Slide 9 — Baseline Progression (v1 → v7)
**Heading:** Iterative Improvement: From v1 to v7

**Content (table or annotated line chart):**
| Version | Key Change | Platform ρ |
|---------|-----------|-----------|
| v1 | Initial LGB + XGB | 0.5222 |
| v2 | Increased estimators | 0.5338 |
| v3 | Optuna hyperparameter tuning | 0.5620 |
| v7 | Sample weighting log1p(tc) | **0.5636** |

**Visual:** score_progression.png (new) — show only up to v7 with arrow pointing right  
**Key message:** "Each engineering decision produced measurable, documented improvement."

---

### Slide 10 — Gap Diagnosis (Exp D)
**Heading:** Why Is There a Gap Between OOF and Platform?

**Content:**
- Observed: OOF Spearman ~0.643, Platform Spearman ~0.564 → gap ~0.079
- Hypothesis 1 (overfitting): Rejected — stronger regularization made scores worse
- **Hypothesis 2 (distribution shift):** Supported
  - Adversarial validation: train vs. test classifier AUC = **0.9999**
  - Temporal CV (train M1–M4, validate M5): gap −0.041
  - Conclusion: train and test data come from different temporal distributions

**Visual:** av_probability_distribution.png or te_distribution_shift.png  
**Key message on slide:** *"We diagnosed the gap before trying to fix it."*

---

## Section 4: Key Innovation — Rank-Target Training
**Duration:** ~3 min | **Slides:** 3

---

### Slide 11 — The Problem with Standard Regression
**Heading:** Misaligned Training Objective

**Content:**
- Standard approach: train with MSE loss → minimizes numerical error
- Evaluation: Spearman ρ → measures rank agreement
- **These objectives are fundamentally different:**
  - MSE: penalizes large numerical deviations
  - Spearman: only cares about relative order

**Visual:** Text diagram: "MSE ≠ Spearman" with a simple example  
**Key message:** *"Training with the wrong objective is training in the wrong direction."*

---

### Slide 12 — Rank-Target Transformation
**Heading:** Solution: Train to Rank, Not to Regress

**Content:**
- **Transformation:** Replace `invalid_ratio` with `y_rank = rankdata(y) / N`
- This converts the target to a uniform [0, 1] distribution of ranks
- The model now learns to predict **relative ordering**, not absolute values
- At inference: predictions are re-ranked → Spearman is computed on ranks

**Visual:** rank_target_diagram.png (new) — side-by-side histogram of raw vs rank target  
**Key message on slide:**
```
y_rank = rankdata(y) / N
```
*"One line of code. The biggest single improvement in our entire pipeline."*

---

### Slide 13 — Results of Rank-Target (Exp C & Exp I-A)
**Heading:** Impact: Rank-Target Delivers Our Largest Gain

**Content (comparison table):**
| | OOF Spearman | Platform Spearman | Change |
|--|---|---|---|
| v7 (baseline) | 0.6429 | 0.5636 | — |
| Exp C (rank-target) | 0.6464 | 0.5698 | **+0.0062** |
| Exp I-A (+iterations) | 0.6478 | **0.5705** | **+0.0069** |

- Further tuning (Exp I): Increased LGB iterations from 10K to 20K → additional +0.0007
- Final result: **Platform 0.5705, Rank #5 globally**

**Visual:** score_progression.png (new, highlighted Exp C and Exp I-A region)  
**Key message:** "Metric alignment gave us the biggest single-step improvement."

---

## Section 5: Experiment Summary & Analysis
**Duration:** ~2 min | **Slides:** 3

---

### Slide 14 — All Experiments at a Glance
**Heading:** 9 Experiments: What Worked and What Did Not

**Visual:** experiment_summary_chart.png (new) — horizontal bar chart

Brief annotations:
- ✅ Rank-target training (Exp C, I-A): **Success — largest gain**
- ❌ Strong regularization (v9): Confirmed gap is NOT overfitting
- ❌ DART boosting: Precision loss exceeded diversity gain
- ❌ TabM deep learning (Exp E): OOF ceiling 0.445 — GBDT far superior
- ❌ Noise removal (Exp H): OOF improved but Platform dropped
- ⚪ Pseudo-labeling (Exp G): Null result — prediction range mismatch

---

### Slide 15 — Why Deep Learning Failed (Exp E)
**Heading:** Deep Learning Does Not Help Here

**Content:**
- Tested TabM (ICLR 2025 — state-of-the-art tabular DL architecture)
- OOF Spearman: **0.4445** vs GBDT **0.6429** — gap of 0.198
- Root cause: only 10 input features, no image/text structure → GBDT advantages dominate
- Conclusion: more data and more features would be needed for DL to compete

**Visual:** tabm_correlation.png or a simple comparison bar  
**Key message:** "Domain structure matters more than model architecture."

---

### Slide 16 — Model Interpretability (SHAP)
**Heading:** Understanding the Model with SHAP

**Content:**
- `total_count`: strong negative effect — busy locations have lower violation rates
- `grid_te`: high-violation areas have elevated TE values
- Weather features (precipitation, temperature): minimal SHAP contribution
- Model is interpretable and consistent with domain knowledge

**Visual:** shap_bar.png + shap_dep_total_count.png  
**Key message:** "Our model's decisions are explainable and align with real-world intuition."

---

## Section 6: Conclusion
**Duration:** ~1 min | **Slides:** 2

---

### Slide 17 — Final Results
**Heading:** Results: Platform 0.5705, Rank #5

**Content:**
- Official baseline: 0.197 | Our result: **0.5705** | Improvement: **+190%**
- Leaderboard: **Rank #5** globally

**Key contributions:**
1. Tier 2 feature engineering with leakage-free Target Encoding
2. **Rank-target training** — aligning training with Spearman evaluation
3. Systematic gap diagnosis via adversarial validation

**Visual (two-panel layout recommended):**
- Left panel: score_progression.png (full journey, start highlighted at 0.5222, end at 0.5705)
- Right panel: **Leaderboard screenshot** (full size, highlight/circle our row at Rank #5 with score 0.5705)

**Screenshot tips:**
- Crop to show top 8–10 rows so Rank #5 is clearly visible
- Use a red rectangle or circle annotation to mark our row
- Include the score column and rank column; hide personal info of other teams if needed

---

### Slide 18 — Takeaways & Future Work
**Heading:** Key Takeaways

**Content:**
- **Lesson 1:** Match your training objective to your evaluation metric
- **Lesson 2:** Diagnose before you optimize — the gap was distribution shift, not overfitting
- **Lesson 3:** Systematic iteration with quantitative baselines outperforms guesswork

**Future directions (brief):**
- Stacking or blending with additional base models
- Richer spatial features (POI density, road type)
- Cross-city generalization (other THESi deployments)

---

## Notes on Slide Design

- **Font:** Use a clean sans-serif (e.g., Arial, Calibri, or Roboto)
- **Color scheme:** Blue (#1565C0) for primary, Orange (#E65100) for highlights, Gray (#616161) for secondary
- **Each slide:** Max 5 bullet points; prefer visuals over text
- **Innovation slides (11–13):** Use a colored callout box for the key formula / key quote
- **Consistent header bar** across all slides with project name on top-right

---

## Slide Count Summary

| Section | Slides | Duration |
|---------|--------|----------|
| 1. Introduction | 3 | 1.5 min |
| 2. EDA & Feature Engineering | 4 | 2.5 min |
| 3. Baseline & Gap Analysis | 3 | 2 min |
| 4. Key Innovation (Rank-Target) | 3 | 3 min |
| 5. Experiments | 3 | 2 min |
| 6. Conclusion | 2 | 1 min |
| **Total** | **18** | **~12 min** |
