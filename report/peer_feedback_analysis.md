# Peer Feedback Analysis

7 feedback items received after presentation. Assessed for report integration.

## Summary Table

| # | Topic | Action | Where in Report |
|---|-------|--------|-----------------|
| 1 | Time-series CV vs K-Fold under temporal shift | Address in Discussion/Limitations | §6 Discussion |
| 2 | Why GBDT over LR/SVM/RF | Add justification in Methodology | §4 Methodology |
| 3 | "GBDT robust to shift" claim too strong | Soften language in Discussion | §6 Discussion |
| 4 | Feature importance ≠ causality | Add caveat in Results | §5 Results |
| 5 | Rank-target robustness under extreme shift | Discuss as limitation | §6 Discussion |
| 6 | Rank-target single-experiment concern | Clarify CV evidence | §5 Results |
| 7 | ML vs DL comparison (positive) | Keep and strengthen | §5 Results |

---

## Detailed Analysis

### Feedback 1: Time-Series CV vs Standard K-Fold

**Claim**: Under severe temporal shift, standard 5-fold CV causes data leakage (future → past), making OOF scores unreliable. Should have used time-series split.

**Assessment**: Partially valid, but our context has nuances.

**Response for report**:
- The temporal distribution shift was discovered *after* the main experiments (Phase 5b, via Adversarial Validation). By that point, the CV pipeline and all model versions (v1–v7) were already established.
- More importantly, each row represents a *location-timeslot observation*, not a sequential time series. The data has no natural ordering — rows from the same hour span different locations. Standard K-fold is the norm for i.i.d. tabular data; time-series split is for sequential forecasting.
- The OOF-Platform gap (~0.077) was diagnosed as TE distribution shift (months 6–12 inflate TE statistics), not as CV leakage. Strong regularization (v9/v9a) did not close the gap, confirming it is not overfitting.
- **Acknowledge**: A grouped time-split (e.g., train on months 1–9, validate on 10–12) could have provided a more conservative OOF estimate. This is a valid point for the Limitations section.

**Report action**: Add 2–3 sentences in §6 Discussion acknowledging that time-aware validation could yield more conservative estimates, while explaining why standard K-fold was appropriate for i.i.d. tabular data and that the gap was confirmed to be TE shift, not leakage.

---

### Feedback 2: Justify GBDT over Other Models

**Claim**: Should compare with non-GBDT models (logistic regression, SVM, RF) and explain why GBDT suits this data.

**Assessment**: Valid. The report should justify model selection.

**Response for report**:
- The official baseline is RF with 10 trees (Spearman 0.197). Our ablation study shows even a basic LightGBM (v1) achieves 0.5815 OOF — a 3× improvement over RF.
- GBDT advantages for this data: (a) handles nonlinear feature interactions (e.g., grid × time × weather), (b) natively handles mixed feature types without scaling, (c) robust to irrelevant features (HauteurNeige, ForceVent contribute minimally), (d) efficient on 6M rows (histogram-based splitting).
- Linear models (LR, SVM) cannot capture the complex spatial-temporal interactions that drive violation patterns. The U-shaped target distribution and nonlinear total_count effect make linear assumptions inappropriate.
- RF comparison exists implicitly: official baseline RF=0.197 vs our GBDT ensemble=0.5705.

**Report action**: Add a paragraph in §4 Methodology explaining why GBDT was chosen, referencing the data characteristics (nonlinearity, mixed types, scale, feature interactions). Mention RF baseline as the comparison point.

---

### Feedback 3: "GBDT Robust to Shift" Claim Too Strong

**Claim**: The presentation's claim that GBDT is "less sensitive to distribution shift" compared to deep learning is not well-founded. Tree models can also degrade under covariate shift.

**Assessment**: Valid criticism. The claim was oversimplified.

**Response for report**:
- The actual finding was: deep learning models (MLP, ResNet, TabM) achieved OOF ~0.42–0.44, far below GBDT's 0.64. This gap is too large to attribute to shift sensitivity alone.
- The more accurate explanation: on this dataset with only 10 raw features and 26 engineered features, GBDT's inductive bias (axis-aligned splits, ensemble averaging) is better matched than neural networks, which excel with high-dimensional or unstructured data.
- Distribution shift affects both GBDT and DL. The OOF-Platform gap (0.077) proves GBDT is not immune.
- **Revised claim**: "GBDT's inductive bias is better suited to low-dimensional tabular data with engineered features, consistent with findings in the tabular learning literature (e.g., Grinsztajn et al., 2022)."

**Report action**: In §6 Discussion, replace any "GBDT is more robust to shift" language with the more precise explanation above. Cite the tabular ML benchmark literature if appropriate.

---

### Feedback 4: Feature Importance ≠ Causality

**Claim**: SHAP feature importance shows correlation, not causation. Some features may be proxies, not causal drivers.

**Assessment**: Valid and standard ML caveat.

**Response for report**:
- This is a correct observation. Target-encoded features (grid_period_te, grid_te) are by construction correlated with the target — they encode historical violation rates for spatial-temporal groups. High SHAP importance reflects predictive power, not causal mechanisms.
- total_count's negative correlation with invalid_ratio is partly mechanical: with more observations, extreme ratios (0 or 1) become less likely (law of large numbers), independent of actual violation behavior.
- Our goal is *prediction* (Spearman correlation), not causal inference. The distinction should be stated explicitly.

**Report action**: Add a brief caveat in §5 Results (SHAP analysis subsection): "Feature importance reflects predictive contribution, not causal influence. Target-encoded features are by construction correlated with the outcome. Causal analysis would require controlled experiments or instrumental variable approaches, which are beyond the scope of this prediction task."

---

### Feedback 5: Rank-Target Robustness Under Extreme Shift

**Claim**: If test distribution is completely different from training, does rank-target training remain robust?

**Assessment**: A theoretical concern, partially addressed by our results.

**Response for report**:
- Rank-target training converts regression into a ranking problem. It is robust when the *relative ordering* of predictions transfers across distributions, even if absolute values shift.
- In our case, rank-target improved Platform score from 0.5636 (v7) to 0.5698 (Exp C), then 0.5705 (Exp I-A) — the OOF-Platform gap actually narrowed slightly (from 0.079 to 0.077), suggesting rank-target does transfer reasonably well.
- However, under extreme distribution shift where the *ranking itself* changes (e.g., features that are positively correlated in training become negatively correlated in test), rank-target offers no special protection.
- This is a legitimate limitation worth mentioning.

**Report action**: Add 1–2 sentences in §6 Discussion: "Rank-target training is robust when relative ordering transfers, but does not guarantee robustness under extreme covariate shift where feature-target relationships reverse."

---

### Feedback 6: Single-Experiment Concern for Rank-Target

**Claim**: The rank-target gain conclusion may come from a single experiment; repeated experiments would be more convincing.

**Assessment**: Partially valid, but our evidence is actually stronger than a single run.

**Response for report**:
- Rank-target was evaluated with **5-fold cross-validation** (SEED=42), not a single train/test split. The OOF score aggregates predictions across all 5 folds, covering the entire training set.
- The gain is consistent: Exp C (rank-target, same hyperparameters as v7) improved OOF from 0.6429 to 0.6464, and Platform from 0.5636 to 0.5698. Exp I-A (rank-target + 20K iterations) further improved to OOF 0.6478 / Platform 0.5705.
- Multiple random seeds would further strengthen the evidence, but with 5-fold CV on 6M rows, the variance is already small.
- **Acknowledge**: Testing with multiple seeds (e.g., 3–5 seeds) would provide confidence intervals and is a valid suggestion for future work.

**Report action**: In §5 Results, clarify that rank-target results are from 5-fold CV (not single split). In §6 Discussion, acknowledge that multi-seed evaluation would provide confidence intervals.

---

### Feedback 7: ML vs DL Comparison (Positive)

**Claim**: The comparison explaining why ML outperforms DL was impressive.

**Assessment**: Positive feedback. Strengthen this section in the report.

**Report action**: Ensure §5 Results includes a clear ML vs DL comparison with:
- Quantitative gap: GBDT OOF 0.64 vs NN OOF 0.42–0.44
- Explanation: low-dimensional tabular data favors GBDT's inductive bias
- Reference to tabular ML literature
- Honest note that DL might excel with richer feature sets (images, text, embeddings)
