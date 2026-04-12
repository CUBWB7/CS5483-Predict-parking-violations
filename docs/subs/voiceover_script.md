# Voiceover Script: Predicting Parking Violation Rates

**Based on:** `presentation/slides.md` (24 slides, April 2026)  
**Target length:** ~1,450 words | ~12 minutes at 120 wpm (AI voice-over)  
**Language:** English  
**AI voice-over notes:** Sentences are short to medium length. No contractions. [PAUSE] marks slide transitions. Section divider slides have a single short transition line.

---

## Section 1: Introduction & Problem Setup

---

**[Slide 1 — Cover]**

In this video, we present our solution to ChallengeData Challenge 163 — predicting parking violation rates in a real-world smart parking system. The heatmap behind us shows actual geographic violation patterns across the streets of Thessaloniki, Greece.

[PAUSE]

---

**[Slide 2 — Section Divider: Introduction & Problem Setup]**

We begin with the problem setup.

[PAUSE]

---

**[Slide 3 — What Are We Predicting?]**

The dataset comes from THESi, a street parking management system deployed in Thessaloniki, Greece — one of the largest real-world parking datasets publicly available. Our training set contains 6.07 million observations. Each record describes a parking location and time slot, with 10 input features including geographic coordinates, weather conditions, and temporal information. The target variable is `invalid_ratio` — the fraction of parking events recorded as violations at a given location and time.

Notice the distribution on the right: there is a heavy mass at exactly 0 and exactly 1. This bimodal pattern is caused by locations where only a single parking event was recorded. We will revisit this noise problem shortly.

[PAUSE]

---

**[Slide 4 — Understanding the Evaluation Metric]**

Before we dive into our methods, it is important to understand what we are optimizing. The evaluation metric for this challenge is Spearman rank correlation. Unlike mean squared error, Spearman does not care about numerical accuracy. It only measures how well the model preserves the relative order of predictions. A model that correctly ranks all locations from lowest to highest violation rate will achieve a perfect score of 1.0, even if its absolute predictions are off.

This insight became the foundation of our entire approach. The official Random Forest baseline achieves a Spearman of 0.197. Our final model achieves 0.5705, placing us at Rank 5 on the global leaderboard — nearly triple the baseline performance.

[PAUSE]

---

## Section 2: Data Exploration & Feature Engineering

---

**[Slide 5 — Section Divider: Data Exploration & Feature Engineering]**

Next, we describe how we explored the data and engineered features.

[PAUSE]

---

**[Slide 6 — What the Data Tells Us]**

Our exploratory analysis revealed that the single strongest predictor is `total_count`, which measures how many parking events occurred at a location. It has a Spearman correlation of negative 0.297 with the target. Locations with more parking activity tend to have lower violation rates, likely because they are well-monitored or attract more compliant behavior. Geographic location is the second most important factor. Temporal features show enforcement cycles. Weather features, on the other hand, contribute very little — their Spearman correlations are below 0.03.

[PAUSE]

---

**[Slide 7 — Challenge: High Noise in Low-Count Observations]**

The data has a significant challenge. Approximately 25 percent of training observations come from locations where `total_count` equals 1. For these records, the violation ratio is forced to be either exactly 0 or exactly 1. This is a binary outcome, not a continuous rate — and it creates severe label noise that can mislead the model.

Our solution was to apply sample weights equal to the natural log of one plus `total_count`. This downweights the noisy single-event observations without discarding them, allowing the model to focus on more reliable multi-event records while preserving the full 6 million training set.

[PAUSE]

---

**[Slide 8 — Tier 2 Feature Engineering Pipeline]**

Raw features alone are insufficient to capture the spatial and temporal structure of parking behavior. We developed a five-step pipeline.

First, we divided the map into geographic grid cells. Second, we applied K-Fold Target Encoding with 5 folds to encode historical violation rates per grid cell and time period. K-Fold prevents data leakage — a critical design decision. Third, we applied cyclic encoding to hour and month using sine and cosine transforms. Fourth, we created cross features by multiplying total_count with the grid encoding. This expanded the feature set from 10 raw inputs to approximately 20 engineered features.

[PAUSE]

---

**[Slide 9 — What Drives Violation Rates?]**

SHAP analysis confirms that `total_count` and `grid_te` — the geographic target encoding — are consistently the top two features. As `total_count` increases, its SHAP value becomes increasingly negative, confirming that busier locations have lower violation rates. Weather and wind features receive near-zero SHAP values, confirming they contribute little predictive power. This interpretability supports the correctness of our feature engineering choices.

[PAUSE]

---

## Section 3: Baseline Development & Gap Analysis

---

**[Slide 10 — Section Divider: Baseline Development & Gap Analysis]**

We now turn to model development and gap analysis.

[PAUSE]

---

**[Slide 11 — Model: LightGBM + XGBoost Ensemble]**

For our model, we chose an ensemble of LightGBM and XGBoost — two gradient boosting frameworks that consistently perform well on tabular data. LightGBM alone achieves an OOF Spearman of approximately 0.630. XGBoost alone achieves approximately 0.618. Their ensemble reaches 0.6429. We evaluated CatBoost as a third model, but its optimized ensemble weight converged to zero, so we excluded it. All models use 5-Fold cross-validation with Spearman correlation as the early stopping criterion.

[PAUSE]

---

**[Slide 12 — Iterative Improvement: v1 to v7]**

We followed a rigorous iterative development process, documenting every change. Starting from version 1 with a platform score of 0.5222, we improved by increasing the number of estimators, which added 0.0116. Optuna hyperparameter tuning then delivered the largest pre-innovation gain of 0.0282, bringing the score to 0.5620. Sample weighting in version 7 added a further 0.0016 to reach our pre-sprint baseline of 0.5636.

Throughout this process, the out-of-fold Spearman was approximately 0.643, while the platform score was approximately 0.564 — a consistent gap of about 0.079.

[PAUSE]

---

**[Slide 13 — Why Is There a Gap Between OOF and Platform?]**

At this point, we asked a critical question: what is causing this gap?

We tested the overfitting hypothesis by applying stronger regularization. The result was worse, not better — so overfitting was ruled out. We then ran an adversarial validation experiment, training a classifier to distinguish training samples from test samples. The classifier achieved an AUC of 0.9999 — essentially perfect separation. Temporal cross-validation further confirmed the finding. The conclusion is clear: the training and test sets come from different temporal distributions. The gap is a structural distribution shift problem, not an overfitting problem.

This diagnosis guided all subsequent experiments. Approaches designed to reduce overfitting were eliminated early, saving significant time.

[PAUSE]

---

## Section 4: Key Innovation — Rank-Target Training

---

**[Slide 14 — Section Divider: Rank-Target Training]**

This brings us to the most important contribution of our work: rank-target training.

[PAUSE]

---

**[Slide 15 — Statement: Training with MSE ≠ Optimizing Spearman]**

Standard gradient boosting minimizes mean squared error. But Spearman correlation measures rank agreement — not numerical proximity. MSE penalizes large numerical deviations. Spearman only cares about relative order. These two objectives are fundamentally different, and training with MSE means training in the wrong direction.

[PAUSE]

---

**[Slide 16 — Solution: Train to Rank, Not to Regress]**

The solution was conceptually simple but highly effective. Instead of training on the raw `invalid_ratio` values, we replaced the training target with its normalized rank.

The formula is: y-rank equals rankdata of y, divided by N.

This transformation converts the target from a skewed distribution to a uniform distribution between 0 and 1. The model now learns to predict the relative ordering of violation rates — which is exactly what Spearman measures. No other change to the pipeline was needed. Just one replacement of the target variable.

This single change represents the most impactful modification in our entire pipeline.

[PAUSE]

---

**[Slide 17 — Impact: Rank-Target Delivers Our Largest Gain]**

The results confirmed our hypothesis. Compared to version 7, the rank-target model — Experiment C — improved the platform Spearman by 0.0062, from 0.5636 to 0.5698. In Experiment I-A, we further increased the LightGBM iteration limit from 10,000 to 20,000 rounds, gaining an additional 0.0007 to reach our final score of 0.5705.

Rank-target training delivered the biggest single-step improvement in the entire project — more than any feature engineering step, and more than Optuna hyperparameter tuning.

[PAUSE]

---

## Section 5: Experiment Summary & Analysis

---

**[Slide 18 — Section Divider: Experiment Summary & Analysis]**

We now summarize our full experiment log.

[PAUSE]

---

**[Slide 19 — 9 Experiments: What Worked and What Did Not]**

Across the project, we conducted 9 experiments, each with a clear hypothesis and evaluated on both OOF and platform scores. The chart summarizes all results: green bars indicate success, red bars indicate failure, and yellow indicates null results.

Successful experiments include rank-target training in Experiment C and the iteration increase in Experiment I-A. Unsuccessful experiments include strong regularization — which confirmed the gap is not overfitting — and noise label removal in Experiment H, which improved the OOF score but lowered the platform score. Pseudo-labeling in Experiment G produced a null result due to a mismatch between the rank-target prediction range and the binary threshold required for labeling.

Documenting failed experiments is as important as documenting successes. Each failure narrowed the search space and informed the next step.

[PAUSE]

---

**[Slide 20 — Deep Learning Does Not Help Here]**

We also evaluated TabM, a state-of-the-art tabular deep learning architecture published at ICLR 2025. Its out-of-fold Spearman score was 0.4445 — approximately 0.20 below our gradient boosting ensemble. The fundamental limitation is clear: our dataset has only 10 input features with no image or text structure. In this regime, gradient boosting machines have well-known advantages. They handle sparse signals efficiently, require less architectural tuning, and are less sensitive to distribution shift. More data and richer features would be needed for deep learning to compete here.

[PAUSE]

---

**[Slide 21 — Understanding the Model with SHAP]**

SHAP analysis provides an additional layer of confidence in our model. The model assigns high importance to `total_count` and geographic target encoding, which align with domain knowledge about parking enforcement. The dependence plot confirms a consistent negative relationship: as `total_count` increases, the SHAP value decreases — busy locations comply more or are better policed. Weather features receive near-zero SHAP values. Our model is interpretable, and its decisions match real-world intuition.

[PAUSE]

---

## Section 6: Conclusion

---

**[Slide 22 — Section Divider: Conclusion]**

We close with our final results and key takeaways.

[PAUSE]

---

**[Slide 23 — Results: Platform 0.5705, Rank #5]**

Our final model achieves a platform Spearman of 0.5705, placing us at Rank 5 on the global leaderboard. The leaderboard on the right confirms our submission at Row 5 with a score of 0.5705. This represents a 190 percent improvement over the official baseline of 0.197.

Four contributions drove this result. First, Tier 2 feature engineering with leakage-free K-Fold target encoding. Second, rank-target training that directly optimizes Spearman correlation. Third, systematic gap diagnosis via adversarial validation, which prevented us from pursuing ineffective regularization strategies. Fourth, sample weighting to handle label noise in low-count observations.

[PAUSE]

---

**[Slide 24 — Key Takeaways]**

We close with three lessons from this project.

First: match your training objective to your evaluation metric. Rank-target training was the single biggest improvement in the project, adding 0.0069 to the platform Spearman score.

Second: diagnose before you optimize. Understanding that the OOF-to-platform gap was caused by distribution shift — not overfitting — prevented us from pursuing the wrong direction.

Third: systematic iteration with quantitative baselines outperforms guesswork. Every version was tracked, every change was measured, and every null result was documented.

These three principles — align your objective, diagnose before you optimize, and iterate with evidence — carried us from a baseline of 0.197 to Rank 5 globally. Thank you.

---

## Word Count Reference

| Section | Slides | Approx. Words |
|---------|--------|--------------|
| Sec 1: Introduction | 1–4 | ~210 |
| Sec 2: EDA & Feature Eng. | 5–9 | ~310 |
| Sec 3: Baseline & Gap | 10–13 | ~270 |
| Sec 4: Rank-Target | 14–17 | ~310 |
| Sec 5: Experiments | 18–21 | ~250 |
| Sec 6: Conclusion | 22–24 | ~185 |
| **Total** | **24** | **~1,535** |

At 120 wpm (typical AI voice speed): ~12.8 minutes.  
At 130 wpm (slightly faster AI setting): ~11.8 minutes.

> **Note:** Section divider slides (2, 5, 10, 14, 18, 22) each carry only a one-sentence transition line.  
> If the AI voice reads faster than expected, the video will land comfortably within 12 minutes.
