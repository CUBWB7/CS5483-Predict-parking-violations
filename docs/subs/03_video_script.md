# Video Script: Predicting Parking Violation Rates

**Target length:** ~1,400 words | ~11–12 minutes at 120 wpm (AI voice-over)  
**Language:** English  
**AI voice-over notes:** Sentences are short to medium length. No contractions. [PAUSE] marks slide transitions.

---

## Section 1: Introduction & Problem Setup
*(Slides 1–3 | ~1.5 min)*

---

**[Slide 1 — Title]**

In this video, we present our solution to ChallengeData Challenge 163 — predicting parking violation rates in a real-world smart parking system.

[PAUSE]

---

**[Slide 2 — Problem & Data]**

The dataset comes from THESi, the street parking management system deployed in Thessaloniki, Greece. It is one of the largest real-world parking datasets publicly available. Our training set contains over 6 million observations, each describing a parking location and time slot, with 10 input features including geographic coordinates, weather conditions, and temporal information. The target variable is `invalid_ratio` — the fraction of parking events that were recorded as violations at a given location and time.

[PAUSE]

---

**[Slide 3 — Why Spearman?]**

Before we dive into our methods, it is important to understand what we are optimizing. The evaluation metric for this challenge is Spearman rank correlation. Unlike mean squared error, Spearman does not care about numerical accuracy. It only measures how well the model preserves the relative order of the predictions. A model that correctly ranks all locations from lowest to highest violation rate will achieve a perfect Spearman score of 1.0, even if its absolute predictions are wrong.

This insight became the foundation of our entire approach. The official Random Forest baseline achieves a Spearman of 0.197. Our final model achieves 0.5705, placing us at rank 5 on the global leaderboard — nearly triple the baseline performance.

[PAUSE]

---

## Section 2: Data Exploration & Feature Engineering
*(Slides 4–7 | ~2.5 min)*

---

**[Slide 4 — Key EDA Findings]**

Our exploratory analysis revealed that the single strongest predictor is `total_count`, which measures how many parking events occurred at a location. It shows a Spearman correlation of negative 0.297 with the target: locations with more parking activity tend to have lower violation rates, likely because they are well-monitored. Geographic location is the second most important factor, as violation rates vary significantly across city districts.

[PAUSE]

---

**[Slide 5 — The Noise Problem]**

However, the data has a significant challenge. Approximately 25 percent of training observations come from locations where `total_count` equals 1. For these records, the violation ratio can only be 0 or 1 — it is a binary outcome, not a continuous rate. This creates severe label noise that can mislead the model.

Our solution was to apply sample weights equal to the log of one plus `total_count`. This down-weights the noisy single-event observations without discarding them entirely, allowing the model to focus on more reliable data points.

[PAUSE]

---

**[Slide 6 — Feature Engineering Pipeline]**

Raw features alone are insufficient to capture the spatial and temporal structure of parking behavior. We developed a Tier 2 feature engineering pipeline with five key steps.

First, we divided the map into geographic grid cells to capture neighborhood-level patterns. Second, we applied K-Fold Target Encoding with 5 folds to encode violation rates for each grid cell and time period. Using K-Fold prevents data leakage — a critical design decision. Third, we applied cyclic encoding to hour and month features using sine and cosine transforms. Fourth, we created cross features by multiplying `total_count` with the grid target encoding. This process expanded the feature set from 10 raw features to approximately 20 engineered features.

[PAUSE]

---

**[Slide 7 — Feature Importance]**

SHAP analysis confirms that `total_count` and `grid_te` — the geographic target encoding — are consistently the top two features. Weather features such as precipitation and wind contribute very little to the model's predictions. This is consistent with domain knowledge: parking enforcement patterns are driven primarily by location and activity volume, not by weather.

[PAUSE]

---

## Section 3: Baseline Development & Gap Analysis
*(Slides 8–10 | ~2 min)*

---

**[Slide 8 — Model Architecture]**

For our model, we chose an ensemble of LightGBM and XGBoost — two gradient boosting frameworks that consistently perform well on tabular data. We evaluated CatBoost as a third model, but its ensemble weight converged to zero during optimization, so we excluded it. All models use 5-Fold cross-validation with Spearman correlation as the early stopping criterion.

[PAUSE]

---

**[Slide 9 — Baseline Progression v1 to v7]**

We followed an iterative development process, documenting every improvement. Starting from version 1 with a platform score of 0.5222, we progressively improved through Optuna hyperparameter tuning, which brought the platform score to 0.5620, and then through sample weighting, which brought it to 0.5636. This became our pre-sprint baseline, version 7.

[PAUSE]

---

**[Slide 10 — Gap Diagnosis]**

At this point, we noticed a persistent gap: our out-of-fold Spearman score was approximately 0.643, but our platform score was only 0.564. The gap was about 0.079. Before trying to close this gap, we first asked: what is causing it?

We tested the overfitting hypothesis by applying stronger regularization. The result was worse, not better. This ruled out overfitting.

We then ran an adversarial validation experiment, training a classifier to distinguish training samples from test samples. The AUC was 0.9999 — nearly perfect separation. This confirmed that the training and test sets come from different temporal distributions. The gap is a distribution shift problem, not an overfitting problem.

This diagnosis guided all subsequent experiments. Approaches designed to reduce overfitting were eliminated early, saving significant time.

[PAUSE]

---

## Section 4: Key Innovation — Rank-Target Training
*(Slides 11–13 | ~3 min)*

---

**[Slide 11 — The Problem with MSE]**

With the baseline established and the gap diagnosed, we turned to the most important question: how can we better align our model with the evaluation metric?

Standard gradient boosting minimizes mean squared error during training. But Spearman correlation measures rank agreement, not numerical proximity. These two objectives are fundamentally different. A model trained with MSE may produce predictions that are numerically close to the true values, yet still rank them poorly. We were training in the wrong direction.

[PAUSE]

---

**[Slide 12 — Rank-Target Transformation]**

The solution was conceptually simple but highly effective: instead of training on the raw `invalid_ratio` values, we replaced the training target with its normalized rank.

The formula is: y-rank equals rankdata of y, divided by N.

This transformation converts the target from a skewed distribution to a uniform distribution of relative ranks between 0 and 1. The model now learns to separate high-violation locations from low-violation ones — exactly what Spearman measures.

This single change represents the most impactful modification in our entire pipeline.

[PAUSE]

---

**[Slide 13 — Results]**

The results confirmed our hypothesis. Compared to version 7, the rank-target model, Experiment C, improved the platform Spearman by 0.0062, from 0.5636 to 0.5698. In Experiment I-A, we further increased the iteration limit for LightGBM from 10,000 to 20,000 rounds, gaining an additional 0.0007 to reach our final score of 0.5705.

To summarize: metric alignment gave us the largest single improvement in the project.

[PAUSE]

---

## Section 5: Experiment Summary & Analysis
*(Slides 14–16 | ~2 min)*

---

**[Slide 14 — All Experiments]**

Across the full project, we conducted 9 systematic experiments. Each experiment was designed with a clear hypothesis and evaluated using both out-of-fold Spearman and platform submission.

Successful experiments include rank-target training in Experiment C and iteration increase in Experiment I-A. Unsuccessful experiments include DART boosting, which reduced precision without adding diversity; noise label removal in Experiment H, which improved OOF score but lowered the platform score; and pseudo-labeling in Experiment G, which produced a null result due to a mismatch between the rank-target prediction range and the required threshold values.

Documenting failed experiments is as important as documenting successes. Each failure narrowed the search space and informed the next step.

[PAUSE]

---

**[Slide 15 — Why Deep Learning Failed]**

We also evaluated TabM, a state-of-the-art tabular deep learning architecture from ICLR 2025. Its out-of-fold Spearman score was 0.4445 — approximately 0.20 below our GBDT ensemble. The fundamental limitation is that our dataset has only 10 input features with no image or text structure. In this regime, gradient boosting machines have well-known advantages over neural networks: they handle sparse signals efficiently, require no extensive tuning of architecture depth, and are less sensitive to distribution shift.

[PAUSE]

---

**[Slide 16 — Model Interpretability]**

SHAP analysis provides an additional layer of confidence in our model. The model assigns high importance to `total_count` and geographic target encoding, which align with domain knowledge about parking enforcement. Weather and wind features receive near-zero SHAP values, confirming they contribute little predictive power. This interpretability supports the correctness of our feature engineering choices.

[PAUSE]

---

## Section 6: Conclusion
*(Slides 17–18 | ~1 min)*

---

**[Slide 17 — Final Results]**

Our final model achieves a platform Spearman of 0.5705, placing us at rank 5 on the global leaderboard. This represents a 190 percent improvement over the official baseline of 0.197. The three contributions that drove this result are: Tier 2 feature engineering with leakage-free target encoding; rank-target training that directly optimizes Spearman correlation; and systematic gap diagnosis that redirected our efforts away from ineffective regularization strategies.

[PAUSE]

---

**[Slide 18 — Takeaways]**

We close with three lessons from this project.

First: match your training objective to your evaluation metric. Training with MSE when you are evaluated on Spearman leaves performance on the table.

Second: diagnose before you optimize. Understanding the root cause of the OOF-to-platform gap prevented us from pursuing the wrong direction.

Third: systematic iteration with documented baselines outperforms trial and error. Every experiment had a hypothesis, a measurement, and a conclusion.

Thank you for watching.

---

## Word Count Reference

| Section | Approx. Words |
|---------|--------------|
| Sec 1: Introduction | ~190 |
| Sec 2: EDA & Feature Eng. | ~310 |
| Sec 3: Baseline & Gap | ~240 |
| Sec 4: Rank-Target | ~310 |
| Sec 5: Experiments | ~260 |
| Sec 6: Conclusion | ~145 |
| **Total** | **~1,455** |

At 120 wpm (typical AI voice speed): ~12.1 minutes.  
At 130 wpm (slightly faster AI setting): ~11.2 minutes.
