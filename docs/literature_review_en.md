# Literature Review

> CS5483 Project 2 — ChallengeData #163 "Predict Parking Violations"
> Group 28

Seven papers shaped this project. They fall into three groups: (A) prior work on the same THESi data source, (B) spatiotemporal parking-violation studies in other cities, and (C) the gradient-boosting methods we used.

---

## Part A — Prior Work on THESi Data

### Paper 1: Vo (2025), "Deep Learning for On-Street Parking Violation Prediction"

*arXiv:2505.06818.* Predicts per-zone violation rates from the THESi system in Thessaloniki — almost certainly the same data source as ChallengeData #163. Uses a 6-layer residual MLP (512→256→128→64→128→32→1) trained with Adamax on 28 engineered features. Achieves MAE 0.146, a 41.8% reduction over the mean predictor.

**Relevance to our project.** Three ideas informed our design: (i) **sine/cosine encoding** of cyclical features (hour, day-of-week, month) so that adjacent values stay adjacent — we use the same encoding in our Tier-1 feature set; (ii) **6-hour rolling-window weather** captures delayed effects of weather on behavior — we considered this but deprioritized because our OOF gain projections were small; (iii) **Gaussian label smoothing** to stabilize noisy observations — we did not use it directly, but it motivated our `log1p(total_count)` sample-weighting strategy (down-weighting the 25% of samples with `total_count=1` that can only produce violation rates of 0 or 1).

### Paper 2: Karantaglis et al. (2022), "Predicting On-Street Parking Violation Rate Using Deep Residual Neural Networks"

*Pattern Recognition Letters, vol. 163.* A companion paper to Vo (2025) with a more rigorous ablation study. Testing feature groups one at a time, they report:

| Feature group added | MAE | Relative improvement |
|---|---|---|
| PoI distances only | 0.211 | baseline |
| + zone capacity | 0.206 | 2.4% |
| + time features | **0.185** | **10.2%** (largest jump) |
| + weather | 0.176 | 4.9% |
| + holiday indicators | 0.173 | 1.7% |
| + COVID indicators | 0.169 | 2.3% |

**Relevance to our project.** This paper is the source of two decisions: (i) our **feature engineering priority order** (temporal > weather > binary indicators) mirrors their contribution ordering, and (ii) we adopt their **incremental ablation methodology** in §5 of our report to quantify each feature group's Spearman contribution.

---

## Part B — Spatiotemporal Analysis in Other Cities

### Paper 3: Gao et al. (2019), "Predicting the Spatiotemporal Legality of On-Street Parking"

*Annals of GIS.* Compares six ML models across four spatial scales (point, street, census block, 1 km grid) on 10.8 million NYC parking citations. **Random Forest wins at every scale** (accuracy 0.82–0.88), and finer spatial granularity always beats coarser. POI features (retail, dining, healthcare) are the strongest predictors; 3-hour time lags show measurable autocorrelation.

**Relevance to our project.** Confirmed that spatial discretization matters — we built `grid_te` via K-Fold Target Encoding on discretized lat/long cells, which became our second-strongest feature (ρ = 0.307). RF's strength here also validated the competition's RF baseline (ρ = 0.197) as a reasonable reference rather than a weak strawman.

### Paper 4: Liu & Chen (2025), "Short-term Parking Violations Demand Dynamic Prediction"

*Transportation (Springer).* Hybrid architecture (MGTWM + GAT + ALSTM) on Chongqing data. Finds commute peaks at 08:00–09:00 and lunch peaks at 12:00–13:00, strong seasonal differences (winter vs. summer), and extensive spatiotemporal heterogeneity — the same feature affects violation rates differently at different locations and times.

**Relevance to our project.** Their commute/lunch peak finding inspired our `time_period` feature (6 bins covering morning peak, mid-morning, lunch, afternoon, evening peak). The spatiotemporal-heterogeneity framing also motivated `grid_period_te` — a target encoding over (grid × time-period) interactions, which became our **strongest single feature** (ρ = 0.311).

### Paper 5: Sui et al. (2025), "Spatio-temporal Heterogeneity in Street Illegal Parking"

*Journal of Transport Geography.* Bayesian hierarchical model (BYM + zero-inflated Poisson + INLA) on 18.7M NYC citations at 500m grid resolution. Two findings stood out: **humidity dominates among weather features** (more predictive than temperature), and SHAP interaction analysis reveals complex nonlinear effects (e.g., crime rate × population density).

**Relevance to our project.** Two choices trace back to this paper: (i) we adopt **500m-class grid discretization** as our spatial unit, and (ii) we use **SHAP analysis** in §5 of the report for interpretability. Our weather features are individually weak (|ρ| < 0.03), consistent with their observation that raw weather matters less than interactions.

---

## Part C — Gradient Boosting Methods

### Paper 6: Ke et al. (2017), "LightGBM"

*NeurIPS 2017.* Introduces GOSS (gradient-based one-sided sampling) and EFB (exclusive feature bundling), combined with a leaf-wise growth strategy. Achieves 6–20× speedup over XGBoost with comparable accuracy on large datasets.

**Relevance to our project.** LightGBM is our primary model. The dataset size (6.07M rows) makes GOSS particularly appropriate, and histogram-based training keeps 5-fold CV tractable on an M4 Mac. Our baseline LGB hyperparameters follow the paper's recommended starting ranges before Optuna refinement.

### Paper 7: Chen & Guestrin (2016), "XGBoost"

*KDD 2016.* Explicit L1+L2 regularization with second-order Taylor expansion of the loss, level-wise growth, sparse-aware split finding.

**Relevance to our project.** XGBoost is our second model. The complementarity between XGBoost (level-wise, explicit regularization) and LightGBM (leaf-wise, implicit regularization via GOSS/EFB) gives the ensemble its limited-but-real diversity (LGB–XGB OOF correlation ≈ 0.965). In the final Exp I-A result, the ensemble OOF (0.6478) exceeds either model individually (LGB 0.6417, XGB 0.6430).

---

## Summary: Where Each Paper Landed in Our Pipeline

| Design choice | Paper |
|---|---|
| Sine/cosine time encoding | 1 |
| Ablation-study format in §5 of report | 2 |
| Feature-engineering priority: time > space > weather | 2 |
| Spatial target encoding (grid cells) | 3, 5 |
| Time-period binning around commute/lunch peaks | 4 |
| Grid × time-period interaction feature | 4, 5 |
| SHAP analysis in §5 of report | 5 |
| LightGBM as primary model | 6 |
| XGBoost as diversifying second model | 7 |

Our novel contribution — rank-target training (§4.3 of the report) — has no direct precedent in this literature and is our main methodological addition.
