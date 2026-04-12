# Q&A Preparation: Peer Review & Class Challenges

Based on the submission rubric ("Can the team challenge other teams successfully for the group presentation?"), this document prepares answers for likely questions from classmates and the instructor.

---

## Category 1: Problem & Metric

**Q1: Why use Spearman correlation instead of RMSE for this problem?**

Spearman correlation is the official evaluation metric set by the challenge organizers (Egis). Beyond that, Spearman is a natural choice for this problem because stakeholders likely care about *relative risk* — which areas have higher violation rates than others — rather than the exact fractional value. Spearman directly measures whether the model correctly identifies high-violation vs. low-violation zones.

---

**Q2: Your Spearman is 0.5705. That is far from 1.0. Why is the score not higher?**

Three factors limit the score. First, the dataset has inherent label noise: 25 percent of observations have `total_count = 1`, meaning the violation ratio is either 0 or 1 with no intermediate information. Second, there is a temporal distribution shift between training (months 1–11) and test (month 12 or later), as confirmed by our adversarial validation AUC of 0.9999. Third, we have only 10 input features with limited temporal and spatial granularity. Given these constraints, 0.5705 is competitive — it ranks 5th globally on the leaderboard.

---

## Category 2: Feature Engineering

**Q3: Why did you use K-Fold Target Encoding instead of a simpler mean encoding?**

Standard mean encoding (computing the mean violation rate for each category on the full training set) introduces data leakage: the model sees information about the target when computing the feature, making OOF evaluation unreliable. K-Fold Target Encoding computes the encoding on held-out folds only, preventing the model from seeing its own label. This is standard practice and is documented in the Kaggle literature on TE.

---

**Q4: Did you try adding more external features, such as Points of Interest or road type data?**

We did not add external data. The challenge provides only the 10 original features, and sourcing reliable POI or road data for Thessaloniki within the project timeline was not feasible. This is a noted limitation. Adding spatial context features (e.g., proximity to commercial zones) would likely improve the model further.

---

**Q5: Your weather features (precipitation, temperature) have near-zero SHAP values. Why keep them?**

Although their mean SHAP values are low, they may still contribute small, non-linear interactions that benefit the ensemble slightly. Removing them would require an ablation study showing a consistent score improvement. Our ablation study (shown in the figures) confirmed that the geographic and count-based features carry the most signal, but removing weather features did not produce a measurable improvement in our experiments.

---

## Category 3: Model & Training

**Q6: Why LightGBM and XGBoost? Why not just use one of them?**

Both models are gradient boosting frameworks, but they differ in their split-finding algorithms, regularization defaults, and handling of missing values. LightGBM uses leaf-wise growth with histogram binning; XGBoost uses level-wise growth. Empirically, their predictions are partially diverse — combining them reduces variance without sacrificing much bias. Our OOF sweep confirmed a weighted ensemble (LGB=0.39, XGB=0.61) outperforms either model alone.

---

**Q7: How did you set the ensemble weights (LGB 0.39, XGB 0.61)?**

We performed a grid search over the ensemble weight parameter α, where the final prediction equals α × LGB_pred + (1 − α) × XGB_pred, with α ranging from 0.30 to 0.70 in steps of 0.01. We selected the weight that maximized out-of-fold Spearman. The OOF evaluation is computed on held-out fold predictions that were not used during training, so this search is statistically valid.

---

**Q8: You said CatBoost weight converged to zero. Can you explain why?**

In our ensemble weight optimization, when we included CatBoost predictions alongside LightGBM and XGBoost, the optimizer consistently assigned CatBoost a weight near zero. This suggests that CatBoost's predictions are either highly correlated with LGB/XGB (offering no diversity) or slightly less accurate on this dataset. Rather than forcing it in, we respected the data-driven signal and removed it.

---

## Category 4: Rank-Target Innovation

**Q9: Did rank-target training change the model's inference procedure as well?**

No. At inference time, the model predicts a value in the rank space (between 0 and 1), which preserves the relative ordering of predictions. Since Spearman only measures rank correlation, the absolute scale of predictions does not matter. The final predictions are submitted as-is; no inverse transformation is needed.

---

**Q10: Is rank-target training a published technique, or did you develop it yourself?**

This is a well-known technique in the machine learning community, especially for competition settings where the evaluation metric is rank-based (e.g., NDCG, Spearman). It is related to the broader category of "learning to rank" methods. We applied it based on first principles: since Spearman evaluates ranks, training on ranks directly aligns the objective. We are not aware of a specific paper that first proposed this exact application, but it is a standard technique in competitive data science practice.

---

**Q11: Why did the score improve further in Exp I-A when you just increased iterations?**

LightGBM's early stopping, when evaluated on Spearman, can terminate prematurely because Spearman is a noisy metric on moderate-sized validation folds. By increasing the hard iteration cap from 10,000 to 20,000, we allowed the model to continue learning past the early stopping point without overfitting. The fact that LightGBM's best iteration was approximately 19,990 out of 20,000 confirms it was still improving — the original cap was the binding constraint.

---

## Category 5: Gap & Generalization

**Q12: Your OOF is 0.6478 but platform is 0.5705. The gap is 0.077. Is that not concerning?**

It is a known limitation. We diagnosed it thoroughly using adversarial validation (AUC = 0.9999) and temporal cross-validation. The gap reflects a genuine distribution shift between training and test data — the model was trained on months 1 through 11 and evaluated on a different time period. This is a structural challenge that no feature engineering or regularization fully resolves. Our rank-target training actually slightly reduced the gap (from 0.079 to 0.077), and the gap remained stable across most experiments, which rules out overfitting.

---

**Q13: Did you try pseudo-labeling to leverage the test data?**

Yes — this was Experiment G. We used the model predictions on the test set as soft pseudo-labels, iteratively retraining the model on the combined dataset. However, the rank-target transformation compresses predictions into a range of approximately [0.079, 0.867], which is incompatible with standard confidence thresholds (e.g., <0.10 or >0.90). As a result, fewer than 200 test samples met the threshold, and the model showed no meaningful improvement (OOF change of −0.0001). The approach was abandoned.

---

## Category 6: Evaluation & Reproducibility

**Q14: How do we know your results are reproducible?**

All code is in `notebooks/06_sprint.ipynb`. Each section is self-contained: the first cell checks whether required variables are already loaded in memory; if not, it reloads from disk. All random seeds are fixed to 42. The notebook is designed to be fully reproducible with a single Restart and Run All operation, given the input data files.

---

**Q15: What would you do differently if you had more time?**

Three directions: First, we would explore stacking — training a meta-model on the OOF predictions of multiple base models. Second, we would investigate richer spatial features such as POI density or proximity to commercial districts, which might reduce the train-test distribution gap. Third, we would explore test-time augmentation by generating multiple perturbed test predictions and averaging their ranks, which can stabilize Spearman on noisy test distributions.
