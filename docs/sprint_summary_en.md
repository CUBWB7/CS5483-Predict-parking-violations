# Sprint Summary: From a Tuned Baseline to Rank 5

> CS5483 Project 2 — ChallengeData #163 "Predict Parking Violations"
> Group 28. Final platform Spearman: **0.5705** (rank 5 on the leaderboard).

This is a focused summary of how we improved the platform score from 0.5222 (first submission) to 0.5705 (final). The key insight is that **the decisive gain did not come from more tuning — it came from recognizing and fixing an objective-metric mismatch**.

---

## The Problem

ChallengeData #163 evaluates submissions by **Spearman rank correlation**. The training target is a continuous rate `invalid_ratio ∈ [0, 1]`. Standard regression (MSE on raw `invalid_ratio`) optimizes *value* accuracy, but the metric only rewards *ranking* accuracy. This gap between training objective and evaluation metric is subtle enough that we did not recognize it for most of the project.

---

## Chapter 1 — Steady Gains from Tuning (v1 → v7, +0.041 on platform)

Standard techniques compounded into a respectable baseline:

| Version | Change | Platform | Delta |
|---|---|---|---|
| v1 | LGB + XGB, 3000 rounds | 0.5222 | (vs. RF baseline 0.197) |
| v2 | 8000 rounds, LR 0.03 | 0.5338 | +0.012 |
| v3 | Optuna tuning (50 trials/model) | 0.5620 | **+0.028** |
| v7 | + `log1p(total_count)` sample weighting | 0.5636 | +0.002 |

After v7, further tuning yielded diminishing returns. The OOF-platform gap (0.079) remained stubbornly constant, suggesting the problem was not under-tuned models but something structural.

---

## Chapter 2 — Seven Hypotheses, Seven Negative Results

To close the OOF-platform gap we tested distinct hypotheses. Each produced a negative result, but collectively they eliminated large regions of the hypothesis space:

| Experiment | Hypothesis | Why it failed |
|---|---|---|
| Step 8 (Spearman ES) | Spearman-based early stopping helps | Subsampled Spearman is too noisy to detect real improvements |
| Step 8b (round cap) | Over-training drives the gap | Platform changed by −0.0002 |
| Step 9 (constrained Optuna) | Over-fitting drives the gap | Stronger regularization dropped platform to 0.5477 |
| Step 11a (M1–5 TE on test) | Test set is M1–5; TE should match | Platform 0.5507 — full-data TE was actually better |
| Step 11b (M1–5 training) | Training only on M1–5 reduces mismatch | Losing 60% of data hurt more than it helped |
| Step 13 (DART) | LGB-XGB correlation 0.968 limits ensemble | LGB OOF dropped 0.019 — diversity gain did not compensate |
| Step 14 (neural network) | Need architectural diversity | NN OOF only 0.42; ensemble weight-search assigned it 0 |

Additional diagnostics: an adversarial-validation classifier reached AUC ≈ 1.0 distinguishing train from test, and a temporal CV split (M1–4 → M5) gave 0.6017 — roughly halfway between random-CV OOF (0.6429) and platform (0.5636). AV-weighted training failed because the near-binary AV probabilities collapsed to discarding most samples.

**What we learned.** The gap was neither an over-training problem, nor an over-fitting problem, nor a TE-encoding problem, nor an architecture-diversity problem. What remained was the training objective.

---

## Chapter 3 — Rank-Target Training (+0.0069 on platform)

With the same hyperparameters as v7, we replaced the regression target:

```
y_rank = rankdata(y) / N    # uniform distribution in [0, 1]
```

MSE against `y_rank` penalizes ranking errors directly, aligning training with Spearman. Implementation is a three-line change; accuracy gain was immediate.

| Experiment | Change | OOF | Platform |
|---|---|---|---|
| v7 (raw target) | baseline | 0.6429 | 0.5636 |
| Exp C | rank target, 10K rounds | 0.6464 | 0.5698 |
| **Exp I-A** | **rank target, LGB 20K rounds** | **0.6478** | **0.5705** |

Exp I-A increased the LGB round budget from 10K to 20K (LGB was still improving at the cap in Exp C) and added a fine-grained ensemble-weight search on the M1–5 OOF subset. A parallel attempt (Exp I-B) re-ran Optuna specifically for rank-target training; the new parameters underperformed v7's because lower learning rates made 20K rounds insufficient for convergence.

**Why this worked.** The training loss now rewards the model for getting the *order* right, even at the cost of value accuracy. This matters most in the noisy `total_count = 1` bucket (25% of training data), where raw values collapse to 0 or 1 but rank-encoded targets preserve finer gradations.

---

## Final Result

- **Platform Spearman: 0.5705**, rank 5 on the ChallengeData #163 leaderboard
- **2.9×** improvement over the official Random Forest baseline (0.197)
- OOF-platform gap: 0.6478 − 0.5705 = **0.077** (unchanged — consistent with our earlier conclusion that the gap is a feature of the data, not of the model)

---

## Key Takeaways

1. **Align the training objective with the evaluation metric.** For rank-based metrics, train on ranks. This is the single most impactful decision in the project.
2. **Negative results are load-bearing.** Every gap-reduction hypothesis we falsified narrowed the remaining search space. Without the Phase 5b failures, we would not have thought to question the training objective.
3. **Disciplined evaluation matters more than model-zoo breadth.** We beat a neural network, CatBoost, and DART boosting with "just" LGB + XGB — because what mattered was the training target, not the learning algorithm.
4. **Systematic experimentation beats intuition.** We used adversarial validation, temporal CV, and per-group error analysis to diagnose where signal and noise lived. These diagnostics told us *what not to try* as much as what to try.
