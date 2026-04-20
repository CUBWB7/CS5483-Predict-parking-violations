# Development Progress Log

> CS5483 Project 2 — Predicting Parking Violations (ChallengeData #163)
> Group 28. Timeline: 2026-04-02 to 2026-04-23.

This log summarizes each phase with its goal, key outcome, and one pivotal decision. The narrative arc: an Optuna-tuned LightGBM + XGBoost baseline reached platform 0.5636, a series of "shrink the OOF-platform gap" experiments all failed, and a pivot to **rank-target training** then drove the platform score to 0.5705 (rank 5).

---

## Phase 0 — Environment Setup (2026-04-02)

**Goal.** Set up reproducible tooling before touching the data.

**Outcome.** Conda environment `parking` (Python 3.11; LightGBM 4.6, XGBoost 3.2, CatBoost 1.2, Optuna, SHAP); private GitHub repo; `SEED = 42` fixed project-wide; raw CSVs (~500 MB) downloaded but excluded from Git.

**Key decision.** Local development on an M4 Mac for iteration speed; GPU jobs delegated to the course group server (NVIDIA) via standalone scripts (`scripts/step_*_gpu.py`) that sync through Git.

---

## Phase 1 — EDA (2026-04-02)

**Goal.** Understand data structure and identify feature-engineering priorities.

**Outcome.** `notebooks/01_eda.ipynb` + 7 figures. Dataset: 6,076,546 training rows, 10 raw features, target `invalid_ratio ∈ [0,1]`. Key findings: target is U-shaped (15.9% = 0, 26.6% = 1); `total_count` is the strongest raw predictor (ρ = −0.298); 25.2% of samples have `total_count = 1`, where the rate can only be 0 or 1 (severe label noise); weather features are individually weak (|ρ| < 0.03); spatial data clusters tightly within a small bounding box.

**Key decision.** Our feature-engineering priority was established here: spatial target encoding > count transforms > cyclical encoding > cross features > weather augmentation — matching the ablation ordering reported by Karantaglis et al. (2022).

---

## Phase 2 — Feature Engineering (2026-04-02)

**Goal.** Expand 10 raw features to a richer representation without leakage.

**Outcome.** 26 engineered features organized into Tier-1 and Tier-2. Strongest additions: **`grid_period_te`** (spatial × time-period target encoding, ρ = +0.311) and **`grid_te`** (spatial target encoding, ρ = +0.307), both computed via 5-fold K-Fold target encoding with smoothing. Log transform of `total_count` and a 5-bin `count_bin` feature, sine/cosine encoding of hour/day/month, time-period binning, and coarse grid statistics completed the set.

**Key decision.** Target encoding was applied with **strict K-Fold out-of-fold computation** to prevent leakage. An early attempt with too coarse a grid (GRID_SIZE = 0.001) produced only 14 cells; refining to 0.00005 produced 742 cells and restored `grid_te` to its expected ρ range.

---

## Phase 3 — Baseline Modeling (2026-04-02) → v1

**Goal.** Establish a defensible ensemble baseline.

**Outcome.** LGB + XGB, 5-fold CV, 3000 rounds. OOF Spearman 0.5880; platform 0.5222 (**first submission**, already ~2.6× the official RF baseline of 0.197). Ensemble weights chosen by grid search on OOF: LGB 0.30, XGB 0.70.

**Key decision.** We switched early-stopping from Spearman to l2/RMSE — computing Spearman on 1.2M rows per round was prohibitively slow. Later experiments confirmed that l2 tracks Spearman closely on this task, so we never had to revisit this choice.

---

## Phase 4 — Evaluation & Interpretability (2026-04-03)

**Goal.** Understand where the model succeeds and fails.

**Outcome.** Ablation study (26 → 10 features degrades OOF by 0.0136); SHAP analysis showing `total_count`, `grid_period_te`, and spatial coordinates as top drivers; per-group error analysis revealing the `count_bin = 0` bucket (`total_count = 1`, 25% of data) has Spearman only 0.41 vs. 0.73 for `count_bin = 4` — i.e., **label noise is the dominant bottleneck on a quarter of the training set**.

**Key decision.** Rather than trying to "fix" the noisy bucket with heuristics, we committed to addressing it through principled sample weighting (later Phase 5a Step 10).

---

## Phase 5a — Hyperparameter Tuning and Sample Weighting (2026-04-03 to 04-06) → v7

**Goal.** Extract all achievable signal from the v1 baseline via tuning and weighting.

**Outcome.** Three improvements compounded:
- **v2 (more rounds, lower LR):** 8000 rounds, LR 0.03 → ensemble OOF 0.6012, platform 0.5338.
- **v3 (Optuna tuning, 50 trials/model):** OOF jumped to 0.6408, platform 0.5620. This was the single largest gain — larger than expected.
- **v4 (Optuna'd CatBoost as a 3rd model on GPU):** CB OOF 0.6175, but ensemble weight = 0. The LGB–XGB–CB correlation structure left CatBoost with no margin to contribute.
- **v7 (log1p sample weighting):** adding `sample_weight = log1p(total_count)` reduced the influence of noisy `total_count = 1` rows. Ensemble OOF 0.6429, **platform 0.5636** — the best we would achieve under raw-target training.

**Key decision.** Freezing v7 as the baseline. We recognized the OOF-platform gap (~0.079) was persistent and would require a conceptually different strategy, not more tuning.

---

## Phase 5b — Gap-Reduction Hypotheses (Steps 8–14, 2026-04-06 to 04-07)

**Goal.** Shrink the 0.079 OOF-platform gap by testing distinct hypotheses about its cause.

**Outcome.** Seven experiments, all falsifying their hypotheses:

| Step | Hypothesis | Result |
|---|---|---|
| 8 | Spearman-based early stopping improves generalization | Subsampled Spearman is too noisy; LGB stopped far too early (best_iter ≈ 2000) |
| 8b | Over-training drives the gap (hard cap at 6000 rounds) | Platform changed by only −0.0002 — over-training was not the cause |
| 9 | Over-fitting drives the gap (stronger regularization via constrained Optuna) | OOF dropped 0.010; platform 0.5477, well below v7 |
| 11a | Test set is M1–5 only, so test-time TE should use M1–5 statistics | Platform 0.5507 (−0.013). Full-data TE was actually closer to the test distribution |
| 11b | Training only on M1–5 reduces mismatch | M1–5 OOF dropped 0.006; losing 60% of training data hurt more than it helped |
| 13 | DART boosting adds LGB-XGB diversity | Diversity goal met (correlation 0.968 → 0.948), but LGB OOF dropped 0.019 — ensemble got worse |
| 14 | A neural network provides orthogonal signal | NN OOF only 0.42; ensemble weight-search assigned it 0 weight |

Additionally, **adversarial validation** (Exp D) achieved AUC ≈ 1.0 distinguishing train from test — proof that the distribution shift is large. Temporal CV (M1–4 → M5) gave 0.6017, roughly halfway between random-CV OOF (0.6429) and platform (0.5636). AV-weighted retraining failed because AUC ≈ 1.0 forced the weights to near-binary, effectively discarding most training data.

**Key decision.** Each negative result ruled out a different explanation of the gap. By the end of Phase 5b, we had systematically eliminated over-training, over-fitting, TE distribution shift, and lack of architectural diversity. What remained was a hypothesis about the **training objective itself** — which led directly to Phase 5c.

---

## Phase 5c — Sprint: Rank-Target Training (2026-04-09 to 04-11) → Exp I-A

**Goal.** Align the training objective with the evaluation metric.

**Outcome.** Spearman is rank-based, but we had been training on raw `invalid_ratio`, whose distribution is dominated by the two spikes at 0 and 1. Exp C replaced the target with `y_rank = rankdata(y) / N`, training LGB + XGB with otherwise-identical v7 hyperparameters. OOF jumped to 0.6464; **platform reached 0.5698** — the first time any experiment cleared v7.

Exp I-A refined Exp C: LGB rounds increased from 10,000 to 20,000 (LGB was still improving at the cap), with a matching fine-grained ensemble-weight search. Final ensemble OOF 0.6478, **platform 0.5705 — rank 5 on the leaderboard**.

Exp I-B re-ran Optuna specifically for rank-target training; the new parameters underperformed v7's parameters because the lower learning rates meant the 15,000–20,000 round budget no longer sufficed for convergence. I-A won.

**Key decision.** Of the seven Phase-5b failures, the most consequential was realizing that none of them addressed the objective–metric mismatch. Rank-target training turns out to be the single most impactful change in the whole project (+0.0069 on platform, from 0.5636 to 0.5705) and is our main methodological contribution.

---

## Phase 6 — Report (2026-04-17 onward)

**Goal.** A ~15-page research report by 2026-04-23.

**Outcome.** Draft written; revisions in progress by a teammate. Final version submitted via the course platform. Structure: Abstract + 7 sections (Introduction, Related Work, Data, Methodology, Results, Discussion, Conclusion) + Appendix (reproducibility). All seven peer-review feedback items incorporated.

---

## Phase 7 — Presentation Video (2026-04-06 to 04-13)

**Goal.** 15-minute video presentation uploaded to Canvas.

**Outcome.** 24-slide Slidev deck + ~1,450-word voiceover script rendered via Edge-TTS + FFmpeg automation. Final `final_v2.mp4` (~14:57) uploaded 2026-04-13.

**Key decision.** Chose Slidev over PowerPoint for version-control friendliness, LaTeX math, and programmatic rendering. Hit a layout bug in the `slidev-theme-neversink` theme (`top-title` slot uses `h-full`, sinking content) and root-caused it by switching all content slides to `top-title-two-cols` with an explicit `columns` prop.

---

## Final Scorecard

| Version | Change | Ensemble OOF | Platform | Note |
|---|---|---|---|---|
| Official RF baseline | — | — | 0.197 | reference |
| v1 | LGB + XGB, 3000 rounds | 0.5880 | 0.5222 | first submission |
| v2 | 8000 rounds, LR 0.03 | 0.6012 | 0.5338 | |
| v3 | Optuna tuning | 0.6408 | 0.5620 | largest single gain |
| v7 | + log1p sample weighting | 0.6429 | 0.5636 | best raw-target |
| Exp C | Rank-target training | 0.6464 | 0.5698 | new best |
| **Exp I-A** | **LGB rounds 10K → 20K** | **0.6478** | **0.5705** | **final, rank 5** |

Total improvement over the official baseline: **2.9× Spearman** (0.197 → 0.5705).
