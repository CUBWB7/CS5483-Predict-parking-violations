# Report Revision Checklist

## 说明
- 当前 `report/report.md` 是框架稿，缺少正文属于正常状态。
- 本清单的目标不是重复 review finding，而是把现有问题整理成“下一步直接可执行的修改动作”。
- 修改 `report/report.md` 时，事实口径以仓库中的实际实验记录为准，优先参考：
  - `docs/logs/progress.md`
  - `README.md`
  - `scripts/step_c_gpu.py`
  - `scripts/step_i_gpu.py`
  - `data/train_features_tier2.parquet`

---

## P0 — 先补成完整报告

- [ ] 把 `report/report.md` 中 1-7 节的大段 HTML 注释提纲改写成正式英文正文。
- [ ] 每个主章节至少包含：
  - 开头段：这一节回答什么问题
  - 主体段：方法、证据、结果或解释
  - 结尾段：这一节的 takeaway
- [ ] 优先补全这些 currently-empty sections：
  - `1. Introduction`
  - `2. Related Work`
  - `3.1 Dataset Overview`
  - `3.3 Key Data Challenges`
  - `4. Methodology`
  - `5.5 Negative Results`
  - `6. Discussion`
  - `7. Conclusion`
  - `Appendix A: Reproducibility`
  - `Appendix B: Team Contribution`
- [ ] 保留注释可以，但注释不能替代正文。
- [ ] 完成后先自查：渲染出来的报告不能只剩标题、Abstract 和图片。

---

## P1 — 统一事实与方法口径

- [x] 在全文冻结一张 canonical results table，只使用这一套核心结果：（已验证全部正确）
  - `v1: OOF 0.5880 / Platform 0.5222`
  - `v3: OOF 0.6408 / Platform 0.5620`
  - `v7: OOF 0.6429 / Platform 0.5636`
  - `Exp C: OOF 0.6464 / Platform 0.5698`
  - `Exp I-A: OOF 0.6478 / Platform 0.5705`
- [x] 将 `5-fold stratified CV` 改为与实际一致的表述：（已修正骨架注释）
  - `5-fold shuffled KFold CV (SEED = 42)`
- [x] 重写 4.1 Feature Engineering，只写最终实际进入建模的 26 特征体系。（已修正骨架注释，Tier 1: 10→19, Tier 2: 19→26）
- [ ] 方法部分应重点描述这些最终特征：
  - 原始特征
  - `log_total_count`
  - `count_bin`
  - `hour_sin/hour_cos`
  - `dow_sin/dow_cos`
  - `month_sin/month_cos`
  - `grid_te`
  - `time_period`
  - `grid_period_te`
  - `is_raining`
  - `has_snow`
  - `grid_avg_count`
  - `grid_violation_std`
  - `grid_sample_count`
- [x] 删除或改写这些与最终模型不一致的描述：（已从骨架注释中删除全部 6 个不存在特征）
  - `1/total_count` — 不存在
  - `grid_hour_te` — 不存在
  - `grid_month_te` — 不存在
  - `dow_period_te` — 不存在
  - `grid_dow_te` — 不存在
  - `weather_bin_te` — 不存在（实际是 `is_raining` + `has_snow` 二值变量）
- [x] 重写 4.3.4 Hyperparameter Optimization，区分清楚三个阶段：（已修正骨架注释）
  - `v3`: Optuna 50 trials per model
  - `Exp C`: rank-target，沿用 v7 tuned params
  - `Exp I-A`: 在 Exp C 基础上提高迭代上限到 `LGB 20000 / XGB 15000`
- [x] 如果 4.4 Ensemble 写的是最终最佳模型，则应使用 final-best 对应权重。（已修正：v1=0.30/0.70, v7=0.35/0.65, Exp I-A=0.48/0.52）
- [x] 不要把 v7、Exp C、Exp I-A 的配置混写在同一段里。（已在骨架注释中区分）

---

## P1 — 补齐结果论证与图表引用

- [ ] 把目前只存在于注释中的表格真正写成 Markdown 表格：
  - overall performance table
  - ablation table
  - GBDT vs DL comparison table
  - negative results summary table
- [ ] 所有图片第一次出现时都补一句正文解释：
  - `Figure X shows ...`
  - `This figure supports ...`
- [ ] 如果正文提到 Fig. 4 / Fig. 5 / Fig. 6，就真的插入对应图，或删掉相关文字。
- [ ] 为每张图增加简洁 caption，不要只放图片不解释。
- [ ] `5.4 Model Comparison: GBDT vs Deep Learning` 中必须统一比较口径，不混用不同实验阶段。
- [ ] 推荐统一成 final-family 或 rank-family，再全部按同一版本填数。
- [x] `TabM` 的数值要与项目日志统一。（已修正：0.4402→0.4445；ResNet 0.4215 已注明）
- [x] `5.5 Negative Results` 表述已谨慎化：（已修正骨架注释）
  - 用 `suggests` / `is consistent with`
  - Exp D 标注为 “local eval, not submitted”
  - v10 DART 标注为 “LGB OOF -0.019 (ensemble -0.002)”
- [x] 关于 OOF-platform gap，已改写为：（已修正骨架注释）
  - 广义的 train/test distribution shift 是主因
  - TE-heavy features 是重要表现之一
  - 不再写成”TE shift 已被严格证明是唯一根因”

---

## P1 — 补齐引用与可复现性

- [ ] 在正文加入 `[1]-[7]` 形式的文内引用，不要只保留文末 references。
- [ ] 至少这些位置要补 citation：
  - 问题背景
  - THESi / ChallengeData 数据来源
  - 相关工作综述
  - LightGBM / XGBoost 方法依据
  - 与论文方法对比的陈述
- [ ] leaderboard 排名加时间限定，避免写成无时间边界的永久事实。
- [ ] 推荐写法：
  - `At the time of submission on April 11, 2026, our score of 0.5705 placed 5th on the public leaderboard.`
- [ ] Appendix A 要写成真正的 reproducibility section，至少包含：
  - 仓库路径或链接
  - 环境安装方式
  - 数据下载方式
  - notebook 执行顺序
  - GPU 脚本说明
  - 随机种子 `SEED = 42`
  - `data/`、`models/`、`submissions/` 不入 Git 的说明
- [ ] Appendix A 中补充实际依赖提示：
  - `pyarrow`
  - `optuna`
  - 可选 `torch` / GPU-only requirements
- [ ] 说明本地 notebook 与 GPU server 脚本的区别，避免评审误以为全部步骤都能在一台普通机器上无条件重现。

---

## P2 — 补齐团队分工与 rubric 要素

- [ ] 把 Appendix B 改成实际团队分工表。
- [ ] 每位成员至少写清楚：
  - 负责的实验/代码模块
  - 报告撰写部分
  - PPT / video / Q&A 准备内容
  - 协作方式
- [ ] 若使用了 peer feedback，可在 Discussion 或附录中简短说明：
  - 哪些 reviewer concern 被采纳
  - 哪些表述因此变得更谨慎
- [ ] 目标是让老师能直接看出：
  - 分工清晰
  - 贡献真实
  - 团队协作存在证据

---

## 推荐修改顺序

- [ ] 第一步：先补正文，不改风格。
- [ ] 第二步：统一方法、特征、参数、分数口径。
- [ ] 第三步：补表格、图注、正文中的 figure/table references。
- [ ] 第四步：补文内引用、Appendix A、Appendix B。
- [ ] 第五步：最后统一语言风格，删除残留占位文字，检查全篇术语一致性。

---

## 最后检查

- [ ] 渲染后的报告主体不再出现“只有图片没有解释”的情况。
- [ ] 所有关键分数都能在 `README.md` 或 `docs/logs/progress.md` 中找到对应记录。
- [ ] 方法部分只描述“实际用于最终结果”的流程。
- [ ] 讨论部分避免过强因果断言。
- [ ] 报告满足课程要求的 research report 形式，而不是 project outline。
