# 项目进度日志

> 按 Phase 组织。每个 phase 记录：日期、目标、产出、关键发现和决策。

---

## Phase 0 — 环境搭建 ✅

**日期**: 2026-04-02  
**状态**: 已完成

### 产出
- [x] GitHub 私有仓库: https://github.com/CUBWB7/CS5483-Predict-parking-violations.git
- [x] Conda 环境 `parking`（Python 3.11, LightGBM, XGBoost, CatBoost 等）
- [x] 项目结构: .gitignore, README.md, CLAUDE.md, environment.yml
- [x] 数据下载: 训练集 607 万行 / 316MB，测试集 203 万行 / 105MB

### 论文研究
- 下载并阅读 7 篇相关论文（3 篇免费 + 4 篇付费）
- **核心发现**: Paper 1 & 2（Vo 2025, Karantaglis 2022）使用的 THESi 数据极可能与 ChallengeData #163 同源
- 提取的可直接应用的技术：
  - Sine 周期编码（Paper 1）
  - 6 小时天气窗口平均（Paper 1）
  - 消融实验方法论（Paper 2）：时间特征贡献 10.2% > 天气 4.9%
  - 500m 网格空间离散化（Paper 5）
  - SHAP 交互分析（Paper 5）
  - LightGBM 起步参数配置（Paper 6）

### 文档产出
- `docs/project_plan.md` — 项目实施方案（含时间线和论文指导）
- `docs/literature_review.md` — 7 篇论文的详细综述
- `docs/plan/detailed_plan.md` — 870 行操作指南
- `research_parking_violations/tutorial_part{1-4}*.md` — 各阶段教程

### 关键决策

| 决策 | 理由 |
|------|------|
| 数据文件不入 Git | CSV 单文件 >100MB，超 GitHub 限制 |
| 本地开发（M4 Mac） | 16GB RAM 足够跑 LightGBM；方便与 Claude Code 交互 |
| LightGBM 作为首选模型 | 直方图算法 + GOSS 采样适合 600 万行 |
| Spearman 作为早停指标 | 与比赛评估指标一致 |
| K-Fold Target Encoding | 防止区域统计特征的数据泄露 |
| Phase 2 引入 Tier 1/2/3 分层 | 降低特征工程超时风险 |

---

## Phase 1 — EDA 探索性数据分析 ✅

**日期**: 2026-04-02  
**状态**: 已完成

### 产出
- [x] `notebooks/01_eda.ipynb` — 完整 EDA notebook（已验证 Restart & Run All）
- [x] 7 张图表保存至 `docs/figures/fig{1-7}_*.png`

### 图表清单

| # | 图表 | 用途 |
|---|------|------|
| 1 | 目标变量分布（U 型 + total_count 分组对比） | 展示小样本二元噪声 |
| 2 | total_count vs 违规率（箱线图 + 柱状图） | 最强预测因子分析 |
| 3 | 特征分布（2×5 子图） | 10 个原始特征概览 |
| 4 | Spearman 相关性（热力图 + 条形图） | 特征重要性排序 |
| 5 | 空间违规率（散点图 + hexbin） | 区域聚类证据 |
| 6 | 时间模式（hour / dow / month） | 周期性模式 |
| 7 | 缺失值柱状图 | 数据质量评估 |

### 关键发现

| 发现 | 数值 |
|------|------|
| 训练集大小 | 6,076,546 行 |
| invalid_ratio = 0 | 15.90% |
| invalid_ratio = 1 | 26.62% |
| total_count = 1 占比 | 25.22%（二元噪声来源） |
| 最强特征 | total_count（ρ = −0.298） |
| 第二强特征 | month_of_year（ρ = −0.091） |
| 天气特征 | 均 ρ < 0.03（单独作用弱） |
| 缺失值 | HauteurNeige 2.70%，ForceVent 0.10% |
| dtype 优化后内存 | 203 MB（原始约 2.4 GB） |
| 坐标离群点 | 少量点在 (0, 0) 附近；99.9%+ 数据在 [0.998, 1.000] × [0.995, 1.000] |

### 对特征工程的启示
1. **区域 Target Encoding 是最高优先级** — hexbin 图清晰展示了违规率的空间聚类
2. **total_count 变换必不可少** — log 和分箱来利用最强预测因子
3. **时间特征需要周期编码** — hour/dow/month 都有周期性模式
4. **天气特征单独很弱** — 可能通过交互或窗口平均增强
5. **total_count=1 的样本噪声极大** — 可能需要特殊处理（占 25% 数据）

---

## Phase 2 — 特征工程 ✅

**日期**: 2026-04-02  
**状态**: 已完成（Tier 1 + Tier 2）

### 产出
- [x] `notebooks/02_feature_engineering.ipynb` — 完整特征工程 notebook（已验证 Restart & Run All）
- [x] `data/train_features_tier1.parquet` / `data/test_features_tier1.parquet` — Tier 1 检查点
- [x] `data/train_features_tier2.parquet` / `data/test_features_tier2.parquet` — Tier 2 检查点
- [x] `data/encoding_maps_tier1.pkl` / `data/encoding_maps_tier2.pkl` — 编码映射

### 特征清单（26 个特征）

| Tier | 特征 | 类型 | Spearman ρ | 备注 |
|------|------|------|-----------|------|
| 原始 | total_count | int32 | -0.295 | 最强原始预测因子 |
| T1 | log_total_count | float32 | -0.295 | 对数变换 |
| T1 | count_bin | int8 | -0.289 | 分箱 (0,1,2,3,4) |
| T1 | grid_te | float32 | **+0.307** | ⭐ 区域 K-Fold Target Encoding |
| T1 | hour_sin/cos | float32 | ~0.01-0.04 | 周期编码 |
| T1 | dow_sin/cos | float32 | ~0.002-0.005 | 周期编码 |
| T1 | month_sin/cos | float32 | ~0.03-0.06 | 周期编码 |
| T2 | grid_period_te | float32 | **+0.311** | ⭐ 区域×时段交叉 TE |
| T2 | time_period | int8 | 0.010 | 时段分箱 (0-5) |
| T2 | is_raining | int8 | 0.001 | 天气二元 |
| T2 | has_snow | int8 | 0.000 | 天气二元 |
| T2 | grid_avg_count | float32 | -0.169 | 区域平均检查数 |
| T2 | grid_sample_count | float32 | -0.158 | 区域样本数 |
| T2 | grid_violation_std | float32 | +0.064 | 区域违规率波动 |

### 关键发现与决策

| 发现 / 决策 | 详情 |
|-------------|------|
| GRID_SIZE 调整 | 0.001→0.00005；原值只产生14个网格（坐标跨度仅~0.002-0.005），调整后产生742个网格 |
| grid_te 验证通过 | ρ=0.307，在预期范围 [0.1, 0.7] 内，无数据泄露 |
| grid_period_te 最强特征 | ρ=0.311，略优于 grid_te，说明区域×时段交叉有增益 |
| 天气二元特征极弱 | is_raining / has_snow 的 ρ < 0.001，但保留给模型判断 |
| grid_id 乘数调整 | 0.00005 grid_size 需要更大的乘数（100000 而非 10000）避免 ID 碰撞 |
| pyarrow 安装 | 环境缺少 pyarrow，已通过 pip 安装 |
| pandas dtype 兼容性 | K-Fold TE 函数使用 numpy 数组避免 float32/64 类型冲突 |

### 未完成项（Tier 3，可选）
- [ ] 区域 × 星期交叉编码
- [ ] 温度离散化
- [ ] KMeans 空间聚类
- [ ] 6 小时天气窗口平均

---

## Phase 3 — 建模 ✅

**日期**: 2026-04-02  
**状态**: 已完成

### 产出
- [x] `notebooks/03_modeling.ipynb` — 建模 notebook（已执行，含输出）
- [x] `models/lgbm_fold{0-4}.txt` — 5 个 LightGBM fold 模型
- [x] `models/xgb_fold{0-4}.json` — 5 个 XGBoost fold 模型
- [x] `models/lgb_oof_preds.npy` / `lgb_test_preds.npy` — LightGBM OOF & 测试预测
- [x] `models/xgb_oof_preds.npy` / `xgb_test_preds.npy` — XGBoost OOF & 测试预测
- [x] `submissions/lgbm_v1.csv` — LightGBM 提交文件
- [x] `submissions/xgb_v1.csv` — XGBoost 提交文件
- [x] `submissions/ensemble_v1.csv` — 集成提交文件
- [x] `figures/lgbm_feature_importance.png` — 特征重要性图

### 模型结果

| 模型 | OOF Spearman | 平台分数 | 备注 |
|------|-------------|---------|------|
| RF Baseline (10 trees, 500K) | 0.937* | — | *训练集评估，非 OOF |
| LightGBM 5-Fold CV | 0.5815 | 0.5182 | 26 特征，3000 轮 |
| **XGBoost 5-Fold CV** | **0.5870** | 待提交 | 26 特征，3000 轮 |
| **Ensemble (LGB=0.30)** | **0.5880** | **0.5223** | 加权平均 |

### LightGBM 详细结果
- Fold scores: 0.5828, 0.5816, 0.5812, 0.5812, 0.5807（极稳定，std=0.0008）
- 早停指标: l2（自定义 Spearman eval 太慢，改用 l2 后每 fold ~3 分钟）
- 所有 fold 跑到 3000 轮上限（早停未触发）

### XGBoost 详细结果
- Fold scores: 0.5879, 0.5876, 0.5866, 0.5862, 0.5866（极稳定，std=0.0006）
- 参数: max_depth=6, tree_method=hist, 3000 轮
- RMSE 从 0.365 下降到 0.292（仍在下降，可增加轮数）
- 每 fold ~5 分钟

### 集成结果
- 最优权重: LGB=0.30, XGB=0.70（XGBoost 贡献更大）
- 集成 OOF Spearman 0.5880（比单模型微幅提升 +0.0010）
- 测试预测范围 [0.041, 1.000]，均值 0.536

### 平台提交
- LightGBM: CV 0.5815 → 平台 0.5182（差距 0.063）
- Ensemble: CV 0.5880 → 平台 0.5223（差距 0.066）
- 集成比单 LGB 平台分提升 +0.004
- CV-平台差距稳定在 ~0.06，可能原因: Target Encoding 训练/测试分布差异 + 模型略过拟合

### 关键决策

| 决策 | 理由 |
|------|------|
| l2/rmse 替代 Spearman 做早停 | 自定义 Spearman eval 在 120 万行上每轮排序太慢 |
| 26 个 Tier 2 特征 | 包含 grid_te、grid_period_te 等区域编码特征 |
| 5-Fold CV + OOF | 无偏评估 + 为集成提供 OOF 预测 |
| 网格搜索最优集成权重 | LGB=0.30 最优，XGBoost 表现更好 |

---

## Phase 4 — 评估分析 ✅

**日期**: 2026-04-03  
**状态**: 已完成

### 产出
- [x] `notebooks/04_evaluation.ipynb` — 完整评估 notebook（26 cells，已验证 Restart & Run All）
- [x] 11 张图表保存至 `figures/`

### 图表清单

| # | 图表 | 文件名 | 用途 |
|---|------|--------|------|
| 1 | 模型对比柱状图 | `model_comparison.png` | 报告 Ch.5 + 视频 |
| 2 | 消融实验柱状图 | `ablation_study.png` | 报告 Ch.5 + 视频 |
| 3 | 分组 Spearman 柱状图 | `grouped_spearman.png` | 报告 Discussion |
| 4 | 预测分布 + 校准曲线 | `prediction_distribution.png` | 报告 Ch.5 |
| 5 | 预测 vs 真实 hexbin 散点图 | `pred_vs_actual.png` | 报告 Ch.5 |
| 6 | TE 分布偏移对比 | `te_distribution_shift.png` | 报告 Discussion |
| 7 | SHAP Summary (beeswarm) | `shap_summary.png` | 报告 Ch.5 |
| 8 | SHAP Bar Plot | `shap_bar.png` | 视频 |
| 9 | SHAP Dep: grid_period_te | `shap_dep_grid_period_te.png` | 报告 Ch.5 |
| 10 | SHAP Dep: grid_te | `shap_dep_grid_te.png` | 报告 Ch.5 |
| 11 | SHAP Dep: total_count | `shap_dep_total_count.png` | 报告 Ch.5 |

### 消融实验结果（3000 rounds, 5-Fold LightGBM）

| 特征组 | Spearman | 增量 |
|--------|---------|------|
| Baseline (10 orig) | 0.5679 | — |
| + count transforms | 0.5712 | +0.0033 |
| + periodic encoding | 0.5677 | -0.0035 |
| + spatial TE | 0.5746 | +0.0069 |
| + cross TE | 0.5771 | +0.0025 |
| Full (26 features) | 0.5815 | +0.0044 |

### 分组误差分析

| 分组 | Spearman | 样本占比 |
|------|---------|---------|
| count_bin=0 (total_count=1) | 0.4106 | 25.2% |
| count_bin=1 | 0.5678 | 18.9% |
| count_bin=2 | 0.6486 | 28.8% |
| count_bin=3 | 0.6811 | 22.1% |
| count_bin=4 | 0.7304 | 4.9% |

### SHAP Top 5 特征
1. `total_count` — 远超其他，count 越大 → 违规率越低
2. `grid_period_te` — 区域×时段交叉编码
3. `longitude_scaled` — 空间经度
4. `latitude_scaled` — 空间纬度
5. `month_of_year` — 季节性

### OOF-Platform Gap 分析
- LGB: 0.5815 → 0.5182 (gap: 0.063)
- Ensemble: 0.5880 → 0.5222 (gap: 0.066)
- LGB-XGB OOF 相关性: 0.9778（模型多样性有限）
- TE 分布偏移 KS stat: grid_te 0.0117, grid_period_te 0.0102（轻微）

### 关键发现
| 发现 | 详情 |
|------|------|
| TE 是最大特征工程贡献 | spatial TE + cross TE 合计 +0.0094 |
| 周期编码单独略降 | 与原始 hour/month 冗余，但完整模型中有用 |
| 小样本噪声是主要瓶颈 | total_count=1 占 25%，Spearman 仅 0.41 |
| 模型能力强 | total_count>=10 时 Spearman 达 0.69 |
| Ensemble 增益有限 | LGB-XGB 相关性 0.98，仅提升 +0.001 |

---

## Phase 5a — 提分改进 ✅

**日期**: 2026-04-03 – 05  
**状态**: 已完成（最终: Ensemble OOF 0.6408, Platform 0.5620）

### 改进计划
- [x] `docs/plan/improvement_plan.md` — 7 步提分计划（英文）

### 第一批：Step 1 + Step 2 + Step 6 ✅

| Step | 改动 | 结果 |
|------|------|------|
| Step 1: 改进 TE | 5-fold→10-fold, smooth 30/50→100/150, KDTree fallback | KS stat 降幅仅 0.001，未达 0.005 目标 |
| Step 2: 增加轮数 | n_estimators 3000→8000, lr 0.05→0.03, ES 50→100 | **主要贡献者，OOF 显著提升** |
| Step 6: Rank Normalization | 排序归一化映射到训练分布 | **反而降分 -0.007，已废弃** |

#### v2 结果

| 模型 | OOF Spearman | vs v1 |
|------|-------------|-------|
| LGB v2 | 0.5959 | +0.0144 |
| XGB v2 | 0.5994 | +0.0124 |
| Ensemble v2 | 0.6012 | +0.0132 |

#### 平台提交

| 提交文件 | Platform 得分 | vs v1 |
|---------|-------------|-------|
| ensemble_v2_raw.csv | **0.5338** | **+0.0116** |
| ensemble_v2_ranked.csv | 0.5266 | +0.0044 |

- OOF-Platform 差距: 0.6012 - 0.5338 = 0.0674（未缩小）
- 两个模型都跑满 8000 轮未触发早停，仍欠训练

#### 关键发现

| 发现 | 详情 |
|------|------|
| Step 2 是主要贡献 | lr 降低 + 更多轮数带来 OOF +0.012~0.014 |
| Step 1 效果有限 | 10-fold + 高 smoothing 对 KS stat 改善微弱，TE 偏移主因非 fold 数 |
| Rank Norm 有害 | Spearman(before, after)=0.988≠1.0，且强制映射到训练分布引入噪声 |
| 模型仍欠训练 | 8000 轮 loss 仍在下降，后续应增加到 10000+ |

### 第二批：Step 3 + Step 4 ✅

**日期**: 2026-04-04  
**状态**: 已完成

#### Step 3: Optuna 超参调优

- optuna 4.8.0, 子采样 1M 行, 3-Fold CV, 50 trials/model
- LGB best subsample Spearman: 0.5853 (trial 47)
- XGB best subsample Spearman: 0.5901 (trial 28)
- 全量 6M 重训: 10000 rounds, ES=150

**Optuna 最优参数:**

| 参数 | LightGBM | XGBoost |
|------|----------|---------|
| num_leaves / max_depth | 100 | 10 |
| learning_rate | 0.0564 | 0.0362 |
| min_child_samples / weight | 69 | 11 |
| reg_lambda | 0.452 | 1.561 |
| reg_alpha | 1.243 | 1.239 |
| feature_fraction / colsample | 0.844 | 0.951 |
| bagging_fraction / subsample | 0.972 | 0.948 |

**v3 全量重训结果:**

| 模型 | OOF Spearman | Fold std | best_iter | 训练时间 |
|------|-------------|----------|-----------|---------|
| LGB v3 | 0.6322 | 0.0006 | 9999-10000 (未触发ES) | 93 min |
| XGB v3 | 0.6379 | 0.0005 | 5909-6234 (触发ES) | 88 min |

#### Step 4: CatBoost 第三模型

- `grid_id` + `grid_period` 作为原生类别特征（ordered TE，无手动 TE 偏移）
- 8000 iterations, depth=6, ES=100（未调参）
- 5 个 fold 均跑满 8000 轮未触发早停，总耗时 378 min（~6.3 小时）

| 模型 | OOF Spearman | Fold std | 备注 |
|------|-------------|----------|------|
| CatBoost | 0.5728 | 0.0007 | 远低于调参后的 LGB/XGB |

#### Inter-Model Correlations

| 模型对 | 相关性 | vs v2 |
|--------|--------|-------|
| LGB-XGB | 0.9652 | 0.9778 → 降低 ✅ |
| LGB-CB | 0.9330 | — (新增) |
| XGB-CB | 0.9109 | — (新增) |

#### 3-Model Ensemble

- 最优权重: LGB=0.35, XGB=0.65, CB=0.00
- CatBoost 权重为 0 — OOF 0.5728 远低于 LGB/XGB (0.63+)，即使多样性高也无法补偿精度差距
- **Ensemble v3 OOF: 0.6408**

#### v3 结果汇总

| 模型 | v1 OOF | v2 OOF | v3 OOF | 提升 (v2→v3) |
|------|--------|--------|--------|-------------|
| LightGBM | 0.5815 | 0.5959 | **0.6322** | **+0.0363** |
| XGBoost | 0.5870 | 0.5994 | **0.6379** | **+0.0385** |
| CatBoost | — | — | 0.5728 | (新增) |
| Ensemble | 0.5880 | 0.6012 | **0.6408** | **+0.0396** |

#### 关键发现

| 发现 | 详情 |
|------|------|
| Optuna 调参效果巨大 | OOF 从 0.60 跳到 0.64，远超预期 (+0.01~0.02) |
| XGBoost 触发早停 | best_iter ~6000, 说明 Optuna 参数在 10000 rounds 下足够收敛 |
| LGB 仍未触发早停 | 10000 轮 loss 仍在下降，可能还有边际提升空间 |
| CatBoost 未调参导致弱 | 默认参数 OOF 0.5728，未对 ensemble 产生贡献 |
| Optuna 总耗时 ~5 小时 | LGB 160 min + XGB 132 min (on M4 Mac) |
| CatBoost 训练极慢 | 原生 categorical (742 categories) 导致每 fold ~75 min |

### 产出文件

| 文件 | 说明 |
|------|------|
| `notebooks/05_improvement.ipynb` | 提分 notebook（31 cells） |
| `docs/plan/improvement_plan.md` | 7 步提分计划 |
| `models/[lgb\|xgb]_[oof\|test]_v2.npy` | v2 预测文件 |
| `models/[lgb\|xgb\|cb]_[oof\|test]_v3.npy` | v3 预测文件 |
| `submissions/ensemble_v2_raw.csv` | v2 提交（平台 0.5338） |
| `submissions/ensemble_v2_ranked.csv` | v2 rank norm 提交（平台 0.5266） |
| `submissions/ensemble_v3.csv` | v3 提交（待提交平台） |

### 第三批：Step 4b — CatBoost Optuna 调参 ✅

**日期**: 2026-04-05  
**状态**: 已完成（GPU 服务器运行，33.7 min）

#### Optuna 最优参数（40 trials, 500K subsample, 3-Fold）

| 参数 | 值 |
|------|----|
| depth | 10 |
| learning_rate | 0.0827 |
| l2_leaf_reg | 4.12 |
| random_strength | 6.24 |
| bagging_temperature | 0.449 |
| border_count | 235 |
| min_data_in_leaf | 21 |

#### CB v4 全量重训结果（5-Fold, GPU, 8000 iterations）

| Fold | Spearman | best_iter |
|------|---------|-----------|
| 0 | 0.6182 | 7999 |
| 1 | 0.6175 | 7999 |
| 2 | 0.6179 | 7999 |
| 3 | 0.6163 | 7999 |
| 4 | 0.6175 | 7998 |
| **OOF** | **0.6175** | 均未触发 ES |

训练时间: **33.7 min**（GPU vs CPU ~6h）

#### Inter-Model Correlations (v4)

| 模型对 | v3 | v4 |
|--------|----|----|
| LGB-XGB | 0.9652 | 0.9652 |
| LGB-CB | 0.9330 | 0.9661 |
| XGB-CB | 0.9109 | 0.9615 |

#### Ensemble v4 结果

- 最优权重: LGB=0.35, XGB=0.65, **CB=0.00**
- **Ensemble v4 OOF: 0.6408**（与 v3 持平，CB 无贡献）

#### 关键发现

| 发现 | 详情 |
|------|------|
| Optuna 调参大幅提升 CB | OOF 0.5728 → 0.6175（+0.0447），GPU 仅 33.7 min |
| CB 仍无法贡献 ensemble | OOF 低于 LGB/XGB 0.015+，且调参后与 LGB/XGB 相关性反而升高（0.93→0.97），多样性降低 |
| 所有 fold 跑满 8000 轮 | loss 仍在缓慢下降，但边际收益极小 |
| Step 5 (Stacking) 放弃 | CB 权重为 0，meta-learner 只能利用 LGB+XGB 两路信号，等价于加权平均，无额外收益 |

### 产出文件

| 文件 | 说明 |
|------|------|
| `notebooks/05_improvement.ipynb` | 提分 notebook（42 cells，含 Step 4b） |
| `scripts/step4b_gpu.py` | GPU 服务器独立运行脚本（含 fold checkpoint） |
| `docs/plan/improvement_plan.md` | 7 步提分计划（含 Step 4b GPU 方案） |
| `models/[lgb\|xgb]_[oof\|test]_v2.npy` | v2 预测文件 |
| `models/[lgb\|xgb\|cb]_[oof\|test]_v3.npy` | v3 预测文件 |
| `models/cb_[oof\|test]_v4.npy` | CB v4 预测文件 |
| `models/cb_v4_fold{0-4}_[oof\|test].npy` | fold checkpoint 文件 |
| `submissions/ensemble_v2_raw.csv` | v2 提交（平台 0.5338） |
| `submissions/ensemble_v3.csv` | v3 提交（待提交平台） |
| `submissions/ensemble_v4.csv` | v4 提交（OOF 0.6408，待提交平台） |

### 完整进度汇总

| 模型 | v1 OOF | v2 OOF | v3 OOF | v4 OOF | v5 OOF | v6 OOF | v7 OOF |
|------|--------|--------|--------|--------|--------|--------|--------|
| LightGBM | 0.5815 | 0.5959 | 0.6322 | 0.6322 | 0.6315 | 0.6098 ❌ | 0.6336 |
| XGBoost | 0.5870 | 0.5994 | 0.6379 | 0.6379 | 0.6382 | 0.6375 | **0.6403** |
| CatBoost | — | — | 0.5728 | **0.6175** | 0.6175 | 0.6175 | 0.6175 |
| **Ensemble** | 0.5880 | 0.6012 | 0.6408 | 0.6408 | 0.6408 | 0.6377 ❌ | **0.6429** |
| Platform | 0.5222 | 0.5338 | 0.5620 | — | — | 0.5618 | **0.5636** |

### 待完成

- [x] 提交 ensemble_v3.csv 到平台 → **0.5620**（OOF-Platform gap: 0.6408 - 0.5620 = 0.0788）
- [x] ~~Step 7: Tier 3 特征工程 — Grid×Month TE~~ — 已完成，收益极微（见下节）
- [x] ~~Step 5: Stacking 元学习器~~ — 已放弃（CB 权重为 0，无收益）

### 第四批：Step 7 — Tier 3 特征工程（Grid×Month TE） ✅

**日期**: 2026-04-05  
**状态**: 已完成

#### 特征工程

- `grid_month = grid_id * 100 + month_of_year`（6,561 unique groups，avg 926 samples/group）
- K-Fold TE：10-fold，smooth=200
- 114 unseen test groups → fallback 使用 `grid_te`（0.022% 行）
- validation gate 通过：Spearman > 0.25，corr with grid_te < 0.96

#### LGB v5 + XGB v5（27 特征，Optuna v3 参数，10000 轮，ES=150）

| 模型 | v3/v4 OOF | v5 OOF | 变化 |
|------|----------|--------|------|
| LightGBM | 0.6322 | 0.6315 | -0.0007 |
| XGBoost | 0.6379 | 0.6382 | +0.0002 |
| Ensemble v5 | 0.6408 | **0.6408** | +0.0001 (flat) |

最优权重：LGB=0.35, XGB=0.65, CB=0.00（CB 复用 v4 预测）

#### 关键发现

| 发现 | 详情 |
|------|------|
| grid_month_te 收益极微 | Ensemble +0.0001，低于预期 +0.003~0.008 |
| KS stat 0.123 极高 | train/test 分布偏移严重（目标 <0.02），26.4% 的组样本 <50，smooth=200 不足以消除 |
| LGB 略微下降 | -0.0007，新特征引入噪声（feature_fraction=0.844 大概率选到它） |
| XGB 基本持平 | +0.0002，在随机波动范围内 |
| CB 权重仍为 0 | v5 ensemble 最优权重不含 CB |
| 特征空间饱和 | 4 种候选交叉 TE 均与现有特征高度相关（grid_dow: 0.993, grid_month: 0.946, grid_hour: 0.975），Tier 3 路径关闭 |

#### 产出文件

| 文件 | 说明 |
|------|------|
| `models/lgb_oof_v5.npy` / `lgb_test_v5.npy` | LGB v5 预测 |
| `models/xgb_oof_v5.npy` / `xgb_test_v5.npy` | XGB v5 预测 |
| `submissions/ensemble_v5.csv` | v5 提交（OOF 0.6408，待提交平台）|

---

## Phase 5b — 缩小 OOF-Platform Gap

**日期**: 2026-04-05  
**状态**: 进行中

### 关键发现：测试集 = 月份 1-5 Only

| 月份范围 | 测试行数 | 训练行数 | Ensemble OOF Spearman |
|---------|---------|---------|----------------------|
| M1-5    | 2,028,750 | 2,470,429 | **0.6492** |
| M6-12   | 0       | 3,606,117 | 0.6297 |

- 模型在 M1-5 上表现更好（OOF 0.6492），但 gap 达 0.087（0.6492→0.5620）
- Gap 来源：LGB 过拟合 + TE 编码偏移，而非月份分布差异

### Gap 根本原因

| 原因 | 证据 |
|------|------|
| LGB 过拟合 | 10,000 轮全部跑完（best_iter=9998-10000），Spearman 早已收敛 |
| TE 编码偏移 | KS stat ~0.01-0.013（系统性偏移）|
| 弱特征引入噪声 | 消融实验：周期 sin/cos 单独加入使 OOF -0.0035 |
| 小样本标签噪声 | 25.2% 样本 total_count=1（Spearman=0.41） |

### Step 8 + Step 9：Spearman ES + 特征剪枝 → v6 ❌ Spearman ES 失败

**日期**: 2026-04-06  
**Step 8**：用 Spearman 替换 l2 作为 LGB 早停指标  
**Step 9**：删除 8 个弱特征（6 个周期 sin/cos + is_raining + has_snow），FEATURES_V6 = 18 个特征

#### v6 结果

| 模型 | v3/v4 OOF | v6 OOF | 变化 | best_iter |
|------|----------|--------|------|-----------|
| LGB v6 | 0.6322 | **0.6098** | **-0.0223** | 1756-2652 (ES triggered) |
| XGB v6 | 0.6379 | 0.6375 | -0.0005 | 5800-6230 (ES triggered) |
| Ensemble v6 | 0.6408 | 0.6377 | -0.0031 | — |
| M1-5 | 0.6492 | 0.6457 | -0.0036 | — |

Ensemble v6 权重：LGB=0.05, XGB=0.90, CB=0.05（LGB 几乎被淘汰）

#### Spearman ES 失败分析

| 发现 | 详情 |
|------|------|
| 200K 子采样引入噪声 | 每轮 eval 用不同随机 200K 行，Spearman 估计轮间波动 ~0.001-0.002，远大于每轮真实提升 ~0.0001 |
| LGB 误判 plateau | best_iter=1756-2652，远低于预期的 5000-7000，模型严重欠训练 |
| 全量计算也不行 | 第一次尝试（120 万行全量 eval）显示 Spearman 一路上升到 10000 轮从未停止，与 l2 行为一致 |
| **核心假设错误** | Spearman 在此任务上不会 plateau——排名质量随 l2 持续下降而单调改善 |
| 特征剪枝中性 | XGB v6 OOF 几乎不变（-0.0005），说明 8 个弱特征对模型影响极小 |

#### 关键教训

- Spearman ES 不可行：要么全量太慢（每轮 0.2s × 10000 轮 × 5 fold），要么子采样噪声导致误停
- LGB 的 Spearman 确实与 l2 同步改善，不存在"l2 下降但 Spearman 停滞"的情况
- 特征剪枝本身无害也无益，可保留

#### 产出文件

| 文件 | 说明 |
|------|------|
| `notebooks/05_improvement.ipynb` | Cells 53-58（Step 8+9） |
| `models/lgb_[oof\|test]_v6.npy` | LGB v6 预测（Spearman ES，已退化） |
| `models/xgb_[oof\|test]_v6.npy` | XGB v6 预测（特征剪枝，基本持平） |
| `submissions/ensemble_v6.csv` | v6 提交（OOF 0.6377，未提交平台） |

#### Step 8 修正：硬上限 n_estimators=6000 → v6b

Spearman ES 失败后，改用硬上限 6000 轮（匹配 XGB 自然停止点）验证过拟合假设。

| 模型 | v3 OOF | v6b OOF | 变化 | best_iter |
|------|--------|---------|------|-----------|
| LGB v6b | 0.6322 | 0.6263 | -0.0058 | 5998-6000 (ran to limit) |
| XGB v6 | 0.6379 | 0.6375 | -0.0005 | 5800-6230 |
| Ensemble v6b | 0.6408 | 0.6392 | -0.0016 | — |
| M1-5 | 0.6492 | 0.6476 | -0.0016 | — |

Ensemble v6b 权重：LGB=0.30, XGB=0.70, CB=0.00

**Platform 提交**: ensemble_v6.csv → **0.5618**（v3: 0.5620, delta: -0.0002）

| 版本 | OOF | Platform | Gap |
|------|-----|----------|-----|
| v3 | 0.6408 | 0.5620 | 0.0788 |
| v6b | 0.6392 | 0.5618 | 0.0774 |

**结论：LGB 轮数不是 gap 的主要来源。** Gap 仅缩小 0.0014，platform 几乎不变。过拟合假设不成立。

#### 下一步方向

恢复 v3 基线配置（10000 轮、26 特征），进入 Step 10（样本加权）：
- 用 `sample_weight = np.log1p(total_count)` 降低 25% 噪声样本的影响
- 详见 `docs/plan/phase5b_gap_reduction_plan.md`

### Step 10：Sample Weighting → v7 ✅ New Best Platform

**日期**: 2026-04-06  
**状态**: 已完成（GPU 服务器运行，scripts/step10_gpu.py，~45 min）

#### 配置

- **基线**: v3 配置（Optuna v3 参数, 10000 rounds, 26 features）
- **唯一变更**: 添加 `sample_weight = np.log1p(total_count)`
  - total_count=1 → weight 0.693 (25.2% of samples)
  - total_count=10 → weight 2.398
  - total_count=50 → weight 3.932

#### v7 结果

| 模型 | v3/v4 OOF | v7 OOF | 变化 | best_iter |
|------|----------|--------|------|-----------|
| LGB v7 | 0.6322 | 0.6336 | +0.0015 | 9997-9999 (ran to limit) |
| XGB v7 | 0.6379 | 0.6403 | +0.0024 | 7079-8041 (ES triggered) |
| Ensemble v7 | 0.6408 | **0.6429** | **+0.0021** | — |
| M1-5 | 0.6492 | **0.6515** | **+0.0022** | — |

Ensemble v7 权重：LGB=0.35, XGB=0.65, CB=0.00

#### Inter-Model Correlations

| 模型对 | v4 | v7 |
|--------|----|----|
| LGB-XGB | 0.9652 | 0.9647 |
| LGB-CB | 0.9661 | 0.9657 |
| XGB-CB | 0.9615 | 0.9563 |

#### Platform 提交

| 版本 | OOF | Platform | Gap |
|------|-----|----------|-----|
| v3 | 0.6408 | 0.5620 | 0.0788 |
| v7 | **0.6429** | **0.5636** | 0.0793 |
| Delta | +0.0021 | **+0.0016** | +0.0005 |

**结论**: OOF 提升几乎 1:1 转化为 Platform 提升（+0.0016），但 **gap 未缩小**（0.0793 vs 0.0788）。Sample weighting 改善了模型质量，但未解决 distribution shift 根本问题。

#### 关键发现

| 发现 | 详情 |
|------|------|
| XGB 受益更大 | +0.0024 vs LGB +0.0015，XGB 对噪声标签更敏感 |
| XGB best_iter 增加 | v3 ~5800 → v7 ~7500，加权后损失面更平滑 |
| CB 权重仍为 0 | CB v4 未加权，与 v7 模型差距加大 |
| 模型间相关性不变 | 加权未增加多样性 |
| Gap 未缩小 | 0.0793 ≈ 0.0788，加权不治 distribution shift |

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step10_gpu.py` | GPU 服务器独立脚本 |
| `models/lgb_[oof\|test]_v7.npy` | LGB v7 预测 |
| `models/xgb_[oof\|test]_v7.npy` | XGB v7 预测 |
| `submissions/ensemble_v7.csv` | v7 提交（**Platform 0.5636，新 best**）|

### Step 11：M1-5 Focused TE + Training → v8a / v8b ❌ M1-5 TE 反而有害

**日期**: 2026-04-06 – 07  
**状态**: 已完成，v8a 平台 0.5507（比 v7 低 0.013）

#### 背景

Gap 0.079 在所有 model-quality 干预中持续存在，剩余假设：**TE 编码偏移被 M6-12 数据放大**，因为测试集仅含月份 1-5。

#### KS 分布诊断

| 特征 | KS stat (orig vs M1-5 TE) | p-value |
|------|--------------------------|---------|
| grid_te | **0.1305** | 0.0000 |
| grid_period_te | **0.1330** | 0.0000 |

- KS stat 远高于此前测量的 0.01-0.013（那个是 train OOF TE vs test full-data TE）
- 说明用全部 12 月计算的 test TE 与用 M1-5 计算的 test TE 差异显著
- M1-5 TE mean 更高（grid_te: 0.5337 vs 0.5017），反映 M1-5 违规率更高

#### 11A: 全量训练 + M1-5 TE 测试（v8a, Low Risk）

训练完全与 v7 相同（全 6M 行、Optuna v3 参数、log1p 加权），仅将 test set 的 `grid_te` 和 `grid_period_te` 替换为 M1-5 统计量（smooth=100/150）。

| 模型 | v7 OOF | v8a OOF | 变化 | best_iter |
|------|--------|---------|------|-----------|
| LGB v8a | 0.6336 | 0.6336 | ±0.0000 | 9997-9999 (ran to limit) |
| XGB v8a | 0.6403 | 0.6403 | ±0.0000 | 7079-8041 (ES triggered) |
| Ensemble v8a | 0.6429 | **0.6429** | ±0.0000 | — |
| M1-5 | 0.6515 | 0.6515 | ±0.0000 | — |

OOF 完全一致（预期：训练数据未变）。权重：LGB=0.35, XGB=0.65, CB=0.00。
LGB 训练 35.2 min，XGB 训练 10.5 min（GPU）。

#### 11B: M1-5 数据训练 + M1-5 K-fold TE（v8b, Medium Risk）

训练仅用 M1-5 行（2.47M，约 40% 原始数据）。K-fold TE 在 M1-5 内计算（smooth=30/50）。test TE 用 M1-5 全量统计。

| 模型 | v7 M1-5 OOF | v8b M1-5 OOF | 变化 | best_iter |
|------|-------------|-------------|------|-----------|
| LGB v8b | 0.6428 | 0.6384 | **-0.0044** | 9987-10000 (ran to limit) |
| XGB v8b | 0.6482 | 0.6417 | **-0.0065** | 4354-4660 (ES ~4500) |
| Ensemble v8b | 0.6515 | **0.6455** | **-0.0060** | — |

Ensemble v8b 权重：LGB=0.35, XGB=0.50, **CB=0.15**（CB 首次获得非零权重）。
LGB 训练 19.9 min，XGB 训练 3.6 min（GPU）。

#### 关键发现

| 发现 | 详情 |
|------|------|
| KS stat 0.13 表明 TE 差异显著 | M1-5 TE 与全量 TE 均值差 ~0.03，有可能缩短 gap |
| 11A OOF 完全不变 | 符合预期——训练未变，仅 test TE 不同，只有平台得分能评估效果 |
| 11B M1-5 OOF 下降 0.006 | 60% 数据损失确实伤害模型，OOF 从 0.6515 降至 0.6455 |
| XGB v8b 提前停止 ~4500 轮 | 数据量减少→更快收敛（v7: ~7500, v8b: ~4500） |
| CB 首次获非零权重 | v8b 中 LGB/XGB 较弱，CB v4（全量训练）提供互补信号 |
| 平台得分待观察 | v8a 是 "低风险高潜力"（test TE 更贴近实际），v8b 是 "OOF 略降但可能 gap 更小" |

#### 评估与建议

**v8a 评估：⏳ 高优先提交**
- 实现质量优秀，诊断完整
- KS = 0.13 表明 M1-5 TE 与全量 TE 有实质性差异，test 预测已明显改变
- OOF 不变符合预期（训练数据未变）
- 只有平台分数能评估 v8a 是否缩短 gap — **必须尽快提交**
- 决策阈值：≥ 0.575 = 成功，≈ 0.564 = 持平需其他手段，< 0.560 = 回退 v7

**v8b 评估：❌ 不推荐提交**
- M1-5 OOF 下降 0.006（0.6515 → 0.6455），未达到计划中 ≥ 0.640 但低于 v7
- 60% 数据损失超过了季节聚焦的收益
- CB 首次获非零权重是因为 LGB/XGB 变弱，而非 CB 变强

**Step 12 与 v8a 的关系：独立正交**
- Step 12 (constrained Optuna) 解决正则化问题，与 TE 编码方式无关
- 两个维度可叠加：最终提交 = 最佳 TE × 最佳正则化
- 不需等 v8a 分数即可开始 Step 12

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step11_gpu.py` | GPU 服务器脚本（含 KS 诊断 + 11A + 11B） |
| `models/lgb_[oof\|test]_v8a.npy` | LGB v8a 预测 |
| `models/xgb_[oof\|test]_v8a.npy` | XGB v8a 预测 |
| `models/lgb_[oof\|test]_v8b.npy` | LGB v8b 预测 |
| `models/xgb_[oof\|test]_v8b.npy` | XGB v8b 预测 |
| `submissions/ensemble_v8a.csv` | v8a 提交（待提交平台） |
| `submissions/ensemble_v8b.csv` | v8b 提交（不推荐提交） |

### Step 12：Constrained Re-Optuna → v9 / v9a ❌ 过度正则化

**日期**: 2026-04-06 – 07  
**状态**: 已完成，v9a 平台 0.5477（比 v7 低 0.016），v9 待提交

#### 背景

Step 12 与 Step 11 正交：Step 11 解决 TE 编码偏移（test 时 TE 的计算方式），Step 12 解决模型过拟合（更强正则化）。两个维度可叠加，v9a = v9 参数 + M1-5 TE，无需等 v8a 结果即可开始训练。

#### Optuna 搜索结果（40 trials，M1-5 子采样 1M 行，3-fold）

**LGB 最优参数对比：**

| 参数 | v7（Step 3 Optuna） | v9（Step 12 约束 Optuna） | 变化 |
|------|---------------------|--------------------------|------|
| num_leaves | 100 | **52** | ↓ 减半，树更简单 |
| learning_rate | 0.0564 | 0.0834 | ↑ 略高 |
| min_child_samples | 69 | 118 | ↑ 更保守 |
| reg_lambda | 0.452 | **9.95** | ↑ 22x，强烈 L2 正则化 |
| reg_alpha | 1.243 | **6.10** | ↑ 5x，强烈 L1 正则化 |
| feature_fraction | 0.844 | 0.799 | ↓ 略降 |
| bagging_fraction | 0.972 | 0.820 | ↓ 略降 |
| 子采样 Spearman | — | **0.5991** | M1-5 1M 行评估 |

**XGB 最优参数对比：**

| 参数 | v7 | v9 | 变化 |
|------|----|----|------|
| max_depth | 10 | **8** | ↓ 降低深度 |
| learning_rate | 0.0362 | 0.0853 | ↑ 略高 |
| min_child_weight | 11 | **180** | ↑ 16x，防小样本过拟合 |
| reg_lambda | 1.561 | **2.41** | ↑ 更强 L2 |
| reg_alpha | 1.239 | **8.34** | ↑ 7x，强烈 L1 正则化 |
| colsample_bytree | 0.951 | **0.565** | ↓↓ 大幅降低特征采样 |
| 子采样 Spearman | — | **0.5966** | M1-5 1M 行评估 |

Optuna 耗时：LGB 53 min + XGB 16 min

#### v9 训练结果（5-Fold，全量 6M 行，log1p 加权）

| 模型 | v7 OOF | v9 OOF | 变化 | best_iter |
|------|--------|--------|------|-----------|
| LGB v9 | 0.6336 | 0.6273 | **-0.0064** | 10000 (ran to limit) |
| XGB v9 | 0.6403 | 0.6283 | **-0.0120** | ~9990 (ran to limit) |
| Ensemble v9 | 0.6429 | **0.6326** | **-0.0103** | — |
| M1-5 | 0.6515 | **0.6419** | **-0.0096** | — |

Ensemble v9 权重：LGB=0.50, XGB=0.50（无 CB，文件丢失，但 CB 权重本来就是 0）
LGB 训练 35 min + XGB 训练 8 min（GPU）
LGB-XGB 相关性：0.9681（v7: 0.9647，略有上升）

#### 评估

**OOF 下降幅度超出预期（计划预期 < 0.005，实际 -0.010）：**

| 预期场景 | Ensemble OOF |
|---------|-------------|
| 轻微正则化（符合预期） | ~0.638-0.643 |
| 实际 v9 | **0.6326（-0.010）** |
| 脚本判断 | ⚠️ "OOF dropped significantly" |

根本原因：reg_lambda=9.95（原 0.452）和 min_child_weight=180（原 11）极度强烈，接近搜索边界。两模型均跑满 10000 轮（强正则化让每棵树更弱，需更多轮收敛）。

**平台分数预期分析（需实际提交才能确认）：**

| gap 缩小幅度 | 预期平台分 | vs v7 (0.5636) |
|------------|----------|----------------|
| gap 缩小 0.005 (0.079→0.074) | ~0.558 | ❌ 低于 v7 |
| gap 缩小 0.013 (0.079→0.066) | ~0.564 | ≈ 持平 v7 |
| gap 缩小 0.020 (0.079→0.059) | ~0.572 | ✅ 超过 v7 |

历史数据：v6b（硬限 6000 轮，OOF -0.0016）gap 仅缩小 0.0014，几乎不变。v9 的强正则化机制不同（约束参数 vs 截断轮数），但下降幅度大得多，结果不确定。

#### 提交结果（2026-04-07）

1. **ensemble_v8a.csv** → Platform **0.5507**（❌ 比 v7 低 0.013，M1-5 TE 有害）
2. **ensemble_v9a.csv** → Platform **0.5477**（❌ 比 v7 低 0.016，强正则化+M1-5 TE 双重伤害）
3. ensemble_v9.csv — 待提交（纯正则化，无 M1-5 TE）

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step12_gpu.py` | GPU 服务器脚本（含 Optuna + 全量重训 + 双版本 submission） |
| `models/lgb_[oof\|test]_v9.npy` | LGB v9 预测（49MB / 16MB） |
| `models/xgb_[oof\|test]_v9.npy` | XGB v9 预测（49MB / 16MB） |
| `submissions/ensemble_v9.csv` | v9 提交，全量 TE（54MB，待提交平台） |
| `submissions/ensemble_v9a.csv` | v9a 提交，M1-5 TE（54MB，待提交平台） |

### Step 13：DART Boosting → v10 ❌ 精度损失过大

**日期**: 2026-04-06 – 07  
**状态**: 已完成，结论为负面结果（不提交平台）

#### 背景

LGB-XGB 相关性 0.968，ensemble 几乎无多样性收益。DART（Dropouts meet Multiple Additive Regression Trees）在每轮 boosting 时随机丢弃已有树，强迫新树独立学习，可降低模型间相关性。

#### DART 参数（第二次调参后）

| 参数 | v7 (GBDT) | v10 (DART) | 说明 |
|------|-----------|------------|------|
| boosting_type | gbdt | **dart** | 核心变更 |
| drop_rate | — | 0.05 | 每轮丢弃 5% 的树（首次 0.1 过于激进） |
| max_drop | — | 30 | 每轮最多丢弃 30 棵 |
| skip_drop | — | 0.5 | 50% 概率跳过 drop（加速） |
| n_estimators | 10000 | 7000 | DART 不支持 early stopping，固定轮数 |

首次运行（drop_rate=0.1, n_estimators=5000）Fold 0 OOF 仅 0.6054，中断后调参重跑。

#### v10 结果（5-Fold, GPU, 182 min）

| 指标 | v7 | v10 (DART) | Delta |
|------|-----|------------|-------|
| LGB OOF | 0.6336 | 0.6147 | **-0.0190** |
| LGB M1-5 OOF | 0.6428 | 0.6266 | -0.0162 |
| LGB-XGB 相关性 | 0.9681 | **0.9476** | -0.0205 (改善) |
| Ensemble OOF | 0.6429 | 0.6406 | -0.0023 |
| Ensemble M1-5 | 0.6515 | 0.6490 | -0.0025 |

Ensemble v10 权重：**LGB=0.10, XGB=0.85, CB=0.05**（DART LGB 权重极低，几乎被忽略）

#### 成功标准检查

| 标准 | 目标 | 实际 | 结果 |
|------|------|------|------|
| DART LGB OOF ≥ 0.625 | 0.625 | 0.6147 | ❌ FAIL |
| LGB-XGB 相关性 < 0.965 | < 0.965 | 0.9476 | ✅ PASS |
| Ensemble OOF ≥ 0.640 | 0.640 | 0.6406 | ✅ PASS (勉强) |

#### 关键发现

| 发现 | 详情 |
|------|------|
| DART 确实增加多样性 | 相关性 0.968→0.948，目标达成 |
| 但精度代价太大 | LGB OOF -0.019，远超预期 ±0.005 |
| 权重搜索回避 DART | 仅给 DART 10% 权重，多样性收益不足以弥补精度损失 |
| 首次参数过于激进 | drop_rate=0.1 → Fold 0 仅 0.6054，需中断调参 |
| DART 不适合此数据集 | 600 万行高噪声数据，dropout 正则化反而削弱了有效信号 |

#### 结论

**有意义的负面结果**：DART 达成了多样性目标，但精度代价过大，不适合当前数据集。不提交 v10/v10a，保留 v7 作为 baseline。此结论可写入报告作为实验对比。

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step13_gpu.py` | GPU 服务器脚本（含 fold checkpoint + dual TE submission） |
| `models/lgb_oof_v10.npy` / `lgb_test_v10.npy` | DART LGB v10 预测 |
| `submissions/ensemble_v10.csv` | v10 提交（不推荐提交平台） |
| `submissions/ensemble_v10a.csv` | v10a 提交（不推荐提交平台） |
| `step13_gpu.log` | 训练日志 |

### Step 14：Neural Network (MLP/ResNet) → v11 ❌ NN 精度不足，无法贡献 Ensemble

**日期**: 2026-04-07  
**状态**: 已完成，结论为负面结果（不提交平台）

#### 背景

LGB-XGB 相关性 0.968，ensemble 多样性极低。训练 NN（基于 Vo 2025 的 6 层残差网络）以提供与 GBDT 正交的预测信号，即使 NN 单模精度较低，低相关性也可能提升 ensemble。

#### 模型架构 (ParkingResNet)

```
Input(26) → BatchNorm → Linear(256) → ReLU → Dropout(0.3)
→ Linear(128) → ReLU → Dropout(0.3)
→ Linear(64) → ReLU → Dropout(0.2)
→ [skip] Linear(64) → ReLU → Dropout(0.2)
→ Linear(64) → ReLU + skip [/skip]
→ Linear(32) → ReLU → Dropout(0.1)
→ Linear(1) → Sigmoid
```

训练配置：Batch=4096, LR=1e-3, WD=1e-4, MaxEpochs=30, ES patience=5, CosineAnnealingLR, Weighted MSE loss

#### NN v1 训练结果（5-Fold, GPU RTX 5880 Ada, 18.7 min）

| Fold | Spearman | best_epoch | ES epoch |
|------|---------|------------|----------|
| 0 | 0.4225 | 9 | 14 |
| 1 | 0.4223 | 4 | 9 |
| 2 | 0.4219 | 4 | 9 |
| 3 | 0.4211 | 4 | 9 |
| 4 | 0.4214 | 6 | 11 |
| **OOF** | **0.4215** | — | — |
| **M1-5** | **0.4378** | — | — |

训练 loss 快速收敛（epoch 2-4 后几乎不下降: 0.0748→0.0740），模型严重 underfitting。

#### Inter-Model Correlations

| 模型对 | 相关性 |
|--------|--------|
| NN-LGB | 0.7318 |
| NN-XGB | 0.7009 |
| NN-Ensemble v7 | **0.7174** (目标 < 0.90 ✅) |

#### 4-Model Ensemble v11 (LGB_v7 + XGB_v7 + CB_v4 + NN_v1)

| 版本 | 权重 (L/X/C/N) | OOF | M1-5 | vs v7 |
|------|---------------|-----|------|-------|
| v7 (baseline) | 0.35/0.65/0.00/— | 0.6429 | 0.6515 | — |
| v11 (full-data TE) | 0.25/0.45/0.30/**0.00** | 0.6429 | 0.6515 | ±0.0000 |
| v11a (M1-5 TE) | 0.25/0.45/0.30/**0.00** | 0.6429 | 0.6515 | ±0.0000 |

NN 权重 = 0 — grid search 判定 NN 精度太低，加入反而引入噪声。

#### 成功标准检查

| 标准 | 目标 | 实际 | 结果 |
|------|------|------|------|
| NN OOF ≥ 0.58 | 0.58 | 0.4215 | ❌ FAIL |
| NN-Ens 相关性 < 0.90 | < 0.90 | 0.7174 | ✅ PASS |
| Ensemble OOF > 0.6429 | > 0.6429 | 0.6429 | ❌ FAIL |

#### 关键发现

| 发现 | 详情 |
|------|------|
| NN 多样性优秀 | 与 GBDT 相关性仅 0.70-0.73，远低于 LGB-XGB 的 0.968 |
| 但精度太低 | OOF 0.42 vs GBDT 0.63-0.64，差距 0.22 |
| NN 严重 underfitting | 训练 loss 在 epoch 2-4 就停止下降，Early stopping 在 9-14 轮触发 |
| 26 个特征不适合 NN | GBDT 的树分裂能天然捕获 TE/count 等特征的非线性关系，NN 需要更多特征工程 |
| Grid search 耗时过长 | 4 模型 step=0.05 → 7770 组合 × 6M 行 Spearman，总运行 134.5 min（NN 训练仅 18.7 min） |

#### 结论

**有意义的负面结果**：NN 达成了多样性目标（相关性 0.72 << 0.90），但准确度远低于 GBDT，无法贡献 ensemble。v11/v11a 不提交平台。此结论可写入报告作为实验对比——展示"多样性 vs 精度"的 tradeoff。

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step14_gpu.py` | GPU 服务器脚本（含 ParkingResNet + fold checkpoint + dual TE submission） |
| `models/nn_oof_v1.npy` | NN v1 OOF 预测 (6,076,546 rows) |
| `models/nn_test_v1.npy` | NN v1 test 预测 (full-data TE) |
| `models/nn_test_v1a.npy` | NN v1 test 预测 (M1-5 TE) |
| `submissions/ensemble_v11.csv` | v11 提交（不推荐提交平台） |
| `submissions/ensemble_v11a.csv` | v11a 提交（不推荐提交平台） |
| `step14_gpu.log` | 训练日志（134.5 min 完整记录） |

### 完整进度汇总（更新至 Step 14）

| 模型 | v3 OOF | v7 OOF | v8a OOF | v9 OOF | v10 OOF | v11 OOF |
|------|--------|--------|---------|--------|---------|---------|
| LightGBM | 0.6322 | 0.6336 | 0.6336 | 0.6273 | 0.6147 (DART) | (reuse v7) |
| XGBoost | 0.6379 | **0.6403** | 0.6403 | 0.6283 | (reuse v7) | (reuse v7) |
| NN | — | — | — | — | — | 0.4215 |
| **Ensemble** | 0.6408 | **0.6429** | **0.6429** | 0.6326 | 0.6406 | 0.6429 (NN wt=0) |
| M1-5 OOF | 0.6492 | **0.6515** | 0.6515 | 0.6419 | 0.6490 | 0.6515 |
| Platform | 0.5620 | **0.5636** | 0.5507 ❌ | ⏳ pending | ❌ not submitted | ❌ not submitted |

#### 平台提交结果（2026-04-07 新增）

| 版本 | OOF | Platform | Gap | 结论 |
|------|-----|----------|-----|------|
| v7 | 0.6429 | **0.5636** | 0.079 | **当前最佳** |
| v8a (M1-5 TE) | 0.6429 | 0.5507 | 0.092 | ❌ M1-5 TE 反而有害 (-0.013) |
| v9a (强正则+M1-5 TE) | 0.6326 | 0.5477 | 0.085 | ❌ 双重伤害 (-0.016) |

**关键结论**：
- M1-5 TE 假设被推翻——全量 TE 比 M1-5 TE 更适合 test 数据
- gap 0.079 并非 TE 偏移或过拟合导致，而是数据本身的 train/test 分布差异
- Steps 8-14 全部未超越 v7，需要转向全新方法

---

## Phase 5c — Sprint Experiments (Final Push)

**日期**: 2026-04-08 起  
**状态**: 进行中  
**基线**: v7 Platform 0.5636, OOF 0.6429, M1-5 OOF 0.6515

### Experiment A: M1-5 Weight Optimization + Fine-Grained Search ❌ 无收益

**日期**: 2026-04-08  
**状态**: 已完成，结论为负面结果（delta +0.0000）

#### 背景

v7 ensemble 权重（LGB=0.35, XGB=0.65）是在全量 12 月 OOF 上以 step=0.05 搜索得到的，但测试集仅含 M1-5。在 M1-5 OOF 子集上用更细粒度（step=0.01）重新搜索，可能找到更适配 test 分布的权重。

#### 实验配置

- OOF 数据：使用 v8a OOF 文件（v8a OOF ≡ v7 OOF，因 step11_gpu.py 训练过程完全相同，仅 test TE 不同）
- 搜索范围：LGB weight ∈ [0.00, 1.00]，step=0.01，CB=0（与 v7 一致）
- 评估：M1-5 子集 Spearman + 全量 OOF Spearman

#### 权重搜索结果

| 搜索范围 | 最优 LGB | 最优 XGB | Spearman |
|---------|---------|---------|----------|
| 全量 OOF（step=0.01） | 0.36 | 0.64 | 0.6429 |
| M1-5 OOF（step=0.01） | 0.39 | 0.61 | **0.6515** |
| v7 原权重（参考） | 0.35 | 0.65 | **0.6515** |

**M1-5 Delta: +0.0000**（新权重 0.39/0.61 与原权重 0.35/0.65 得到完全相同的 Spearman）

#### 关键发现

| 发现 | 详情 |
|------|------|
| M1-5 Spearman 处于 plateau | 多种权重组合（~0.35-0.42 范围）给出相同 Rho=0.6515 |
| 全量搜索几乎一致 | 0.36/0.64 vs v7 的 0.35/0.65，差异在噪声范围内 |
| 细粒度搜索无价值 | step=0.01 相比 step=0.05 没有找到更好的权重 |
| CB 仍无贡献 | CB v4 OOF 文件不在本地，但 v7 已确认 CB weight=0 |

#### v12 提交说明

`ensemble_v12.csv` 已生成，但使用了 v8a test 预测（M1-5 TE，非 v7 全量 TE），因 v7 test .npy 文件不在本地（在 GPU 服务器上）。鉴于：
- 权重优化无改善（+0.0000）
- v8a test TE 已被证实有害（platform 0.5507 vs v7 0.5636）

**不建议提交 v12**。

#### 结论

**确定的负面结果**：M1-5 OOF 上的 Spearman 对 LGB/XGB 权重不敏感（plateau），细粒度权重搜索无法产生收益。此结论与 test 预测文件无关（纯 OOF 分析）。

#### 产出文件

| 文件 | 说明 |
|------|------|
| `notebooks/06_sprint.ipynb` Section A | Sprint notebook，带完整输出 |
| `submissions/ensemble_v12.csv` | v12 提交（v8a test + M1-5 权重，不建议提交平台） |

### Sprint Experiment C — Rank-Based Target Training ✅

**日期**: 2026-04-09  
**状态**: 已完成（GPU 训练 + 本地分析）  
**Notebook**: `notebooks/06_sprint.ipynb` Section C（自包含，可独立运行）  
**GPU 脚本**: `scripts/step_c_gpu.py`  
**GPU 运行时间**: LGB ~32 min + XGB ~10 min + Ensemble B ~15 min ≈ 57 min

#### 核心思路

当前模型训练目标是原始 `invalid_ratio`（双峰分布：25% 样本 total_count=1，违规率只有 0 或 1）。
由于 Spearman 只关心排名，改用 `y_rank = rankdata(y) / N` 作为训练目标（均匀分布），
MSE 直接优化排名误差，与评估指标一致。

- 模型参数：与 v7 完全一致（Optuna v3 + log1p 加权）
- 唯一变更：target = rank(invalid_ratio) / N

#### Stage 1 结果：✓ PASS

| 模型 | OOF Spearman | M1-5 OOF | vs v7 OOF |
|------|-------------|----------|-----------|
| LGB (rank target) | 0.6373 | 0.6440 | +0.0037 |
| XGB (rank target) | 0.6430 | 0.6486 | +0.0027 |
| v7 LGB (baseline) | 0.6336 | 0.6428 | — |
| v7 XGB (baseline) | 0.6403 | 0.6482 | — |

- Stage 1 pass criterion (OOF ≥ 0.635): **✓ PASS** (best: 0.6430)
- LGB best_iter 全部在 9998-10000，说明 10000 轮不够，可能还有提升空间

#### 模型相关性

| 组合 | Spearman 相关 |
|------|-------------|
| rank LGB — rank XGB | 0.9628 |
| v7 LGB — rank LGB | 0.9899 |
| v7 XGB — rank XGB | 0.9915 |
| v7 LGB — v7 XGB (ref) | 0.9650 |

- rank 模型与 v7 高度相关（≥ 0.97），stacking 收益有限

#### Ensemble 结果

| Ensemble | 权重 | OOF | M1-5 OOF | vs v7 |
|----------|------|-----|----------|-------|
| A: rank-only (LGB+XGB) | LGB=0.39, XGB=0.61 | **0.6464** | 0.6527 | +0.0035 |
| B: 4-model (v7+rank) | v7LGB=0.00, v7XGB=0.05, rLGB=0.40, rXGB=0.55 | **0.6465** | 0.6529 | +0.0036 |

- v7 模型权重接近 0 → rank-target 模型已完全替代 v7
- 4-model ensemble 相比 rank-only 几乎无额外收益（+0.0001）
- **推荐提交**: `ensemble_c_rank.csv`（更简洁，OOF 差异可忽略）

#### Platform 提交结果

- **`ensemble_c_rank.csv` → Platform 0.5698** 🎉 **NEW BEST** (v7 was 0.5636, +0.0062)
  - OOF 0.6464 → Platform 0.5698, gap = 0.0766 (v7 gap was 0.0793, 缩小了 0.0027)
  - rank-target 训练同时提升了 OOF 和 platform，且 gap 也略有收窄

#### 待确认

- [x] ~~提交 `ensemble_c_rank.csv` 到 platform~~ → **0.5698 (NEW BEST)**
- [ ] Stage 2 torchsort: Stage 1 已 PASS，技术上可行，但距报告截止时间紧迫，优先级降低

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step_c_gpu.py` | GPU 服务器完整训练脚本 |
| `step_c_gpu.log` | GPU 运行完整日志 |
| `models/lgb_rank_oof.npy` / `lgb_rank_test.npy` | LGB rank-target 预测 |
| `models/xgb_rank_oof.npy` / `xgb_rank_test.npy` | XGB rank-target 预测 |
| `submissions/ensemble_c_rank.csv` | rank-only ensemble 提交 (OOF=0.6464) |
| `submissions/ensemble_c_combined.csv` | v7 + rank 4-model ensemble (OOF=0.6465) |
| `notebooks/06_sprint.ipynb` Section C | 分析 notebook，带完整输出 |

---

### Sprint Experiment D — Adversarial Validation + Temporal CV

**日期**: 2026-04-08  
**Notebook**: `notebooks/06_sprint.ipynb` Section D  
**目标**: 诊断 train/test 分布偏移，建立更准确的本地评估指标

#### Part 1: Adversarial Validation

训练 LightGBM 二分类器区分 train vs test：

| 变体 | 说明 | 5-Fold AUC |
|------|------|-----------|
| V1 | 全量 train vs test | 0.9999 |
| V2 | M1-5 train vs test | 0.9995 |

- AUC 接近 1.0 → train/test 分布差异**极其显著**
- V1-V2 gap 仅 0.0003 → 月份差异只占很小比例，主要差异来自其他特征（可能与 target encoding 有关）
- 所有 fold 500 轮未触发早停 → 分类器轻松达到完美区分
- AV 概率按月分布：M5=0.0024（最像 test），M6-7≈0.002，M4=0.112，M1-3/M11-12≈0.19

**Top AV 特征**: grid_te, grid_period_te, grid_avg_count（target encoding 特征主导）

#### Part 2: Temporal CV (M1-4 → M5)

| 模型 | Random 5-Fold (v7) | Temporal (M1-4→M5) | Delta |
|------|--------------------|--------------------|-------|
| LGB | 0.6336 | 0.5972 | -0.0364 |
| XGB | 0.6403 | 0.5970 | -0.0433 |
| Ensemble | 0.6429 | 0.6017 | -0.0412 |

- Temporal CV gap = 0.0412，占 platform gap（0.0793）的约 52%
- 说明**时间分布偏移是 OOF-platform 差距的主要原因之一**

#### Part 2b: AV-Weighted Temporal CV

| 模型 | 无 AV 权重 | + AV 权重 | Delta |
|------|-----------|----------|-------|
| LGB | 0.5972 | 0.5902 | -0.0070 |
| XGB | 0.5970 | 0.5896 | -0.0074 |
| Ensemble | 0.6017 | 0.5937 | -0.0080 |

- AV 权重**有害**（-0.0080）
- 原因：AUC≈1.0 导致 AV 概率接近二值化，权重方案变成了"丢弃大部分训练样本"

#### 结论

1. **Temporal CV (0.6017) 是比 Random CV (0.6429) 更接近 platform 的本地评估指标**
2. **AV 权重不可行** — 分布差异太大，简单加权无法弥合
3. 后续实验应以 Temporal CV 作为辅助参考指标

#### 产出文件

| 文件 | 说明 |
|------|------|
| `notebooks/06_sprint.ipynb` Section D | AV + Temporal CV 完整代码和输出 |
| `figures/av_feature_importance.png` | AV 特征重要性对比图 |
| `figures/av_probability_distribution.png` | AV 概率分布图 |

### Sprint Experiment H — GBDT Label Noise Handling ✅

**日期**: 2026-04-09  
**状态**: 已完成（GPU 训练 + 本地分析）  
**Notebook**: `notebooks/06_sprint.ipynb` Section H  
**GPU 脚本**: `scripts/step_h_gpu.py`  
**GPU 运行时间**: 3 策略共 ~2.5h（LGB ~33 min × 3 + XGB ~10-28 min × 3）

#### 核心思路

25% 训练样本 total_count=1（违规率只有 0 或 1），属于高噪声标签。v7 的 log1p 加权已部分缓解（weight=0.693），但标签本身未处理。  
利用 v7 ensemble OOF 预测识别"自信错误"的 tc=1 样本作为噪声候选，测试三种处理策略。

#### 噪声识别

| 指标 | 值 |
|------|-----|
| tc=1 样本数 | 1,532,442 (25.2%) |
| 噪声: pred<0.15 & y=1 | 2,168 (0.04%) |
| 噪声: pred>0.85 & y=0 | 34,340 (0.57%) |
| **总噪声候选** | **36,508 (0.60%)** |
| tc=1 Spearman | 0.4521 |
| 清洁 tc=1 Spearman | 0.5230 |
| tc≥2 Spearman | 0.7243 |

- 噪声候选高度不对称：pred>0.85 & y=0 占 94%

#### 三种策略结果

| Strategy | LGB OOF | XGB OOF | Ens OOF | M1-5 OOF | delta vs v7 |
|----------|---------|---------|---------|----------|-------------|
| v7 baseline | 0.6336 | 0.6403 | 0.6429 | 0.6515 | — |
| **(a) Remove** | 0.6342 | 0.6421 | **0.6442** | **0.6526** | **+0.0013** |
| (b) Down-weight | 0.6340 | 0.6420 | 0.6441 | 0.6526 | +0.0012 |
| (c) Label smooth | 0.6341 | 0.6410 | 0.6435 | 0.6521 | +0.0006 |

- Success criterion (OOF ≥ 0.643): **✓ PASS**（三种策略均通过）
- 最佳策略: **(a) Remove**，OOF 0.6442，M1-5 0.6526

#### 与 v7 相关性

| Strategy | 与 v7 ensemble 相关 |
|----------|-------------------|
| (a) Remove | 0.9918 |
| (b) Down-weight | 0.9944 |
| (c) Label smooth | 0.9971 |

- 所有策略与 v7 高度相关（>0.99），对 ensemble diversity 几乎无贡献

#### 关键发现

| 发现 | 详情 |
|------|------|
| 噪声候选仅占 0.6% | 影响面小，OOF 增益有限（+0.001 级别） |
| 策略 a ≈ b > c | Remove 和 Down-weight 效果接近，Label smooth 最弱 |
| XGB 受益更大 | XGB delta +0.0018 vs LGB +0.0006，XGB 对噪声更敏感 |
| LGB 全部 ran to limit | 10000 轮未早停（用 L2 作 ES 指标），可能还有微量提升空间 |
| 收益不如 Exp C | Exp C +0.0035 vs Exp H +0.0013，rank-target 性价比更高 |

#### Platform 提交结果

- **`ensemble_ha.csv` → Platform 0.5613** ❌ 低于 v7 (0.5636, -0.0023)
  - OOF 0.6442 → Platform 0.5613, gap = 0.0829 (v7 gap was 0.0793, 扩大了 0.0036)
  - OOF 提升 +0.0013 未能转化为 platform 收益，反而 gap 扩大
  - **原因分析**: 噪声识别基于 v7 OOF 预测，去掉的 36K 样本可能包含了对 test 分布有用的信息

#### 待确认

- [x] ~~提交 `ensemble_ha.csv` 到 platform~~ → **0.5613 (低于 v7)**
- [ ] 探索 Exp C + Exp H 结合 — 鉴于 H 单独提交 platform 下降，需谨慎评估

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step_h_gpu.py` | GPU 服务器完整训练脚本 |
| `step_h_gpu.log` | GPU 运行完整日志 |
| `models/lgb_h{a,b,c}_oof.npy` / `lgb_h{a,b,c}_test.npy` | LGB 三种策略预测 |
| `models/xgb_h{a,b,c}_oof.npy` / `xgb_h{a,b,c}_test.npy` | XGB 三种策略预测 |
| `submissions/ensemble_ha.csv` | 最佳策略提交 (a) Remove (OOF=0.6442) |
| `submissions/ensemble_hb.csv` | (b) Down-weight (OOF=0.6441) |
| `submissions/ensemble_hc.csv` | (c) Label smooth (OOF=0.6435) |
| `notebooks/06_sprint.ipynb` Section H | 分析 notebook，带完整输出 |
| `docs/figures/fig_h_noise_diagnosis.png` | 噪声诊断可视化 |

### Sprint Experiment E — TabM Deep Learning Model ❌ OOF 不达标

**日期**: 2026-04-09  
**状态**: 已完成（GPU 训练 + 本地分析）  
**Notebook**: `notebooks/06_sprint.ipynb` Section E  
**GPU 脚本**: `scripts/step_e_gpu.py`  
**GPU 运行时间**: 5 折共 ~119 min（每折 20-26 min）

#### 核心思路

TabM (ICLR 2025) 是当前 SOTA tabular deep learning 模型，使用 BatchEnsemble 参数共享高效模拟 MLP ensemble。目标是获得与 GBDT 互补的深度学习预测，通过 ensemble 提升整体表现。

- 架构: BatchEnsemble MLP, K=32 ensemble members, 3 blocks, d=256, dropout=0.1
- 训练: MSE loss, AdamW, lr=1e-3, batch_size=4096, max_epochs=50, patience=7
- 早停指标: Spearman (非 RMSE)
- 样本加权: log1p(total_count)

#### 训练过程

| Fold | Best Epoch | OOF Spearman | 训练时间 |
|------|-----------|-------------|---------|
| 0 | 14 | 0.4417 | 21.9 min |
| 1 | 18 | 0.4458 | 25.5 min |
| 2 | 14 | 0.4378 | 19.4 min |
| 3 | 20 | 0.4530 | 26.0 min |
| 4 | 18 | 0.4452 | 25.8 min |

- 各折表现稳定（std < 0.005），非偶然
- 典型训练曲线：14-20 epoch 达到最佳后 val_spearman 开始下降（过拟合）

#### 结果

| 指标 | TabM | v7 baseline | Delta |
|------|------|-------------|-------|
| OOF Spearman (all) | 0.4445 | 0.6429 | -0.1984 |
| M1-5 OOF Spearman | 0.4601 | 0.6515 | -0.1914 |
| Success criterion (OOF ≥ 0.55) | **FAIL** | — | — |

#### 多样性检查

| 组合 | Spearman 相关 |
|------|-------------|
| TabM — v7 LGB | 0.7593 |
| TabM — v7 XGB | 0.7299 |
| TabM — v7 Ensemble | 0.7457 |
| TabM — rank ensemble | 0.7234 |
| v7 LGB — v7 XGB (ref) | 0.9650 |

- Diversity criterion (corr < 0.85): **PASS** — 与 GBDT 确实在"看"不同模式
- 但准确度太低导致多样性无法转化为 ensemble 收益

#### Ensemble 搜索

Grid search over [rank_LGB, rank_XGB, TabM] weights (step=0.05):

| 排名 | rank_LGB | rank_XGB | TabM | OOF |
|------|---------|---------|------|-----|
| 1 | 0.40 | 0.60 | **0.00** | 0.6464 |
| 2 | 0.35 | 0.65 | **0.00** | 0.6464 |
| 3 | 0.45 | 0.55 | **0.00** | 0.6464 |

- **TabM 最优权重 = 0**，对 ensemble 完全无贡献
- Best ensemble OOF 与 Exp C rank-only 完全一致（0.6464）

#### 关键发现

| 发现 | 详情 |
|------|------|
| DL 在此数据集上限约 0.44-0.45 | 两次 DL 尝试（ResNet 0.42, TabM 0.44）均远不如 GBDT |
| **预测方差被严重压缩** | TabM std=0.187 vs 目标 std=0.368（仅一半），预测缩向均值 → 排名区分力不足 |
| 10 特征 + 整数主力因子 = 不利于 DL | total_count 是整数计数器，GBDT 用 tree split 天然处理；DL 缺乏足够特征宽度学习非线性交互 |
| MSE loss 不直接优化 Spearman | Exp C rank-target 直接优化排名；TabM 用 MSE 被 tc=1 极端值（y∈{0,1}）主导，干扰中间值排名 |
| 多样性高但准确度不足 | corr=0.74，但 OOF 差 0.20，任何正权重都会降低 ensemble 分 |
| 不建议继续投入 DL | 两次独立实验撞上同一天花板，属结构性上限而非 hyperparameter 问题 |

#### Bug 修复

- GPU 脚本末尾生成 submission CSV 时报错 `ValueError: Usecols... ['id']`
- 原因：fallback 分支尝试从原始 CSV 读 `id` 列，但该 CSV 无名索引
- 修复 (commit e46e30f): 改用 `test_df.index`，与 step_h_gpu.py 一致
- Notebook Section E 同步修复：加 `np.clip(best_ens_test, 0, 1)`

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step_e_gpu.py` | GPU 服务器完整训练脚本 |
| `step_e_gpu.log` | GPU 运行完整日志 |
| `models/tabm_oof.npy` | TabM OOF 预测 (6,076,546 rows) |
| `models/tabm_test.npy` | TabM test 预测 (2,028,750 rows) |
| `models/tabm_fold{0-4}.pt` | 5 折模型权重（在服务器上，未下载） |
| `submissions/ensemble_tabm.csv` | TabM 单模型 submission |
| `submissions/ensemble_e_tabm.csv` | Notebook ensemble 版 (=rank-only, TabM weight=0) |
| `figures/tabm_correlation.png` | 相关性可视化 |
| `notebooks/06_sprint.ipynb` Section E | 分析 notebook，带完整输出 |

### Sprint Experiment G — Pseudo-Labeling with Curriculum Strategy ✅ COMPLETED — Null Result

**日期**: 2026-04-10  
**状态**: 完成，**null result**（OOF 无提升，不进入最终 ensemble）  
**Notebook**: `notebooks/06_sprint.ipynb` Section G  
**GPU 脚本**: `scripts/step_g_gpu.py`  
**GPU 日志**: `step_g_gpu.log`

#### 核心思路

测试集仅含 M1-5，而训练集涵盖全年。Adversarial Validation（Section D）确认分布差异极显著（AUC≈1.0）。
以高置信度的 test 预测作为伪标签加入训练集，使模型获得更多 M1-5 样本——直接针对 gap 根本原因。
基线：Exp C rank-target ensemble（OOF 0.6464, Platform 0.5698）。

**Curriculum 策略**:
- Layer 1（高置信）: `rank_test_avg < 0.02 or > 0.98`，weight=0.7
- Layer 2（中置信）: `0.10 > or < 0.90`（排除 Layer 1），weight=0.3

#### 实验结果

| 模型 | OOF Spearman | M1-5 | vs Exp C |
|------|-------------|------|---------|
| Exp C Ensemble（基准） | 0.6464 | 0.6527 | — |
| Layer 1 LGB | 0.6373 | 0.6440 | -0.0000 |
| Layer 1 XGB | 0.6430 | 0.6485 | -0.0000 |
| **Layer 1 Ensemble** | **0.6463** | 0.6525 | **-0.0001** |
| Layer 2 LGB | 0.6369 | 0.6436 | -0.0004 |
| Layer 2 XGB | 0.6430 | 0.6485 | -0.0000 |
| **Layer 2 Ensemble** | **0.6462** | 0.6524 | **-0.0002** |

Safety check: ✓ PASS（Layer 1 OOF 0.6463 ≥ 阈值 0.6434）

#### 根本原因分析：Null Result 的成因

**Pseudo-label 数量极少（近乎为零）**：
- rank_test_avg 实际范围：[0.0254, 1.0045]
- Layer 1 阈值（< 0.02 or > 0.98）筛出：**1 个**（占 test 的 0.00%）
- Layer 2 阈值（< 0.10 or > 0.90）筛出：198 个（占 test 的 0.01%）
- 两层合计：**199 个 pseudo-label** / 2,028,750 行 test

**根本原因**：Exp C 使用 rank-target 训练，训练集 y_rank 的实际范围被压缩在 [0.079, 0.867] 之间（见 step_c_gpu.log）。模型以此范围内的值进行预测，导致测试预测几乎不可能超出此范围，自然无法满足 < 0.02 或 > 0.98 的高置信阈值。这是 **threshold 设计与 rank-target prediction range 的结构性不匹配**——如果使用 raw invalid_ratio 预测（范围 0/1）则阈值设计成立，但 rank-target 预测天然被压缩。

即便 199 个 pseudo-label 被加入 607 万行训练集，比例仅为 0.003%，统计上没有任何作用。

#### 结论与决策

- **不提交 Exp G 文件**（与 Exp C 几乎相同，没有意义）
- **不纳入最终 ensemble**（OOF 降低而非提升）
- **Exp G null result 本身有科研价值**：证明了 rank-target 的预测范围压缩特性会破坏标准 pseudo-labeling 阈值设计，为后续工作提供警示
- **Exp C（Platform 0.5698）维持最高分**，Exp F 将以此为核心

#### 产出文件

| 文件 | 说明 |
|------|------|
| `models/g1_lgb_oof.npy` / `g1_lgb_test.npy` | Layer 1 LGB 预测 |
| `models/g1_xgb_oof.npy` / `g1_xgb_test.npy` | Layer 1 XGB 预测 |
| `models/g2_lgb_oof.npy` / `g2_lgb_test.npy` | Layer 2 LGB 预测 |
| `models/g2_xgb_oof.npy` / `g2_xgb_test.npy` | Layer 2 XGB 预测 |
| `submissions/ensemble_g1.csv` | Layer 1 提交文件（不提交至平台）|
| `submissions/ensemble_g2.csv` | Layer 2 提交文件（不提交至平台）|
| `step_g_gpu.log` | 完整训练日志 |

---

### Sprint Experiment F — Final Ensemble Combination ✅ COMPLETED

**日期**: 2026-04-10  
**状态**: 完成，最终提交文件已生成  
**Notebook**: `notebooks/06_sprint.ipynb` Section F（本地 CPU，< 1 min）

#### 核心思路

仅纳入有 platform 收益的模型，对 Exp C rank LGB/XGB 进行精细权重搜索（step=0.01 vs Exp C 脚本的 step=0.05），并条件性加入 Exp G（阈值 OOF ≥ 0.6464）。

#### 实验结果

| 策略 | OOF | M1-5 | 决策 |
|------|-----|------|------|
| Exp C LGB 单模型 | 0.6373 | 0.6440 | — |
| Exp C XGB 单模型 | 0.6430 | 0.6486 | — |
| **Strategy 1: Exp C rank ensemble** | **0.6464** | **0.6527** | ✅ 最终选择 |
| Exp G Layer 1（参考） | 0.6463 | 0.6525 | ✗ OOF 低于阈值 0.6464 |
| Strategy 2: C+G blend | — | — | 未触发（G 不达标）|

**最终权重**：LGB=0.39，XGB=0.61（精搜从 step=0.05 的 LGB=0.40 微调，OOF 不变）

#### 分析

Exp F 无法超越 Exp C，原因：
1. Exp G OOF=0.6463 比阈值 0.6464 低 0.0001，`_g_qualifies=False`，Strategy 2 未触发
2. 精细权重搜索（step=0.01）未找到比 Exp C 更好的组合，OOF 维持 0.6464
3. `ensemble_final.csv` 实质上是用更精细权重重生成的 Exp C，与已提交的 `ensemble_c_rank.csv` 几乎等价

**最终结论：Platform 0.5698（Exp C）是本次 Sprint 天花板。** 所有可行方向均已探索完毕：
- rank-target 方向（Exp C）✅ 最优
- pseudo-labeling（Exp G）❌ null result（threshold/range 不匹配）
- label noise 去除（Exp H）❌ platform 反降
- 深度学习（Exp E）❌ OOF 上限 0.44
- AV 加权（Exp D）❌ 相当于丢弃大量训练样本

#### 产出文件

| 文件 | 说明 |
|------|------|
| `submissions/ensemble_final.csv` | 最终提交文件（OOF=0.6464，LGB=0.39，XGB=0.61）|
| `submissions/ensemble_f_expC.csv` | Exp C baseline 参考提交 |
| `notebooks/06_sprint.ipynb` Section F | 完整分析 + 权重搜索可视化 |

### Sprint Experiment I — Rank-Target GBDT Re-tuning ✅ COMPLETED

**日期**: 2026-04-10  
**状态**: 完成（Part A + Part B 均已运行）  
**GPU 脚本**: `scripts/step_i_gpu.py`  
**GPU 日志**: `step_i_gpu.log`  
**GPU 运行时间**: Part A ~79 min (LGB 68 min + XGB 11 min) + Part B ~315 min (Optuna ~220 min + retrain ~94 min)

#### 核心思路

Exp C 使用 v7 的 Optuna 参数（为 raw bimodal target 调优），但 rank-target 的 loss surface 完全不同。
关键证据：Exp C 所有 5 个 LGB fold 均命中 n_estimators=10000 上限（best_iter=9998-10000），模型仍在学习。

**Part A**: 增加 LGB n_estimators 10000→20000，XGB 10000→15000，ES patience 150→200  
**Part B**: Optuna 60 trials 在 rank-target 上重新搜索超参（1M M1-5 子采样，3-fold CV）

#### Part A 结果：✓ PASS

| 模型 | OOF Spearman | M1-5 | vs Exp C |
|------|-------------|------|---------|
| LGB I-A (n=20000) | 0.6417 | 0.6476 | +0.0044 |
| XGB I-A (n=15000) | 0.6430 | 0.6486 | +0.0000 |
| **Ensemble I-A** | **0.6478** | **0.6537** | **+0.0014** |

- **LGB 受益显著**：OOF 0.6373→0.6417（+0.0044），5 个 fold 仍全部撞限（best_iter≈19990+），模型仍未收敛
- **XGB 无变化**：ES 在 ~7900-8100 触发（与 Exp C 一致），已达最优
- Ensemble 权重（M1-5 OOF 搜索，step=0.01）：LGB=0.48, XGB=0.52
- Success criterion (OOF ≥ 0.6470): **✓ PASS**

#### Part B 结果：✗ 不如 Part A

**Optuna 搜索结果**（60 trials, 1M M1-5 rows, 3-fold）:

| 模型 | Best trial Spearman | 搜索耗时 |
|------|-------------------|---------|
| LGB | 0.6135 | 162.4 min |
| XGB | 0.6152 | 57.5 min |

**Optuna 最优参数 vs v7**:

| 参数 | v7 (Exp C) | Optuna best | 变化 |
|------|-----------|-------------|------|
| LGB num_leaves | 100 | 123 | +23 |
| LGB learning_rate | 0.0564 | 0.0325 | -0.024 (更慢) |
| LGB min_child_samples | 69 | 49 | -20 |
| LGB reg_lambda | 0.452 | 1.933 | +1.48 (更强) |
| LGB reg_alpha | 1.243 | 0.422 | -0.82 |
| XGB max_depth | 10 | 9 | -1 |
| XGB learning_rate | 0.0362 | 0.0284 | -0.008 (更慢) |
| XGB min_child_weight | 11 | 25 | +14 |

**Part B 全量重训结果**:

| 模型 | OOF Spearman | M1-5 | vs Exp C | vs Part A |
|------|-------------|------|---------|----------|
| LGB I-B | 0.6415 | 0.6479 | +0.0042 | -0.0002 |
| XGB I-B | 0.6426 | 0.6482 | -0.0004 | -0.0004 |
| **Ensemble I-B** | **0.6474** | **0.6536** | **+0.0010** | **-0.0004** |

- LGB I-B 略逊于 I-A（0.6415 vs 0.6417）：Optuna 选择更低 lr 导致收敛更慢，20000 轮仍不够
- XGB I-B 略逊于 I-A（0.6426 vs 0.6430）：同理，XGB 全部 5 fold 撞 15000 上限（Exp C 仅 ~8000）
- **结论：v7 参数在 rank-target 下依然（近似）最优，Optuna 未能找到更好组合**

#### 模型相关性

| 组合 | Spearman 相关 |
|------|-------------|
| rank LGB_i — rank XGB_i | 0.9663 |
| Exp C LGB — Exp I LGB | 0.9957 |
| Exp C XGB — Exp I XGB | 0.9999 |

- Exp I 与 Exp C 高度相关（I 只是多跑了迭代），不增加多样性

#### 关键发现

| 发现 | 详情 |
|------|------|
| LGB 确实欠训练 | 从 10000→20000 带来 +0.0044 单模型提升 |
| 20000 轮仍不够 | 5 个 fold 全部再次撞限，理论上可继续增加 |
| XGB 已稳定 | ES 在 ~8000 触发，更多迭代无意义 |
| Optuna 无收益 | rank-target 的最优参数与 raw-target 非常接近 |
| 根本限制来自迭代上限 | LGB 的收益来自更多迭代而非新参数 |
| Part B 失败根因 | Optuna 选择更低 lr（LGB 0.032 vs 0.056, XGB 0.028 vs 0.036）但 n_estimators 上限不变，导致 Part B 两个模型全部 5/5 fold 撞限，实际在更差的收敛点停下 |

#### 最终决策

**Part A > Part B > Exp C**：已提交 `ensemble_i_a.csv`（OOF=0.6478, M1-5=0.6537）。

**平台结果（2026-04-11）**：**0.5705** 🎉（排名第 5）
- vs Exp C: **+0.0007**（0.5705 vs 0.5698）
- OOF-Platform gap: 0.6478 - 0.5705 = **0.077**（与历史 gap 一致）
- **新项目最佳**：Platform 0.5705

#### 产出文件

| 文件 | 说明 |
|------|------|
| `scripts/step_i_gpu.py` | GPU 脚本（Part A + B + ensemble） |
| `step_i_gpu.log` | 完整训练日志 |
| `models/lgb_rank_i_oof.npy` / `lgb_rank_i_test.npy` | Part A LGB 预测 |
| `models/xgb_rank_i_oof.npy` / `xgb_rank_i_test.npy` | Part A XGB 预测 |
| `models/lgb_rank_ib_oof.npy` / `lgb_rank_ib_test.npy` | Part B LGB 预测 |
| `models/xgb_rank_ib_oof.npy` / `xgb_rank_ib_test.npy` | Part B XGB 预测 |
| `submissions/ensemble_i_a.csv` | Part A ensemble（OOF=0.6478）**推荐提交** |
| `submissions/ensemble_i_b.csv` | Part B ensemble（OOF=0.6474）|
| `docs/plan/exp_ij_tuning_plan.md` | 实验计划 |

---

## Phase 6 — 报告

**日期**: 2026-04-09 – 22（计划）  
**状态**: 未开始

---

## Phase 7 — 展示视频 ✅

**日期**: 2026-04-06 – 13  
**状态**: 已完成（视频已上传 Canvas）

### 7.1 前期策划（04-06 ~ 04-11）

在 `docs/subs/` 下完成 4 份策划文档：

| 文档 | 内容 |
|------|------|
| `01_figure_checklist.md` | 可视化图表清单，确认每张图对应哪页 slide |
| `02_ppt_structure.md` | 24 页 PPT 结构规划（6 Section + 18 Content） |
| `03_video_script.md` | 初版视频脚本草稿 |
| `04_qa_prep.md` | Peer Review Q&A 准备 |

### 7.2 PPT 制作与迭代（04-12）

**技术栈选型**：选用 [Slidev](https://sli.dev/) (Vite + Vue + Markdown) + `slidev-theme-neversink` 学术主题，而非传统 PowerPoint。
- **优势**：代码高亮、LaTeX 公式、版本控制友好、一键部署为网页
- **安装**：`@slidev/cli@52.14.2`, `slidev-theme-neversink@0.4.1`, `playwright-chromium`

**PPT 内容**：24 页，覆盖完整项目流程：
- Section 1: Introduction & Problem Setup（Slide 1-4）
- Section 2: Data Exploration & Feature Engineering（Slide 5-9）
- Section 3: Baseline Development & Gap Analysis（Slide 10-13）
- Section 4: Key Innovation — Rank-Target Training（Slide 14-17）
- Section 5: Experiment Summary & Analysis（Slide 18-21）
- Section 6: Conclusion（Slide 22-24）

**经历 4 轮布局修复**（commit `064e2d4` → `e3fe92f`）：
1. **Round 1**: 图片压缩失真（改用 `w-full` 替代 `h-*` 限高）、Section 颜色过深（navy → sky/blue/teal/slate）、p11/12/13 内容溢出（`text-sm` + 减 `<br>`）
2. **Round 2**: Cover 徽章颜色改黑色、p4 内容下沉
3. **Round 3**: 发现 neversink 主题 `top-title` 布局 bug（`:: default ::` slot 使用 `h-full` 导致内容下沉），**根因修复**：全部改为 `top-title-two-cols` + `columns` prop
4. **Round 4**: p8/p19 下沉修复（转 `top-title-two-cols` / `:: content ::`）、p6 图片大小优化、p20 表格显示不全（加宽列）

**关键技术发现**：
- `top-title.vue` 的 `:: default ::` slot 有 `h-full`（100% 父高度），在 `flex-col` 里导致内容下沉
- `top-title-two-cols.vue` 用 `flex-1 min-h-0`，正确填充剩余高度
- **结论**：内容页永远用 `top-title-two-cols` + `columns` prop 控制列宽，禁用 `top-title` + 自定义 flex

### 7.3 旁白脚本（04-12 ~ 04-13）

编写英文旁白脚本 `docs/subs/voiceover_script.md`：
- 目标：~1,450 词，~12 分钟（120 wpm，AI 配音语速）
- 每页 slide 对应一段旁白，标注 `[PAUSE]` 作为转场点
- 句子保持短到中等长度，无缩写，适合 TTS 朗读

### 7.4 视频自动化合成（04-12 ~ 04-13）

开发 `video_production/make_video.py` 自动化流水线：

```
voiceover_script.md → TTS 音频 → slide PNG 截图 → FFmpeg 合成 → 最终 MP4
```

**流水线步骤**：
1. 解析 `voiceover_script.md`，提取每页 slide 的旁白文本
2. 使用 **edge-tts**（Microsoft 免费 TTS）生成 24 段音频（voice: `en-US-AriaNeural`）
3. 使用 Slidev `export --format png` 导出 24 张 1920×1080 PNG
4. FFmpeg 将每张 slide 图片 + 对应音频合成短片段（含 0.3s fade 过渡）
5. 拼接所有片段为最终 MP4 + 自动生成 SRT 字幕

**视频参数**：
- 分辨率：1920×1080, 30fps, H.264 + AAC
- 每段结尾留 1.0s 静音（`TAIL_SECONDS`），0.3s fade-out 过渡
- TTS 语速：+0%（正常速度）

**产出文件**：

| 文件 | 说明 |
|------|------|
| `video_production/output/final_v2.mp4` | 最终视频（22 MB, ~13 分 18 秒） |
| `video_production/output/final_v2.srt` | 自动生成字幕 |
| `video_production/audio_v2/slide_*.mp3` | 24 段 TTS 音频（每页一段） |
| `video_production/slides_png/slide/*.png` | 24 张 slide 截图 |
| `video_production/make_video.py` | 视频合成脚本 |

### 7.5 封面更新 & PDF 导出（04-13）

- 封面增加团队信息：Group 28 + 4 名成员（commit 后脱敏处理）
- ChallengeData #163 移至 Platform Spearman 徽章
- 尝试 PDF 导出（`slidev export`）：p6/12/13 有渲染 bug（CSS 在无头 Chromium 中未完整加载），最终通过分段导出 PDF 规避

### 7.6 部署尝试（04-12 ~ 04-13）

- 创建 GitHub Actions workflow（`.github/workflows/deploy.yml`）自动部署 Slidev 到 GitHub Pages
- **失败原因**：私有仓库 + GitHub Free 计划不支持 Pages
- 暂未改为 Public，待课程结束后考虑

### 7.7 视频上传（04-13）

- 最终视频 `final_v2.mp4` 已上传 Canvas "Group presentation assignment"
- 视频时长 ~14:57，符合 15 分钟限制

---

### Commit 记录（Phase 7 相关）

| Commit | 描述 |
|--------|------|
| `9d99217` | 生成 5 张缺失的可视化图表 |
| `ec79ad1` | 初版 Slidev 18 页（academic 主题） |
| `9f24049` | 修复图片路径 + 升级 neversink 主题 |
| `c7cccdb` | 修复 9 项布局/内容问题 |
| `064e2d4` ~ `e3fe92f` | 4 轮布局修复（颜色、图片、内容下沉） |
| `2d6967c` | p6/p8/p19/p20 修复 |
| `bff330f` | 去重 score_progression、p12 改单列 |
| `b6ce570` / `f1305ab` | GitHub Actions workflow（未成功） |
| `e99239e` | .gitignore 更新 + voiceover script |
| `fa2da1e` / `9693036` | 封面团队信息（脱敏版） |
| `fda0ad8` | Phase 7 状态更新、best score 修正、gitignore exports |
| `097b4a9` | 修复 slides.md 测试集大小 1.5M→2.03M、补齐 environment.yml 依赖、同学反馈分析 |
| `7aa944e` | 报告骨架：7 节 + 附录、图表引用、同学反馈整合点 |

---

## Phase 6: Report Writing（04-17 开始）

**状态**: 进行中
**截止日期**: 2026-04-23

### 6.1 报告前准备（04-17）

- 项目全面审查：代码质量优秀、实验完整、文档充分，无阻塞性问题
- 修复一致性问题：
  - `slides.md` 测试集大小 1.5M → 2.03M（事实错误）
  - `environment.yml` 补齐 pyarrow、optuna（缺失依赖）、torch 标为可选
  - `README.md` Phase 7 状态 `[ ]` → `[x]`
  - `CLAUDE.md` 最佳成绩更新为 0.5705（Exp I-A）
- 创建 `report/` 目录，专放报告相关材料
- 分析 7 条同学反馈 → `report/peer_feedback_analysis.md`
  - 4 条值得写进报告（时序 CV、GBDT 选择理由、GBDT 抗偏移说法修正、特征重要性≠因果）
  - 3 条简单回应（rank-target 鲁棒性、单实验担忧澄清、ML vs DL 正面反馈）
- 搭建报告骨架 `report/report.md`：
  - Abstract（已写初稿）、Introduction、Related Work、Data Description、Methodology、Results、Discussion、Conclusion、References（已完成）、Appendix A/B
  - 每节含 HTML 注释提纲 + 图表引用 + 同学反馈整合标记
  - 约 15 页，格式 Markdown + Pandoc → PDF（可选转 LaTeX）

### 6.2 报告撰写（计划）

- 预计按 §3 Data → §4 Methodology → §5 Results → §6 Discussion → §1-2 Introduction/Related Work → §7 Conclusion → Abstract 定稿 的顺序撰写
- 需补充：团队分工信息（Appendix B）
