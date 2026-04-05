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

| 模型 | v1 OOF | v2 OOF | v3 OOF | v4 OOF | v5 OOF |
|------|--------|--------|--------|--------|--------|
| LightGBM | 0.5815 | 0.5959 | **0.6322** | 0.6322 | 0.6315 |
| XGBoost | 0.5870 | 0.5994 | **0.6379** | 0.6379 | 0.6382 |
| CatBoost | — | — | 0.5728 | **0.6175** | 0.6175 (reused) |
| **Ensemble** | 0.5880 | 0.6012 | **0.6408** | **0.6408** | **0.6408** |
| Platform | 0.5222 | 0.5338 | **0.5620** | 待提交 | 待提交 |

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

## Phase 5b — 可视化

**日期**: 待定  
**状态**: 未开始

---

## Phase 6 — 报告

**日期**: 2026-04-09 – 22（计划）  
**状态**: 未开始

---

## Phase 7 — 视频

**日期**: 2026-04-06 – 08（计划）  
**状态**: 未开始
