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

## Phase 6 — 报告

**日期**: 2026-04-09 – 22（计划）  
**状态**: 未开始

---

## Phase 7 — 视频

**日期**: 2026-04-06 – 08（计划）  
**状态**: 未开始
