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

## Phase 3 — 建模

**日期**: 2026-04-07 – 09（计划）  
**状态**: 未开始

---

## Phase 4 — 评估分析

**日期**: 2026-04-10（计划）  
**状态**: 未开始

---

## Phase 5 — 可视化

**日期**: 2026-04-11（计划）  
**状态**: 未开始

---

## Phase 6 — 报告

**日期**: 2026-04-15 – 22（计划）  
**状态**: 未开始

---

## Phase 7 — 视频

**日期**: 2026-04-12 – 14（计划）  
**状态**: 未开始
