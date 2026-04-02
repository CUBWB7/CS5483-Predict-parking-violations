# CS5483 Project 2 — #163 停车违章预测 项目实施方案

> 选题：ChallengeData #163 "Predict Parking Violations" by Egis
> 初始版本：2026-04-02 | 最后更新：2026-04-02
> 数据来源：希腊塞萨洛尼基 THESi 路边停车系统（与 Paper 1&2 数据源一致）

---

## 一、项目时间线

| 日期 | 阶段 | 核心产出 |
|------|------|---------|
| **4/2** | Phase 0 ✅ | Git 仓库、conda 环境、项目框架 |
| **4/2** | Phase 1 ✅ | EDA notebook + 7 张图表 |
| **4/4-6** | Phase 2 | 特征工程管道（最重要阶段） |
| **4/7-9** | Phase 3 | LightGBM/XGBoost 模型 + 调参 |
| **4/10-11** | Phase 4-5 | 评估分析 + 可视化 |
| **4/12-14** | Phase 7 | 录制 15 分钟展示视频 |
| **⚠️ 4/15** | **截止** | **视频提交** |
| **4/15-22** | Phase 6 | 撰写 ~15 页报告 |
| **⚠️ 4/23** | **截止** | **报告提交** |

---

## 二、数据分析关键发现

### 数据规模
- 训练集：~607 万行，10 个特征
- 测试集：~203 万行，同样 10 个特征
- 目标变量：`invalid_ratio`（0-1 连续值，违规率）
- 评估指标：**Spearman 相关系数**（衡量排序一致性，非数值精度）
- 官方基线：0.197（Random Forest, 10 棵树，仅使用 count≥45 的记录）

### 特征清单

| 特征 | 类型 | 说明 | Spearman ρ | 论文启示 |
|------|------|------|-----------|---------|
| `total_count` | int | 检查的停车数量 | **-0.297**（最强） | 对应 Paper 1 的"区域容量" |
| `longitude_scaled` | float | 缩放经度 (0.98-1.0) | 0.061 | 需网格化/聚类转为区域 ID |
| `latitude_scaled` | float | 缩放纬度 (0.99-1.0) | 0.080 | 同上 |
| `Precipitations` | float | 降水量 | 0.003 | Paper 5：湿度比温度更重要 |
| `HauteurNeige` | float | 积雪（二元：0/1） | ~0 | 填充 0 |
| `Temperature` | float | 温度 | 0.026 | Paper 1：6 小时窗口平均更有效 |
| `ForceVent` | float | 风力 | -0.010 | 填充中位数 |
| `day_of_week` | int | 星期几 (1-6，无周日) | 0.004 | Paper 1：sine 编码 sin(2π×w/7) |
| `month_of_year` | int | 月份 (1-12) | **-0.091** | Paper 1：sine 编码 sin(2π×(m-1)/12) |
| `hour` | int | 小时 (6-19) | 0.010 | Paper 4：通勤/午休高峰 |

### 关键洞察

1. **`total_count` 是最强预测因子**：检查车辆数越多 → 违规率越低/越稳定
2. **小样本噪声严重**：`total_count=1` 时违规率只能是 0 或 1（均值 0.664）；`total_count≥30` 时均值 ~0.30，方差小得多
3. **目标变量 U 型分布**：15.86% = 0，26.74% = 1（Paper 5 建议零膨胀处理）
4. **月份有季节性**：违规率随月份变化（ρ=-0.091），Paper 4 发现冬夏模式不同
5. **空间位置有模式**：经纬度组合可定位具体区域，Paper 3 发现 RF 在所有空间尺度最优
6. **天气单独作用弱但有交互效应**：Paper 5 SHAP 分析揭示天气×时间非线性交互
7. **数据只覆盖工作日白天**：周一至周六、6:00-19:00（与 THESi 执法时段一致）
8. **论文特征重要性排序**（Paper 2 消融实验）：时间特征(10.2%) > 天气(4.9%) > 指标(1.7-2.3%)

### total_count 与违规率的关系

| total_count 分组 | 样本数 | 平均违规率 | 标准差 |
|-----------------|--------|-----------|--------|
| 1 | 126,289 | 0.664 | 0.472 |
| 2-3 | 94,785 | 0.582 | 0.382 |
| 4-10 | 144,112 | 0.459 | 0.296 |
| 11-30 | 110,449 | 0.354 | 0.213 |
| 31-50 | 18,611 | 0.300 | 0.169 |
| 51-100 | 5,308 | 0.294 | 0.168 |
| 101+ | 446 | 0.362 | 0.260 |

---

## 三、项目开展流程

### Phase 0: 环境与代码框架搭建 ✅ 已完成（4/2）
- [x] 创建 GitHub 仓库：https://github.com/CUBWB7/CS5483-Predict-parking-violations.git
- [x] 创建 conda 环境 `parking`（lightgbm 4.6, xgboost 3.2, catboost 1.2 等）
- [x] 项目结构搭建（.gitignore、README、CLAUDE.md、environment.yml）
- [x] 随机种子：`SEED = 42`

### Phase 1: EDA 探索性数据分析（4/2-3）
- [ ] 各特征分布可视化
- [ ] 目标变量 U 型分布分析（展示 total_count 与噪声的关系）
- [ ] 特征与目标的 Spearman 相关性分析
- [ ] 空间分布散点图（经纬度着色为违规率）
- [ ] 时间模式分析（按小时、星期、月份的平均违规率）
- [ ] 缺失值检查与处理策略确认
- **产出**：`notebooks/01_eda.ipynb` + 关键图表（复用到报告和视频）

### Phase 2: 数据预处理与特征工程（4/4-6）— 最重要的阶段

> **论文指导的优先级**：空间区域特征 > 时间周期编码 > 交叉特征 > 天气增强

#### 2.1 基础预处理
- [ ] 缺失值填充（`HauteurNeige` 用 0，`ForceVent` 用中位数）
- [ ] 数据类型优化（float64 → float32，减少内存占用）

#### 2.2 空间特征（最高优先级，Paper 1,2,3,5）
- [ ] 坐标网格化：将经纬度划分为网格（参考 Paper 5 的 500m 网格）
- [ ] 区域聚类：DBSCAN/KMeans 对坐标聚类，生成区域 ID
- [ ] **区域 Target Encoding**（K-Fold 防泄露）：每个区域的历史平均违规率
- [ ] 区域统计特征：平均 total_count、违规率标准差等

#### 2.3 时间特征（高优先级，Paper 1,2 消融实验表明贡献最大 10.2%）
- [ ] **Sine/Cosine 周期编码**：
  - `hour_sin = sin(2π × hour / 24)`，`hour_cos = cos(2π × hour / 24)`
  - `dow_sin = sin(2π × day_of_week / 7)`，`dow_cos = cos(2π × day_of_week / 7)`
  - `month_sin = sin(2π × (month - 1) / 12)`，`month_cos = cos(2π × (month - 1) / 12)`
- [ ] 时段分箱：早高峰(7-9)、上午(9-12)、午休(12-14)、下午(14-17)、傍晚(17-19)
  - Paper 4 发现 08-09 和 12-13 是违章高峰

#### 2.4 交叉/交互特征（中优先级，Paper 3,4,5）
- [ ] 区域 × 时段的平均违规率（Target Encoding，K-Fold 防泄露）
- [ ] 区域 × 星期几
- [ ] 天气 × 时段（Paper 5 SHAP 揭示非线性交互）
- [ ] total_count 分箱 × 区域

#### 2.5 天气特征增强（中优先级，Paper 1,5）
- [ ] 是否下雨（Precipitations > 0 → 二元特征）
- [ ] 温度离散化
- [ ] 如可行：6 小时窗口天气平均（Paper 1 的做法）

#### 2.6 total_count 特殊处理
- [ ] `log(total_count + 1)` 对数变换
- [ ] total_count 分箱（1, 2-3, 4-10, 11-30, 31+）
- [ ] 考虑对小 total_count 样本降权

- **产出**：`notebooks/02_feature_engineering.ipynb`

### Phase 3: 建模（4/7-9）

#### 3.1 Baseline 模型
- [ ] Random Forest（10 棵树）— 复现官方 baseline (Spearman ~0.197)
- [ ] 线性回归 — 最简单 baseline

#### 3.2 主力模型

**LightGBM（首选）— Paper 6 起步参数**：
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'lambda_l2': 1.0,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}
# 早停指标用 Spearman（不是 RMSE）
```

**调参优先级**（Paper 6&7 指导）：
1. `num_leaves`（10-150）→ 控制模型复杂度
2. `learning_rate`（0.01-0.2）→ 收敛速度
3. `min_child_samples`（10-50）→ 防叶子过拟合
4. `feature_fraction`（0.7-1.0）→ 特征子采样
5. `lambda_l2`（0-10）→ L2 正则化

- [ ] **LightGBM** — 5-fold CV，Spearman 早停
- [ ] **XGBoost** — `tree_method='hist'` 应对大数据
- [ ] **CatBoost** — 天然处理类别特征

#### 3.3 进阶模型（可选，有时间再做）
- [ ] LGBMRanker — 直接优化排序指标
- [ ] 深度残差网络 — 参考 Paper 1 架构（512→256→128→64→128→32→1）
- [ ] GPBoost — 高斯过程 + LightGBM

#### 3.4 模型集成
- [ ] 加权平均 LightGBM + XGBoost + CatBoost
- [ ] 或 Stacking（第二层用 Ridge 回归）

- **产出**：`notebooks/03_modeling.ipynb`

### Phase 4: 评估与分析（4/10-11）
- [ ] **主指标**：Spearman 相关系数（5-fold CV + 平台提交）
- [ ] 辅助指标：MAE, MSE, R²
- [ ] 各模型对比表格
- [ ] **消融实验**（参考 Paper 2）：逐步添加特征组，展示每组的 Spearman 贡献
  - 仅原始特征 → +空间特征 → +时间编码 → +交叉特征 → +天气增强
- [ ] **特征重要性分析**：LightGBM feature importance (gain) + SHAP
- [ ] **SHAP 交互分析**（参考 Paper 5）：展示特征间的非线性交互
- [ ] 分组误差分析：按 total_count / 区域 / 时段分析模型弱点
- [ ] 与 baseline (0.197) 的对比
- **产出**：`notebooks/04_evaluation.ipynb`

### Phase 5: 可视化与展示准备（4/10-11，与 Phase 4 并行）
- [ ] 违规率空间散点图/热力图
- [ ] 时间趋势图（按小时/月份）
- [ ] 模型性能对比柱状图
- [ ] 消融实验柱状图
- [ ] 特征重要性图 / SHAP Summary Plot
- [ ] 预测 vs 真实散点图

### Phase 6: 报告撰写（4/15-22）

**结构（~15 页）**：

| 章节 | 页数 | 内容 |
|------|------|------|
| Abstract | 0.5 | 问题、方法、结果一段话概括 |
| 1. Introduction | 1-1.5 | 停车违章预测的背景和意义 |
| 2. Related Work | 1.5-2 | 引用 7 篇论文（Paper 1-2 直接相关，3-5 领域，6-7 模型） |
| 3. Data Description | 1.5-2 | 数据来源、特征、EDA 关键发现（配图表） |
| 4. Methodology | 3-4 | 特征工程 + 模型选择 + 训练策略 |
| 5. Results | 2-3 | 模型对比 + 消融实验 + SHAP 分析 |
| 6. Discussion | 1-1.5 | 局限性、与论文方法的对比、改进方向 |
| 7. Conclusion | 0.5 | 主要贡献和发现 |
| References | 0.5-1 | 至少 7 篇论文 |

### Phase 7: 展示视频录制（4/12-14）
- [ ] 15 分钟视频
- [ ] 结构：问题动机(2min) → EDA 可视化(3min) → 方法(4min) → 结果(4min) → 总结(2min)

---

## 四、预期目标

| 指标 | 基线 | 预期保底 | 预期冲高 |
|------|------|---------|---------|
| Spearman ρ | 0.197 | 0.40-0.50 | 0.55+ |
| 提升倍数 | — | 2-2.5x | 2.8x+ |

**课程得分预估**：保底 12-13/15，冲高 14-15/15

---

## 五、关键文件路径

```
CS5483_Data_Project2-forCC_2/
├── 163-Predict parking violations/     # 数据文件（不入 Git）
│   ├── x_train_final_asAbTs5.csv       # 训练特征 (~607万行, 316MB)
│   ├── y_train_final_YYyFil7.csv       # 训练目标 (82MB)
│   └── x_test_final_fIrnA7Q.csv        # 测试特征 (105MB)
├── notebooks/                          # Jupyter Notebooks（主要代码）
│   ├── 01_eda.ipynb                    # EDA
│   ├── 02_feature_engineering.ipynb    # 特征工程
│   ├── 03_modeling.ipynb               # 建模
│   └── 04_evaluation.ipynb             # 评估
├── docs/
│   ├── project_plan.md                 # 本文档
│   ├── literature_review.md            # 论文综述
│   └── logs/                           # 每日开发日志
├── research_parking_violations/
│   ├── papers/                         # 7 篇相关论文 PDF
│   └── tutorial_part*.md               # 项目教程系列
├── background/                         # 课程指南
├── CLAUDE.md                           # 项目编码规范
├── environment.yml                     # Conda 环境配置
└── README.md                           # 项目说明
```

---

## 六、验证方案

1. **Phase 1 验证**：EDA notebook 可运行，图表完整
2. **Phase 2 验证**：特征工程后数据 shape 正确，无数据泄露（CV 分数合理，非异常高）
3. **Phase 3 验证**：5-fold CV Spearman 稳定，且 > baseline 0.197
4. **Phase 4 验证**：提交测试集预测到 challengedata.ens.fr，查看排行榜得分
5. **最终验证**：代码从头到尾可复现（notebook Restart & Run All）

---

## 七、参考论文

| # | 论文 | 来源 | 主要借鉴 |
|---|------|------|---------|
| 1 | Deep Learning for On-Street Parking Violation Prediction (Vo, 2025) | arXiv:2505.06818 | **同数据源**，sine 编码、6h 天气窗口、高斯平滑 |
| 2 | Predicting on-street parking violation rate using deep ResNN (Karantaglis+, 2022) | PRL Vol.163 | 消融实验方法、特征重要性排序 |
| 3 | Predicting spatiotemporal legality of parking (Gao+, 2019) | Annals of GIS | RF 多尺度最优、POI 特征、3h 时间自相关 |
| 4 | Short-term parking violations demand dynamic prediction (Liu & Chen, 2025) | Transportation | 通勤/午休高峰、建成环境时空异质性 |
| 5 | Spatio-temporal heterogeneity in street illegal parking (Sui+, 2025) | J Transport Geog | SHAP 交互分析、500m 网格、零膨胀处理 |
| 6 | LightGBM (Ke+, 2017) | NeurIPS | GOSS+EFB+直方图，首选模型 |
| 7 | XGBoost (Chen & Guestrin, 2016) | KDD | 正则化目标、二阶泰勒展开 |
