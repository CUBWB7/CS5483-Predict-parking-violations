# CS5483 Project 2 — #163 停车违章预测 项目实施方案

> 选题：ChallengeData #163 "Predict Parking Violations" by Egis
> 生成日期：2026-04-02
> 数据已下载至 `163-Predict parking violations/` 目录

---

## 一、数据分析关键发现

### 数据规模
- 训练集：~607 万行，10 个特征
- 测试集：~203 万行，同样 10 个特征
- 目标变量：`invalid_ratio`（0-1 连续值，违规率）
- 评估指标：**Spearman 相关系数**（衡量排序一致性）
- 官方基线：0.197（Random Forest, 10 棵树）

### 特征清单

| 特征 | 类型 | 说明 | 与目标相关性 (Spearman ρ) |
|------|------|------|--------------------------|
| `total_count` | int | 检查的停车数量 | **-0.297**（最强） |
| `longitude_scaled` | float | 缩放经度 (0.98-1.0) | 0.061 |
| `latitude_scaled` | float | 缩放纬度 (0.99-1.0) | 0.080 |
| `Precipitations` | float | 降水量 | 0.003（极弱） |
| `HauteurNeige` | float | 积雪（实质是二元：0/1） | ~0（无关） |
| `Temperature` | float | 温度 | 0.026 |
| `ForceVent` | float | 风力 | -0.010 |
| `day_of_week` | int | 星期几 (1-6，无周日) | 0.004（极弱） |
| `month_of_year` | int | 月份 (1-12) | **-0.091** |
| `hour` | int | 小时 (6-19) | 0.010 |

### 关键洞察

1. **`total_count` 是最强预测因子**：检查车辆数越多 → 违规率越低/越稳定
2. **小样本噪声严重**：`total_count=1` 时，违规率只能是 0 或 1（均值 0.664）；`total_count≥30` 时均值 ~0.30，方差小得多
3. **月份有季节性**：违规率随月份有变化（ρ=-0.091）
4. **空间位置有模式**：经纬度虽单独相关性不高，但组合起来可定位到具体区域
5. **天气特征单独作用弱**：但可能通过交互效应发挥作用
6. **缺失值少**：仅 `HauteurNeige`（~2.7%）和 `ForceVent`（~0.1%）有缺失
7. **数据只覆盖工作日白天**：周一至周六、6:00-19:00

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

## 二、需要学习的知识

### A. 核心必学（项目依赖）

#### 1. Spearman 相关系数（评估指标）
- **是什么**：衡量预测值与真实值的**排序一致性**，不关心精确数值，只看排名
- **为什么重要**：模型不需要精确预测违规率的绝对值，只要能正确排序（哪些时段/地点违规更多）就能得高分
- **学习资源**：
  - [Understanding Spearman's Rho - Kaggle](https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-spearman-s-rho)
  - scipy.stats.spearmanr 文档

#### 2. LightGBM / XGBoost（主力模型）
- **是什么**：梯度提升决策树 (GBDT)，表格数据上最强的传统 ML 方法
- **为什么选它**：600 万行数据，LightGBM 用直方图算法训练快、内存省
- **学习资源**：
  - [LightGBM 官方文档](https://lightgbm.readthedocs.io/en/latest/)
  - [XGBoost 教程](https://xgboost.readthedocs.io/en/latest/tutorials/index.html)
  - [NVIDIA: Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) — Kaggle 大师的建模技巧

#### 3. 特征工程（表格数据核心竞争力）
- **空间特征**：GPS 坐标网格化/聚类、区域 ID、区域级统计量
- **时间特征**：周期编码（sin/cos）、时段分箱
- **交叉特征**：区域×时段、区域×星期、天气×时间
- **聚合统计**：各区域/时段的历史均值、中位数等（Target Encoding 思路）
- **学习资源**：
  - [Geospatial Feature Engineering - Kaggle](https://www.kaggle.com/code/camnugent/geospatial-feature-engineering-and-visualization)
  - [Uber H3 六边形网格](https://h3geo.org/) — 空间离散化的现代方案
  - [Cyclical Encoding 教程](https://towardsdatascience.com/cyclical-encoding-an-alternative-to-one-hot-encoding-for-time-series-features-4db46248ebba/)

#### 4. 交叉验证策略
- **为什么重要**：600 万行数据需要合理的验证方案来评估模型
- 推荐 5-fold CV 或按时间/空间分组的 GroupKFold
- **学习资源**：scikit-learn cross_validation 文档

### B. 进阶提升（冲高分用）

#### 5. 模型集成 (Ensemble)
- Stacking/Blending 多个模型的预测结果
- **学习资源**：[Kaggle Ensemble Guide](https://mlwave.com/kaggle-ensembling-guide/)

#### 6. 地理可视化（Presentation 评分加分项）
- Folium 热力图 + Plotly 交互式地图
- **学习资源**：
  - [Folium + GeoPandas 教程](https://geopandas.org/en/stable/gallery/plotting_with_folium.html)
  - [Creating Geospatial Heatmaps - TDS](https://towardsdatascience.com/creating-geospatial-heatmaps-with-pythons-plotly-and-folium-libraries-4159e98a1ae8/)

#### 7. 深度学习方法（可选）
- 有一篇**问题设定几乎完全一致**的论文：[Deep Learning for On-Street Parking Violation Prediction (arXiv:2505.06818)](https://arxiv.org/abs/2505.06818)，使用 6 层残差网络
- [Predicting on-street parking violation rate using deep residual neural networks (Pattern Recognition Letters, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002938)

---

## 三、推荐阅读的论文

| # | 论文 | 来源 | 用途 |
|---|------|------|------|
| 1 | Deep Learning for On-Street Parking Violation Prediction (2025) | arXiv:2505.06818 | **最核心参考**，问题设定完全一致 |
| 2 | Short-term parking violations demand dynamic prediction (2025) | Transportation | 时空异质性建模方法 |
| 3 | Predicting the spatiotemporal legality of on-street parking (2019) | Int. J. Geogr. Inf. Sci. | RF 在多尺度上效果最好 |
| 4 | Spatio-temporal heterogeneity in street illegal parking (2025) | ScienceDirect | 纽约违章时空模式分析 |
| 5 | Predicting on-street parking violation rate using deep residual neural networks (2022) | Pattern Recognition Letters | 深度残差网络方法 |
| 6 | LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al., 2017) | NeurIPS | 主力模型原论文 |
| 7 | XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016) | KDD | 对比模型原论文 |

---

## 四、项目开展流程

### Phase 0: 环境与代码框架搭建（0.5 天）
- [ ] 创建项目 Jupyter Notebook 结构
- [ ] 安装依赖：lightgbm, xgboost, catboost, h3, folium, plotly, scipy
- [ ] 设置随机种子和数据读取管道（分块读取 600 万行）
- [ ] 建立可复现的代码框架

### Phase 1: EDA 探索性数据分析（1 天）
- [ ] 各特征分布可视化
- [ ] 目标变量分布分析（U 型分布，大量 0 和 1）
- [ ] 特征与目标的相关性分析
- [ ] 空间分布热力图（还原坐标后画地图）
- [ ] 时间模式分析（按小时、星期、月份）
- [ ] `total_count` 与目标噪声的关系
- **产出**：EDA notebook + 关键图表（复用到报告和展示中）

### Phase 2: 数据预处理与特征工程（2 天）— 最重要的阶段

#### 2.1 基础预处理
- [ ] 缺失值填充（`HauteurNeige` 用 0，`ForceVent` 用中位数）
- [ ] 数据类型优化（减少内存占用）

#### 2.2 空间特征（重点！）
- [ ] 坐标还原或在缩放空间操作
- [ ] 网格化：将坐标区域划分为网格（等距网格或 H3 六边形）
- [ ] 区域聚类：用 DBSCAN/KMeans 对坐标聚类，生成区域 ID
- [ ] 区域统计特征：每个区域的历史平均违规率、平均 total_count 等

#### 2.3 时间特征
- [ ] 周期编码：hour, day_of_week, month 的 sin/cos 变换
- [ ] 时段分箱：早高峰/上午/午休/下午/晚
- [ ] 是否月初/月末

#### 2.4 交叉/交互特征
- [ ] 区域 × 时段的平均违规率（Target Encoding，防泄露）
- [ ] 区域 × 星期几
- [ ] 天气 × 时段
- [ ] total_count 分箱 × 区域

#### 2.5 天气特征增强
- [ ] 是否下雨（二元）、是否极端温度
- [ ] 温度离散化

### Phase 3: 建模（2 天）

#### 3.1 Baseline 模型
- [ ] Random Forest（10 棵树）— 复现官方 baseline (Spearman ~0.197)
- [ ] 线性回归 — 最简单 baseline

#### 3.2 主力模型
- [ ] **LightGBM** — 首选，参数调优 (num_leaves, learning_rate, feature_fraction 等)
- [ ] **XGBoost** — 第二模型，hist 树方法
- [ ] **CatBoost** — 第三模型

#### 3.3 进阶模型（可选）
- [ ] LGBMRanker — 直接优化排序指标
- [ ] 深度残差网络 — 参考 arXiv:2505.06818
- [ ] GPBoost — 高斯过程 + LightGBM

#### 3.4 模型集成
- [ ] 加权平均 LightGBM + XGBoost + CatBoost
- [ ] 或 Stacking（第二层用简单线性模型）

### Phase 4: 评估与分析（1 天）
- [ ] **主指标**：Spearman 相关系数（5-fold CV + 平台提交）
- [ ] 辅助指标：MAE, MSE, R²
- [ ] 各模型对比表格
- [ ] 特征重要性分析（feature importance + SHAP）
- [ ] 预测结果的空间/时间误差分析
- [ ] 与 baseline (0.197) 的对比

### Phase 5: 可视化与展示准备（1 天）
- [ ] 违规率空间热力图
- [ ] 时间趋势图（按小时/月份）
- [ ] 模型性能对比图
- [ ] 特征重要性图 / SHAP 图
- [ ] 预测 vs 真实散点图

### Phase 6: 报告撰写（2 天）
- [ ] Abstract → Introduction → Related Work → Data & Preprocessing → Methodology → Results → Conclusion
- [ ] ~15 页，充分引用数据来源和论文
- [ ] 代码附录或 Git 仓库链接

### Phase 7: 展示视频录制（1 天）
- [ ] 15 分钟视频
- [ ] 重点：问题动机、EDA 可视化、方法、结果对比

---

## 五、预期目标

| 指标 | 基线 | 预期保底 | 预期冲高 |
|------|------|---------|---------|
| Spearman ρ | 0.197 | 0.40-0.50 | 0.55+ |
| 提升倍数 | — | 2-2.5x | 2.8x+ |

**课程得分预估**：保底 12-13/15，冲高 14-15/15

---

## 六、关键文件路径

```
CS5483_Data_Project2-forCC_2/
├── 163-Predict parking violations/     # 数据文件
│   ├── x_train_final_asAbTs5.csv       # 训练特征 (~607万行)
│   ├── y_train_final_YYyFil7.csv       # 训练目标
│   └── x_test_final_fIrnA7Q.csv        # 测试特征 (~203万行)
├── docs/
│   ├── topic_selection.md              # 选题推荐文档
│   ├── project_plan.md                 # 本文档：项目实施方案
│   ├── project2_guide.md               # 课程项目指南
│   └── project_submit.md               # 提交要求
└── research_parking_violations/
    └── papers/                         # 相关论文
```

---

## 七、验证方案

1. **Phase 1 验证**：EDA notebook 可运行，图表完整
2. **Phase 2 验证**：特征工程后的数据 shape 正确，无数据泄露
3. **Phase 3 验证**：5-fold CV Spearman 稳定，且 > baseline 0.197
4. **Phase 4 验证**：提交测试集预测到 challengedata.ens.fr，查看排行榜得分
5. **最终验证**：代码从头到尾可复现（notebook restart & run all）
