# CS5483 数据挖掘项目 -- Challenge #163 停车违章预测 资源汇总

## 一、挑战赛专属资源

### 1.1 官方挑战页面
- **[Predict parking violations! by Egis](https://challengedata.ens.fr/challenges/163)**
  - ENS ChallengeData 平台上的官方页面
  - 问题：利用 GPS 坐标、天气数据、时间特征预测不同城市区域的停车违章率 (invalid_ratio)
  - 评估指标：Spearman 相关系数（因为业务上关注的是区域排序，而非精确数值）
  - 基线：Random Forest (10 棵树) 达到 0.197
  - 训练集约 600 万行，测试集约 200 万行，10 个特征
  - 数据过滤说明：基线模型只使用了停车数 >= 45 的控制记录来提高数据质量
  - 2025 赛季获胜者：Christophe Leroux, Antoine Li, Martin Deldicque

### 1.2 直接相关论文（极可能基于同一数据集/同一公司 Egis）
- **[Deep Learning for On-Street Parking Violation Prediction (arXiv:2505.06818)](https://arxiv.org/abs/2505.06818)**
  - 作者：Vo Thien Nhan，2025 年 5 月
  - 使用深度学习（6 层残差网络，512-256-128-64-128-32 神经元）预测停车违章率
  - 提出了数据增强 + 平滑技术来处理缺失/噪声数据
  - 在希腊 Thessaloniki 的 390 万条扫描数据上验证
  - **高度相关**：问题设定几乎完全一致（间接预测、天气+时间+位置特征）

- **[Predicting on-street parking violation rate using deep residual neural networks (ScienceDirect, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002938)**
  - Pattern Recognition Letters 期刊论文
  - 使用深度残差神经网络预测街道停车违章率
  - 提出了处理噪声数据的数据增强和平滑方法

### 1.3 未找到的资源
- 暂未找到 Challenge #163 的公开解题博客、GitHub 仓库或论坛讨论
- 这可能因为该挑战仍在进行中（2025 赛季延长至 2026 年）

---

## 二、相关学术论文与技术

### 2.1 时空停车违章预测
- **[Short-term parking violations demand dynamic prediction (Transportation, 2025)](https://link.springer.com/article/10.1007/s11116-025-10632-7)**
  - 使用 MGTWR-GAT-ALSTM 混合模型（多尺度地理时间加权回归 + 图注意力网络 + 注意力 LSTM）
  - 考虑建成环境的时空异质性效应
  - **方法参考价值高**

- **[Spatio-temporal heterogeneity in street illegal parking (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S096669232500153X)**
  - 分析纽约街道非法停车的时空异质性
  - 使用贝叶斯层次模型

- **[Predicting the spatiotemporal legality of on-street parking using open data and ML (2019)](https://www.tandfonline.com/doi/full/10.1080/19475683.2019.1679882)**
  - 使用纽约开放数据预测停车合法性
  - 比较了四种空间分析尺度（点、街道、人口普查区、网格）
  - **发现 Random Forest 在所有尺度上表现最好**

- **[Semi-supervised learning for parking violation prediction using GCN](https://www.researchgate.net/publication/382943245_Semi-supervised_learning_for_on-street_parking_violation_prediction_using_graph_convolutional_networks)**
  - 使用图卷积网络进行半监督学习

### 2.2 Spearman 相关系数优化
- **[Differentiable Spearman in PyTorch - Numerai Forum](https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287)**
  - 详细讨论了如何直接优化 Spearman 相关系数
  - 推荐使用 `torchsort` 库实现可微分的 soft ranking
  - 实用建议：先用 MSE loss 预训练，再用 Spearman loss 微调
  - 正则化强度的权衡：值低 -> 排名更准确但梯度弱；值高 -> 梯度强但排名精度略降

- **[torchsort - Fast, differentiable sorting and ranking in PyTorch](https://github.com/teddykoker/torchsort)**
  - 纯 PyTorch 实现，支持 GPU
  - 基于 Google Research 的 Fast Differentiable Sorting 论文

- **[Fine-grained Correlation Loss for Regression (arXiv:2207.00347)](https://ar5iv.labs.arxiv.org/html/2207.00347)**
  - 提出细粒度相关性损失函数
  - 将排名学习转化为样本相似性关系学习

- **[Understanding The Metric: Spearman's Rho - Kaggle Notebook](https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-spearman-s-rho)**
  - Kaggle 上的教学 notebook，清晰解释 Spearman 相关系数

- **实用策略总结**：
  - 对于 GBDT 模型：直接优化 MSE/MAE，因为 Spearman 关注排序，而好的数值预测通常带来好的排序
  - 可以使用 LightGBM 的 LGBMRanker（LambdaRank 目标函数）来间接优化排序
  - 对于神经网络：可使用 torchsort 直接优化可微分的 Spearman

---

## 三、实用 ML 技术与教程

### 3.1 梯度提升树（XGBoost / LightGBM）
- **[NVIDIA: Kaggle Grandmasters Playbook - 7 Battle-Tested Modeling Techniques](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)**
  - Kaggle 大师总结的表格数据建模技巧
  - 包括交叉验证、特征工程、集成学习等

- **[Build XGBoost/LightGBM models on large datasets (Medium)](https://medium.com/data-science/build-xgboost-lightgbm-models-on-large-datasets-what-are-the-possible-solutions-bf882da2c27d)**
  - 针对大数据集的训练策略
  - 子采样参数：`subsample=0.8`, `colsample_bytree=0.8`
  - 使用 `hist` 树方法提高效率
  - LightGBM 的直方图算法天然更适合大数据

- **[Training XGBoost model by big data (Medium)](https://medium.com/@jwang.ml/training-xgboost-model-by-big-data-a5bcf5d54e19)**
  - 大数据集上训练 XGBoost 的实战经验

- **[A Simple and Fast Baseline for Tuning Large XGBoost Models (arXiv)](https://arxiv.org/abs/2111.06924)**
  - 使用均匀子采样加速超参数调优

- **[LightGBM: Enhancing Feature Engineering](https://www.numberanalytics.com/blog/lightgbm-feature-engineering)**
  - LightGBM 特征工程最佳实践

### 3.2 地理空间特征工程
- **[Geospatial Feature Engineering and Visualization - Kaggle](https://www.kaggle.com/code/camnugent/geospatial-feature-engineering-and-visualization)**
  - Kaggle notebook，包含代码示例
  - 距离特征、空间聚类、角度差异等

- **[Leveraging Geolocation Data for ML - Towards Data Science](https://towardsdatascience.com/leveraging-geolocation-data-for-machine-learning-essential-techniques-192ce3a969bc/)**
  - 经纬度数据的 ML 处理技巧
  - 坐标转换、3D 投影（sin/cos 变换）

- **[Geospatial Feature Engineering via Clustering (Medium)](https://medium.com/@eanunez85/geospatial-feature-engineering-139b4d8b4d4d)**
  - 使用聚类将坐标转化为分类特征
  - DBSCAN 和层次聚类通常比 K-means 更适合地理数据

- **[Uber H3 Hexagonal Grid System](https://github.com/uber/h3-py)** / **[H3 官方文档](https://h3geo.org/)**
  - 六边形层次空间索引系统
  - 将 GPS 坐标映射到六边形网格单元
  - 统一距离属性，适合空间聚合和特征生成
  - **教程**: [Analytics Vidhya H3 Guide](https://www.analyticsvidhya.com/blog/2025/03/ubers-h3-for-spatial-indexing/)
  - **教程**: [Uber H3 for Data Analysis with Python (Medium)](https://medium.com/data-science/uber-h3-for-data-analysis-with-python-1e54acdcc908)

- **[Spatial Feature Engineering - Geographic Data Science with Python](https://geographicdata.science/book/notebooks/12_feature_engineering.html)**
  - 系统性的空间特征工程教程

- **[Tree-Boosting for Spatial Data - Towards Data Science](https://towardsdatascience.com/tree-boosting-for-spatial-data-789145d6d97d/)**
  - 讨论是否将坐标作为普通特征 vs 使用 GPBoost 处理空间效应

- **[GPBoost: Combining tree-boosting with Gaussian process (GitHub)](https://github.com/fabsig/GPBoost)**
  - 基于 LightGBM 的扩展，结合高斯过程随机效应
  - 能更好地建模空间相关性

### 3.3 时间特征工程
- **[Cyclical Encoding for Time Series Features - Towards Data Science](https://towardsdatascience.com/cyclical-encoding-an-alternative-to-one-hot-encoding-for-time-series-features-4db46248ebba/)**
  - 用 sin/cos 编码周期性特征（小时、星期、月份）
  - 优点：每个周期特征只需 2 列（vs one-hot 的 24/7/12 列）
  - 注意：对神经网络效果好，对树模型需要额外调优

- **[Encoding Cyclical Features for Deep Learning - Kaggle Notebook](https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning)**
  - Kaggle 上的实战 notebook

- **[Feature-Engine CyclicalFeatures](https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html)**
  - Python 库，自动化周期性特征编码

- **[Skforecast: Cyclical Features in Time Series](https://skforecast.org/latest/faq/cyclical-features-time-series.html)**
  - 时间序列中周期性特征的处理指南

### 3.4 Target Encoding（位置特征编码）
- **[scikit-learn TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)**
  - 官方文档，内置平滑防止过拟合

- **[Encoding Categorical Variables: Target Encoding - TDS](https://towardsdatascience.com/encoding-categorical-variables-a-deep-dive-into-target-encoding-2862217c2753/)**
  - 详细教程，包含平滑处理和防止目标泄漏的技巧

### 3.5 地理可视化
- **[Creating Geospatial Heatmaps with Plotly and Folium - TDS](https://towardsdatascience.com/creating-geospatial-heatmaps-with-pythons-plotly-and-folium-libraries-4159e98a1ae8/)**
  - 同时覆盖 Folium 和 Plotly 的热力图教程

- **[CloudThat: Heatmaps with Plotly and Folium](https://www.cloudthat.com/resources/blog/visualizing-geospatial-data-creating-heatmaps-with-plotly-and-folium-in-python)**
  - 另一个详细教程

- **[GeoPandas: Plotting with Folium](https://geopandas.org/en/stable/gallery/plotting_with_folium.html)**
  - GeoPandas 官方 Folium 集成教程

---

## 四、类似 Kaggle 竞赛与项目

### 4.1 直接相关竞赛
- **[Predict Parking Demand - Kaggle](https://www.kaggle.com/competitions/predict-parking-demand)**
  - IST TU Lisbon 的停车需求预测竞赛
  - 与本挑战最相似的 Kaggle 竞赛

### 4.2 时空预测类竞赛
- **[Predict Future Sales - Kaggle](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)**
  - 经典的时空预测竞赛（店铺 x 商品 x 时间）
  - [Top 7% 解题方案 (Medium)](https://shivang-ahd.medium.com/predict-future-sales-kaggle-competition-68061ef7abb5)
  - 关键技巧：滞后特征 (lag features) 是最重要的特征

- **[Tabular Playground Series (Kaggle)](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022)**
  - 月度表格数据竞赛系列，适合练习
  - 获胜方案常用：K-fold 交叉验证、多模型 stacking、特征工程

### 4.3 相关项目与 Notebook
- **[NYC Parking Violations Analysis (GitHub)](https://github.com/nickdcox/learn-nyc-parking-violations)**
  - 用纽约停车违章数据学习 Python 数据分析

- **[Parking Lot Prediction - Kaggle Notebook](https://www.kaggle.com/code/denistopallaj/parking-lot-prediction)**
  - 停车场占用率预测实战

- **[Traffic Violations Prediction (GitHub)](https://github.com/tadowney/ticket_analysis)**
  - 使用多种 ML 模型预测交通违章

---

## 五、针对本挑战的实用建议总结

### 特征工程思路
1. **空间特征**：
   - 使用 H3 六边形网格将 GPS 坐标映射到空间单元
   - 对空间单元做 Target Encoding（每个区域的历史违章率）
   - 空间聚类（DBSCAN/K-means）创建区域类别
   - GPS 坐标的 sin/cos 变换

2. **时间特征**：
   - 周期性编码：hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
   - 是否为上下班高峰、是否为周末
   - 时间段分桶（早高峰/午间/晚高峰/夜间）

3. **天气交互特征**：
   - 温度 x 小时交互（天气对违章的影响可能因时间而异）
   - 极端天气标志（大雨、降雪、大风）
   - 降水量分桶

4. **交叉特征**：
   - 区域 x 小时的历史统计量
   - 区域 x 星期几的历史统计量
   - 天气条件 x 区域

### 模型选择
- **首选 LightGBM**：600 万行数据上训练速度快，直方图算法内存效率高
- 考虑使用 **LGBMRanker** 直接优化排序（因为评估指标是 Spearman）
- 集成多个模型（LightGBM + XGBoost + CatBoost）可能进一步提升
- 如果时间允许，尝试 **GPBoost** 来建模空间效应

### 训练策略
- 使用 `subsample` 和 `colsample_bytree` 参数加速训练
- K-fold 交叉验证，确保验证集的 Spearman 分数稳定
- 使用 Optuna 进行超参数搜索
