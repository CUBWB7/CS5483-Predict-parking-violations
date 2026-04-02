# Project: CS5483 Predict Parking Violations

## 项目背景
- ChallengeData #163 停车违章预测，由 Egis 公司发起
- 数据来自希腊塞萨洛尼基 THESi 路边停车系统
- 训练集 ~607 万行，10 特征，目标变量 `invalid_ratio`（0-1）
- 评估指标：**Spearman 相关系数**（排序一致性，非数值精度）
- 官方基线：0.197（RF 10 棵树）
- 目标：Spearman ≥ 0.40（保底），冲 0.55+

## 截止日期
- 展示视频：2026-04-15（15 分钟）
- 报告：2026-04-23（~15 页）

## 技术栈
- Python (Anaconda)，Jupyter Notebook
- 主力模型：LightGBM、XGBoost、CatBoost
- 关键库：pandas, numpy, scipy, lightgbm, xgboost, catboost, scikit-learn, shap, matplotlib, seaborn, folium, plotly
- Conda 环境名：`parking`

## 数据文件
- 位于 `163-Predict parking violations/` 目录
- `x_train_final_asAbTs5.csv` — 训练特征
- `y_train_final_YYyFil7.csv` — 训练目标
- `x_test_final_fIrnA7Q.csv` — 测试特征
- **数据文件不纳入 Git**（太大），已在 .gitignore 中排除

## 特征清单（原始 10 个）
| 特征 | 类型 | Spearman ρ | 备注 |
|------|------|-----------|------|
| total_count | int | -0.297 | 最强预测因子 |
| longitude_scaled | float | 0.061 | 缩放经度 |
| latitude_scaled | float | 0.080 | 缩放纬度 |
| Precipitations | float | 0.003 | 降水量 |
| HauteurNeige | float | ~0 | 积雪，本质二元 |
| Temperature | float | 0.026 | 温度 |
| ForceVent | float | -0.010 | 风力 |
| day_of_week | int(1-6) | 0.004 | 无周日 |
| month_of_year | int(1-12) | -0.091 | 季节性 |
| hour | int(6-19) | 0.010 | 白天执法时段 |

## 关键技术决策
1. **特征工程优先级**：区域 Target Encoding > 坐标网格化 > 周期编码 > 交叉特征 > 天气增强
2. **Target Encoding 必须用 K-Fold 防泄露**
3. **早停指标用 Spearman**，不是 RMSE
4. **total_count=1 的样本噪声极大**（违规率只能 0 或 1），需特殊处理
5. **缺失值**：HauteurNeige 填 0，ForceVent 填中位数

## 项目结构
```
├── 163-Predict parking violations/   # 数据（不入 Git）
├── background/                       # 课程指南
├── docs/
│   ├── project_plan.md               # 项目概览（高层时间线和策略）
│   ├── literature_review.md          # 7 篇论文综述
│   ├── plan/detailed_plan.md         # 详细实施计划（操作指南，按此执行）
│   └── logs/                         # 每日开发日志
├── research_parking_violations/      # 研究资料、教程、论文 PDF
├── notebooks/                        # Jupyter Notebooks（主要代码）
├── environment.yml                   # Conda 环境
└── CLAUDE.md                         # 本文件
```

## 编码规范
- Notebook 中不要出现中文，因为这是英文课程的项目，要提交的文件的语言应当是英文
- 固定随机种子 `SEED = 42`
- 代码要能 Restart & Run All 复现
- 训练集/测试集特征工程流程必须一致
- 不要在 notebook 中留 debug 输出

## 参考论文（已下载到 research_parking_violations/papers/）
1. Vo (2025) — 深度学习停车违章预测，与本项目数据源一致
2. Karantaglis et al. (2022) — 深度残差网络预测违章率
3. Gao et al. (2019) — RF 多尺度停车合法性预测
4. Liu & Chen (2025) — 时空异质性动态预测
5. Sui et al. (2025) — NYC 违章时空模式
6. Ke et al. (2017) — LightGBM 原论文
7. Chen & Guestrin (2016) — XGBoost 原论文
