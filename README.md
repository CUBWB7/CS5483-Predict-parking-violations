# CS5483 Project 2 — Predict Parking Violations

> ChallengeData #163 "Predict Parking Violations" by Egis

## 项目概述

利用 GPS 坐标、天气数据和时间特征，预测城市不同区域的停车违章率（invalid_ratio）。

- **数据来源**：[ChallengeData #163](https://challengedata.ens.fr/challenges/163)
- **评估指标**：Spearman 相关系数
- **官方基线**：0.197（Random Forest, 10 棵树）

## 数据下载

数据文件较大（~500MB），未包含在仓库中。请从 ChallengeData 下载后放到 `163-Predict parking violations/` 目录：

1. 访问 https://challengedata.ens.fr/challenges/163
2. 下载以下文件并放入 `163-Predict parking violations/` 目录：
   - `x_train_final_asAbTs5.csv` — 训练特征（~607 万行）
   - `y_train_final_YYyFil7.csv` — 训练目标
   - `x_test_final_fIrnA7Q.csv` — 测试特征（~203 万行）

## 项目结构

```
├── 163-Predict parking violations/   # 数据文件（需手动下载）
├── background/                       # 课程指南与参考资料
├── docs/                             # 项目文档
│   ├── project_plan.md              # 项目实施方案
│   └── literature_review.md         # 论文综述
├── research_parking_violations/      # 研究资料与教程
│   ├── papers/                      # 相关论文 PDF
│   └── tutorial_part*.md            # 项目教程系列
├── environment.yml                   # Conda 环境配置
└── README.md
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate parking
```

## 团队成员

- [待填写]
