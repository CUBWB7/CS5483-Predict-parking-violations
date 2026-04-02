# CS5483 停车违章预测 — 详细实施计划

> 基于 `docs/project_plan.md` 的可执行操作指南
> 创建日期：2026-04-02 | 项目截止：视频 4/15，报告 4/23
> 本文档定位：**操作指南**（告诉你做什么、怎么做）；`project_plan.md` 是概览，`tutorial_part*.md` 是教程

---

## 一、修订后的时间线

```
4/2  ✅ Phase 0: 环境搭建（已完成）
4/2  ✅ Phase 1: EDA（已完成，产出 01_eda.ipynb + 7 张图表）
4/4-6   Phase 2: 特征工程（3 天，⚠️ 最关键阶段，分 Tier 1/2/3）
4/7-9   Phase 3: 建模（3 天，含首次平台提交）
4/10    Phase 4: 评估分析（1 天，消融实验 + SHAP）
4/11    Phase 5: 可视化（1 天，生成报告/视频共用图表）
4/12-14 Phase 7: 视频录制（3 天）
⚠️ 4/15 视频提交截止
4/15-22 Phase 6: 报告撰写（8 天）
⚠️ 4/23 报告提交截止
```

### 关键决策点（No-Go Gates）

| 时间点 | 检查项 | 未通过的应对 |
|--------|--------|-------------|
| **4/5 EOD** | Phase 2 Tier 1 是否完成？ | 是 → 继续 Tier 2；否 → 集中精力完成 Tier 1，放弃 Tier 2/3 |
| **4/6 EOD** | 特征工程是否可用？ | 是 → 正常进入建模；否 → 仅用 Tier 1 特征进入建模 |
| **4/9 EOD** | 是否有 Spearman > 0.30 的模型？ | 是 → 正常评估；否 → 回查数据泄露/特征 bug |
| **4/11 EOD** | 图表是否足够支撑视频？ | 是 → 开始录制；否 → 用现有图表精简视频结构 |

---

## 二、可行性评估摘要

### 现有计划的优势
- ✅ 论文指导的特征工程优先级非常合理
- ✅ LightGBM 作为首选模型适合 6M 行数据
- ✅ Spearman 早停指标与比赛一致
- ✅ K-Fold Target Encoding 防泄露策略正确

### 识别的风险与改进

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|---------|
| Phase 2 超时（最大风险） | 高 | 高 | Tier 1/2/3 优先级系统 + No-Go Gate |
| 6M 行内存溢出 | 高 | 高 | dtype 优化 + parquet 中间保存 + gc.collect() |
| CV 分数与平台分数不一致 | 中 | 高 | Phase 3 Day 1 就提交 baseline 建立反馈环 |
| K-Fold Target Encoding 实现错误 | 中 | 高 | 检查编码后特征与目标的相关性是否合理（~0.3-0.5） |
| SHAP 在 6M 行上太慢 | 中 | 低 | 采样 5K-10K 行计算 |
| 视频录制拖延 | 中 | 高 | 先写脚本大纲再录；一次过，不求完美 |

---

## 三、Phase 1 — EDA 探索性数据分析（4/3）

### 目标
完成 `notebooks/01_eda.ipynb`，产出 6-8 张可复用于报告和视频的图表。

### 检查点
- [ ] Notebook 可 Restart & Run All
- [ ] 6-8 张图表完整
- [ ] 关键发现用 Markdown cell 记录

### 步骤

#### Step 1: 数据加载（内存高效方式）

```python
import pandas as pd
import numpy as np

SEED = 42
DATA_DIR = '163-Predict parking violations/'

# 指定 dtype 减少内存（原始 float64 → float32，int64 → int16/int32）
dtype_dict = {
    'total_count': 'int32',
    'longitude_scaled': 'float32',
    'latitude_scaled': 'float32',
    'Precipitations': 'float32',
    'HauteurNeige': 'float32',
    'Temperature': 'float32',
    'ForceVent': 'float32',
    'day_of_week': 'int8',
    'month_of_year': 'int8',
    'hour': 'int8',
}

x_train = pd.read_csv(f'{DATA_DIR}x_train_final_asAbTs5.csv', dtype=dtype_dict)
y_train = pd.read_csv(f'{DATA_DIR}y_train_final_YYyFil7.csv')
x_test = pd.read_csv(f'{DATA_DIR}x_test_final_fIrnA7Q.csv', dtype=dtype_dict)

train_df = x_train.copy()
train_df['invalid_ratio'] = y_train['invalid_ratio'].astype('float32')
```

> **内存估算**：优化后 train_df 约 600MB（vs 原始 ~2.4GB），16GB RAM 完全够用。

#### Step 2: 基本统计信息
- `train_df.info()`, `train_df.describe()`
- `train_df.isnull().sum()` — 确认缺失值（HauteurNeige ~2.7%, ForceVent ~0.1%）
- `train_df.shape`, `x_test.shape` — 确认行数

#### Step 3: 目标变量分布（重要图表 ⭐）
- 直方图展示 `invalid_ratio` 的 U 型分布
- 分别标注 =0（15.86%）和 =1（26.74%）的占比
- **关键故事**：为什么是 U 型？因为 total_count=1 时只能取 0 或 1

#### Step 4: total_count 与违规率关系（重要图表 ⭐）
- 按 total_count 分组的违规率箱线图（分组：1, 2-3, 4-10, 11-30, 31+）
- 展示小样本噪声大、大样本噪声小的规律
- 计算并展示每组的样本数、均值、标准差

#### Step 5: 特征分布
- 各特征的直方图/柱状图（2×5 子图排列）
- 特别关注：hour（6-19 均匀分布？）、month（是否有缺失月份？）、day_of_week（1-6）

#### Step 6: 特征相关性
- Spearman 相关系数热力图（用 `scipy.stats.spearmanr` 或 `train_df.corr(method='spearman')`）
- 每个特征与 `invalid_ratio` 的单变量 Spearman ρ 条形图

#### Step 7: 空间分布（重要图表 ⭐）
- 经纬度散点图，颜色 = 平均违规率（采样 50K-100K 点避免过密）
- 可选：用 hexbin 热力图替代散点图，更清晰

#### Step 8: 时间模式（重要图表 ⭐）
- 三张子图：按 hour / day_of_week / month_of_year 的平均违规率折线图
- 标注高峰时段（预期 12-14 点午休高峰、月份冬夏差异）

#### Step 9: 缺失值可视化
- 柱状图展示各特征的缺失率
- 确认处理策略：HauteurNeige → 填 0，ForceVent → 填中位数

### EDA 关键发现模板（用 Markdown cell 总结）

```markdown
## EDA 关键发现
1. **目标变量 U 型分布**：...% 为 0，...% 为 1，根本原因是小 total_count
2. **total_count 是最强预测因子**：Spearman ρ = -0.297
3. **空间位置有聚类模式**：某些区域违规率显著高于其他区域
4. **时间有周期性**：月份 ρ = -0.091，小时有午休高峰
5. **天气单独影响弱**：降水/温度/风力的单变量相关性 < 0.03
6. **缺失值极少**：仅 HauteurNeige(2.7%) 和 ForceVent(0.1%)
```

---

## 四、Phase 2 — 特征工程（4/4-6）⚠️ 最关键阶段

### 目标
构建特征工程管道，产出 `notebooks/02_feature_engineering.ipynb`，将训练集和测试集的特征从 10 个扩展到 20-30 个。

### 检查点
- [ ] train/test 特征 parquet 已保存
- [ ] encoding maps 已保存（用于测试集）
- [ ] 无 NaN 残留
- [ ] 新特征与目标的 Spearman ρ 合理（区域 TE 约 0.3-0.5，不应超过 0.7——超过说明有泄露）

### 总体原则
1. **训练集和测试集特征工程流程必须一致**
2. Target Encoding 必须用 K-Fold（训练集）/ 全量训练集统计（测试集）
3. 每完成一个 Tier，保存一次 parquet 检查点
4. 用 `gc.collect()` 及时释放中间变量

---

### Tier 1：必做项（Day 1，4/4）

完成 Tier 1 就足以让模型显著超越 baseline。

#### T1.1 缺失值填充 + dtype 优化

```python
# 缺失值填充
train_df['HauteurNeige'] = train_df['HauteurNeige'].fillna(0)
train_df['ForceVent'] = train_df['ForceVent'].fillna(train_df['ForceVent'].median())

# 测试集也一样（用训练集的中位数！）
forcevent_median = train_df['ForceVent'].median()  # 保存这个值
test_df['HauteurNeige'] = test_df['HauteurNeige'].fillna(0)
test_df['ForceVent'] = test_df['ForceVent'].fillna(forcevent_median)
```

#### T1.2 total_count 变换

```python
# 对数变换：压缩极端值，让模型更好学习
train_df['log_total_count'] = np.log1p(train_df['total_count']).astype('float32')

# 分箱：将连续值离散化，匹配违规率的阶梯式变化
bins = [0, 1, 3, 10, 30, np.inf]
labels = [0, 1, 2, 3, 4]  # 用整数编码
train_df['count_bin'] = pd.cut(train_df['total_count'], bins=bins, labels=labels).astype('int8')
```

#### T1.3 Sin/Cos 周期编码

```python
# 小时周期（周期=24，虽然数据只有 6-19，但 24 保持语义正确）
train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24).astype('float32')
train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24).astype('float32')

# 星期周期（周期=7）
train_df['dow_sin'] = np.sin(2 * np.pi * train_df['day_of_week'] / 7).astype('float32')
train_df['dow_cos'] = np.cos(2 * np.pi * train_df['day_of_week'] / 7).astype('float32')

# 月份周期（周期=12）
train_df['month_sin'] = np.sin(2 * np.pi * (train_df['month_of_year'] - 1) / 12).astype('float32')
train_df['month_cos'] = np.cos(2 * np.pi * (train_df['month_of_year'] - 1) / 12).astype('float32')
```

#### T1.4 坐标网格化 + 区域 Target Encoding（最关键特征 ⭐⭐⭐）

```python
# --- 网格化 ---
# 先观察坐标范围，选择合适的 grid_size
# longitude_scaled 范围 ~0.98-1.00，latitude_scaled 范围 ~0.99-1.00
# grid_size = 0.001 大约对应城市中的 ~100m 网格
GRID_SIZE = 0.001  # 可能需要实验调整

train_df['grid_lon'] = (train_df['longitude_scaled'] / GRID_SIZE).astype('int32')
train_df['grid_lat'] = (train_df['latitude_scaled'] / GRID_SIZE).astype('int32')
# 用整数组合作为 grid_id（比字符串拼接更快更省内存）
train_df['grid_id'] = train_df['grid_lon'] * 10000 + train_df['grid_lat']
```

```python
# --- K-Fold Target Encoding（防泄露的核心） ---
from sklearn.model_selection import KFold

def kfold_target_encode(train_df, test_df, col, target, n_splits=5, smooth=30):
    """
    K-Fold Target Encoding：
    - 训练集：每个 fold 只用其他 fold 的统计来编码当前 fold
    - 测试集：用全部训练集的统计来编码
    - smooth 参数：样本少的类别向全局均值收缩（贝叶斯平滑）

    返回: train_encoded, test_encoded, encoding_map
    """
    global_mean = train_df[target].mean()
    train_encoded = pd.Series(np.nan, index=train_df.index, dtype='float32')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train_df):
        # 用训练折的数据计算统计
        fold_stats = train_df.iloc[train_idx].groupby(col)[target].agg(['mean', 'count'])
        # 贝叶斯平滑
        smoothed = (fold_stats['count'] * fold_stats['mean'] + smooth * global_mean) / (fold_stats['count'] + smooth)
        # 用这个统计编码验证折
        train_encoded.iloc[val_idx] = train_df.iloc[val_idx][col].map(smoothed)

    # 填充未映射到的值（极少见）
    train_encoded = train_encoded.fillna(global_mean).astype('float32')

    # 测试集：用全部训练集统计
    full_stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    full_smoothed = (full_stats['count'] * full_stats['mean'] + smooth * global_mean) / (full_stats['count'] + smooth)
    test_encoded = test_df[col].map(full_smoothed).fillna(global_mean).astype('float32')

    # 保存 encoding map 用于复现
    encoding_map = full_smoothed.to_dict()

    return train_encoded, test_encoded, encoding_map

# 使用
train_df['grid_te'], test_df['grid_te'], grid_encoding = kfold_target_encode(
    train_df, test_df, 'grid_id', 'invalid_ratio'
)
```

> **验证**：`train_df['grid_te']` 与 `invalid_ratio` 的 Spearman ρ 应在 0.3-0.5 之间。如果 >0.7 → 数据泄露！如果 <0.1 → grid_size 太大或实现有 bug。

#### T1.5 保存 Tier 1 检查点

```python
import pickle, gc

# 保存 encoding maps
encoding_maps = {'grid': grid_encoding, 'forcevent_median': forcevent_median, 'grid_size': GRID_SIZE}
with open('data/encoding_maps_tier1.pkl', 'wb') as f:
    pickle.dump(encoding_maps, f)

# 保存特征 DataFrame
train_df.to_parquet('data/train_features_tier1.parquet', index=False)
test_df.to_parquet('data/test_features_tier1.parquet', index=False)

gc.collect()
```

> **Tier 1 完成后预期**：10 原始特征 + log_total_count + count_bin + 6 周期编码 + grid_id + grid_te = **约 20 个特征**。仅此就应让模型大幅超越 baseline。

---

### Tier 2：应做项（Day 2，4/5）

如果 Tier 1 按时完成，继续 Tier 2 进一步提升。

#### T2.1 时段分箱

```python
# 基于 Paper 4 发现的通勤/午休高峰
def hour_to_period(hour):
    if hour < 7: return 0    # 早晨（数据中极少）
    elif hour < 9: return 1  # 早高峰
    elif hour < 12: return 2 # 上午
    elif hour < 14: return 3 # 午休（高峰）
    elif hour < 17: return 4 # 下午
    else: return 5           # 傍晚

# 用 np.select 向量化（比 apply 快 100 倍！）
conditions = [
    train_df['hour'] < 7,
    train_df['hour'] < 9,
    train_df['hour'] < 12,
    train_df['hour'] < 14,
    train_df['hour'] < 17,
    train_df['hour'] >= 17
]
choices = [0, 1, 2, 3, 4, 5]
train_df['time_period'] = np.select(conditions, choices, default=5).astype('int8')
```

#### T2.2 区域 × 时段交叉 Target Encoding

```python
# 组合键：同一区域在不同时段的违规率可能差异很大
train_df['grid_period'] = train_df['grid_id'] * 10 + train_df['time_period']

train_df['grid_period_te'], test_df['grid_period_te'], gp_encoding = kfold_target_encode(
    train_df, test_df, 'grid_period', 'invalid_ratio', smooth=50  # 组合键样本更少，平滑更强
)
```

#### T2.3 天气二元特征

```python
train_df['is_raining'] = (train_df['Precipitations'] > 0).astype('int8')
train_df['has_snow'] = (train_df['HauteurNeige'] > 0).astype('int8')
```

#### T2.4 区域统计特征

```python
# 每个 grid_id 的历史统计（用全量训练集计算，注意这不是 target encoding，是 count/std）
grid_stats = train_df.groupby('grid_id').agg(
    grid_count=('total_count', 'mean'),         # 该区域的平均检查数
    grid_violation_std=('invalid_ratio', 'std'), # 该区域违规率波动
    grid_sample_count=('invalid_ratio', 'count') # 该区域样本数
).reset_index()

# 注意：std 对测试集用训练集统计是安全的（不是 target 本身）
# 但 violation_std 用到了 target → 严格来说需要 K-Fold，但 std 泄露风险较低
# 为简化，这里用全量训练集统计，但要在报告中说明

train_df = train_df.merge(grid_stats, on='grid_id', how='left')
test_df = test_df.merge(grid_stats, on='grid_id', how='left')
# 测试集中未见过的 grid → 填全局统计
```

#### T2.5 保存 Tier 2 检查点

```python
train_df.to_parquet('data/train_features_tier2.parquet', index=False)
test_df.to_parquet('data/test_features_tier2.parquet', index=False)
# 更新 encoding_maps
```

---

### Tier 3：锦上添花（Day 3，4/6，仅有余力时做）

#### T3.1 区域 × 星期交叉编码
- `grid_dow = grid_id * 10 + day_of_week` → K-Fold TE

#### T3.2 温度离散化
- 分箱（<5°C, 5-15, 15-25, >25°C）

#### T3.3 KMeans 空间聚类
- 对 (longitude_scaled, latitude_scaled) 做 KMeans（k=20-50）
- 生成 `cluster_id` → Target Encoding
- 比网格更灵活，但需要调参 k

#### T3.4 6 小时天气窗口（复杂，需谨慎）
- 需要按位置+时间排序后滚动平均
- 数据可能没有时间连续性（不同区域混在一起），实现复杂
- **建议：除非前面完成得很顺利，否则跳过**

---

### 测试集特征工程流水线

**核心原则**：测试集的特征工程必须使用训练集的统计量，不能使用测试集自身的统计。

```python
# 测试集流水线模板
def apply_features_to_test(test_df, encoding_maps):
    """将训练集推导出的特征工程应用到测试集"""
    # 1. 缺失值填充（用训练集的中位数）
    test_df['HauteurNeige'] = test_df['HauteurNeige'].fillna(0)
    test_df['ForceVent'] = test_df['ForceVent'].fillna(encoding_maps['forcevent_median'])

    # 2. total_count 变换
    test_df['log_total_count'] = np.log1p(test_df['total_count']).astype('float32')
    test_df['count_bin'] = pd.cut(test_df['total_count'],
                                   bins=[0, 1, 3, 10, 30, np.inf],
                                   labels=[0, 1, 2, 3, 4]).astype('int8')

    # 3. 周期编码（不依赖训练集统计）
    test_df['hour_sin'] = np.sin(2 * np.pi * test_df['hour'] / 24).astype('float32')
    # ... 其余 sin/cos 同理

    # 4. 网格化 + Target Encoding（用训练集的 map）
    GRID_SIZE = encoding_maps['grid_size']
    test_df['grid_lon'] = (test_df['longitude_scaled'] / GRID_SIZE).astype('int32')
    test_df['grid_lat'] = (test_df['latitude_scaled'] / GRID_SIZE).astype('int32')
    test_df['grid_id'] = test_df['grid_lon'] * 10000 + test_df['grid_lat']

    global_mean = encoding_maps['global_mean']
    test_df['grid_te'] = test_df['grid_id'].map(encoding_maps['grid']).fillna(global_mean).astype('float32')

    return test_df
```

### 容错方案

| 问题 | 应对 |
|------|------|
| 内存不足 (MemoryError) | 减少 float 精度 → float16；或分批处理后拼接 |
| K-Fold TE 太慢 | 减少 fold 数（5→3）；或用 numpy array 代替 pandas groupby |
| grid_size 选择困难 | 先用 0.001，跑完 Phase 3 看效果；有时间再试 0.0005 和 0.002 |
| 测试集有新 grid_id | 已用 `.fillna(global_mean)` 处理 |

---

## 五、Phase 3 — 建模（4/7-9）

### 目标
训练 LightGBM/XGBoost 模型，5-Fold CV Spearman > 0.30（保底），冲 0.40+。

### 检查点
- [ ] Baseline RF Spearman ~0.197 复现
- [ ] LightGBM 5-Fold CV Spearman > 0.30
- [ ] 至少一次 ChallengeData 平台提交
- [ ] 模型文件已保存

### 步骤

#### Step 1: 加载特征数据

```python
train_df = pd.read_parquet('data/train_features_tier2.parquet')  # 或 tier1
# 定义特征列和目标列
TARGET = 'invalid_ratio'
# 排除辅助列（grid_lon, grid_lat 等中间变量）
FEATURES = [col for col in train_df.columns
            if col not in [TARGET, 'grid_lon', 'grid_lat', 'grid_id', 'grid_period']]
```

#### Step 2: Baseline 复现（Day 1 上午）

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

# 官方 baseline: RF 10 棵树，仅 count>=45 的记录
# 我们先用全量数据 + 原始 10 特征复现
ORIG_FEATURES = ['total_count', 'longitude_scaled', 'latitude_scaled',
                 'Precipitations', 'HauteurNeige', 'Temperature',
                 'ForceVent', 'day_of_week', 'month_of_year', 'hour']

rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
# 抽样训练（6M 行 RF 太慢，用 50 万行估算）
sample_idx = train_df.sample(500000, random_state=42).index
rf.fit(train_df.loc[sample_idx, ORIG_FEATURES], train_df.loc[sample_idx, TARGET])
preds = rf.predict(train_df.loc[sample_idx, ORIG_FEATURES])
rho, _ = spearmanr(train_df.loc[sample_idx, TARGET], preds)
print(f'RF Baseline Spearman: {rho:.4f}')  # 应接近 0.197
```

#### Step 3: LightGBM 5-Fold CV（Day 1 下午 - Day 2）

```python
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import numpy as np

params = {
    'objective': 'regression',
    'metric': 'rmse',          # 内置指标用 rmse 监控收敛
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,     # 比 0.1 更保守，配合更多轮数
    'n_estimators': 2000,      # 配合早停
    'reg_lambda': 1.0,
    'min_child_samples': 50,   # 大数据集用大值防过拟合
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': 8,               # M4 有 10 核，留 2 给系统
    'random_state': 42,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))    # Out-of-Fold 预测
test_preds = np.zeros(len(test_df))    # 测试集预测（5 fold 平均）
fold_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f'\n--- Fold {fold+1}/5 ---')
    X_tr = train_df.iloc[train_idx][FEATURES]
    y_tr = train_df.iloc[train_idx][TARGET]
    X_val = train_df.iloc[val_idx][FEATURES]
    y_val = train_df.iloc[val_idx][TARGET]

    model = lgb.LGBMRegressor(**params)

    # 自定义 Spearman 回调
    def spearman_eval(y_pred, dataset):
        y_true = dataset.get_label()
        rho, _ = spearmanr(y_true, y_pred)
        return 'spearman', rho, True  # True = higher is better

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=spearman_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ]
    )

    # 记录分数
    val_pred = model.predict(X_val)
    rho, _ = spearmanr(y_val, val_pred)
    fold_scores.append(rho)
    print(f'Fold {fold+1} Spearman: {rho:.4f}')

    # OOF 预测
    oof_preds[val_idx] = val_pred
    # 测试集预测（累加后平均）
    test_preds += model.predict(test_df[FEATURES]) / 5

    # 保存模型
    model.booster_.save_model(f'models/lgbm_fold{fold}.txt')
    models.append(model)

    gc.collect()

# 汇总
oof_rho, _ = spearmanr(train_df[TARGET], oof_preds)
print(f'\nOverall OOF Spearman: {oof_rho:.4f}')
print(f'Fold Spearman scores: {[f"{s:.4f}" for s in fold_scores]}')
print(f'Mean ± Std: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')
```

#### Step 4: 首次平台提交（Day 1 完成后立即提交）

```python
# 生成提交文件
# 注意：需要确认 ChallengeData 的具体提交格式
# 通常是 CSV，一列 ID 一列预测值
submission = pd.DataFrame({
    'ID': range(len(test_preds)),  # 确认 ID 格式
    'invalid_ratio': test_preds
})
submission.to_csv('submissions/lgbm_v1.csv', index=False)
```

> **重要**：首次提交后，比较平台分数与 CV 分数。如果差距 > 0.05 → 可能有数据泄露或 train/test 分布不一致。

#### Step 5: XGBoost（Day 2-3）

```python
import xgboost as xgb

xgb_params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',      # 大数据必须用 hist
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'random_state': 42,
    'n_jobs': 8,
}
# 同样的 5-Fold CV 流程，替换模型即可
```

#### Step 6: 模型集成（Day 3）

```python
# 简单加权平均（权重按 CV Spearman 分配）
lgb_weight = lgb_oof_spearman / (lgb_oof_spearman + xgb_oof_spearman)
xgb_weight = 1 - lgb_weight
ensemble_preds = lgb_weight * lgb_test_preds + xgb_weight * xgb_test_preds

# 验证：集成后 OOF Spearman 应 >= 单模型
ensemble_oof = lgb_weight * lgb_oof_preds + xgb_weight * xgb_oof_preds
rho, _ = spearmanr(train_df[TARGET], ensemble_oof)
print(f'Ensemble OOF Spearman: {rho:.4f}')
```

#### 调参优先级（如果初始分数不满意）

1. `num_leaves`: 31 → 尝试 15, 63, 127
2. `learning_rate`: 0.05 → 尝试 0.01（配合更多 n_estimators）
3. `min_child_samples`: 50 → 尝试 20, 100
4. `feature_fraction`: 0.8 → 尝试 0.6, 0.9
5. `reg_lambda`: 1.0 → 尝试 0.1, 5.0, 10.0

> **注意**：不要在特征未定型前花时间调参！先确保特征工程最优，再微调参数。

---

## 六、Phase 4 — 评估分析（4/10）

### 目标
完成 `notebooks/04_evaluation.ipynb`，产出消融实验表、SHAP 分析图、分组误差分析。

### 检查点
- [ ] 消融实验表格完成（5+ 行对比）
- [ ] SHAP Summary Plot + 至少 2 个 Dependence Plot
- [ ] 模型对比表格（RF / LightGBM / XGBoost / Ensemble）

### 消融实验设计（参考 Paper 2）

| 实验 | 使用特征 | 预期 Spearman |
|------|---------|--------------|
| Baseline | 原始 10 特征 | ~0.20 |
| + total_count 变换 | + log_count, count_bin | ~0.22-0.25 |
| + 周期编码 | + sin/cos ×6 | ~0.25-0.30 |
| + 空间特征 | + grid_te | ~0.35-0.45 |
| + 交叉特征 | + grid_period_te | ~0.40-0.50 |
| + 天气增强 | + is_raining 等 | ~0.40-0.50 |

> 实现方式：对每组特征分别跑 LightGBM 5-Fold CV，记录 Spearman。

### SHAP 分析

```python
import shap

# 采样避免内存/速度问题
sample = train_df.sample(5000, random_state=42)
explainer = shap.TreeExplainer(models[0])  # 用第一个 fold 的模型
shap_values = explainer.shap_values(sample[FEATURES])

# Summary Plot（特征重要性排序 + 影响方向）
shap.summary_plot(shap_values, sample[FEATURES])

# Dependence Plot（展示非线性关系）
shap.dependence_plot('grid_te', shap_values, sample[FEATURES])
shap.dependence_plot('log_total_count', shap_values, sample[FEATURES])
```

### 分组误差分析

```python
# 按 total_count 分组分析模型表现
for bin_val in sorted(train_df['count_bin'].unique()):
    mask = train_df['count_bin'] == bin_val
    rho, _ = spearmanr(train_df.loc[mask, TARGET], oof_preds[mask])
    print(f'count_bin={bin_val}: n={mask.sum()}, Spearman={rho:.4f}')
```

---

## 七、Phase 5 — 可视化（4/11）

### 目标
生成报告和视频共用的 8-10 张高质量图表，保存到 `figures/` 目录。

### 图表清单

| # | 图表 | 用途 | 来源 |
|---|------|------|------|
| 1 | 目标变量 U 型分布直方图 | 报告 3章 + 视频 EDA | Phase 1 |
| 2 | total_count vs 违规率箱线图 | 报告 3章 + 视频 EDA | Phase 1 |
| 3 | 空间违规率散点图/热力图 | 报告 3章 + 视频 EDA | Phase 1 |
| 4 | 时间趋势图（hour/month） | 报告 3章 + 视频 EDA | Phase 1 |
| 5 | 消融实验柱状图 | 报告 5章 + 视频结果 | Phase 4 |
| 6 | 模型对比柱状图 | 报告 5章 + 视频结果 | Phase 4 |
| 7 | SHAP Summary Plot | 报告 5章 + 视频结果 | Phase 4 |
| 8 | SHAP Dependence Plot | 报告 5章 | Phase 4 |
| 9 | 特征重要性（LightGBM gain） | 报告 4章 + 视频方法 | Phase 3 |
| 10 | 预测 vs 真实散点图 | 报告 5章 | Phase 4 |

### 图表规范

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.dpi'] = 150

# 每张图保存
plt.savefig('figures/01_target_distribution.png', dpi=300, bbox_inches='tight')
```

---

## 八、Phase 7 — 视频录制（4/12-14）

### 时间分配（15 分钟）

| 段落 | 时长 | 内容 |
|------|------|------|
| 1. 问题动机 | 2 min | 停车违章预测的意义、数据来源（THESi 系统）、评估指标 |
| 2. EDA 发现 | 3 min | U 型分布、total_count 关系、空间/时间模式（用图 1-4） |
| 3. 方法论 | 4 min | 特征工程（重点讲区域 TE）、模型选择（LightGBM）、训练策略 |
| 4. 结果 | 4 min | 消融实验、模型对比、SHAP 分析（用图 5-9） |
| 5. 总结 | 2 min | 主要贡献、局限性、未来改进 |

### 录制建议
- **工具**：Zoom（共享屏幕 + 录制）或 OBS
- **准备**：先写每段的要点提纲（3-5 个 bullet points）
- **策略**：分段录制后拼接，比一镜到底压力小
- **注意**：不需要完美，内容清晰即可

---

## 九、Phase 6 — 报告撰写（4/15-22）

### 章节规划

| 章节 | 页数 | 重点内容 |
|------|------|---------|
| Abstract | 0.5 | 一段话概括问题、方法、主要结果 |
| 1. Introduction | 1-1.5 | 停车违章预测的实际意义、THESi 系统背景、研究目标 |
| 2. Related Work | 1.5-2 | 7 篇论文综述（Paper 1-2 直接相关，3-5 领域相关，6-7 算法） |
| 3. Data Description | 1.5-2 | 数据概况 + EDA 关键发现（复用图 1-4） |
| 4. Methodology | 3-4 | 特征工程详解 + 模型选择 + CV 策略 + 早停/防泄露 |
| 5. Results | 2-3 | 消融实验 + 模型对比 + SHAP 分析（复用图 5-9） |
| 6. Discussion | 1-1.5 | 局限性、与论文方法的对比、改进方向 |
| 7. Conclusion | 0.5 | 三句话：做了什么、结果如何、未来方向 |
| References | 0.5-1 | 至少 7 篇（必须包含 Paper 1-7） |

### 引用论文清单

1. Vo (2025) — arXiv:2505.06818
2. Karantaglis et al. (2022) — Pattern Recognition Letters, Vol.163
3. Gao et al. (2019) — Annals of GIS
4. Liu & Chen (2025) — Transportation
5. Sui et al. (2025) — J. Transport Geography
6. Ke et al. (2017) — NeurIPS (LightGBM)
7. Chen & Guestrin (2016) — KDD (XGBoost)

---

## 附录 A：内存管理备忘

### 加载时优化

```python
# float64 → float32 节省 50% 内存
# int64 → int8/int16/int32 节省 75-87% 内存
dtype_dict = {
    'total_count': 'int32',       # 范围 1-200+, int32 足够
    'longitude_scaled': 'float32',
    'latitude_scaled': 'float32',
    'Precipitations': 'float32',
    'HauteurNeige': 'float32',
    'Temperature': 'float32',
    'ForceVent': 'float32',
    'day_of_week': 'int8',        # 范围 1-6
    'month_of_year': 'int8',      # 范围 1-12
    'hour': 'int8',               # 范围 6-19
}
```

### 运行时管理

```python
import gc, psutil

def print_memory():
    """打印当前内存使用"""
    mem = psutil.virtual_memory()
    print(f'内存使用: {mem.percent}% ({mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB)')

# 删除不需要的 DataFrame 后立即回收
del intermediate_df
gc.collect()
print_memory()
```

### 中间保存

```python
# Parquet 比 CSV 快 10x，体积小 50-70%
df.to_parquet('data/checkpoint.parquet', index=False)  # 保存
df = pd.read_parquet('data/checkpoint.parquet')         # 加载
```

---

## 附录 B：常见陷阱与规避

| # | 陷阱 | 后果 | 规避方法 |
|---|------|------|---------|
| 1 | 对 grid_id 做 One-Hot Encoding | 特征数爆炸（几千列），内存溢出 | 用 Target Encoding 或 LightGBM 的 categorical 处理 |
| 2 | Target Encoding 不用 K-Fold | 严重数据泄露，CV 虚高 | 必须用 K-Fold（训练集）+ 全量统计（测试集） |
| 3 | 用 `df.apply()` 处理 6M 行 | 极慢（分钟级 vs 向量化的秒级） | 用 `np.where`/`np.select`/向量化运算 |
| 4 | SHAP 在 6M 行上计算 | 内存溢出或跑几小时 | 采样 5K-10K 行 |
| 5 | 测试集用自身统计做特征 | 信息泄露 | 测试集必须用训练集导出的 encoding map |
| 6 | 早停指标用 RMSE 而非 Spearman | 优化的不是比赛指标 | 自定义 `eval_metric` 用 Spearman |
| 7 | 不保存中间结果 | Kernel 崩溃后重跑数小时 | 每个 Tier 结束保存 parquet |
| 8 | `.copy()` 创建隐性副本 | 6M 行 ×30 列 ≈ 1.4GB → 被无意翻倍 | 原地操作或明确管理副本生命周期 |
| 9 | 调参时间投入过多 | 挤占评估和视频时间 | 特征 > 调参；只在 Phase 3 Day 3 简单调参 |
| 10 | 忘记设 random_state | 结果不可复现 | 全局 `SEED = 42`，所有模型/KFold 都用 |

---

## 附录 C：检查点保存命令速查

```python
import pickle, os

# 创建目录
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

# ===== Phase 2 保存 =====
# 特征 DataFrame
train_df.to_parquet('data/train_features.parquet', index=False)
test_df.to_parquet('data/test_features.parquet', index=False)

# Encoding Maps（用于测试集 & 复现）
encoding_maps = {
    'grid': grid_encoding,
    'grid_period': gp_encoding,      # Tier 2
    'forcevent_median': forcevent_median,
    'grid_size': GRID_SIZE,
    'global_mean': train_df['invalid_ratio'].mean(),
}
with open('data/encoding_maps.pkl', 'wb') as f:
    pickle.dump(encoding_maps, f)

# ===== Phase 3 保存 =====
# LightGBM 模型
model.booster_.save_model(f'models/lgbm_fold{fold}.txt')

# XGBoost 模型
model.save_model(f'models/xgb_fold{fold}.json')

# OOF 预测（用于消融实验和集成）
np.save('data/lgbm_oof_preds.npy', oof_preds)
np.save('data/lgbm_test_preds.npy', test_preds)

# ===== Phase 4 保存 =====
# 提交文件
submission.to_csv('submissions/ensemble_v1.csv', index=False)

# SHAP values（计算耗时，保存备用）
np.save('data/shap_values.npy', shap_values)
```
