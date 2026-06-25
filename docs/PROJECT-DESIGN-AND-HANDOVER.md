# A-Share Short-Term Alpha Model 项目设计与交接文档

> 版本：v1.0  
> 更新时间：2026-06-25  
> 项目目录：`projects/stock-alpha-model`  
> 当前主数据目录：`data_real_2000_10pct`  
> 当前主配置：`config/pipeline.strategy_10pct_2000_offline.json`

---

## 1. 项目定位

本项目是一个 **A 股短线 Alpha 选股与交易计划生成框架**。

当前核心目标不是做一个最终可直接实盘托管的量化系统，而是先构建一条可运行、可扩展、可替换模型的研究/验证流水线：

1. 拉取 A 股日线数据；
2. 构建可交易股票池；
3. 生成日线技术特征；
4. 使用短线标签训练分类模型；
5. 生成上涨/下跌/震荡概率；
6. 根据概率、风险分、股票池约束生成候选；
7. 用 A 股交易规则做简化回测；
8. 输出次日交易计划和报告。

一句话概括：

> 用工程化方式把「数据 → 特征 → 标签 → 模型 → 预测 → 回测 → 交易计划 → 报告」串成一条可复用流水线。

---

## 2. 当前项目状态

### 2.1 已完成状态

当前已经完成：

- 数据源抽象；
- AkShare / BaoStock 双 Provider；
- Fallback Provider；
- 本地 CSV DataLake；
- 批量下载与失败重试；
- 下载状态记录；
- 股票池过滤；
- 数据质量检查；
- V1 日线特征；
- Triple Barrier 三分类标签；
- V1 日线分类模型；
- LightGBM / sklearn / heuristic fallback；
- 预测结果生成；
- A 股简化回测；
- 风险标签；
- 候选解释；
- 次日交易计划；
- Markdown / HTML 报告；
- run 目录归档；
- 离线 pipeline 运行。

### 2.2 当前已跑通的数据规模

当前数据目录：

```text
data_real_2000_10pct/
```

当前数据结果：

| 项 | 数量/结果 |
|---|---:|
| 股票基础名单 | 2000 |
| 成功落库日线 | 1994 |
| 失败日线 | 6 |
| 日线周期 | 2023-01-03 ~ 2025-12-31 |
| 最终交易股票池 | 342 |
| 预测样本数 | 246,870 |
| 最新信号日期 | 2025-12-31 |

剩余失败股票：

```text
001220, 001237, 001257, 001312, 001365, 001393
```

这些失败不阻塞当前训练和预测。

### 2.3 最近一次完整 pipeline 结果

运行 ID：

```text
20260625_183409_09cac68e
```

输出摘要：

| 指标 | 结果 |
|---|---:|
| 总收益 | 224.14% |
| 年化收益 | 50.33% |
| 最大回撤 | -20.37% |
| Sharpe | 2.48 |
| 交易次数 | 4930 |
| 完整买卖回合 | 2460 |
| 胜率 | 45.41% |
| 盈亏比 | 1.71 |
| 最大连续亏损 | 11 |

注意：该结果是基于当前简化撮合、当前样本、当前参数的研究回测，不能直接等同于实盘收益。

---

## 3. 设计思想

### 3.1 核心设计原则

项目整体设计有几个明确原则。

#### 3.1.1 分层解耦

项目把量化流程拆成多个相对独立层：

```text
数据源层 → 数据湖层 → 特征层 → 标签层 → 模型层 → 预测层 → 回测层 → 报告层
```

每一层只依赖上一层的标准输出，不直接耦合具体实现。

好处：

- 换数据源不影响模型；
- 换模型不影响下载和回测；
- 换回测逻辑不影响特征和训练；
- 后续可以逐层优化，而不是推倒重来。

#### 3.1.2 Provider 抽象优先

数据源统一走 `MarketDataProvider` 抽象接口。

当前实现：

```text
stock_alpha/data/providers/base.py
stock_alpha/data/providers/akshare_provider.py
stock_alpha/data/providers/baostock_provider.py
stock_alpha/data/providers/fallback_provider.py
stock_alpha/data/providers/level2_csv_provider.py
```

设计目的：

- 免费数据源不稳定，不能让模型逻辑绑定某一个接口；
- 未来接入 Wind、同花顺 iFinD、聚宽、米筐、券商 Level-2，都只需要新增 Provider；
- 模型训练、回测、报告不用改。

#### 3.1.3 先用 CSV DataLake，保证低依赖可跑

当前 `DataLake` 用 CSV 做本地缓存：

```text
stock_alpha/storage/cache.py
```

虽然方法名保留了 `write_parquet/read_parquet`，但实际落地是 CSV。

设计目的：

- 避免初期强依赖 DuckDB / PyArrow / Parquet；
- 任何环境都容易跑；
- 文件可直接查看、排查、手工修复；
- 后续可以平滑升级到 DuckDB/Parquet。

当前数据目录结构类似：

```text
data_real_2000_10pct/
├── daily/                  # 每只股票一个 CSV
├── meta/                   # 股票基础信息、下载状态
├── universe/               # 股票池过滤结果
├── quality/                # 数据质量检查结果
├── predictions/            # 模型预测结果
├── evaluation/             # 模型评估和特征重要性
├── backtest/               # 回测结果
├── analysis/               # 风险标签、候选解释、信号稳定性
├── orders/                 # 次日交易计划
└── runs/                   # 每次 pipeline 运行归档
```

#### 3.1.4 先做可解释的 V1，不直接上复杂深度模型

V1 采用日线技术指标 + 三分类模型。

原因：

- 日线数据易得；
- 特征可解释；
- 训练速度快；
- 便于验证完整闭环；
- 便于后续替换模型作为 baseline。

当前模型不是最终模型，而是一个 baseline 框架。

#### 3.1.5 交易计划和模型预测分离

模型输出的是概率和分数：

```text
up_probability
down_probability
neutral_probability
risk_score
final_score
suggest_action
```

交易计划再根据资金、仓位、止盈止损、股票池、风险过滤生成：

```text
orders/next_day_orders.csv
```

这样设计是为了避免把「预测」和「执行」混在一起。

模型可以认为某只股票强，但交易层仍可以因为流动性、风险、仓位、涨跌停等因素过滤。

---

## 4. 总体架构

### 4.1 模块结构

```text
stock_alpha/
├── config/                 # Pipeline 配置
├── data/                   # 数据下载、Provider、增量逻辑、质量检查
│   └── providers/          # AkShare / BaoStock / Fallback / Level2 CSV
├── storage/                # 本地 DataLake 与任务状态
├── universe.py             # 可交易股票池构建
├── features/               # 特征工程
├── labels/                 # 标签生成
├── models/                 # 模型封装
├── training/               # 训练、评估、walk-forward
├── backtest/               # 回测、交易统计、持仓快照
├── analysis_signal.py      # 信号稳定性、换手分析
├── reports/                # Markdown/HTML 报告
├── trading_plan.py         # 次日交易计划
├── runtime/                # 完整 pipeline 编排、run log、归档
└── scripts/run_production.py # 生产入口 CLI
```

### 4.2 运行入口

主要入口：

```text
stock_alpha/scripts/run_production.py
```

核心命令：

```bash
.venv/bin/python -m stock_alpha.scripts.run_production pipeline --config <config.json>
```

常用命令：

```bash
# 批量下载
.venv/bin/python -m stock_alpha.scripts.run_production batch-download \
  --provider fallback \
  --start 20230101 \
  --end 20251231 \
  --data-root data_real_2000_10pct \
  --limit 2000 \
  --batch-size 50

# 重试失败下载
.venv/bin/python -m stock_alpha.scripts.run_production retry-failed \
  --provider fallback \
  --start 20230101 \
  --end 20251231 \
  --data-root data_real_2000_10pct \
  --batch-size 50

# 离线完整流程：不再下载，只基于本地数据训练/预测/回测/报告
.venv/bin/python -m stock_alpha.scripts.run_production pipeline \
  --config config/pipeline.strategy_10pct_2000_offline.json

# 只训练 V1
.venv/bin/python -m stock_alpha.scripts.run_production train-v1 \
  --data-root data_real_2000_10pct \
  --train-end 2024-12-31 \
  --valid-end 2025-06-30
```

---

## 5. 完整 Pipeline 流程

核心编排类：

```text
stock_alpha/runtime/pipeline.py
FullPipeline.run()
```

完整流程：

```text
1. pipeline start
2. download daily，或 skip_download
3. build trade universe
4. quality check
5. train model
6. predict
7. backtest
8. candidate analysis
9. walk-forward
10. grid-search，可选
11. report markdown/html
12. archive outputs
13. pipeline done
```

### 5.1 下载阶段

相关文件：

```text
stock_alpha/data/downloader.py
stock_alpha/data/batch.py
stock_alpha/storage/status.py
```

功能：

- 获取股票基础名单；
- 批量下载日线；
- 检查本地缓存是否覆盖目标日期；
- 支持增量补缺；
- 失败任务写入 `download_status.csv`；
- 支持 `retry-failed` 重试失败。

当前已验证：

- 2000 股票基础名单成功；
- 1994 只股票日线成功；
- 免费数据源存在远端断连、空返回、Broken pipe 等问题。

### 5.2 股票池阶段

相关文件：

```text
stock_alpha/universe.py
```

股票池过滤维度：

| 维度 | 说明 |
|---|---|
| 最近 60 日交易天数 | 排除停牌/数据不足股票 |
| 最新价格区间 | 排除过低价/过高价股票 |
| 20 日成交额 | 保证短线流动性 |
| 60 日成交额 | 保证长期流动性 |
| 20 日换手率 | 保证活跃度 |
| 20 日振幅 | 过滤过冷/过热 |
| 20 日波动率 | 控制风险 |
| 20 日最大回撤 | 排除近期极端下跌 |
| ST 标记 | 默认排除 ST |
| 60 日零成交日 | 排除流动性异常 |

当前配置：

```text
config/pipeline.strategy_10pct_2000_offline.json
```

核心参数：

```json
{
  "use_universe_filter": true,
  "universe_max_size": 800,
  "min_avg_amount_20": 200000000.0,
  "min_avg_amount_60": 100000000.0,
  "min_turnover_20": 3.0,
  "min_amplitude_20": 0.04,
  "max_amplitude_20": 0.15,
  "min_volatility_20": 0.02,
  "max_volatility_20": 0.10
}
```

最近一次结果：

```text
2000 名单 → 1994 有日线 → 342 进入最终交易股票池
```

### 5.3 数据质量检查

相关文件：

```text
stock_alpha/data/quality.py
```

输出：

```text
data_real_2000_10pct/quality/daily_issues.csv
data_real_2000_10pct/quality/daily_summary.csv
```

最近结果：

```text
severity=medium
issue_count=46
affected_codes=46
```

说明：当前有中等级别数据质量问题，不阻塞训练，但后续需要清理。

### 5.4 特征工程

相关文件：

```text
stock_alpha/features/v1_daily.py
stock_alpha/features/technical.py
stock_alpha/features/risk_filters.py
```

当前 V1 特征主要是日线技术指标：

| 特征类型 | 示例 |
|---|---|
| 收益率 | `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d` |
| 均线偏离 | `close_ma5_ratio`, `close_ma10_ratio`, `close_ma20_ratio` |
| 均线斜率 | `ma5_slope`, `ma10_slope` |
| 量能 | `volume_ratio_5`, `volume_ratio_20`, `amount_ratio_20` |
| K 线结构 | `amplitude`, `upper_shadow`, `lower_shadow` |
| 动量指标 | `rsi_6`, `rsi_14`, `macd_*`, `kdj_*` |
| 风险指标 | `atr_14`, `turnover_rate` |

当前特征列定义：

```text
stock_alpha/features/v1_daily.py
V1_FEATURE_COLUMNS
```

### 5.5 标签设计

相关文件：

```text
stock_alpha/labels/triple_barrier.py
```

当前使用三分类标签：

| 标签 | 含义 |
|---:|---|
| 1 | 看涨 |
| 0 | 震荡 |
| -1 | 看跌 |

标签生成逻辑：

```text
给定当前 close，向后看 horizon 个交易日：
- 如果未来最高价达到 profit_take，且未来最低价没有跌破 stop_loss，则 label=1
- 如果未来最低价跌破 stop_loss，或未来最高收益不足 profit_take/3，则 label=-1
- 其他情况 label=0
```

当前主配置：

```json
{
  "label_profit_take": 0.10,
  "label_stop_loss": 0.05,
  "label_horizon": 5
}
```

意思是：

- 未来 5 天内目标止盈 10%；
- 风险阈值 5%；
- 用这个目标训练短线分类器。

### 5.6 模型训练

相关文件：

```text
stock_alpha/models/v1_daily_model.py
stock_alpha/training/train_v1.py
```

模型封装：

```text
V1DailyAlphaModel
```

后端优先级：

```text
LightGBM → sklearn HistGradientBoosting → heuristic fallback
```

当前设计点：

- 如果安装了 LightGBM，优先用 LightGBM 多分类；
- 如果没有 LightGBM，尝试 sklearn；
- 如果都没有，使用启发式模型，保证 smoke/demo 可跑。

当前训练逻辑：

```text
features + labels merge
↓
抽取 V1_FEATURE_COLUMNS
↓
label: -1/0/1 映射为 0/1/2
↓
fit model
↓
predict all features
↓
生成 predictions/v1_latest.csv
↓
保存模型文件
↓
输出 feature_importance
```

当前预测输出字段：

```text
code
date
down_probability
neutral_probability
up_probability
risk_score
final_score
suggest_action
```

当前 `final_score` 公式：

```text
final_score = 0.7 * up_probability
            - 0.2 * down_probability
            - 0.1 * risk_score
```

当前建议动作：

```text
final_score >= 0.45 → BUY
final_score <= 0.10 → AVOID
其他 → WATCH
```

注意：交易计划可以使用 `selection_mode=topn`，因此即使模型 `suggest_action=AVOID`，也可能因为排序进入 TopN 交易计划。这是当前项目一个需要优化的点。

### 5.7 回测设计

相关文件：

```text
stock_alpha/backtest/ashare_backtest.py
stock_alpha/backtest/metrics.py
stock_alpha/backtest/holdings.py
```

当前回测模型：

```text
AShareBacktester
```

核心假设：

| 规则 | 当前实现 |
|---|---|
| 信号 | 使用前一交易日信号 |
| 买入 | 下一交易日开盘买入 |
| 卖出 | 止盈/止损/持有期到期 |
| T+1 | 简化体现为次日开盘买 |
| 涨停 | 涨停开盘不买 |
| 跌停 | 跌停时不卖 |
| 手续费 | 买入 0.03%，卖出 0.13% |
| 滑点 | 默认 0.1% |
| 最小单位 | 100 股 |
| 仓位 | 单票最大仓位比例 |

当前主配置交易参数：

```json
{
  "top_n": 10,
  "hold_days": 5,
  "min_score": 0.45,
  "buy_fee": 0.0003,
  "sell_fee": 0.0013,
  "slippage": 0.001,
  "max_position_pct": 0.1,
  "take_profit": 0.10,
  "stop_loss": 0.05,
  "selection_mode": "topn",
  "score_quantile": 0.95,
  "max_down_probability": 0.60,
  "max_risk_score": 0.40,
  "max_daily_buys": 5
}
```

### 5.8 候选分析与交易计划

相关文件：

```text
stock_alpha/reports/candidate_analysis.py
stock_alpha/trading_plan.py
```

候选分析：

- 风险标签：高波动、下跌概率高、综合分偏低、风控阻断等；
- 候选解释：根据特征重要性输出关键特征快照。

交易计划输出：

```text
data_real_2000_10pct/orders/next_day_orders.csv
```

字段：

```text
signal_date
code
action
ref_price
shares
planned_amount
score
up_probability
down_probability
take_profit_price
stop_loss_price
note
```

注意：这是「计划」，不是自动交易指令。实盘前必须复核：

- 是否停牌；
- 是否涨跌停；
- 是否公告黑天鹅；
- 开盘是否极端高开/低开；
- 实际盘口流动性；
- 当日市场整体风险。

### 5.9 报告输出

相关文件：

```text
stock_alpha/reports/daily_report.py
stock_alpha/reports/html_report.py
stock_alpha/reports/runs_index.py
```

当前输出：

```text
reports/daily_report_20251231.md
reports/daily_report_20251231.html
```

run 归档：

```text
data_real_2000_10pct/runs/<run_id>/
```

---

## 6. 当前做了什么

### 6.1 工程层面

已经完成：

1. 搭建 Python 项目结构；
2. 实现数据源抽象；
3. 接入 AkShare；
4. 接入 BaoStock；
5. 实现 fallback provider；
6. 实现 CSV 本地数据湖；
7. 实现下载状态记录；
8. 实现失败重试；
9. 实现完整 pipeline；
10. 实现离线运行配置；
11. 实现 run log 和归档。

### 6.2 数据层面

已经完成：

1. 获取 2000 个股票基础名单；
2. 下载 1994 只股票日线；
3. 周期覆盖 2023-01-03 ~ 2025-12-31；
4. 生成下载状态；
5. 发现 6 只失败股票；
6. 做了数据质量检查。

### 6.3 策略层面

已经完成：

1. 股票池过滤；
2. 日线技术特征；
3. 三分类短线标签；
4. V1 模型训练；
5. 全样本预测；
6. A 股规则简化回测；
7. 次日交易计划；
8. 风险标签；
9. 解释字段；
10. 报告输出。

### 6.4 Bug 修复

修复了一个 Pandas 版本兼容问题：

问题：

```text
pd.NA 导致 KDJ 中 ewm() 聚合时 dtype 变成 object，报错：
DataError: No numeric types to aggregate
```

修复：

```text
stock_alpha/features/technical.py
```

将 `pd.NA` 替换为 `np.nan`，并引入 `numpy`。

验证结果：

```text
features_ok (246870, 46)
```

---

## 7. 当前没做什么

这个项目还不是完整生产级量化交易系统，以下内容尚未完成或只是占位。

### 7.1 没做实时行情

当前基于历史日线 CSV，不接实时行情。

未完成：

- 实时盘口；
- 实时分钟线补齐；
- 实时涨跌停状态；
- 实时风控；
- 实时交易信号刷新。

### 7.2 没做真实交易接入

当前只生成交易计划 CSV，不接券商交易接口。

未完成：

- 券商 API；
- 自动下单；
- 撤单；
- 成交回报；
- 账户同步；
- 真实持仓管理；
- 实盘风控。

### 7.3 没做严格复权/除权除息校验

当前依赖 Provider 返回的前复权数据。

未完成：

- 复权因子持久化；
- 除权除息事件处理；
- 多数据源复权一致性校验；
- 停复牌事件校验。

### 7.4 没做完整财务/基本面/事件因子

当前主要是日线技术指标。

未完成：

- 财务因子；
- 估值因子；
- 机构持仓；
- 公告事件；
- 龙虎榜；
- 研报；
- 北向资金；
- 板块/题材热度；
- 新闻情绪。

### 7.5 V2/V4 还不是生产可用

README 中提到：

- V2：分钟级量能 / VWAP / 分时强弱特征；
- V4：Level-2 / 盘口订单簿抽象接口、DeepLOB 框架占位。

当前它们更多是框架占位或可扩展方向，不是本次主流程的核心生产模型。

### 7.6 没做严格样本外检验

当前有 train/valid split 和 walk-forward，但还不够严肃。

未完成：

- 完整滚动训练；
- 防止数据泄漏的系统性校验；
- 多市场周期样本外测试；
- 多参数稳定性分析；
- 交易成本敏感性分析；
- 幸存者偏差处理。

### 7.7 没做组合优化

当前回测以 TopN + 固定仓位为主。

未完成：

- 风险平价；
- 行业中性；
- 风格中性；
- 波动率目标；
- Kelly / 半 Kelly；
- 最大回撤约束；
- 相关性约束；
- 换手率约束。

### 7.8 没做严格模型解释

当前解释只是特征重要性快照，不是 SHAP。

未完成：

- SHAP；
- 单票贡献分解；
- 特征方向解释；
- 横截面对比解释；
- 历史相似样本解释。

---

## 8. 当前主要问题与风险

### 8.1 数据源稳定性风险

这次下载过程中出现过：

```text
RemoteDisconnected
Broken pipe
BaoStock empty
```

说明免费数据源不稳定。

建议：

- 生产环境必须换付费/稳定数据源；
- 免费源只适合研究验证；
- 下载需要限流、重试、断点续传、定时补缺。

### 8.2 数据日期不是真实当前日期

当前数据最新日期是：

```text
2025-12-31
```

因此当前报告中的「次日交易计划」是针对这个样本最后日期的模拟次日，不是实时 2026-06-25 的明日交易计划。

如果要做真实明日预测，需要先把日线数据更新到最新交易日。

### 8.3 selection_mode=topn 会放大弱信号

当前配置：

```json
"selection_mode": "topn"
```

含义是：即使整体分数不高，也会选 TopN。

这会导致交易计划中出现：

- 模型 `suggest_action=AVOID`；
- 但交易计划仍列为 BUY。

这是当前最大策略逻辑问题之一。

建议：

- 将交易计划层增加 `min_score` 硬阈值；
- 或 `topn + min_score` 同时生效；
- 或只输出 `WATCH`，不输出 `BUY`。

### 8.4 当前标签偏激进

当前标签：

```text
5 日止盈 10%，止损 5%
```

这在 A 股短线里属于比较激进的目标，可能导致：

- 正样本偏少；
- 模型更偏向捕捉高波动票；
- 回测换手率较高；
- 实盘滑点/冲击成本可能显著高于当前假设。

### 8.5 回测仍然简化

虽然已考虑部分 A 股规则，但仍然简化：

- 没有真实盘口成交；
- 没有集合竞价逻辑；
- 没有涨跌停排队；
- 没有冲击成本；
- 没有真实可买量；
- 没有停牌/临停细节；
- 没有实盘持仓状态同步。

---

## 9. 后续可优化点

### 9.1 数据层优化

优先级：高。

建议：

1. 接入稳定数据源；
2. 数据存储从 CSV 升级到 DuckDB/Parquet；
3. 建立交易日历表；
4. 建立股票主数据表；
5. 建立停复牌/涨跌停/复权因子表；
6. 每日增量更新；
7. 自动补缺；
8. 多数据源交叉校验；
9. 数据质量规则细化为 blocking / warning / info。

推荐目录演进：

```text
data_lake/
├── raw/               # 原始数据，不改动
├── normalized/        # 标准化数据
├── features/          # 特征快照
├── labels/            # 标签快照
├── predictions/       # 预测快照
└── marts/             # 给报告/交易使用的数据集市
```

### 9.2 特征层优化

当前特征主要是传统技术指标。

可新增：

#### 行情结构特征

- 跳空缺口；
- 连板/断板；
- 涨停炸板；
- 大阳线/大阴线；
- N 日新高；
- 量价背离；
- 缩量回踩；
- 尾盘拉升。

#### 横截面特征

- 当日收益排名；
- 行业内排名；
- 20 日强度分位；
- 成交额分位；
- 换手率分位；
- 波动率分位。

#### 市场环境特征

- 指数趋势；
- 全市场涨跌家数；
- 涨停家数；
- 跌停家数；
- 市场成交额；
- 情绪周期。

#### 板块/题材特征

- 所属行业强度；
- 概念板块热度；
- 近期主线持续性；
- 板块内龙头强度。

### 9.3 标签层优化

当前 Triple Barrier 是合理起点，但可优化：

1. 加入先触达顺序，而不是只看未来 max/min；
2. 加入交易成本后的净收益标签；
3. 标签按市场环境动态调整；
4. 不同波动率股票使用动态止盈止损；
5. 尝试 ranking label，而不是 classification label；
6. 尝试未来收益回归；
7. 尝试 meta-labeling：先判断能不能交易，再判断方向。

更合理的短线标签可以是：

```text
label = 未来 N 天是否能在不触发止损的情况下达到目标收益
```

并且必须考虑：

- 次日是否涨停无法买入；
- 买入价不是今日 close，而是次日 open；
- 卖出价也不一定能按 high/low 成交。

### 9.4 模型层优化

当前模型是 baseline。

可替换方向：

| 模型 | 适用场景 |
|---|---|
| LightGBM Ranker | 横截面选股排序 |
| CatBoost | 类别特征/稳健性 |
| XGBoost | 树模型 baseline |
| TabNet | 表格深度学习 |
| Temporal CNN | 时间序列局部模式 |
| Transformer | 多日序列建模 |
| DeepLOB | Level-2 盘口 |
| Ensemble | 多模型融合 |

我建议下一阶段优先做：

```text
LightGBM Ranker + Walk-forward + 横截面特征
```

原因：

- 对选股问题更自然；
- 不必强行把问题变成三分类；
- 输出排序比输出绝对概率更适合 TopN 组合；
- 工程复杂度可控。

### 9.5 回测层优化

建议增强：

1. 严格撮合引擎；
2. 使用次日 open/high/low/close 真实触发顺序模拟；
3. 加入涨跌停不可成交逻辑；
4. 加入停牌；
5. 加入真实成交量约束；
6. 加入冲击成本模型；
7. 加入卖出排队失败；
8. 加入持仓穿透分析；
9. 加入行业暴露；
10. 加入净值归因。

### 9.6 风控层优化

建议增加：

- 单票最大亏损；
- 单日最大亏损；
- 单周最大亏损；
- 最大连续亏损熔断；
- 市场环境熔断；
- 行业集中度；
- 题材集中度；
- 高开过滤；
- 跌破均线过滤；
- 财务/退市风险过滤。

### 9.7 报告层优化

当前报告可读，但还不够决策级。

建议增加：

1. 今日市场环境；
2. 候选强弱分层；
3. 强信号/弱信号明确区分；
4. 每只票入选原因；
5. 风险原因；
6. 历史相似样本胜率；
7. 推荐仓位；
8. 开盘执行条件；
9. 不买条件；
10. 盘后复盘。

---

## 10. 如果换一个模型，如何理解和改造项目

如果新模型开发者接手，建议按以下顺序理解。

### 10.1 第一层：先理解数据格式

先看这些文件：

```text
data_real_2000_10pct/daily/*.csv
data_real_2000_10pct/meta/stock_basic.csv
data_real_2000_10pct/universe/selected.csv
data_real_2000_10pct/predictions/v1_latest.csv
```

日线字段：

```text
date
code
open
high
low
close
volume
amount
turnover_rate
pct_chg
isST
```

预测输出字段：

```text
code
date
down_probability
neutral_probability
up_probability
risk_score
final_score
suggest_action
```

只要新模型能输出类似结构，就可以复用后面的回测、报告、交易计划。

### 10.2 第二层：理解 pipeline 接口

核心入口：

```text
stock_alpha/runtime/pipeline.py
```

模型训练调用点：

```python
trainer = V1Trainer(self.lake)
train_res = trainer.train(...)
pred = self.lake.read_parquet("predictions", "v1_latest")
```

也就是说，如果要换模型，最小改造点是：

```text
stock_alpha/training/train_v1.py
stock_alpha/models/v1_daily_model.py
```

只要保持输出：

```text
predictions/v1_latest.csv
```

后续：

- backtest；
- analysis；
- orders；
- report；

都可以不改。

### 10.3 第三层：新模型必须遵守的输出协议

新模型输出 DataFrame 至少包含：

```text
code: 6 位股票代码
date: 信号日期
final_score: 排序分
```

建议包含：

```text
up_probability
down_probability
neutral_probability
risk_score
suggest_action
```

如果没有概率，也可以这样映射：

```text
final_score = model_score
up_probability = rank_percentile
DOWN/风险字段用风险模型或规则生成
```

但回测和交易计划中以下字段会被使用：

```text
final_score
down_probability
risk_score
```

所以最好保留。

### 10.4 第四层：替换模型推荐做法

#### 方案 A：最小替换

只改：

```text
stock_alpha/models/v1_daily_model.py
```

保持 `V1DailyAlphaModel.fit()` 和 `predict()` 方法签名不变。

优点：

- 改动小；
- pipeline 不用动；
- 报告不变。

缺点：

- 新模型仍被命名为 V1；
- 如果模型需要序列输入，接口可能不够自然。

#### 方案 B：新增 V3 模型，保留 V1

新增：

```text
stock_alpha/features/v3_xxx.py
stock_alpha/models/v3_xxx_model.py
stock_alpha/training/train_v3.py
```

然后在 `runtime/pipeline.py` 增加配置项：

```json
"model_version": "v3"
```

根据配置选择 trainer。

优点：

- 结构清晰；
- V1 baseline 保留；
- 便于 A/B 对比。

缺点：

- pipeline 要稍微改造；
- 报告可能要支持更多字段。

#### 方案 C：做模型注册表

新增：

```text
stock_alpha/models/registry.py
```

形式：

```python
TRAINER_REGISTRY = {
    "v1_daily": V1Trainer,
    "v3_ranker": V3RankerTrainer,
    "v4_lob": V4LobTrainer,
}
```

配置：

```json
{
  "model_name": "v3_ranker"
}
```

这是长期最优方案。

### 10.5 新模型开发流程建议

如果换模型，建议按这个流程做：

```text
1. 固定数据集快照
2. 固定训练/验证/测试区间
3. 固定股票池
4. 复用当前 V1 输出作为 baseline
5. 开发新特征
6. 开发新标签/目标
7. 训练新模型
8. 输出同协议 predictions
9. 复用相同 backtest 比较
10. 做 walk-forward
11. 做交易成本敏感性测试
12. 再考虑上线
```

不能一上来只看单次回测收益，容易过拟合。

---

## 11. 推荐下一阶段路线图

### 阶段 1：把当前 V1 做稳

优先级最高。

任务：

1. 固化离线数据快照；
2. 修复 `topn` 弱信号问题；
3. 增加 `topn + min_score`；
4. 清理 46 个 medium 数据质量问题；
5. 修正报告中 BUY/AVOID 不一致；
6. 补充单元测试；
7. 建立每日增量更新脚本。

建议变更：

```text
交易计划生成时：
selection_mode=topn 也必须满足 min_score 或 up_probability/down_probability 条件。
```

### 阶段 2：增强回测可信度

任务：

1. 引入真实交易日历；
2. 加入停牌；
3. 加入涨跌停价；
4. 加入成交量约束；
5. 加入冲击成本；
6. 输出更细的回撤和持仓分析。

### 阶段 3：做 Ranker 模型

建议新增：

```text
V3 LightGBM Ranker
```

训练目标：

```text
按交易日分组，对股票未来 5 日收益/风险收益比排序
```

输出：

```text
rank_score
final_score
```

优势：

- 更符合选股；
- 不需要强行估计概率；
- TopN 组合更自然。

### 阶段 4：引入市场环境和板块特征

任务：

1. 指数特征；
2. 行业特征；
3. 题材强度；
4. 市场情绪；
5. 风格暴露。

### 阶段 5：实盘前工程化

如果进入准实盘，需要：

1. 稳定数据源；
2. 每日自动更新；
3. 任务调度；
4. 日志告警；
5. 数据质量阻断；
6. 交易前检查；
7. 人工确认；
8. 持仓同步；
9. 复盘归档。

---

## 12. 关键配置说明

当前主配置：

```text
config/pipeline.strategy_10pct_2000_offline.json
```

核心参数解释：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `data_root` | `data_real_2000_10pct` | 数据目录 |
| `start` | `20230101` | 数据开始日期 |
| `end` | `20251231` | 数据结束日期 |
| `limit` | `2000` | 股票名单数量 |
| `skip_download` | `true` | 离线运行，不下载 |
| `top_n` | `10` | 最多候选数量 |
| `hold_days` | `5` | 最长持有 5 天 |
| `take_profit` | `0.10` | 止盈 10% |
| `stop_loss` | `0.05` | 止损 5% |
| `max_position_pct` | `0.1` | 单票 10% 仓位 |
| `selection_mode` | `topn` | 按排名选 TopN |
| `max_down_probability` | `0.60` | 下跌概率过滤 |
| `max_risk_score` | `0.40` | 风险分过滤 |
| `label_profit_take` | `0.10` | 标签止盈阈值 |
| `label_stop_loss` | `0.05` | 标签止损阈值 |
| `label_horizon` | `5` | 标签向后看 5 日 |

---

## 13. 常用文件说明

### 13.1 输入数据

```text
data_real_2000_10pct/daily/*.csv
```

每只股票一个日线文件。

### 13.2 下载状态

```text
data_real_2000_10pct/meta/download_status.csv
```

记录每个股票下载状态。

### 13.3 股票池

```text
data_real_2000_10pct/universe/selected.csv
data_real_2000_10pct/universe/metrics.csv
```

`selected.csv` 是最终进入模型的可交易股票池。

### 13.4 预测结果

```text
data_real_2000_10pct/predictions/v1_latest.csv
```

全量预测结果。

### 13.5 交易计划

```text
data_real_2000_10pct/orders/next_day_orders.csv
```

次日交易计划。

### 13.6 回测结果

```text
data_real_2000_10pct/backtest/metrics_latest.csv
data_real_2000_10pct/backtest/trades_latest.csv
data_real_2000_10pct/backtest/equity_latest.csv
data_real_2000_10pct/backtest/holdings_latest.csv
```

### 13.7 报告

```text
reports/daily_report_20251231.md
reports/daily_report_20251231.html
```

### 13.8 每次运行归档

```text
data_real_2000_10pct/runs/<run_id>/
```

---

## 14. 新人接手阅读顺序

如果换一个模型开发或新工程师接手，建议按以下顺序读：

```text
1. README.md
2. docs/PROJECT-DESIGN-AND-HANDOVER.md
3. config/pipeline.strategy_10pct_2000_offline.json
4. stock_alpha/scripts/run_production.py
5. stock_alpha/runtime/pipeline.py
6. stock_alpha/storage/cache.py
7. stock_alpha/data/downloader.py
8. stock_alpha/universe.py
9. stock_alpha/features/v1_daily.py
10. stock_alpha/labels/triple_barrier.py
11. stock_alpha/models/v1_daily_model.py
12. stock_alpha/training/train_v1.py
13. stock_alpha/backtest/ashare_backtest.py
14. stock_alpha/trading_plan.py
15. stock_alpha/reports/daily_report.py
```

如果只想换模型，重点看：

```text
stock_alpha/features/v1_daily.py
stock_alpha/models/v1_daily_model.py
stock_alpha/training/train_v1.py
stock_alpha/runtime/pipeline.py
```

---

## 15. 最小复现命令

### 15.1 环境

```bash
cd /Users/admin/.openclaw/workspaces/workspace-engineer-jay/projects/stock-alpha-model
```

### 15.2 运行离线完整流程

```bash
.venv/bin/python -m stock_alpha.scripts.run_production pipeline \
  --config config/pipeline.strategy_10pct_2000_offline.json
```

### 15.3 查看次日计划

```bash
cat data_real_2000_10pct/orders/next_day_orders.csv
```

### 15.4 查看回测指标

```bash
cat data_real_2000_10pct/backtest/metrics_latest.csv
cat data_real_2000_10pct/backtest/trade_stats_latest.csv
```

### 15.5 查看报告

```bash
open reports/daily_report_20251231.html
```

---

## 16. 如何运行项目

本节给接手人直接使用，不讲原理，只讲怎么跑、跑完看什么。

### 16.1 进入项目目录

```bash
cd /Users/admin/.openclaw/workspaces/workspace-engineer-jay/projects/stock-alpha-model
```

### 16.2 推荐运行方式：离线完整流程

当前本地已经有 `data_real_2000_10pct` 数据，正常情况下不要重新下载，直接跑离线完整流程：

```bash
.venv/bin/python -m stock_alpha.scripts.run_production pipeline \
  --config config/pipeline.strategy_10pct_2000_offline.json
```

这个命令会执行：

```text
读取本地数据
→ 构建股票池
→ 数据质量检查
→ 训练模型
→ 生成预测
→ 回测
→ 生成风险标签/解释
→ 生成次日交易计划
→ 生成 Markdown/HTML 报告
→ 写入 run 归档
```

### 16.3 如果要重新下载 2000 只股票日线

只有在需要更新数据或重建数据目录时才执行。

```bash
.venv/bin/python -m stock_alpha.scripts.run_production batch-download \
  --provider fallback \
  --start 20230101 \
  --end 20251231 \
  --data-root data_real_2000_10pct \
  --limit 2000 \
  --batch-size 50
```

如果下载有失败，继续重试：

```bash
.venv/bin/python -m stock_alpha.scripts.run_production retry-failed \
  --provider fallback \
  --start 20230101 \
  --end 20251231 \
  --data-root data_real_2000_10pct \
  --batch-size 50
```

注意：免费数据源不稳定，出现 `RemoteDisconnected`、`Broken pipe`、`empty` 是正常现象，通常需要多次重试。

### 16.4 如果只想训练模型和生成预测

不跑完整回测和报告，只训练 V1 并生成预测：

```bash
.venv/bin/python -m stock_alpha.scripts.run_production train-v1 \
  --data-root data_real_2000_10pct \
  --train-end 2024-12-31 \
  --valid-end 2025-06-30
```

输出核心文件：

```text
data_real_2000_10pct/predictions/v1_latest.csv
models/v1_daily_lgb.pkl
```

但实际使用建议跑完整 pipeline，因为完整 pipeline 会同步生成交易计划、回测和报告。

### 16.5 如何判断是否跑成功

完整 pipeline 成功时，终端会看到类似输出：

```text
[pipeline_id] pipeline: start
[pipeline_id] download: skipped
[pipeline_id] universe: done
[pipeline_id] quality: done
[pipeline_id] train: done
[pipeline_id] backtest: done
[pipeline_id] analysis: done
[pipeline_id] walk_forward: done
[pipeline_id] report: done
[pipeline_id] pipeline: done
```

并且最后会输出类似：

```text
{
  'markdown': PosixPath('reports/daily_report_YYYYMMDD.md'),
  'html': PosixPath('reports/daily_report_YYYYMMDD.html'),
  'summary': PosixPath('data_real_2000_10pct/runs/<run_id>/summary.json'),
  'run_id': '<run_id>'
}
```

如果看到 `Process exited with code 0`，说明进程正常结束。

### 16.6 当前最近一次成功运行

最近一次成功完整运行：

```text
run_id = 20260625_183409_09cac68e
```

对应归档目录：

```text
data_real_2000_10pct/runs/20260625_183409_09cac68e/
```

---

## 17. 最终产出物是什么，要看哪些文件

### 17.1 最重要：次日交易计划

如果只看一个文件，看这个：

```text
data_real_2000_10pct/orders/next_day_orders.csv
```

这是最终交易计划，字段含义：

| 字段 | 含义 |
|---|---|
| `signal_date` | 信号日期 |
| `code` | 股票代码 |
| `action` | 操作，当前主要是 BUY |
| `ref_price` | 参考价，通常是最新收盘价 |
| `shares` | 建议股数，已按 100 股取整 |
| `planned_amount` | 计划买入金额 |
| `score` | 模型综合分 |
| `up_probability` | 上涨概率 |
| `down_probability` | 下跌概率 |
| `take_profit_price` | 止盈价 |
| `stop_loss_price` | 止损价 |
| `note` | 执行备注 |

查看命令：

```bash
cat data_real_2000_10pct/orders/next_day_orders.csv
```

或用表格工具打开 CSV。

### 17.2 第二重要：HTML 报告

给人看的报告：

```text
reports/daily_report_20251231.html
```

打开命令：

```bash
open reports/daily_report_20251231.html
```

报告里包含：

- 回测摘要；
- Top 候选；
- 次日交易计划；
- 风险标签；
- 入选解释；
- 特征重要性；
- 数据质量摘要。

如果不方便打开 HTML，看 Markdown：

```text
reports/daily_report_20251231.md
```

查看命令：

```bash
cat reports/daily_report_20251231.md
```

### 17.3 模型预测全量结果

```text
data_real_2000_10pct/predictions/v1_latest.csv
```

这是所有股票、所有日期的预测结果。

字段：

| 字段 | 含义 |
|---|---|
| `code` | 股票代码 |
| `date` | 信号日期 |
| `down_probability` | 下跌概率 |
| `neutral_probability` | 震荡概率 |
| `up_probability` | 上涨概率 |
| `risk_score` | 风险分 |
| `final_score` | 综合排序分 |
| `suggest_action` | 模型建议：BUY/WATCH/AVOID |

如果只想看最新一天 Top 20，可以执行：

```bash
.venv/bin/python - <<'PY'
import pandas as pd
p='data_real_2000_10pct/predictions/v1_latest.csv'
df=pd.read_csv(p,dtype={'code':str})
latest=df['date'].max()
print('latest_date=', latest)
print(df[df['date']==latest].sort_values('final_score', ascending=False).head(20).to_string(index=False))
PY
```

### 17.4 回测核心指标

```text
data_real_2000_10pct/backtest/metrics_latest.csv
```

字段：

| 字段 | 含义 |
|---|---|
| `total_return` | 总收益 |
| `annual_return` | 年化收益 |
| `max_drawdown` | 最大回撤 |
| `sharpe` | 夏普比率 |
| `trade_count` | 交易次数 |
| `final_equity` | 最终权益 |

查看命令：

```bash
cat data_real_2000_10pct/backtest/metrics_latest.csv
```

### 17.5 交易统计

```text
data_real_2000_10pct/backtest/trade_stats_latest.csv
```

字段：

| 字段 | 含义 |
|---|---|
| `round_trips` | 完整买卖回合数 |
| `win_rate` | 胜率 |
| `avg_pnl` | 平均盈亏 |
| `profit_loss_ratio` | 盈亏比 |
| `max_consecutive_losses` | 最大连续亏损 |

查看命令：

```bash
cat data_real_2000_10pct/backtest/trade_stats_latest.csv
```

### 17.6 回测明细

```text
data_real_2000_10pct/backtest/trades_latest.csv
```

每一笔买卖记录。

```text
data_real_2000_10pct/backtest/equity_latest.csv
```

每日权益曲线。

```text
data_real_2000_10pct/backtest/holdings_latest.csv
```

持仓快照。

### 17.7 股票池结果

```text
data_real_2000_10pct/universe/selected.csv
```

这是最终进入模型训练/预测的股票池。

当前最近一次是 342 只。

```text
data_real_2000_10pct/universe/metrics.csv
```

这是所有股票的股票池过滤指标。

### 17.8 数据质量结果

```text
data_real_2000_10pct/quality/daily_summary.csv
data_real_2000_10pct/quality/daily_issues.csv
```

`daily_summary.csv` 看总体问题数。  
`daily_issues.csv` 看具体哪些股票、哪些日期有问题。

### 17.9 风险标签和候选解释

```text
data_real_2000_10pct/analysis/candidate_risk_tags.csv
```

每个候选股票的风险标签。

```text
data_real_2000_10pct/analysis/candidate_explanations.csv
```

每个候选股票的入选解释。

注意：当前解释不是 SHAP，只是基于特征重要性的工程化快照。

### 17.10 每次运行归档

每次 pipeline 会生成一个 run 目录：

```text
data_real_2000_10pct/runs/<run_id>/
```

例如：

```text
data_real_2000_10pct/runs/20260625_183409_09cac68e/
```

里面通常包括：

```text
events.jsonl        # 流程事件日志
summary.json        # 本次运行摘要
models/             # 本次归档模型
reports/            # 本次归档报告
orders/             # 本次归档交易计划
analysis/           # 本次归档分析文件
```

如果要追溯某次运行，用 run 目录最可靠。

---

## 18. 最快阅读方式

如果你只想看结论：

```text
1. data_real_2000_10pct/orders/next_day_orders.csv
2. reports/daily_report_20251231.html
3. data_real_2000_10pct/backtest/metrics_latest.csv
4. data_real_2000_10pct/backtest/trade_stats_latest.csv
```

如果你想看模型为什么这么选：

```text
1. data_real_2000_10pct/predictions/v1_latest.csv
2. data_real_2000_10pct/analysis/candidate_risk_tags.csv
3. data_real_2000_10pct/analysis/candidate_explanations.csv
4. data_real_2000_10pct/evaluation/feature_importance.csv
```

如果你想排查数据问题：

```text
1. data_real_2000_10pct/meta/download_status.csv
2. data_real_2000_10pct/quality/daily_summary.csv
3. data_real_2000_10pct/quality/daily_issues.csv
4. data_real_2000_10pct/daily/*.csv
```

如果你想复盘某次运行：

```text
1. data_real_2000_10pct/runs/<run_id>/summary.json
2. data_real_2000_10pct/runs/<run_id>/events.jsonl
3. data_real_2000_10pct/runs/<run_id>/reports/
4. data_real_2000_10pct/runs/<run_id>/orders/
```

---

## 19. 当前结论摘要

当前项目已经具备「研究闭环」能力：

```text
数据下载 → 股票池 → 特征 → 标签 → 模型 → 预测 → 回测 → 交易计划 → 报告
```

但还不是生产级实盘系统。

当前最重要的下一步不是换复杂模型，而是：

1. 修正弱信号也进入 BUY 计划的问题；
2. 强化数据质量；
3. 提升回测真实性；
4. 加入市场/行业/题材特征；
5. 再做 Ranker 模型替换。

如果目标是更好地开发新模型，核心原则是：

> 不要改掉整条 pipeline，先让新模型输出兼容 `predictions/v1_latest.csv` 的协议，然后复用现有回测和报告做横向对比。

这样可以保证新模型和旧模型在同一数据、同一股票池、同一回测条件下比较，避免因为工程差异导致结论失真。
