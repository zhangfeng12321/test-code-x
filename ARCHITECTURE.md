# A股短线Alpha选股系统 - 架构设计文档

> 版本：v2.0  
> 更新时间：2026-06-26  
> 项目目录：`projects/stock-alpha-model`

---

## 1. 系统定位

A股短线/中线（1-2周）Alpha选股与交易计划生成系统。

核心目标：每日自动拉取行情数据 → 模型打分 → 生成明日交易计划 → 输出报告。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline 编排层                           │
│                   stock_alpha/runtime/pipeline.py                 │
└────────┬────────┬────────┬────────┬────────┬────────┬───────────┘
         │        │        │        │        │        │
    ┌────▼──┐ ┌──▼───┐ ┌──▼──┐ ┌──▼───┐ ┌──▼──┐ ┌──▼───┐
    │ 数据层 │ │特征层│ │标签层│ │模型层│ │回测层│ │报告层│
    └───────┘ └──────┘ └─────┘ └──────┘ └─────┘ └──────┘
```

---

## 3. 三级股票池设计

```
全市场 ~5300
    │ 首轮预过滤（ST + 北交所）
    ▼
┌──────────────────────────────┐
│ 下载池 ~5000                  │  拉取日线数据
└──────────────────────────────┘
    │ min_train_days >= 250
    ▼
┌──────────────────────────────┐
│ 训练池 ~4500                  │  模型训练 + 全量打分
└──────────────────────────────┘
    │ Universe Filter
    ▼
┌──────────────────────────────┐
│ 交易池 ~800-1500              │  回测 + 生成交易计划
└──────────────────────────────┘
    │ Ranker 排名 + 风控
    ▼
  TopN (5-10只) → 明日交易计划
```

### 3.1 下载池过滤规则

- 排除北交所（代码 8/4 开头，30%涨跌停）
- 排除ST股（名称含ST）

### 3.2 训练池门控

- 交易天数 >= 250（约1年数据）
- 自动清洗：去空日期 + 去重 + 排序

### 3.3 交易池（Universe Filter）

| 条件 | 阈值 |
|---|---|
| 20日均成交额 | 2亿 ~ 50亿 |
| 60日均成交额 | >= 1亿 |
| 20日换手率 | >= 3% |
| 20日振幅 | 4% ~ 15% |
| 20日波动率 | 2% ~ 10% |
| 60日交易天数 | >= 45天 |
| 60日零成交天数 | = 0 |
| 最新股价 | 2 ~ 200元 |
| 20日最大回撤 | > -35% |
| ST | 排除 |

---

## 4. 特征体系（60个特征）

### 4.1 个股时序特征（26个）

| 类型 | 特征 |
|---|---|
| 收益率 | ret_1d, ret_3d, ret_5d, ret_10d, ret_20d |
| 均线偏离 | close_ma5_ratio, close_ma10_ratio, close_ma20_ratio |
| 均线斜率 | ma5_slope, ma10_slope |
| 量能 | volume_ratio_5, volume_ratio_20, amount_ratio_20 |
| K线结构 | amplitude, upper_shadow, lower_shadow |
| 动量 | rsi_6, rsi_14, macd_dif, macd_dea, macd_hist, kdj_k, kdj_d, kdj_j |
| 风险 | atr_14, turnover_rate |

### 4.2 量价结构特征（8个）

| 特征 | 含义 |
|---|---|
| gap_up / gap_down | 跳空高开/低开幅度 |
| consecutive_up / consecutive_down | 连涨/连跌天数 |
| vol_price_diverge | 量价背离（+1/-1/0） |
| new_high_20d / new_low_20d | 20日新高/新低 |
| close_position | 收盘在K线中的位置 |

### 4.3 横截面排名特征（10个）

| 特征 | 含义 |
|---|---|
| ret_1d_rank / ret_5d_rank | 当日/5日收益排名分位 |
| amount_rank / turnover_rank | 成交额/换手率排名 |
| volume_ratio_5_rank / amplitude_rank | 量比/振幅排名 |
| close_ma20_rank / rsi_14_rank | 均线偏离/RSI排名 |
| strength_20d / weakness_5d | 20日强度/5日弱势 |

### 4.4 市场环境特征（10个）

| 特征 | 含义 |
|---|---|
| market_ret_1d / market_ret_5d | 全市场当日/5日收益 |
| market_breadth | 上涨家数占比 |
| market_vol_ratio | 全市场成交额/20日均值 |
| up_limit_count / down_limit_count | 涨停/跌停家数 |
| market_amplitude | 全市场平均振幅 |
| day_of_week | 周几 |
| large_amount_ratio | 大资金方向代理 |
| hot_sector_count | 热门板块数量 |

### 4.5 板块强度特征（6个）

| 特征 | 含义 |
|---|---|
| sector_ret_1d / sector_ret_5d | 所属板块当日/5日收益 |
| sector_rank | 板块强度排名 |
| stock_vs_sector | 个股相对板块强度 |
| sector_money_flow | 板块资金流排名 |
| sector_breadth | 板块内上涨比例 |

---

## 5. 模型设计

### 5.1 主模型：LightGBM Ranker

- 目标：LambdaRank 排序学习
- 标签：未来N日收益率横截面五档排名（0-4）
- 输出：rank_score → 归一化为 final_score (0~1)
- Fallback：sklearn HistGradientBoostingRegressor → 启发式模型

### 5.2 双时间框架

| 模型 | 标签 | 持有期 | 用途 |
|---|---|---|---|
| 10日模型（主） | 未来10日收益排名 | 10天 | 交易计划 |
| 5日模型（辅） | 未来5日收益排名 | 5天 | 对比参考 |

### 5.3 输出协议

```
code, date, rank_score, final_score, up_probability, down_probability,
neutral_probability, risk_score, suggest_action
```

---

## 6. 风控体系

### 6.1 市场级风控（回测内置）

| 规则 | 阈值 |
|---|---|
| 市场熔断 | 全市场均跌 > 2% 暂停买入 |
| 恐慌熔断 | 跌停家数 > 100 暂停 |
| 组合回撤暂停 | 5日回撤 > 10% 暂停 |
| 连亏熔断 | 连续亏损6次暂停5天 |

### 6.2 个股级风控

| 规则 | 说明 |
|---|---|
| 综合分门槛 | final_score < 0.45 不买 |
| ATR过高 | atr_14 > 8% 不买 |
| 弱势反抽 | 20日跌>10%且当日涨>7% 不追 |
| 涨停不追 | 开盘涨停不买入 |
| 跌停不卖 | 跌停时不卖出 |

### 6.3 组合级风控

| 规则 | 阈值 |
|---|---|
| 单票最大仓位 | 10% |
| 行业集中度 | 同板块 < 40% |
| 每日最多买入 | 5只 |
| 止盈 | 15% |
| 止损 | 7% |

---

## 7. 回测设计

- T+1：前一日信号 → 次日开盘买入
- 涨停不买、跌停不卖
- 手续费：买0.03% + 卖0.13%
- 滑点：0.1%
- 最小单位：100股
- 持有期：10天（到期开盘卖）

---

## 8. 数据层

### 8.1 数据源

- AkShare（主）+ BaoStock（备）
- Fallback 自动降级

### 8.2 存储

- 本地 CSV DataLake：`data_real_2000_10pct/daily/{code}.csv`
- 每只股票一个文件，增量更新

### 8.3 数据规格

| 字段 | 类型 | 说明 |
|---|---|---|
| date | 日期 | 交易日 |
| code | 字符串 | 6位代码 |
| open/high/low/close | 浮点 | OHLC |
| volume | 整数 | 成交量（股） |
| amount | 浮点 | 成交额（元） |
| turnover_rate | 浮点 | 换手率% |
| pct_chg | 浮点 | 涨跌幅% |

---

## 9. 运行方式

### 9.1 一键日常运行

```bash
./run_daily.sh                  # 完整运行（拉数据+模型+报告）
./run_daily.sh --skip-download  # 跳过下载，只跑模型
```

### 9.2 Pipeline 命令

```bash
.venv/bin/python -m stock_alpha.scripts.run_production pipeline \
  --config config/pipeline.strategy_10pct_2000_offline.json
```

### 9.3 单独下载

```bash
.venv/bin/python -m stock_alpha.scripts.run_production batch-download \
  --provider fallback --start 20220626 --end 20260626 \
  --data-root data_real_2000_10pct --limit 5000 --batch-size 50
```

---

## 10. 关键文件索引

| 模块 | 路径 |
|---|---|
| Pipeline 编排 | `stock_alpha/runtime/pipeline.py` |
| 配置 | `stock_alpha/config/settings.py` |
| 主配置 | `config/pipeline.strategy_10pct_2000_offline.json` |
| 数据下载 | `stock_alpha/data/downloader.py` |
| 预过滤 | `stock_alpha/data/pre_filter.py` |
| 批量下载 | `stock_alpha/data/batch.py` |
| 交易日历 | `stock_alpha/data/trade_calendar.py` |
| 质量检查 | `stock_alpha/data/quality.py` |
| 股票池 | `stock_alpha/universe.py` |
| V1特征 | `stock_alpha/features/v1_daily.py` |
| 横截面特征 | `stock_alpha/features/cross_sectional.py` |
| 市场环境特征 | `stock_alpha/features/market_env.py` |
| 板块特征 | `stock_alpha/features/sector_features.py` |
| 排序标签 | `stock_alpha/labels/ranking_label.py` |
| Ranker模型 | `stock_alpha/models/v2_ranker_model.py` |
| Classifier模型 | `stock_alpha/models/v1_daily_model.py` |
| 训练 | `stock_alpha/training/train_v1.py` |
| 回测 | `stock_alpha/backtest/ashare_backtest.py` |
| 风控规则 | `stock_alpha/risk_rules.py` |
| 交易计划 | `stock_alpha/trading_plan.py` |
| 市场分段分析 | `stock_alpha/backtest/regime_analysis.py` |
| 日报生成 | `stock_alpha/reports/daily_report.py` |
| 启动脚本 | `run_daily.sh` |

---

## 11. 核心指标（最新回测）

| 指标 | 10日模型 |
|---|---|
| 总收益 | 99.2% |
| 年化收益 | 19.6% |
| 最大回撤 | -22.7% |
| Sharpe | 1.13 |
| 胜率 | 50% |
| 盈亏比 | 1.59 |
| 交易次数 | 2,385 |
| 特征数 | 60 |
| 训练池 | ~4500只 |
| 交易池 | ~800-1500只 |

---

## 12. 后续优化方向

1. 接入真实北向资金/龙虎榜数据替换代理特征
2. 精确行业分类替换代码前缀粗分
3. 多模型融合（Ranker + Classifier 加权）
4. Walk-forward 严格无泄漏检验
5. 交易成本敏感性分析
6. 实盘模拟验证（paper trading）
