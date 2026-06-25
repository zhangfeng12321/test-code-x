# A-Share Short-Term Alpha Model

实现范围：

- **V1**：日线 + 技术指标 + LightGBM/备用模型短线选股框架
- **V2**：分钟级量能 / VWAP / 分时强弱特征与评分框架
- **V4**：Level-2 / 盘口订单簿抽象接口、CSV Provider、盘口特征、DeepLOB 框架占位

> V4 真实 Level-2 数据源通常需要付费/券商权限。本项目先完成抽象接口、数据结构、特征和模型框架。

## 快速验证

```bash
cd projects/stock-alpha-model
python3 -m stock_alpha.cli smoke
```

## 可选依赖

```bash
pip install -e '.[data,ml,dev]'
```

未安装 `lightgbm/scikit-learn` 时，V1 会自动降级为纯 numpy/pandas 的启发式模型，保证框架可运行。

## 目录

```text
stock_alpha/
├── data/providers/       # 数据源抽象 + AkShare/BaoStock/CSV Level2 Provider
├── features/             # V1/V2/V4 特征工程
├── labels/               # Triple Barrier 标签
├── models/               # V1/V2/V4 模型框架
├── backtest/             # 简单组合回测
├── framework/            # Pipeline 编排
└── cli.py                # smoke/demo CLI
```

## 设计原则

业务层只依赖 `MarketDataProvider` 抽象接口，后续替换付费数据源不改模型逻辑。
