#!/bin/bash
# ============================================================
# A股短线Alpha模型 - 每日运行脚本
# 
# 功能：
#   1. 增量拉取最新日线数据
#   2. 跑模型打分（Ranker）
#   3. 生成明日交易计划
#   4. 输出报告（HTML/Markdown）
#   5. 打印关键结论
#
# 用法：
#   ./scripts/run_daily.sh
# ============================================================

set -eo pipefail

# === 配置 ===
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${PROJECT_DIR}/data_real_2000_10pct"
CONFIG="${PROJECT_DIR}/config/pipeline.strategy_10pct_2000_offline.json"
VENV="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
TODAY=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/daily_${TODAY}.log"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# === 初始化 ===
mkdir -p "${LOG_DIR}"

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo -e "${msg}" | tee -a "${LOG_FILE}"
}

step() {
    local ts=$(date '+%H:%M:%S')
    echo "" | tee -a "${LOG_FILE}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "${LOG_FILE}"
    echo -e "${GREEN}[${ts}] ▶ $1${NC}" | tee -a "${LOG_FILE}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "${LOG_FILE}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}" | tee -a "${LOG_FILE}"
}

success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}" | tee -a "${LOG_FILE}"
}

cd "${PROJECT_DIR}"

# === 开始 ===
echo "" | tee -a "${LOG_FILE}"
echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${LOG_FILE}"
echo "║     A股短线Alpha模型 - 每日运行 ${TODAY}                    ║" | tee -a "${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${LOG_FILE}"
log "项目目录: ${PROJECT_DIR}"
log "数据目录: ${DATA_ROOT}"
log "日志文件: ${LOG_FILE}"

# 检查 Python 环境
if [ ! -f "${VENV}" ]; then
    error "Python 虚拟环境不存在: ${VENV}"
    exit 1
fi

# === Step 1: 数据增量更新 ===
# 设计原则：每次运行都尝试更新数据，从不整体跳过
#   - 终点始终是"最近已收盘的交易日"，与今天是否是交易日无关
#   - 各数据源内部自带缓存检测，自动跳过已下载部分（增量补缺）
step "Step 1/5: 数据增量更新"

# 计算下载终点：最近已收盘的交易日
# 今天15:15后则含今天；否则以前一天向前找最近交易日
END_DATE=$(${VENV} -c "
import datetime
import pandas as pd
from pathlib import Path

now = datetime.datetime.now()
today = now.date()
already_closed = now.hour > 15 or (now.hour == 15 and now.minute >= 15)
cutoff = today if already_closed else today - datetime.timedelta(days=1)

cal_path = Path('${DATA_ROOT}/meta/trade_calendar.csv')
last_trade_day = None
if cal_path.exists():
    try:
        cal = pd.read_csv(cal_path)
        if 'date' in cal.columns and not cal.empty:
            trade_dates = sorted(
                pd.to_datetime(cal['date'], errors='coerce').dt.date.dropna().tolist()
            )
            for d in reversed(trade_dates):
                if d <= cutoff:
                    last_trade_day = d
                    break
    except: pass

if last_trade_day is None:
    d = cutoff
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    last_trade_day = d

print(last_trade_day.strftime('%Y%m%d'))
" 2>/dev/null || echo "${TODAY}")

# 起点：往前两年（各 downloader 内部会对比本地缓存，只拉缺失部分）
START_DATE=$(${VENV} -c "
import datetime
end = datetime.datetime.strptime('${END_DATE}', '%Y%m%d').date()
print((end - datetime.timedelta(days=730)).strftime('%Y%m%d'))
" 2>/dev/null || echo "20240101")

log "数据更新范围: ${START_DATE} ~ ${END_DATE}"
log "（各数据源自动跳过已缓存部分，仅补充缺失数据段）"

# --- 日线增量更新 ---
log "► 更新日线数据..."
${VENV} -u -m stock_alpha.scripts.run_production batch-download \
    --provider fallback \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --data-root "${DATA_ROOT}" \
    --limit 5000 \
    --batch-size 50 2>&1 | tee -a "${LOG_FILE}"

if [ $? -eq 0 ]; then
    success "日线数据更新完成"
else
    warn "日线数据部分失败（不阻塞后续流程）"
fi

# --- extra 数据增量更新（龙虎榜 / 北向资金 / 基本面 / 融资融券）---
log "► 更新 extra 数据..."
${VENV} -u -m stock_alpha.scripts.run_production download-extra \
    --provider fallback \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --data-root "${DATA_ROOT}" 2>&1 | tee -a "${LOG_FILE}"

if [ $? -eq 0 ]; then
    success "extra 数据更新完成"
else
    warn "extra 数据部分失败（不阻塞后续流程）"
fi

# === Step 2-4: 跑完整 Pipeline（模型打分 + 交易计划 + 报告） ===
step "Step 2/5: 模型训练与打分 + 交易计划 + 报告"

log "启动完整 Pipeline..."
log "（训练 → 预测 → 回测 → 交易计划 → 报告，约需5-15分钟）"

${VENV} -u -c "
import json
from stock_alpha.config.settings import PipelineConfig
from stock_alpha.runtime.pipeline import FullPipeline

cfg = PipelineConfig.from_file('${CONFIG}')
cfg.skip_download = True  # 数据已由 Step 1 增量更新，pipeline 无需重复下载
cfg.min_train_days = 250
cfg.model_type = 'ranker'

result = FullPipeline(cfg).run()
print(json.dumps({'run_id': result['run_id'], 'markdown': str(result.get('markdown','')), 'html': str(result.get('html',''))}))
" 2>&1 | tee -a "${LOG_FILE}"

if [ $? -ne 0 ]; then
    error "Pipeline 运行失败！查看日志: ${LOG_FILE}"
    exit 1
fi

success "Pipeline 完成"

# === Step 5: 打印关键结论 + 操作清单 ===
step "Step 5/5: 操作清单"

${VENV} -u -c "
import pandas as pd
import glob, os
from stock_alpha.storage.cache import DataLake

lake = DataLake('${DATA_ROOT}')

# --- 获取股票名称 + 板块映射 ---
name_map = {}
sector_map = {}  # code -> 板块名
try:
    basics = lake.read_parquet('meta', 'stock_basic')
    if basics.empty:
        from stock_alpha.data.providers.akshare_provider import AkShareProvider
        basics = AkShareProvider().get_stock_basic()
    if not basics.empty:
        basics['code'] = basics['code'].astype(str).str.extract(r'(\d{6})', expand=False)
        name_map = dict(zip(basics['code'], basics['name']))
        if 'industry' in basics.columns:
            sector_map = dict(zip(basics['code'], basics['industry'].fillna('未知')))
except Exception:
    pass

# === 回测摘要（一行） ===
metrics = lake.read_parquet('backtest', 'metrics_latest')
stats = lake.read_parquet('backtest', 'trade_stats_latest')
if not metrics.empty and not stats.empty:
    m, t = metrics.iloc[0], stats.iloc[0]
    print(f'  📊 模型表现: 年化{m.get(\"annual_return\", 0):.1%} | 最大回撤{m.get(\"max_drawdown\", 0):.1%} | 胜率{t.get(\"win_rate\", 0):.1%} | 盈亏比{t.get(\"profit_loss_ratio\", 0):.2f} | Sharpe {m.get(\"sharpe\", 0):.2f}')
    print()

# === 明日操作清单 ===
orders = lake.read_parquet('orders', 'next_day_orders')
if not orders.empty:
    print('━' * 80)
    print(f'  🎯 明日操作清单（共 {len(orders)} 只，按模型分数排序）')
    print('━' * 80)
    print()
    for idx, (_, r) in enumerate(orders.head(10).iterrows(), 1):
        code = str(r['code']).zfill(6)
        name = name_map.get(code, '---')
        sector = sector_map.get(code, '未知')
        ref_price = float(r.get('ref_price', 0))
        tp = r.get('take_profit_price', None)
        sl = r.get('stop_loss_price', None)
        tp_val = float(tp) if pd.notna(tp) and tp else ref_price * 1.15
        sl_val = float(sl) if pd.notna(sl) and sl else ref_price * 0.93
        shares = int(r.get('shares', 0))
        amount = shares * ref_price
        score = float(r.get('score', 0))
        up_prob = r.get('up_probability', None)
        down_prob = r.get('down_probability', None)
        up_str = f'{float(up_prob):.0%}' if pd.notna(up_prob) else '---'
        down_str = f'{float(down_prob):.0%}' if pd.notna(down_prob) else '---'
        # 预计算确认条件
        min_open = ref_price * 0.98  # 不低开超过2%
        print(f'  [{idx}] {code} {name}    板块: {sector}')
        print(f'      模型分: {score:.4f} | 上涨概率: {up_str} | 下跌概率: {down_str}')
        print(f'      买入参考价: {ref_price:.2f}  |  股数: {shares}股  |  金额: {amount/10000:.1f}万')
        print(f'      止盈卖出: {tp_val:.2f} (+15%)  |  止损卖出: {sl_val:.2f} (-7%)  |  持有上限: 10个交易日')
        print(f'      ┈┈┈ 开盘确认条件（全部满足才买）┈┈┈')
        print(f'      ✓ 开盘价 >= {min_open:.2f} 元（不低开超2%）')
        print(f'      ✓ 9:30-9:45 不出现放量下杀（量比>2 且股价持续下跌 → 放弃）')
        print(f'      ✓ 9:45前股价站上 {ref_price:.2f}（昨收）')
        print(f'      ✓ 板块「{sector}」不弱势（见下方判断方法）')
        print()
    if len(orders) > 10:
        print(f'  ... 还有 {len(orders)-10} 只（详见完整报告）')
    print()

    # === 观察池摘要 ===
    watchlist = lake.read_parquet('orders', 'watchlist')
    if watchlist is not None and not watchlist.empty:
        print('━' * 80)
        print(f'  👀 观察池（{len(watchlist)} 只，盘中确认后可作为替补）')
        print('━' * 80)
        for _, w in watchlist.head(5).iterrows():
            wcode = str(w['code']).zfill(6)
            wname = name_map.get(wcode, '---')
            wscore = float(w.get('score', 0))
            print(f'      {wcode} {wname}  分数:{wscore:.4f}')
        if len(watchlist) > 5:
            print(f'      ... 还有 {len(watchlist)-5} 只')
        print()

else:
    print('  ⚠️  明日无交易计划（未通过风控或无信号，空仓等待）')
    print()

# === 操作指南 ===
print('━' * 80)
print('  📋 操作指南')
print('━' * 80)
print('''
  【时间节奏】
  ┌─────────┬────────────────────────────────────────────────────┐
  │ 15:15后  │ 运行本脚本，生成次日操作清单                         │
  │ 晚上     │ 阅读清单，标记拟买入标的                             │
  │ 次日9:15 │ 集合竞价观察：看开盘价是否满足最低开盘价条件            │
  │ 次日9:45 │ 运行确认脚本（自动检查4条件）：                        │
  │          │   .venv/bin/python scripts/confirm_open.py             │
  │          │ 输出「可买入」的标的 → 下单                            │
  │ 持有期间  │ 每日收盘后检查：是否触及止盈/止损价                    │
  └─────────┴────────────────────────────────────────────────────┘

  【买入规则】
  · 4个确认条件必须全部满足，缺一不买
  · 每日最多买入 5 只，优先买模型分数最高的
  · 单只仓位不超过总资金 10%

  【卖出规则】（三选一，哪个先到执行哪个）
  · 止盈：股价 >= 止盈价 → 当日收盘前卖出
  · 止损：股价 <= 止损价 → 立即卖出，不犹豫
  · 到期：持有满 10 个交易日 → 次日开盘卖出

  【板块强弱判断方法】
  本脚本已自动判断，运行 confirm_open.py 即可（无需手动查看）
  原理：通过东方财富实时板块行情接口自动获取：
  · 板块涨跌幅 < -1% → 弱势
  · 板块内下跌家数 > 上涨家数的 2 倍 → 弱势
  如匹配不到板块，需手动在同花顺/东财确认

  【特殊情况处理】
  · 大盘跳水（沪指跌>2%）→ 全部暂停买入
  · 开盘即涨停 → 不追，放弃
  · 集合竞价一字跌停 → 持仓不卖（卖不出），次日再处理
''')

# 报告路径
reports = sorted(glob.glob('reports/daily_report_*.html'))
if reports:
    abs_report = os.path.abspath(reports[-1])
    print(f'  📄 完整报告: open {abs_report}')
    print()
" 2>&1 | tee -a "${LOG_FILE}"

# === 完成 ===
echo "" | tee -a "${LOG_FILE}"
echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${LOG_FILE}"
echo "║                    运行完成！                                ║" | tee -a "${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${LOG_FILE}"
log "完整日志: ${LOG_FILE}"
