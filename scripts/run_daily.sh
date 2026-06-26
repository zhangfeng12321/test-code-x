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
#   ./scripts/run_daily.sh --skip-download   # 跳过下载，只跑模型
# ============================================================

set -e

# === 配置 ===
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
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
    echo "" | tee -a "${LOG_FILE}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "${LOG_FILE}"
    echo -e "${GREEN}▶ $1${NC}" | tee -a "${LOG_FILE}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}" | tee -a "${LOG_FILE}"
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "${LOG_FILE}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "${LOG_FILE}"
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

# === Step 1: 增量拉取最新日线数据 ===
SKIP_DOWNLOAD=false
if [[ "$1" == "--skip-download" ]]; then
    SKIP_DOWNLOAD=true
fi

if [ "$SKIP_DOWNLOAD" = false ]; then
    step "Step 1/5: 增量拉取最新日线数据"
    
    # 计算日期范围：从最近数据到今天
    END_DATE="${TODAY}"
    # 往前推7天确保覆盖（增量逻辑会自动跳过已有数据）
    START_DATE=$(date -v-7d +%Y%m%d 2>/dev/null || date -d "7 days ago" +%Y%m%d 2>/dev/null || echo "${TODAY}")
    
    log "下载范围: ${START_DATE} ~ ${END_DATE}"
    log "开始增量下载..."
    
    ${VENV} -u -m stock_alpha.scripts.run_production batch-download \
        --provider fallback \
        --start "${START_DATE}" \
        --end "${END_DATE}" \
        --data-root "${DATA_ROOT}" \
        --limit 5000 \
        --batch-size 50 2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        success "数据下载完成"
    else
        warn "部分数据下载失败（不阻塞后续流程）"
    fi
else
    step "Step 1/5: 跳过数据下载 (--skip-download)"
    log "使用本地已有数据"
fi

# === Step 2-4: 跑完整 Pipeline（模型打分 + 交易计划 + 报告） ===
step "Step 2/5: 模型训练与打分 (Ranker)"
step "Step 3/5: 生成明日交易计划"
step "Step 4/5: 输出报告"

log "启动完整 Pipeline..."
log "（训练 → 预测 → 回测 → 交易计划 → 报告，约需5-15分钟）"

PIPELINE_OUTPUT=$(${VENV} -u -c "
import json
from stock_alpha.config.settings import PipelineConfig
from stock_alpha.runtime.pipeline import FullPipeline

cfg = PipelineConfig.from_file('${CONFIG}')
cfg.skip_download = True  # 数据已在 Step 1 更新
cfg.min_train_days = 250
cfg.model_type = 'ranker'

result = FullPipeline(cfg).run()
print(json.dumps({'run_id': result['run_id'], 'markdown': str(result.get('markdown','')), 'html': str(result.get('html',''))}))
" 2>&1 | tee -a "${LOG_FILE}")

if [ $? -ne 0 ]; then
    error "Pipeline 运行失败！查看日志: ${LOG_FILE}"
    exit 1
fi

success "Pipeline 完成"

# === Step 5: 打印关键结论 ===
step "Step 5/5: 关键结论"

${VENV} -u -c "
import pandas as pd
from stock_alpha.storage.cache import DataLake

lake = DataLake('${DATA_ROOT}')

# 回测指标
metrics = lake.read_parquet('backtest', 'metrics_latest')
if not metrics.empty:
    m = metrics.iloc[0]
    print(f'  📊 回测指标:')
    print(f'     总收益: {m.get(\"total_return\", 0):.2%}')
    print(f'     年化收益: {m.get(\"annual_return\", 0):.2%}')
    print(f'     最大回撤: {m.get(\"max_drawdown\", 0):.2%}')
    print(f'     Sharpe: {m.get(\"sharpe\", 0):.3f}')
    print()

# 交易统计
stats = lake.read_parquet('backtest', 'trade_stats_latest')
if not stats.empty:
    t = stats.iloc[0]
    print(f'  📈 交易统计:')
    print(f'     胜率: {t.get(\"win_rate\", 0):.2%}')
    print(f'     盈亏比: {t.get(\"profit_loss_ratio\", 0):.3f}')
    print()

# 明日交易计划
orders = lake.read_parquet('orders', 'next_day_orders')
if not orders.empty:
    print(f'  🎯 明日交易计划 ({len(orders)} 只):')
    print()
    for _, r in orders.head(10).iterrows():
        print(f'     {r[\"code\"]}  {r.get(\"action\",\"BUY\")}  参考价:{r.get(\"ref_price\",0):.2f}  股数:{int(r.get(\"shares\",0))}  分数:{r.get(\"score\",0):.4f}')
    if len(orders) > 10:
        print(f'     ... 还有 {len(orders)-10} 只')
else:
    print('  ⚠️  明日无交易计划（未通过风控或无信号）')

print()

# 报告路径
import glob
reports = sorted(glob.glob('reports/daily_report_*.html'))
if reports:
    print(f'  📄 报告路径:')
    print(f'     {reports[-1]}')
    print(f'     打开: open {reports[-1]}')
" 2>&1 | tee -a "${LOG_FILE}"

# === 完成 ===
echo "" | tee -a "${LOG_FILE}"
echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${LOG_FILE}"
echo "║                    运行完成！                                ║" | tee -a "${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${LOG_FILE}"
log "完整日志: ${LOG_FILE}"
