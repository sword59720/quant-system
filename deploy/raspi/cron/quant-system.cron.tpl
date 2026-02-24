SHELL=/bin/bash
CRON_TZ=Asia/Shanghai

# quant-system Raspberry Pi cron template
# Placeholders in command lines:
# - ROOT path placeholder
# - PYTHON path placeholder
#
# Timezone fixed to Asia/Shanghai for A-share trading.

# 1) Stock ETF cycle (workdays, manual execution mode)
#   - 16:05 拉取收盘数据
#   - 16:10 计算目标仓位（T日收盘）
#   - 16:12 生成交易指令（基于本地持仓差分）
#   - 16:13 推送交易指令到企业微信（人工在券商端执行）
5 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/fetch_stock_etf_data.py >> logs/cron_stock_fetch.log 2>&1
10 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/run_stock_etf.py >> logs/cron_stock_signal.log 2>&1
12 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/generate_trades_stock_etf.py >> logs/cron_stock_trade_gen.log 2>&1
13 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/notify_stock_trades_wecom.py >> logs/cron_stock_trade_notify.log 2>&1

# Optional: Single-stock cycle (enable when config/stock_single.yaml enabled=true)
# 5 15 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_single/fetch_stock_single_data.py >> logs/cron_stock_single_fetch.log 2>&1
# 15 15 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_single/run_stock_single.py --task pool >> logs/cron_stock_single_pool.log 2>&1
# 45 9,10,13,14 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_single/run_stock_single.py --task hourly >> logs/cron_stock_single_hourly.log 2>&1
# 15 11 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_single/run_stock_single.py --task hourly >> logs/cron_stock_single_hourly.log 2>&1
# */5 9-11,13-14 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_single/run_stock_single.py --task risk >> logs/cron_stock_single_risk.log 2>&1

# 2) Crypto cycle (every 4 hours)
5 */4 * * * cd __ROOT__ && source ./setenv.sh && __PYTHON__ scripts/crypto/fetch_crypto_data.py >> logs/cron_crypto_fetch.log 2>&1
7 */4 * * * cd __ROOT__ && source ./setenv.sh && __PYTHON__ scripts/crypto/run_crypto.py >> logs/cron_crypto.log 2>&1

# 3) Heavy validation cycle (weekly)
30 17 * * 6 cd __ROOT__ && __PYTHON__ scripts/stock_etf/backtest_stock_etf.py >> logs/cron_backtest_stock_etf.log 2>&1
40 17 * * 6 cd __ROOT__ && __PYTHON__ scripts/crypto/backtest_crypto.py >> logs/cron_backtest_crypto.log 2>&1
50 17 * * 6 cd __ROOT__ && __PYTHON__ scripts/stock_etf/backtest_stock_etf_cpcv.py >> logs/cron_cpcv.log 2>&1

# 4) Health notification (daily)
10 13 * * * cd __ROOT__ && __PYTHON__ scripts/notify_healthcheck_wecom.py >> logs/cron_health.log 2>&1
