SHELL=/bin/bash

# quant-system Raspberry Pi cron template
# Placeholders in command lines:
# - ROOT path placeholder
# - PYTHON path placeholder
#
# Suggested timezone: Asia/Shanghai
# Example:
#   CRON_TZ=Asia/Shanghai
#   10 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/run_stock_etf.py >> logs/cron_stock.log 2>&1

# 1) Stock daily cycle (workdays)
5 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/fetch_stock_etf_data.py >> logs/cron_stock_fetch.log 2>&1
10 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/stock_etf/run_stock_etf.py >> logs/cron_stock.log 2>&1

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
