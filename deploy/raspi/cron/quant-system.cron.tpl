SHELL=/bin/bash

# quant-system Raspberry Pi cron template
# Placeholders in command lines:
# - ROOT path placeholder
# - PYTHON path placeholder
#
# Suggested timezone: Asia/Shanghai
# Example:
#   CRON_TZ=Asia/Shanghai
#   10 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/run_stock.py >> logs/cron_stock.log 2>&1

# 1) Stock daily cycle (workdays)
5 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/fetch_stock_data.py >> logs/cron_stock_fetch.log 2>&1
10 16 * * 1-5 cd __ROOT__ && __PYTHON__ scripts/run_stock.py >> logs/cron_stock.log 2>&1

# 2) Crypto cycle (every 4 hours)
5 */4 * * * cd __ROOT__ && __PYTHON__ scripts/fetch_crypto_data.py >> logs/cron_crypto_fetch.log 2>&1
7 */4 * * * cd __ROOT__ && __PYTHON__ scripts/run_crypto.py >> logs/cron_crypto.log 2>&1

# 3) Heavy validation cycle (weekly)
30 17 * * 6 cd __ROOT__ && __PYTHON__ scripts/backtest_v3.py >> logs/cron_backtest.log 2>&1
50 17 * * 6 cd __ROOT__ && __PYTHON__ scripts/validate_stock_cpcv.py >> logs/cron_cpcv.log 2>&1

# 4) Health notification (daily)
10 13 * * * cd __ROOT__ && __PYTHON__ scripts/notify_healthcheck_wecom.py >> logs/cron_health.log 2>&1
