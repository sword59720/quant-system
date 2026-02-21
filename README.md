# Quant System 使用说明（股票 + 币圈）

> 适用环境：树莓派 4B（2GB）+ OpenClaw Gateway 已运行
>
> 当前策略模式：
> - 股票：A股 ETF（周频调仓）
> - 币圈：HTX 直连（每4小时计算）
> - 全局：Paper 模式（模拟）

---

## 1. 项目目录

项目根目录：

```bash
/home/haojc/.openclaw/workspace/quant-system
```

核心目录：

- `config/`：配置文件
- `scripts/`：执行脚本
- `scripts/stock_etf/`：ETF 股票策略逻辑（已隔离）
- `scripts/stock_single/`：个股策略逻辑（已隔离）
- `data/`：行情数据
- `outputs/orders/`：目标仓位输出
- `outputs/reports/`：巡检报告
- `logs/`：运行日志

---

## 2. 首次安装（仅一次）

```bash
cd /home/haojc/.openclaw/workspace/quant-system
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

---

## 3. 配置说明

### 3.1 总开关（强烈建议）

文件：`config/runtime.yaml`

```yaml
enabled: true
```

- `true`：系统运行
- `false`：系统停机（即使 cron 触发也会立即退出）

### 3.2 关键配置

- `config/stock.yaml`：股票策略参数（ETF池、动量窗口、top_n）
- `config/stock_single.yaml`：个股策略参数（池化与小时信号，默认关闭）
- `config/crypto.yaml`：币圈参数（现货/合约统一配置，当前默认 `exchange: htx_direct`）
- `config/risk.yaml`：风控参数（仓位上限、回撤阈值等）

`config/crypto.yaml` 新增自动下单开关（默认关闭）：

```yaml
execution:
  auto_place_order: false
  min_order_notional: 10
```

- `auto_place_order=true` 且 `config/runtime.yaml` 中 `env: live` 时，`scripts/crypto/run_crypto.py` 会在生成目标仓位后自动生成并执行 `crypto_trades.json`。
- `min_order_notional` 用于过滤过小订单（单位 USDT）。

---

## 4. 手动运行

## 4.1 股票（拉数据 + 生成目标）

```bash
cd /home/haojc/.openclaw/workspace/quant-system
./.venv/bin/python scripts/stock_etf/fetch_stock_etf_data.py
./.venv/bin/python scripts/stock_etf/run_stock_etf.py
```

> 说明：ETF 数据/信号/研究脚本统一位于 `scripts/stock_etf/`。

## 4.2 个股模型（构建股票池 + 小时信号，默认关闭）

```bash
cd /home/haojc/.openclaw/workspace/quant-system
./.venv/bin/python scripts/stock_single/fetch_stock_single_data.py
./.venv/bin/python scripts/stock_single/run_stock_single.py
```

> 启用前请先在 `config/stock_single.yaml` 设置 `enabled: true`（用于实盘信号）。
> 数据抓取和回测可在 `enabled: false` 下执行。
> `scripts/stock_single/run_stock_single.py` 默认执行 `full`（建池 + 快风控 + 小时信号）。
> `stock_single` 仅允许A股个股代码（按交易所前缀校验），ETF和指数代码会被自动过滤/拒绝。
> 信号触发支持 `signal.trigger_mode: threshold | quantile | topk`，默认使用 `quantile` 动态阈值。

按任务运行：

```bash
# 日终建池（建议收盘后）
./.venv/bin/python scripts/stock_single/run_stock_single.py --task pool

# 选股逻辑（build_pool）
# 1) 硬过滤: ST/历史不足/停牌特征/近端极端涨跌
# 2) 流动性: 近20日均成交额与成交量门槛
# 3) 因子预筛: 与回测同口径打分（动量/估值/资金流/波动/布林）
# 4) 入池出池: entry_quantile / exit_quantile + min_pool_hold_days
# 5) 行业约束: max_industry_weight（可选 industry_file）
# 配置位置: config/stock_single.yaml -> pool.selection / liquidity_filter / industry_diversification

# 仅抓个股回测/信号数据（支持增量续抓）
./.venv/bin/python scripts/stock_single/fetch_stock_single_data.py

# 仅抓 PE/PB + 资金流因子层
./.venv/bin/python scripts/stock_single/fetch_stock_single_data.py --only-factors

# 临时指定标的抓取（不依赖 universe.csv）
./.venv/bin/python scripts/stock_single/fetch_stock_single_data.py --only-factors --symbols 600519.SH,000001.SZ,601318.SH

# 抓分钟线（在 daily 基础上追加）
./.venv/bin/python scripts/stock_single/fetch_stock_single_data.py --with-minute

# 个股回测（日线代理小时信号）
./.venv/bin/python scripts/stock_single/backtest_stock_single.py

# 临时指定标的回测（不依赖 universe.csv）
./.venv/bin/python scripts/stock_single/backtest_stock_single.py --symbols 600519.SH,000001.SZ,601318.SH

# 强制要求 PE/资金流覆盖率达标（否则回测直接失败）
./.venv/bin/python scripts/stock_single/backtest_stock_single.py --require-factors

# 小时信号（盘中每小时）
./.venv/bin/python scripts/stock_single/run_stock_single.py --task hourly

# 快风控（建议每5分钟）
./.venv/bin/python scripts/stock_single/run_stock_single.py --task risk
```

输入文件约定：

```text
data/stock_single/hourly_scores_latest.csv
  必填列: symbol, score, last_price, atr14
  说明: 由 `scripts/stock_single/eval_hourly.py` 基于日线+估值+资金流自动生成（回测/实盘同口径）

data/stock_single/intraday_risk_latest.csv
  必填列: symbol, ret_5m
  可选列: weight, vol_zscore

data/stock_single/valuation/*.csv
  估值列: pe_ttm, pb

data/stock_single/fund_flow_real/*.csv
  真实资金流（评分默认读取）: main_net_inflow, main_net_inflow_ratio

data/stock_single/fund_flow_proxy/*.csv
  代理资金流（baostock 估算，主要用于历史回补）

data/stock_single/fund_flow/*.csv
  合并资金流（按 `fund_flow_merge_mode` 生成；可用于诊断/实验）
```

## 4.3 币圈（HTX直连拉数据 + 生成目标）

```bash
cd /home/haojc/.openclaw/workspace/quant-system
./.venv/bin/python scripts/crypto/fetch_crypto_data.py
./.venv/bin/python scripts/crypto/run_crypto.py
```

> 当 `execution.auto_place_order=true` 且 `env=live` 时，`run_crypto.py` 会自动下单并同步持仓。

## 4.4 一键全流程

```bash
cd /home/haojc/.openclaw/workspace/quant-system
./.venv/bin/python scripts/run_quant.py
```

模块开关可选（默认都为 `true`，即 ETF + 个股 + Crypto 全部运行）：

```bash
# 仅 ETF
./.venv/bin/python scripts/run_quant.py --stock-etf=true --stock-single=false --crypto=false

# 仅个股
./.venv/bin/python scripts/run_quant.py --stock-etf=false --stock-single=true --crypto=false

# 仅 Crypto
./.venv/bin/python scripts/run_quant.py --stock-etf=false --stock-single=false --crypto=true

# ETF + Crypto（关闭个股）
./.venv/bin/python scripts/run_quant.py --stock-single=false
```

---

## 5. 自动运行（cron，树莓派）

统一模板位置：

- `deploy/raspi/cron/quant-system.cron.tpl`

推荐运行频率（Asia/Shanghai）：

- 股票数据：工作日 16:05
- 股票信号：工作日 16:10
- 个股数据：工作日 15:05
- 个股建池：工作日 15:15
- 个股小时信号：09:45 / 10:45 / 11:15 / 13:45 / 14:45
- 个股快风控：交易时段每 5 分钟
- 币圈数据：每4小时第5分钟（执行前 `source setenv.sh`）
- 币圈信号：每4小时第7分钟（执行前 `source setenv.sh`）
- 重型验证：每周六 17:30（`backtest_stock_etf`）+ 17:40（`backtest_crypto`）+ 17:50（`backtest_stock_etf_cpcv`）
- 健康告警：每天 13:10

安装到树莓派 crontab：

```bash
cd /home/pi/quant-system
./scripts/install_raspi_cron.sh \
  --target-root /home/pi/quant-system \
  --python /home/pi/quant-system/.venv/bin/python \
  --apply
```

仅渲染 cron 文件（不安装）：

```bash
./scripts/install_raspi_cron.sh --target-root /home/pi/quant-system
```

查看当前计划：

```bash
crontab -l
```

---

## 6. 停止 / 启动系统

已提供快捷脚本：

- `scripts/stop_system.sh`
- `scripts/start_system.sh`

用法：

```bash
cd /home/haojc/.openclaw/workspace/quant-system
./scripts/stop_system.sh
./scripts/start_system.sh
```

这两个脚本本质是切换 `config/runtime.yaml` 中的 `enabled`。

---

## 7. 输出文件说明

- 股票目标仓位：`outputs/orders/stock_targets.json`
- 币圈目标仓位：`outputs/orders/crypto_targets.json`
- 币圈交易指令：`outputs/orders/crypto_trades.json`
- 实盘执行回执：`outputs/orders/execution_record_*.json`
- 巡检结果：`outputs/reports/healthcheck_latest.log`
- 股票回测快照：`outputs/reports/stock_backtest_snapshot.csv`
- 股票回测快照指纹：`outputs/reports/stock_backtest_snapshot_meta.json`

查看示例：

```bash
cat outputs/orders/stock_targets.json
cat outputs/orders/crypto_targets.json
cat outputs/reports/healthcheck_latest.log
```

---

## 8. 日志与排错

常看日志：

```bash
tail -n 100 logs/cron_stock.log
tail -n 100 logs/cron_crypto.log
tail -n 100 logs/cron_health.log
```

快速筛错：

```bash
grep -E "Traceback|ERROR|failed" logs/cron_*.log | tail -n 50
```

---

## 9. 当前策略逻辑（简述）

- 股票：ETF **多因子**（动量+低波动+回撤+流动性）综合打分，选 Top N（当前2个）
- 股票风险开关：基准ETF（510300）若低于 MA120，则股票总仓位减半
- 币圈：BTC/ETH **多因子**（动量+低波动+回撤）打分，选 Top N（当前1个）
- 币圈模型分档：
  - `spot_model`：用于现货模式（`trade.market_type=spot`）
  - `contract_model`：用于合约模式（`trade.market_type=swap/future`）
- 合约模型（`engine: advanced_rmm`）：
  - 核心：趋势过滤（MA）+ 风险调整动量（momentum/vol）+ 目标波动动态杠杆 + 回撤节流
  - 目标：在保持回撤可控的前提下提升年化收益
- 币圈交易模式：
  - `trade.market_type: spot`：现货交易，risk_off 可转 USDT
  - `trade.market_type: swap/future`：合约交易，支持杠杆/保证金模式；risk_off 默认平仓到 0 仓
  - 合约可选 `trade.allow_short=true` + `signal.short_on_risk_off=true`：risk_off 时切换做空组合
- 仓位：按 `capital_alloc_pct` 与单标的上限分配

> 注意：当前为 Paper 模式，输出的是目标仓位，不是实盘下单回执。

---

## 10. 常见问题

### Q1：为什么脚本触发了但没动作？
A：检查 `config/runtime.yaml` 是否 `enabled: false`。

### Q2：币圈没有数据或分数为空？
A：先手动执行：

```bash
./.venv/bin/python scripts/crypto/fetch_crypto_data.py
```

再看 `data/crypto/*.csv` 是否生成。

### Q3：如何临时停机但保留 cron？
A：执行：

```bash
./scripts/stop_system.sh
```

恢复时：

```bash
./scripts/start_system.sh
```

### Q4：为什么 backtest 和 CPCV 有时看起来不是同一批数据？
A：默认流程是先运行 `backtest_stock_etf.py` 生成 `stock_backtest_snapshot.csv`，再由 `backtest_stock_etf_cpcv.py` 复用同一快照。若你想让 CPCV 强制重建快照，可执行：

```bash
./.venv/bin/python scripts/stock_etf/backtest_stock_etf_cpcv.py --force-refresh-snapshot
```

---

## 11. 实盘切换前检查清单（强烈建议）

> 当前系统默认 `env: paper`。切换实盘前，请逐项确认：

- [ ] 过去 2~4 周 paper 运行稳定（无连续报错）
- [ ] `logs/cron_*.log` 无高频 `Traceback/ERROR/failed`
- [ ] 股票/币圈数据文件持续更新（非陈旧文件）
- [ ] 风控阈值已确认（`config/risk.yaml`）
- [ ] 交易所/券商 API 权限最小化（只开交易，禁提现）
- [ ] API Key 不写入代码仓库（仅本机安全存储）
- [ ] 已设置最大单日亏损/回撤处理预案
- [ ] 已完成小额实盘演练（先小资金）
- [ ] 明确人工接管流程（紧急停机：`./scripts/stop_system.sh`）

### 实盘切换最小步骤

1. 修改 `config/runtime.yaml`：

```yaml
env: live
enabled: true
```

2. 手动单次试跑（先非交易时段验证流程）：

```bash
./.venv/bin/python scripts/crypto/fetch_crypto_data.py
./.venv/bin/python scripts/crypto/run_crypto.py
```

3. 确认输出与日志正常，再开放自动调度。

> 建议：先用小仓位实盘 1~2 周，再逐步增加资金。
