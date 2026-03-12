# stock_single 交易核心逻辑说明

- 文档版本: `v2026.03.03`
- 文档日期: `2026-03-03`
- 代码基线: `1bf75d8`
- 适用配置: `config/stock_single.yaml`
- 适用范围: A 股个股策略 `stock_single`（不包含 ETF 逻辑）

## 1. 系统定位

`stock_single` 采用规则化量化框架，主链路为:

1. 日终构建股票池（Pool）。
2. 盘中按小时计算评分并生成 `BUY/SELL/HOLD` 信号。
3. 每 5 分钟执行快风控，覆盖信号动作（禁开仓/强制卖出）。
4. 执行层将信号映射为目标权重和订单。

主入口脚本:
- `scripts/stock_single/run_stock_single.py`

## 2. 任务编排

`run_stock_single.py` 支持 4 个任务:

1. `pool`: 仅构建股票池。
2. `hourly`: 仅做小时信号评估。
3. `risk`: 仅做 5 分钟快风控。
4. `full`: 依次执行 `pool -> risk -> hourly`。

## 3. 选股池逻辑（build_pool）

脚本: `scripts/stock_single/build_pool.py`

### 3.1 候选来源

1. 从 `pool.source_file` 读取候选（默认 `data/stock_single/universe.csv`）。
2. 仅保留 A 股个股代码，自动剔除 ETF/指数/非法代码。

### 3.2 硬过滤

按 `pool.selection` 配置执行:

1. 历史长度不足 `min_history_days` 剔除。
2. `st_filter=true` 时剔除名称含 `ST`。
3. `suspension_lookback_days` 内零成交比例超阈值剔除。
4. `recent_limit_lookback_days` 内涨跌停命中次数超 `max_recent_limit_hits` 剔除。

### 3.3 流动性过滤

按 `liquidity_filter` 执行:

1. 近 `lookback_days` 平均成交额 >= `min_avg_turnover`。
2. 近 `lookback_days` 平均成交量 >= `min_avg_volume`。

### 3.4 打分入池与滞回

1. 使用与回测同口径的因子评分（见第 4 节）。
2. 按得分排序后，使用:
   `entry_quantile / exit_quantile + min_pool_hold_days`
   实现“入池/出池滞回”，降低换手抖动。
3. 目标池大小由 `target_pool_size` 控制，并受 `max_symbols_in_pool` 上限约束。

### 3.5 行业约束

若 `industry_diversification.enabled=true`:

1. 单行业占比不超过 `max_industry_weight`。
2. 尽量满足最少行业数 `min_industry_count`。

## 4. 评分因子逻辑（scoring_core）

脚本: `scripts/stock_single/scoring_core.py`

### 4.1 核心因子

1. 价格动量: `mom20`, `mom60`, `rev5`
2. 波动与量能: `vol20`, `vol_spike`, `vol_change`, `volume_change`
3. 估值: `ep(1/PE)`, `bp(1/PB)`
4. 资金流: `main_flow`（5日滚动）
5. 情绪: `sentiment`（价格变化 * 量能变化）
6. BOLL 派生: `boll_percent`, `boll_width`, `boll_break`

### 4.2 计算方式

1. 所有因子先做横截面 zscore 标准化。
2. 使用 `backtest.score_weights` 线性加权求总分 `score_df`。
3. 当前配置下 BOLL 三项权重均为 `0.00`，等效关闭。

### 4.3 实时修正

1. 若 `volatility_adjusted_signal=true`，分数会除以当期 `vol20`（风险归一化）。
2. 若 `time_series_momentum_window` 窗口内时序动量为负，分数乘 `0.5` 惩罚。

## 5. 小时信号逻辑（eval_hourly）

脚本: `scripts/stock_single/eval_hourly.py`

### 5.1 阈值模式

`signal.trigger_mode` 支持:

1. `threshold`: 固定阈值
2. `quantile`: 分位阈值
3. `topk`: Top-K 动态阈值（当前配置）

当前配置（`topk`）:

1. 买入阈值 = 第 `buy_top_k` 名得分（默认 15）。
2. 卖出阈值 = 第 `hold_top_k` 名得分（默认 25）。

### 5.2 动作判定

单只股票按以下顺序判断:

1. 若在 `force_sell_symbols` 中，动作 `SELL`。
2. 否则 `score >= buy_threshold`，动作 `BUY`。
3. 否则 `score <= sell_threshold`，动作 `SELL`。
4. 其余为 `HOLD`。

若快风控触发 `block_new_buys=true`，所有 `BUY` 会被改写为 `HOLD`。

### 5.3 价格与仓位

1. 买入参考价: `entry_price = last_price * (1 + buy_buffer_bps)`
2. 卖出参考价: `exit_price = last_price * (1 - sell_buffer_bps)`
3. 止损/止盈参考: 基于 `ATR14` 计算
4. 目标权重: `min(per_signal_target_weight, single_max_pct)`

## 6. 快风控逻辑（risk_check_fast）

脚本: `scripts/stock_single/risk_check_fast.py`

输入快照: `data/stock_single/intraday_risk_latest.csv`

触发条件（三选一触发即可）:

1. 组合 5 分钟收益 <= `trigger_portfolio_ret_5m`
2. 任一个股 5 分钟收益 <= `trigger_single_ret_5m`
3. 任一个股 `vol_zscore >= trigger_vol_zscore`

触发后动作:

1. `block_new_buys`（禁开新仓）
2. `force_sell_symbols`（对暴跌个股强制卖出，可配置关闭）

输出:

1. `outputs/orders/stock_single_risk_state.json`
2. `outputs/orders/stock_single_risk_alerts.json`

## 7. 执行层映射（QMT 原生脚本）

脚本: `strategies/qmt_stock_single_native.py`

执行要点:

1. 根据目标权重与当前持仓权重差生成订单。
2. 先卖后买，减少资金占用。
3. 100 股整数手取整。
4. 卖出不超过可用持仓。
5. 支持最小订单金额和单次最多订单数限制。
6. 可用 `DRY_RUN` 先联调再实盘。

## 8. 当前关键参数快照（来自配置）

1. `max_positions=10`
2. `target_pool_size=60`
3. `buy_top_k=15`
4. `hold_top_k=25`
5. `per_signal_target_weight=0.10`
6. `single_max_pct=0.15`
7. `capital_alloc_pct=1.00`
8. 快风控阈值: `-1.2% / -2.5% / zscore 3.5`

## 9. 产物文件

1. 股票池: `outputs/orders/stock_single_pool.json`
2. 小时评分快照: `data/stock_single/hourly_scores_latest.csv`
3. 小时信号: `outputs/orders/stock_single_signals.json`
4. 快风控状态: `outputs/orders/stock_single_risk_state.json`
5. 快风控告警: `outputs/orders/stock_single_risk_alerts.json`

## 10. 版本维护建议

后续每次修改核心逻辑（选股、因子、阈值、风控、执行规则）时，至少同步更新:

1. 本文档头部 `文档版本`。
2. 本文档头部 `文档日期`。
3. 本文档头部 `代码基线`（提交哈希）。
