# 个股量化系统顶层设计 V1

## 1. 目标与边界
- 目标: 在不破坏现有 ETF 生产系统的前提下，新增一套面向个股的日内小时级评估交易系统。
- 交易对象: A 股个股，股票池动态生成，池内上限 100 只。
- 评估频率: 盘中每小时，输出买卖建议、价格区间与目标仓位。
- 执行模式: 提醒模式 / 半自动模式 / 全自动模式（分阶段上线）。
- 非目标: 当前版本不做高频毫秒级交易，不做全市场全自动高换手策略。

## 2. 与 ETF 系统隔离原则
- ETF 逻辑目录: `scripts/stock_etf/`
- 个股逻辑目录: `scripts/stock_single/`
- ETF 运行主入口: `scripts/stock_etf/run_stock_etf.py`（研究/回测脚本同目录）。
- 个股独立入口: `scripts/stock_single/run_stock_single.py`
- 配置分离:
  - ETF: `config/stock.yaml`
  - 个股: `config/stock_single.yaml`
- 输出分离:
  - ETF: `outputs/orders/stock_targets.json`
  - 个股: `outputs/orders/stock_single_pool.json`, `outputs/orders/stock_single_signals.json`

## 3. 系统分层
- Universe Layer: 股票池生成与维护（EOD 运行）。
- Feature Layer: 多源特征构建（价格、成交量、资金流、估值、风险）。
- Alpha Layer: 个股打分与动作判定（BUY/SELL/HOLD）。
- Price & Sizing Layer: 生成入场/离场区间、止损、目标仓位。
- Risk Layer: 组合级与个股级风控总闸（可否决信号）。
- Execution Layer: 交易路由（提醒/自动），订单切片与成交回报。
- Ops Layer: 监控、告警、审计、日报周报。

## 4. 关键数据契约
### 4.1 股票池文件
- 路径: `outputs/orders/stock_single_pool.json`
- 字段:
  - `ts`
  - `mode`
  - `symbol_count`
  - `symbols`

### 4.2 小时评分快照
- 路径: `data/stock_single/hourly_scores_latest.csv`
- 必填列:
  - `symbol`
  - `score`
  - `last_price`
  - `atr14`
- 说明: 后续可扩展为 `score_long`, `score_short`, `confidence`, `feature_version`。

### 4.3 信号输出文件
- 路径: `outputs/orders/stock_single_signals.json`
- 字段:
  - `ts`, `market`, `mode`, `pool_size`
  - `signals[]`:
    - `symbol`
    - `action` (`BUY`/`SELL`/`HOLD`)
    - `score`
    - `last_price`
    - `entry_price`
    - `exit_price`
    - `stop_price`
    - `target_weight`
    - `reason`

### 4.4 快风控快照文件
- 路径: `data/stock_single/intraday_risk_latest.csv`
- 必填列:
  - `symbol`
  - `ret_5m`
- 可选列:
  - `weight`（组合权重）
  - `vol_zscore`（波动异常 z 分数）

### 4.5 估值与资金流因子文件
- 路径:
  - `data/stock_single/valuation/<symbol>.csv`
  - `data/stock_single/fund_flow_real/<symbol>.csv`（真实资金流，评分默认使用）
  - `data/stock_single/fund_flow_proxy/<symbol>.csv`（代理资金流，历史回补）
  - `data/stock_single/fund_flow/<symbol>.csv`（合并结果）
- 关键列:
  - 估值: `pe_ttm`, `pb`
  - 资金流: `main_net_inflow`, `main_net_inflow_ratio`

## 5. 调度与运行时序
### 5.1 日终任务（15:10-15:40）
- 任务 1: 数据补齐与质量检查（行情、资金流、基础面）。
- 任务 2: 生成下一交易日候选池（最多 100 只）。
- 任务 3: 保存池文件与变更日志（新增/移除/原因）。

### 5.2 盘中任务（每小时）
- 建议时点: 09:45, 10:45, 11:15, 13:45, 14:45。
- 流程:
  - 读取股票池与小时特征快照。
  - 计算信号分数与动作。
  - 计算价格区间、止损价、目标仓位。
  - 通过风控闸门后输出提醒或下单指令。

## 6. 信号与仓位模型（V1）
### 6.1 信号评分
- 采用线性可解释模型起步:
  - 动量（20/60）
  - 成交量异常（1h/5h）
  - 资金流（主力/北向）
  - 风险惩罚（短期波动、跳空风险）
- 输出 `score`，通过阈值映射为 `BUY/SELL/HOLD`。

### 6.2 价格区间
- BUY:
  - `entry_price = last_price * (1 + buy_buffer_bps)`
  - `stop_price = entry_price - k1 * ATR14`
  - `exit_price = entry_price + k2 * ATR14`
- SELL:
  - `exit_price = last_price * (1 - sell_buffer_bps)`

### 6.3 仓位控制
- 单票上限: `single_max_pct`
- 信号仓位: `per_signal_target_weight`
- 组合持仓上限: `max_positions`
- 组合总资金: `capital_alloc_pct`

## 7. 风控体系（必须独立）
- 预交易:
  - ST/停牌/涨跌停可交易性过滤
  - 最低流动性与成交金额过滤
  - 黑名单（公告风险、连续异常波动）
- 盘中:
  - 单票止损
  - 组合回撤阈值（20d/60d）
  - 日内风险事件总闸（异常波动、接口故障）
- 盘后:
  - 成交偏离分析（滑点、拒单率、漏单率）
  - 模型漂移监控（信号命中率、IC 衰减）

## 8. 执行模式
- `alert_only`:
  - 仅推送提醒，不自动下单。
- `semi_auto`:
  - 生成订单草案，人工确认后执行。
- `auto_trade`:
  - 自动下单，但保留风控总闸与人工紧急停机开关。

## 9. 可观测性与审计
- 每次评估落盘:
  - 触发时间
  - 输入快照版本
  - 信号结果
  - 风控判定
  - 订单与回执
- 报告:
  - 日报: 信号数、成交率、胜率、滑点、净值变化
  - 周报: 因子表现、池子周转、风险事件复盘

## 10. 上线路径（建议）
- Phase A（1-2 周）: 仅提醒，验证信号稳定性。
- Phase B（2-4 周）: 半自动，小资金验证执行质量。
- Phase C（4 周+）: 自动交易，逐步放量。

上线门槛建议:
- Paper 阶段:
  - 信号可用率 >= 99%
  - 严重风控漏拦截 = 0
  - 日志完整率 >= 99.5%
- 实盘阶段:
  - 拒单率 <= 5%
  - 平均滑点在目标阈值内
  - 回撤与 VaR 不突破预设阈值

## 11. 开发任务拆分
- T1: 完成股票池自动构建（质量检查 + 滞回机制）。
- T2: 完成小时特征快照生产脚本。
- T3: 完成信号引擎与风险闸门联动。
- T4: 完成提醒推送（企业微信）与订单草案输出。
- T5: 对接自动交易适配器（仅在 `auto_trade` 开启时）。
- T6: 回测 + Paper 验证 + 报告闭环。

## 12. 当前仓库映射（V1 已落地部分）
- 已有:
  - `scripts/stock_single/fetch_stock_single_data.py`
  - `scripts/stock_single/backtest_stock_single.py`
  - `scripts/stock_single/build_pool.py`
  - `scripts/stock_single/eval_hourly.py`
  - `scripts/stock_single/risk_check_fast.py`
  - `scripts/stock_single/run_stock_single.py`
  - `scripts/stock_single/fetch_stock_single_data.py`
  - `scripts/stock_single/backtest_stock_single.py`
  - `scripts/stock_single/run_stock_single.py --task hourly`
  - `scripts/stock_single/run_stock_single.py --task risk`
  - `config/stock_single.yaml`
- 下一步重点:
  - 新增小时特征生产脚本
  - 新增风险闸门模块
  - 新增信号 -> 订单草案转换模块
