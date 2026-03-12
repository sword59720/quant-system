# stock_etf 量化交易逻辑说明

- 文档版本: `v2026.03.03`
- 文档日期: `2026-03-03`（Asia/Shanghai）
- 策略代码基线: `1bf75d8`（`stock_etf应用候选A并新增按代码限仓能力`）
- 适用配置: `config/stock.yaml`（当前主配置）

## 1. 策略目标与范围

- 交易对象: A 股/跨市场 ETF 组合（当前池见第 8 节）。
- 模式: `global_momentum`（中低频调仓）。
- 目标优先级:
  1. 年化收益率最大化
  2. MDD 约束（当前控制目标 `>= -15%`）
  3. Sharpe 作为监控指标

## 2. 执行入口

- 数据拉取: `scripts/stock_etf/fetch_stock_etf_data.py`
- 信号与目标权重生成: `scripts/stock_etf/run_stock_etf.py`
- 回测（与生产逻辑对齐）: `scripts/stock_etf/backtest_stock_etf.py`
- 生产回放核心引擎: `scripts/stock_etf/paper_forward_stock.py`

## 3. 信号与选股逻辑

核心参数来自 `config/stock.yaml -> global_model`。

- 基础参数:
  - `rebalance_days=5`
  - `momentum_lb=160`
  - `ma_window=200`
  - `vol_window=35`
  - `top_n=2`（再被 regime/phase2 动态调整）
- 评分:
  - 长动量 + 结构化动量混合:
    - 结构化动量窗口: `[20,60,180]`
    - 权重: `[0.5,0.3,0.2]`
    - `score_blend=0.15`
  - 风险调整分数: `score = momentum / vol`
- 状态识别:
  - 基准趋势（`510300` 相对 `MA200`）
  - 广度（趋势向上且复合动量为正的比例）
  - 状态: `risk_on / neutral / risk_off`

## 4. 仓位与权重分配

### 4.1 状态参数（regime profiles）

- `risk_on`: `top_n=3, min_score=0.01, score_power=0.85, alloc_mult=1.05`
- `neutral`: `top_n=1, min_score=0.06, score_power=1.20, alloc_mult=0.70`

### 4.2 结构化仓位调节（structural_upgrade）

- 趋势与广度乘数调节（上下限约束）。
- Phase2 波动/恢复调节:
  - `phase2_vol_mult_stress=0.90`
  - `phase2_total_mult_min=0.60`
  - `phase2_total_mult_max=1.20`
- 自适应 top_n 与 score_power（由波动状态与分散度触发）。

### 4.3 风险治理（risk_governor）

- 目标波动: `target_daily_vol=0.0095`
- 总乘数范围: `[0.45, 0.70]`
- 回撤触发: `dd_trigger=-0.08`，触发后上限压到 `dd_alloc_mult=0.20`
- 动量触发: `momentum_trigger=0.0`（当前不额外收缩，因为 `momentum_alloc_mult=1.0`）

### 4.4 最终权重落地

- 候选资产先按风险权重分配，再应用单标的上限。
- 剩余仓位补到防御资产（`518880`）。
- 当前主配置 `symbol_weight_caps` 未启用（可选能力已实现）。

## 5. 交易执行保护

- `execution_guard`:
  - 最短调仓间隔: `20` 天
  - 最小换手阈值: `1%`
  - 仅当达到阈值才执行调仓，降低噪声交易。
- `risk_overlay`（独立防御开关）:
  - 若 20 日超额或策略回撤触发阈值，强制转防守仓位
  - 满足释放条件后再恢复风险仓位
  - `sticky_mode=true`（防抖）

## 6. 回测计算口径

- 交易成本: `fee=0.0008`（按换手扣费）。
- 关键指标:
  - 年化收益 `annual_return`
  - 最大回撤 `max_drawdown`
  - Sharpe
  - 相对 alloc 基准超额收益
- OOS 窗口:
  - `oos_2023`: start `2023-01-03`
  - `oos_2024`: start `2024-01-02`
  - `oos_2025`: start `2025-01-02`

## 7. 当前基线表现（最新回测快照）

来源: `outputs/reports/backtest_stock_etf_report.json`（2026-03-03 生成）。

- Full period（2014-05-27 ~ 2026-02-13）:
  - 年化: `32.97%`
  - MDD: `-14.94%`
  - Sharpe: `0.844`
- OOS:
  - `oos_2023`: 年化 `40.26%`，MDD `-11.53%`，Sharpe `2.13`
  - `oos_2024`: 年化 `54.73%`，MDD `-11.53%`，Sharpe `2.50`
  - `oos_2025`: 年化 `49.28%`，MDD `-11.53%`，Sharpe `2.14`

## 8. 当前交易池（主配置）

`config/stock.yaml -> universe`

- `159633`（新能车ETF）
- `159915`（创业板ETF）
- `159949`（创业板50ETF）
- `159819`（人工智能ETF）
- `510300`（沪深300ETF）
- `510500`（中证500ETF）
- `512100`（中证1000ETF）
- `513100`（纳指ETF）
- `518880`（黄金ETF）

## 9. 维护约定

- 每次策略结构变更或关键参数变更，更新本文件的:
  - 文档版本
  - 文档日期
  - 策略代码基线（commit）
- 每次基线切换后，附上最新 `backtest_stock_etf_report.json` 指标快照。

