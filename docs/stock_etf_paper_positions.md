# stock_etf 纸面持仓维护

当前 ETF 策略采用：
- `config/runtime.yaml` 中建议显式配置 `strategy_execution.stock_etf: paper_manual`
- 策略生成目标仓位
- 系统生成调仓指令
- 飞书发送人工跟单消息
- 人工执行后，手动更新纸面持仓

纸面持仓文件：

```text
outputs/state/stock_positions.json
```

## 查看当前纸面持仓
```bash
python3 scripts/stock_etf/update_stock_positions.py --show
```

## 通过 CSV 导入持仓
```bash
python3 scripts/stock_etf/update_stock_positions.py --from-csv docs/examples/stock_positions.template.csv
```

## 通过命令行直接设置
```bash
python3 scripts/stock_etf/update_stock_positions.py --set 159949:0.49568:32000 --set 518880:0.49203:4500
```

## 一键运行纸面调仓流程
```bash
python3 scripts/stock_etf/run_stock_etf_paper_cycle.py
```

## 调仓通知中的估算数量
调仓通知除金额外，还会附带：
- 最新收盘价作为参考价
- 按 A 股 100 股一手估算的下单数量

说明：这是方便人工跟单的估算值，实际成交股数仍以你下单时价格为准。
