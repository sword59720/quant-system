# 国投证券实盘交易切换指南

## 🎯 当前状态

- ✅ 交易适配器已创建 (`adapters/guotou_trader.py`)
- ✅ 交易执行脚本已创建 (`scripts/stock_etf/execute_trades_stock_etf.py`)
- ✅ 券商配置模板已创建 (`config/broker.yaml`)
- ✅ 一键交易脚本已创建 (`scripts/stock_etf/trade_stock_etf.py`)

## 📝 切换实盘前必须完成的步骤

### 步骤 1: 获取国投证券 API 信息

联系国投证券客户经理或拨打客服电话，获取以下信息：

1. **API 类型确认**（以下之一）：
   - QMT/MiniQMT（迅投）
   - 国投自研 API
   - 算法交易 SDK

2. **账户信息**：
   - 资金账号
   - API Key / Secret（如有）
   - API 端点地址
   - 相关文档或 SDK

### 步骤 2: 配置券商信息

编辑 `config/broker.yaml`：

```yaml
guotou:
  name: "国投证券"
  account_id: "您的资金账号"        # 必填
  api_key: "您的API Key"            # 如有
  api_secret: "您的API Secret"      # 如有
  api_endpoint: "API端点地址"       # 如有
  trade_mode: "live"
  order_type: "MARKET"
```

### 步骤 3: 修改交易适配器

编辑 `adapters/guotou_trader.py` 中的 `GuotouLiveTrader` 类。

根据国投提供的 API 类型，替换以下 TODO 部分：

#### 如果是 QMT/MiniQMT：

```python
# 在 connect() 方法中
from xtquant.xttrader import XtQuantTrader
from xtquant import xtconstant

self.trade_client = XtQuantTrader(self.api_endpoint, self.account_id)
self.trade_client.start()
connect_result = self.trade_client.connect()
if connect_result == 0:
    self.is_connected = True
    return True

# 在 place_order() 方法中
order_id = self.trade_client.order_stock(
    self.account_id,
    stock_code=order.symbol,
    order_type=xtconstant.STOCK_BUY if order.side == OrderSide.BUY else xtconstant.STOCK_SELL,
    order_volume=int(order.amount / current_price / 100) * 100,
    price_type=xtconstant.LATEST_PRICE,
    price=0
)
```

#### 如果是国投自研 REST API：

```python
import requests

# 在 connect() 方法中
response = requests.post(
    f"{self.api_endpoint}/auth/login",
    json={"account_id": self.account_id, "api_key": self.api_key}
)
self.token = response.json()["token"]

# 在 place_order() 方法中
response = requests.post(
    f"{self.api_endpoint}/trade/order",
    headers={"Authorization": f"Bearer {self.token}"},
    json={
        "symbol": order.symbol,
        "side": order.side.value,
        "amount": order.amount,
        "order_type": "MARKET"
    }
)
order_id = response.json()["order_id"]
```

### 步骤 4: 切换 runtime 配置

编辑 `config/runtime.yaml`：

```yaml
# 将 env 从 paper 改为 live
env: live
```

### 步骤 5: 测试

#### 5.1 模拟测试（推荐先跑一周）

```bash
cd /home/haojc/.openclaw/workspace/quant-system

# 执行模拟交易（不实际下单）
./.venv/bin/python scripts/stock_etf/trade_stock_etf.py --dry-run
```

#### 5.2 查看生成的交易指令

```bash
cat outputs/orders/stock_trades.json
```

#### 5.3 实盘试运行（小额）

```bash
# 修改 config/runtime.yaml 中的 total_capital 为小额资金（如 10000）
# 然后执行实盘交易
./.venv/bin/python scripts/stock_etf/trade_stock_etf.py
```

### 步骤 6: 自动化运行（可选）

添加到 cron 定时任务：

```bash
# 编辑 crontab
crontab -e

# 添加（每周一 15:00 收盘后执行）
0 15 * * 1 cd /home/haojc/.openclaw/workspace/quant-system && ./.venv/bin/python scripts/stock_etf/trade_stock_etf.py --yes >> logs/cron_trade.log 2>&1
```

## ⚠️ 风险控制

### 已内置的风控措施

1. **执行守护 (Execution Guard)**
   - 最小再平衡间隔: 30 天
   - 最小换手率: 3%
   - 防止过度交易

2. **风险覆盖层 (Risk Overlay)**
   - 策略回撤超过 12% 触发防御
   - 超额收益连续 20 天低于 -2% 触发防御

3. **曝光门 (Exposure Gate)**
   - CPCV 交叉验证
   - 不达标时自动降低仓位

### 实盘额外建议

1. **设置止损线**
   - 单日最大亏损: 5%
   - 策略最大回撤: 15%

2. **资金分批**
   - 首次实盘建议使用 20% 资金
   - 稳定运行一个月后再增加

3. **监控报警**
   - 已配置企业微信通知
   - 每笔交易都会发送通知

## 🆘 紧急处理

如需立即停止自动交易：

```bash
# 方法 1: 停止系统
./scripts/stop_system.sh

# 方法 2: 切换到模拟模式
# 编辑 config/runtime.yaml
env: paper

# 方法 3: 禁用 cron
# 注释掉 crontab 中的相关行
crontab -e
```

## 📞 需要帮助？

1. **国投证券技术支持**: 联系您的客户经理
2. **quant-system 文档**: 查看 README.md
3. **日志排错**: `tail -f logs/execute_trades_*.log`

## ✅ 实盘前检查清单

- [ ] 已获取国投证券 API 信息
- [ ] 已配置 `config/broker.yaml`
- [ ] 已修改 `adapters/guotou_trader.py` 中的 API 调用
- [ ] 已设置 `config/runtime.yaml` 中 `env: live`
- [ ] 已完成模拟测试（至少一周）
- [ ] 已设置合理的初始资金（建议小额）
- [ ] 已配置监控通知
- [ ] 已知悉紧急停止方法

---

**准备好后，运行以下命令开始实盘：**

```bash
cd /home/haojc/.openclaw/workspace/quant-system
./.venv/bin/python scripts/stock_etf/trade_stock_etf.py
```
