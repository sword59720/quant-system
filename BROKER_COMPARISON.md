# 实盘接入方案对比与选择指南

## 📊 三种方案对比

| 维度 | 国投证券EMP | 掘金量化(MyQuant) | 国投GRT |
|------|-------------|-------------------|---------|
| **类型** | 券商自研策略托管 | 第三方量化平台 | 券商自研量化平台 |
| **AlphaT(T0)** | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| **ACT(算法)** | ✅ 支持 | ⚠️ 部分支持 | ✅ 支持 |
| **数据服务** | 基础 | ✅ 丰富 | 基础 |
| **回测引擎** | 无 | ✅ 完善 | ✅ 有 |
| **接入难度** | 中 | 低 | 低 |
| **资金门槛** | 50万+ | 5-20万 | 10万+ |
| **适用策略** | 中高频、T0 | 中低频、多因子 | 全频段 |
| **稳定性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 方案选择建议

### 选择【掘金量化】如果您：
- ✅ 需要丰富的历史数据和回测功能
- ✅ 策略频率不高（日频/周频）
- ✅ 希望快速接入，不想开发太多代码
- ✅ 资金量中等（5-20万）
- ✅ 需要可视化界面监控策略

### 选择【国投证券EMP】如果您：
- ✅ 需要T0算法增强收益（AlphaT）
- ✅ 资金量大（50万+）
- ✅ 有中高频策略需求
- ✅ 需要算法交易执行（ACT）
- ✅ 希望券商级别的低延迟

### 选择【国投GRT】如果您：
- ✅ 已经是国投证券客户
- ✅ 希望一站式解决（数据+回测+交易）
- ✅ 使用通达信/大智慧风格的公式选股
- ✅ 不想切换多个平台

---

## 🚀 掘金量化接入指南

### 第一步：注册与签约

1. **注册掘金量化账号**
   - 访问: https://www.myquant.cn/
   - 完成注册和实名认证

2. **选择合作券商开户/签约**
   
   掘金量化支持的券商列表：
   - 东方财富、广发证券、招商证券
   - 国盛证券、东兴证券、万和证券
   - 英大证券、华源证券、信达证券
   - 申港证券、华鑫证券、东北证券
   - 五矿证券、华林证券、华龙证券
   - 华创证券、天风证券
   
   ⚠️ **注意**：国投证券（原安信证券）**暂不支持**掘金量化
   
   如果您的账户在国投证券，需要：
   - 方案A：在掘金量化支持的券商新开账户
   - 方案B：使用国投EMP或GRT平台

3. **申请实盘权限**
   - 不同券商资金门槛不同（通常5-20万）
   - 联系券商客户经理申请

---

### 第二步：安装掘金SDK

```bash
# 安装掘金量化Python SDK
pip install gm

# 验证安装
python -c "import gm; print(gm.__version__)"
```

---

### 第三步：配置quant-system

编辑 `config/broker.yaml`：

```yaml
myquant:
  name: "掘金量化"
  platform: "myquant"
  token: "您的掘金token"        # 从掘金官网获取
  account_id: "您的资金账号"
  trade_mode: "live"            # paper | live
```

编辑 `config/runtime.yaml`：

```yaml
# 选择掘金量化平台
broker: "myquant"               # guotou | myquant
env: "live"                     # paper | live
```

---

### 第四步：修改执行脚本

编辑 `scripts/stock_etf/execute_trades_stock_etf.py`，修改 `load_config()` 函数：

```python
def load_config() -> dict:
    """加载配置"""
    import yaml
    
    runtime_file = "./config/runtime.yaml"
    with open(runtime_file, 'r', encoding='utf-8') as f:
        runtime = yaml.safe_load(f)
    
    broker = runtime.get("broker", "guotou")
    
    # 加载对应券商配置
    broker_file = "./config/broker.yaml"
    with open(broker_file, 'r', encoding='utf-8') as f:
        broker_config = yaml.safe_load(f)
    
    if broker == "myquant":
        config = broker_config.get("myquant", {})
        config["platform"] = "myquant"
    else:
        config = broker_config.get("guotou", {})
        config["platform"] = broker_config.get("guotou", {}).get("platform", "emp")
    
    config["env"] = runtime.get("env", "paper")
    config["total_capital"] = runtime.get("total_capital", 20000)
    
    return config
```

---

### 第五步：修改交易适配器导入

编辑 `scripts/stock_etf/execute_trades_stock_etf.py`，修改导入部分：

```python
from adapters.guotou_trader import create_trader as create_guotou_trader
from adapters.myquant_trader import create_trader as create_myquant_trader

def create_trader(config: dict):
    """根据配置创建对应交易器"""
    platform = config.get("platform", "guotou")
    if platform == "myquant":
        return create_myquant_trader(config)
    else:
        return create_guotou_trader(config)
```

---

### 第六步：测试与运行

```bash
cd /home/haojc/.openclaw/workspace/quant-system

# 1. 模拟测试
./.venv/bin/python scripts/stock_etf/trade_stock_etf.py --dry-run

# 2. 实盘运行
./.venv/bin/python scripts/stock_etf/trade_stock_etf.py
```

---

## ⚠️ 重要提醒

### 关于国投证券与掘金量化

**国投证券（原安信证券）目前不在掘金量化的合作券商列表中**。

如果您坚持要在国投证券账户上跑量化，您有两个选择：

#### 选择A：使用国投自有平台（推荐）

| 平台 | 特点 | 门槛 |
|------|------|------|
| **EMP** | 策略托管、T0算法、算法交易 | 50万+ |
| **GRT** | 可视化策略、回测、交易 | 10万+ |

#### 选择B：在掘金量化合作券商开户

如果您想使用掘金量化的完善功能，可以：
1. 在掘金量化官网查看合作券商列表
2. 选择一家券商新开账户
3. 资金转入新账户运行策略

---

## 📋 三种方案详细文档

- [国投EMP接入指南](./GUOTOU_EMP_GUIDE.md)
- [掘金量化接入指南](#掘金量化接入指南)（本文档）
- 国投GRT接入指南（待补充）

---

## 🤔 如何选择？

### 决策树

```
您的账户在哪家券商？
│
├─ 国投证券 ────────┐
│                   │
│  资金 > 50万？    │
│  ├─ 是 → 国投EMP │
│  └─ 否 → 国投GRT │
│
└─ 其他券商 ────────┐
    │               │
    是否在掘金列表？│
    ├─ 是 → 掘金量化│
    └─ 否 → 该券商自有平台或换券商
```

### 我的建议

针对您的 **quant-system ETF轮动策略**：

| 如果您的资金 | 建议方案 | 理由 |
|-------------|----------|------|
| < 10万 | 掘金量化+新开账户 | 门槛低，功能完善 |
| 10-50万 | 国投GRT | 一站式，无需换券商 |
| > 50万 | 国投EMP | 可用ACT算法执行调仓 |

---

## 📞 联系方式

- **掘金量化客服**: https://www.myquant.cn/
- **国投证券客服**: 95517
- **国投EMP/GRT**: 联系您的客户经理

---

**您决定使用哪个方案？我可以帮您进一步完善对应平台的接入代码。**
