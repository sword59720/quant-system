#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行交易指令脚本
读取 stock_trades.json 并执行实际下单 (支持国投/掘金)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入适配器
from adapters.guotou_trader import create_trader as create_guotou_trader, Order, OrderSide
from adapters.myquant_trader import create_trader as create_myquant_trader


def create_trader_factory(config: dict):
    """
    根据配置创建对应的交易器实例
    工厂函数根据 'broker' 字段分发
    """
    broker = config.get("broker", "guotou")
    
    # 根据 broker 类型选择适配器
    if broker == "myquant":
        # 掘金量化适配器
        return create_myquant_trader(config)
    else:
        # 默认国投证券适配器 (EMP/GRT)
        return create_guotou_trader(config)


def setup_logging():
    """设置日志"""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"execute_trades_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("execute_trades")


def load_trades(file_path: str) -> dict:
    """加载交易指令"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_config() -> dict:
    """加载配置"""
    import yaml
    
    # 1. 加载 runtime.yaml (环境配置)
    runtime_file = "./config/runtime.yaml"
    with open(runtime_file, 'r', encoding='utf-8') as f:
        runtime = yaml.safe_load(f)
    
    # 获取运行时参数
    broker_type = runtime.get("broker", "guotou")  # 默认国投
    env = runtime.get("env", "paper")
    total_capital = runtime.get("total_capital", 20000)
    
    # 2. 加载 broker.yaml (账户配置)
    broker_file = "./config/broker.yaml"
    broker_full_config = {}
    if os.path.exists(broker_file):
        with open(broker_file, 'r', encoding='utf-8') as f:
            broker_full_config = yaml.safe_load(f)
    
    # 3. 提取对应券商的配置
    if broker_type == "myquant":
        config = broker_full_config.get("myquant", {})
        config["platform"] = "myquant"
    else:
        config = broker_full_config.get("guotou", {})
        # 国投可能有 emp 或 traditional 平台
        config["platform"] = config.get("platform", "emp")
    
    # 4. 合并运行时参数
    config["env"] = env
    config["total_capital"] = total_capital
    config["broker"] = broker_type
    
    return config


def execute_trades(trades_file: str, dry_run: bool = False):
    """
    执行交易指令主流程
    """
    logger = setup_logging()
    
    # 加载配置
    config = load_config()
    
    # dry_run 强制覆盖环境为 paper
    if dry_run:
        config["env"] = "paper"
    
    env = config.get("env", "paper")
    broker_name = config.get("broker", "unknown").upper()
    
    logger.info(f"=" * 60)
    logger.info(f"开始执行交易")
    logger.info(f"环境: {env.upper()}")
    logger.info(f"券商: {broker_name}")
    logger.info(f"=" * 60)
    
    # 实盘时间检查 (仅 live 模式)
    if env == "live":
        now = datetime.now()
        current_time = now.time()
        is_trading_hours = (
            (current_time.hour == 9 and current_time.minute >= 30) or
            (current_time.hour == 10) or
            (current_time.hour == 11 and current_time.minute <= 30) or
            (current_time.hour == 13) or
            (current_time.hour == 14)
        )
        if not is_trading_hours:
            logger.warning("⚠️ 当前不在A股交易时间内，实盘订单可能无法成交")
    
    # -------------------------------------------------
    # 核心：创建交易器实例
    # -------------------------------------------------
    try:
        trader = create_trader_factory(config)
    except Exception as e:
        logger.error(f"❌ 创建交易器失败: {e}")
        return False
    
    # 连接交易服务器
    if not trader.connect():
        logger.error("❌ 连接交易服务器失败")
        return False
    
    try:
        # 加载交易指令
        if not os.path.exists(trades_file):
            logger.error(f"❌ 交易指令文件不存在: {trades_file}")
            return False
        
        trades = load_trades(trades_file)
        orders = trades.get("orders", [])
        
        if not orders:
            logger.info("ℹ️ 没有需要执行的交易指令")
            return True
        
        logger.info(f"📋 发现 {len(orders)} 条交易指令")
        
        # 获取账户信息
        try:
            account = trader.get_account_info()
            logger.info(f"💰 账户可用资金: ¥{account.get('available_cash', 0):.2f}")
        except Exception as e:
            logger.warning(f"⚠️ 获取资金失败: {e}")
        
        # 执行交易循环
        executed_orders = []
        failed_orders = []
        
        for i, trade in enumerate(orders, 1):
            symbol = trade.get("symbol")
            action = trade.get("action")
            amount = trade.get("amount_quote", 0)
            
            logger.info(f"\n[{i}/{len(orders)}] 处理交易: {action} {symbol} ¥{amount:.2f}")
            
            # 构造订单对象
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                amount=amount
            )
            
            # 下单
            result = trader.place_order(order)
            
            if result.status.value in ["filled", "submitted", "partial_filled"]:
                executed_orders.append({
                    "symbol": symbol,
                    "action": action,
                    "amount": amount,
                    "order_id": result.order_id,
                    "status": result.status.value
                })
                logger.info(f"  ✅ 成功 - 订单ID: {result.order_id}")
            else:
                failed_orders.append({
                    "symbol": symbol,
                    "action": action,
                    "amount": amount,
                    "error": result.error_msg
                })
                logger.error(f"  ❌ 失败 - {result.error_msg}")
            
            # 实盘限流
            if env == "live":
                import time
                time.sleep(0.5)
        
        # 同步持仓
        pos_file = "./outputs/state/stock_positions.json"
        try:
            trader.sync_positions(pos_file)
            logger.info(f"\n💾 持仓已同步到: {pos_file}")
        except Exception as e:
            logger.warning(f"⚠️ 同步持仓失败: {e}")
        
        # 统计结果
        logger.info(f"\n" + "=" * 60)
        logger.info(f"执行结果统计")
        logger.info(f"=" * 60)
        logger.info(f"✅ 成功: {len(executed_orders)} 笔")
        logger.info(f"❌ 失败: {len(failed_orders)} 笔")
        
        # 保存记录
        execution_record = {
            "ts": datetime.now().isoformat(),
            "env": env,
            "broker": broker_name,
            "trades_file": trades_file,
            "total_orders": len(orders),
            "success": len(executed_orders),
            "failed": len(failed_orders),
            "executed": executed_orders,
            "failed_details": failed_orders
        }
        
        record_file = f"./outputs/orders/execution_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(execution_record, f, ensure_ascii=False, indent=2)
        logger.info(f"\n📝 执行记录已保存: {record_file}")
        
        return len(failed_orders) == 0
        
    finally:
        trader.disconnect()


def main():
    parser = argparse.ArgumentParser(description="执行交易指令")
    parser.add_argument(
        "--file", "-f",
        default="./outputs/orders/stock_trades.json",
        help="交易指令文件路径"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="模拟执行（不实际下单）"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认，直接执行"
    )
    
    args = parser.parse_args()
    
    # 预览
    if os.path.exists(args.file):
        with open(args.file, 'r', encoding='utf-8') as f:
            trades = json.load(f)
        
        print("\n" + "=" * 60)
        print("交易指令预览")
        print("=" * 60)
        print(f"文件: {args.file}")
        print(f"总资金: ¥{trades.get('capital_total', 0):.2f}")
        
        orders = trades.get("orders", [])
        if not orders:
            print("  (无)")
        else:
            for i, order in enumerate(orders, 1):
                print(f"  {i}. {order['action']:4} {order['symbol']}  "
                      f"金额: ¥{order.get('amount_quote', 0):,.2f}")
        print("=" * 60)
        
        # 确认
        if not args.yes and not args.dry_run:
            config = load_config()
            env = config.get("env", "paper")
            broker = config.get("broker", "guotou")
            
            if env == "live":
                print(f"\n⚠️ 警告: 当前配置为 LIVE 实盘模式！券商: {broker}")
            
            confirm = input(f"\n确认执行{' (模拟)' if args.dry_run else ''}? [y/N]: ")
            if confirm.lower() != 'y':
                print("已取消")
                return
    else:
        print(f"❌ 文件不存在: {args.file}")
        return
    
    # 执行
    success = execute_trades(args.file, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
