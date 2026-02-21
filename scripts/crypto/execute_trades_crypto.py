#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行币圈交易指令脚本（现货/合约）。"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.crypto_trader import Order as CryptoOrder
from adapters.crypto_trader import OrderSide as CryptoOrderSide
from adapters.crypto_trader import create_trader as create_crypto_trader


def create_crypto_market_trader(config: dict):
    return create_crypto_trader(config)


def setup_logging():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"execute_crypto_trades_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("execute_crypto_trades")


def load_trades(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict:
    import yaml

    with open("./config/runtime.yaml", "r", encoding="utf-8") as f:
        runtime = yaml.safe_load(f)

    env = runtime.get("env", "paper")
    total_capital = runtime.get("total_capital", 20000)

    broker_full_config = {}
    if os.path.exists("./config/broker.yaml"):
        with open("./config/broker.yaml", "r", encoding="utf-8") as f:
            broker_full_config = yaml.safe_load(f)

    config = broker_full_config.get("crypto", {})
    config["broker"] = "crypto"
    config["env"] = env
    config["total_capital"] = total_capital
    config["_runtime_paths"] = runtime.get("paths", {})

    if os.path.exists("./config/crypto.yaml"):
        with open("./config/crypto.yaml", "r", encoding="utf-8") as f:
            crypto_cfg = yaml.safe_load(f)
        if isinstance(crypto_cfg, dict):
            config.update(crypto_cfg)

    return config


def build_crypto_order(trade: dict):
    symbol = trade.get("symbol")
    action_raw = str(trade.get("action", "BUY")).strip().upper()
    amount = float(trade.get("amount_quote", 0) or 0)

    buy_actions = {"BUY", "OPEN_LONG", "CLOSE_SHORT"}
    sell_actions = {"SELL", "OPEN_SHORT", "CLOSE_LONG"}
    if action_raw not in buy_actions | sell_actions:
        raise ValueError(f"unsupported crypto action: {action_raw}")

    side = CryptoOrderSide.BUY if action_raw in buy_actions else CryptoOrderSide.SELL
    reduce_only = action_raw in {"CLOSE_LONG", "CLOSE_SHORT"}
    position_side = None
    if action_raw in {"OPEN_LONG", "CLOSE_LONG"}:
        position_side = "LONG"
    elif action_raw in {"OPEN_SHORT", "CLOSE_SHORT"}:
        position_side = "SHORT"

    return CryptoOrder(
        symbol=symbol,
        side=side,
        amount=amount,
        reduce_only=reduce_only,
        position_side=position_side,
    )


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _status_value(order_obj):
    status = getattr(order_obj, "status", None)
    if hasattr(status, "value"):
        return str(status.value).strip().lower()
    return str(status or "unknown").strip().lower()


def _load_latest_close(csv_file: str):
    if not os.path.exists(csv_file):
        return None
    last = None
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = _safe_float(row.get("close"))
            if x is not None and x > 0:
                last = x
    return last


def _reference_price(config: dict, symbol: str, cache: dict):
    key = f"crypto:{symbol}"
    if key in cache:
        return cache[key]

    data_dir = config.get("_runtime_paths", {}).get("data_dir", "./data")
    sym = str(symbol).replace("/", "_").replace(":", "_")
    fp = os.path.join(data_dir, "crypto", f"{sym}.csv")
    px = _load_latest_close(fp)
    cache[key] = px
    return px


def _calc_slippage_bps(action: str, ref_price, fill_price):
    ref = _safe_float(ref_price)
    fill = _safe_float(fill_price)
    if ref is None or fill is None or ref <= 0:
        return None

    a = str(action or "").strip().upper()
    side = 1.0 if a in {"BUY", "OPEN_LONG", "CLOSE_SHORT"} else -1.0
    raw_bps = (fill - ref) / ref * 10000.0
    return float(raw_bps * side)


def _build_execution_metrics(order_results: list):
    status_count = {}
    for x in order_results:
        s = str(x.get("status", "unknown")).strip().lower()
        status_count[s] = status_count.get(s, 0) + 1

    total = len(order_results)
    success = sum(status_count.get(s, 0) for s in ["filled", "submitted", "partial_filled"])
    filled = sum(status_count.get(s, 0) for s in ["filled", "partial_filled"])
    rejected = sum(status_count.get(s, 0) for s in ["rejected", "error", "cancelled"])

    latency_vals = [float(x["latency_ms"]) for x in order_results if x.get("latency_ms") is not None]
    slip_vals = [float(x["slippage_bps"]) for x in order_results if x.get("slippage_bps") is not None]
    abs_slip_vals = [abs(x) for x in slip_vals]

    return {
        "orders_total": int(total),
        "success_rate": float(success / total) if total else 0.0,
        "fill_rate": float(filled / total) if total else 0.0,
        "reject_rate": float(rejected / total) if total else 0.0,
        "avg_latency_ms": float(sum(latency_vals) / len(latency_vals)) if latency_vals else None,
        "latency_samples": int(len(latency_vals)),
        "avg_abs_slippage_bps": float(sum(abs_slip_vals) / len(abs_slip_vals)) if abs_slip_vals else None,
        "slippage_samples": int(len(slip_vals)),
        "status_count": status_count,
    }


def _update_execution_quality_daily(logger):
    try:
        from scripts.report_execution_quality import generate_execution_quality_report

        out_file = "./outputs/reports/execution_quality_daily.json"
        report = generate_execution_quality_report(output_file=out_file)
        summary = report.get("summary", {})
        logger.info(
            "📊 成交质量日报已更新: %s (orders=%s, success_rate=%.2f%%, fill_rate=%.2f%%, reject_rate=%.2f%%)",
            out_file,
            summary.get("orders_total", 0),
            float(summary.get("success_rate", 0.0)) * 100.0,
            float(summary.get("fill_rate", 0.0)) * 100.0,
            float(summary.get("reject_rate", 0.0)) * 100.0,
        )
    except Exception as e:
        logger.warning(f"⚠️ 更新成交质量日报失败: {e}")


def execute_trades(trades_file: str, dry_run: bool = False):
    logger = setup_logging()

    if not os.path.exists(trades_file):
        logger.error(f"❌ 交易指令文件不存在: {trades_file}")
        return False

    trades = load_trades(trades_file)
    market = str(trades.get("market", "crypto")).strip().lower()
    if market not in {"", "crypto"}:
        logger.error(f"❌ 非币圈交易文件: market={market}")
        return False

    config = load_config()
    if dry_run:
        config["env"] = "paper"

    env = config.get("env", "paper")
    venue = str(config.get("exchange", config.get("broker", "crypto"))).upper()

    logger.info("=" * 60)
    logger.info("开始执行币圈交易")
    logger.info("环境: %s", env.upper())
    logger.info("交易所: %s", venue)
    logger.info("=" * 60)

    try:
        trader = create_crypto_market_trader(config)
    except Exception as e:
        logger.error(f"❌ 创建交易器失败: {e}")
        return False

    if not trader.connect():
        logger.error("❌ 连接交易服务器失败")
        return False

    try:
        orders = trades.get("orders", [])
        if not orders:
            logger.info("ℹ️ 没有需要执行的交易指令")
            return True

        logger.info(f"📋 发现 {len(orders)} 条交易指令")
        try:
            account = trader.get_account_info()
            avail = account.get("available_cash", account.get("available", 0))
            logger.info(f"💰 账户可用资金: {avail}")
        except Exception as e:
            logger.warning(f"⚠️ 获取资金失败: {e}")

        executed_orders = []
        failed_orders = []
        order_results = []
        ref_price_cache = {}

        for i, trade in enumerate(orders, 1):
            symbol = trade.get("symbol")
            action = trade.get("action")
            amount = float(trade.get("amount_quote", 0) or 0)

            logger.info(f"\n[{i}/{len(orders)}] 处理交易: {action} {symbol} {amount:.2f}")
            requested_at = datetime.now().isoformat()
            ref_px = _reference_price(config, symbol, ref_price_cache)

            try:
                order = build_crypto_order(trade)
            except Exception as e:
                failed_orders.append(
                    {"symbol": symbol, "action": action, "amount": amount, "status": "error", "error": str(e)}
                )
                order_results.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "amount_quote": amount,
                        "status": "error",
                        "order_id": None,
                        "requested_at": requested_at,
                        "finished_at": datetime.now().isoformat(),
                        "latency_ms": None,
                        "reference_price": ref_px,
                        "filled_price": None,
                        "slippage_bps": None,
                        "error_msg": str(e),
                    }
                )
                logger.error(f"  ❌ 失败 - 构造订单失败: {e}")
                continue

            place_start = time.time()
            try:
                result = trader.place_order(order)
            except Exception as e:
                latency_ms = (time.time() - place_start) * 1000.0
                failed_orders.append(
                    {"symbol": symbol, "action": action, "amount": amount, "status": "error", "error": str(e)}
                )
                order_results.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "amount_quote": amount,
                        "status": "error",
                        "order_id": None,
                        "requested_at": requested_at,
                        "finished_at": datetime.now().isoformat(),
                        "latency_ms": round(float(latency_ms), 3),
                        "reference_price": ref_px,
                        "filled_price": None,
                        "slippage_bps": None,
                        "error_msg": str(e),
                    }
                )
                logger.error(f"  ❌ 失败 - 下单异常: {e}")
                continue

            latency_ms = (time.time() - place_start) * 1000.0
            status = _status_value(result)
            fill_px = _safe_float(getattr(result, "price", None))
            slippage_bps = _calc_slippage_bps(action, ref_px, fill_px)
            row = {
                "symbol": symbol,
                "action": action,
                "amount_quote": amount,
                "status": status,
                "order_id": getattr(result, "order_id", None),
                "requested_at": requested_at,
                "finished_at": datetime.now().isoformat(),
                "latency_ms": round(float(latency_ms), 3),
                "reference_price": ref_px,
                "filled_price": fill_px,
                "slippage_bps": slippage_bps,
                "filled_amount": _safe_float(getattr(result, "filled_amount", None)),
                "error_msg": str(getattr(result, "error_msg", "") or "").strip(),
            }
            order_results.append(row)

            if status in ["filled", "submitted", "partial_filled"]:
                executed_orders.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "amount": amount,
                        "order_id": row["order_id"],
                        "status": status,
                        "latency_ms": row["latency_ms"],
                        "filled_price": row["filled_price"],
                        "slippage_bps": row["slippage_bps"],
                    }
                )
                logger.info(f"  ✅ 成功 - 订单ID: {row['order_id']}")
            else:
                failed_orders.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "amount": amount,
                        "status": status,
                        "error": row["error_msg"] or "unknown error",
                        "latency_ms": row["latency_ms"],
                        "filled_price": row["filled_price"],
                        "slippage_bps": row["slippage_bps"],
                    }
                )
                logger.error(f"  ❌ 失败 - {row['error_msg'] or 'unknown error'}")

            if env == "live":
                time.sleep(0.5)

        pos_file = "./outputs/state/crypto_positions.json"
        try:
            trader.sync_positions(pos_file)
            logger.info(f"\n💾 持仓已同步到: {pos_file}")
        except Exception as e:
            logger.warning(f"⚠️ 同步持仓失败: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("执行结果统计")
        logger.info("=" * 60)
        logger.info(f"✅ 成功: {len(executed_orders)} 笔")
        logger.info(f"❌ 失败: {len(failed_orders)} 笔")

        execution_record = {
            "ts": datetime.now().isoformat(),
            "env": env,
            "market": "crypto",
            "broker": venue,
            "trades_file": trades_file,
            "total_orders": len(orders),
            "success": len(executed_orders),
            "failed": len(failed_orders),
            "executed": executed_orders,
            "failed_details": failed_orders,
            "order_results": order_results,
            "metrics": _build_execution_metrics(order_results),
        }

        record_file = f"./outputs/orders/execution_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(execution_record, f, ensure_ascii=False, indent=2)
        logger.info(f"\n📝 执行记录已保存: {record_file}")
        _update_execution_quality_daily(logger)

        return len(failed_orders) == 0
    finally:
        trader.disconnect()


def main():
    parser = argparse.ArgumentParser(description="执行币圈交易指令")
    parser.add_argument("--file", "-f", default="./outputs/orders/crypto_trades.json", help="交易指令文件路径")
    parser.add_argument("--dry-run", "-d", action="store_true", help="模拟执行（不实际下单）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认，直接执行")
    args = parser.parse_args()

    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            trades = json.load(f)

        print("\n" + "=" * 60)
        print("币圈交易指令预览")
        print("=" * 60)
        print(f"文件: {args.file}")
        print(f"总资金: {trades.get('capital_total', 0):.2f}")

        orders = trades.get("orders", [])
        if not orders:
            print("  (无)")
        else:
            for i, order in enumerate(orders, 1):
                print(f"  {i}. {order['action']:11} {order['symbol']}  金额: {order.get('amount_quote', 0):,.2f}")
        print("=" * 60)

        if not args.yes and not args.dry_run:
            config = load_config()
            env = config.get("env", "paper")
            exchange = config.get("exchange", "unknown")
            if env == "live":
                print(f"\n⚠️ 警告: 当前配置为 LIVE 实盘模式！交易所: {exchange}")
            confirm = input(f"\n确认执行{' (模拟)' if args.dry_run else ''}? [y/N]: ")
            if confirm.lower() != "y":
                print("已取消")
                return
    else:
        print(f"❌ 文件不存在: {args.file}")
        return

    success = execute_trades(args.file, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
