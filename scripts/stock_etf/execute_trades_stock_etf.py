#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行股票交易指令脚本（国投/掘金）。"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.guotou_trader import Order as StockOrder
from adapters.guotou_trader import OrderSide as StockOrderSide
from adapters.guotou_trader import create_trader as create_guotou_trader
from adapters.myquant_trader import create_trader as create_myquant_trader
from core.stock_broker import (
    SUPPORTED_STOCK_BROKERS,
    resolve_runtime_stock_broker,
    resolve_strategy_account_config,
)


def create_stock_trader(config: dict):
    broker = config.get("broker", "guotou")
    if broker == "myquant":
        return create_myquant_trader(config)
    return create_guotou_trader(config)


def setup_logging():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"execute_stock_trades_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("execute_stock_trades")


def load_trades(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_positions_snapshot(path: str = "./outputs/state/stock_positions.json") -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        pos = data.get("positions", [])
        return pos if isinstance(pos, list) else []
    except Exception:
        return []


def _positions_to_weight_map(positions: list) -> dict:
    out = {}
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        sym = str(p.get("symbol", "")).strip()
        if not sym:
            continue
        try:
            w = float(p.get("weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        out[sym] = out.get(sym, 0.0) + w
    return out


def _apply_filled_orders_to_positions(before_positions: list, order_results: list) -> list:
    after_map = _positions_to_weight_map(before_positions)
    for x in order_results or []:
        if str(x.get("status", "")).lower() not in {"filled", "submitted", "partial_filled"}:
            continue
        sym = str(x.get("symbol", "")).strip()
        if not sym:
            continue
        dw = float(x.get("delta_weight", 0.0) or 0.0)
        after_map[sym] = after_map.get(sym, 0.0) + dw
        if abs(after_map[sym]) < 1e-8:
            after_map.pop(sym, None)
    return [{"symbol": k, "weight": round(float(v), 6)} for k, v in sorted(after_map.items()) if abs(v) > 1e-8]


def load_config(*, force_paper: bool = False) -> dict:
    import yaml

    with open("./config/runtime.yaml", "r", encoding="utf-8") as f:
        runtime = yaml.safe_load(f) or {}

    env = str(runtime.get("env", "paper")).strip().lower()
    if force_paper:
        env = "paper"
    broker_type, broker_source = resolve_runtime_stock_broker(runtime, strategy="stock_etf")
    total_capital = runtime.get("total_capital", 20000)
    timezone_name = str(runtime.get("timezone", "Asia/Shanghai")).strip() or "Asia/Shanghai"

    if env == "live" and not broker_type:
        raise ValueError("live 模式必须配置 stock_brokers.stock_etf（或兼容字段 broker）: myquant 或 guotou")
    if not broker_type:
        broker_type = "guotou"
        broker_source = "default(guotou)"
    if broker_type not in SUPPORTED_STOCK_BROKERS:
        raise ValueError(f"不支持的 broker: {broker_type}")

    broker_full_config = {}
    if os.path.exists("./config/broker.yaml"):
        with open("./config/broker.yaml", "r", encoding="utf-8") as f:
            broker_full_config = yaml.safe_load(f) or {}

    config, account_source = resolve_strategy_account_config(
        broker_full_config,
        broker=broker_type,
        strategy="stock_etf",
    )
    if not isinstance(config, dict):
        config = {}

    if broker_type == "myquant":
        config["platform"] = "myquant"
    else:
        config["platform"] = config.get("platform", "emp")

    config["env"] = env
    config["total_capital"] = total_capital
    config["broker"] = broker_type
    config["_runtime_broker_source"] = broker_source or "unknown"
    config["_runtime_account_source"] = account_source or broker_type
    config["_runtime_paths"] = runtime.get("paths", {})
    config["_runtime_timezone"] = timezone_name

    if env == "live" and broker_type == "guotou":
        platform = str(config.get("platform", "emp")).strip().lower()
        hosting_mode = str(config.get("emp", {}).get("hosting_mode", "signal")).strip().lower()
        if platform == "emp" and hosting_mode == "signal":
            raise ValueError(
                "当前代码未实现 guotou EMP signal 实盘连接；"
                "请切换 stock_brokers.stock_etf=myquant，或将 guotou.emp.hosting_mode 设为 hosted 并接入可用通道"
            )
    return config


def _now_in_timezone(tz_name: str) -> datetime:
    tz = str(tz_name or "Asia/Shanghai").strip() or "Asia/Shanghai"
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz))
        except Exception:
            pass
    if tz == "Asia/Shanghai":
        return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    return datetime.now()


def build_stock_order(trade: dict):
    symbol = trade.get("symbol")
    action = str(trade.get("action", "BUY")).strip().upper()
    amount = float(trade.get("amount_quote", 0) or 0)
    if action not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported stock action: {action}")
    side = StockOrderSide.BUY if action == "BUY" else StockOrderSide.SELL
    return StockOrder(symbol=symbol, side=side, amount=amount)


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
    key = f"stock:{symbol}"
    if key in cache:
        return cache[key]

    data_dir = config.get("_runtime_paths", {}).get("data_dir", "./data")
    fp = os.path.join(data_dir, "stock", f"{symbol}.csv")
    px = _load_latest_close(fp)
    cache[key] = px
    return px


def _calc_slippage_bps(action: str, ref_price, fill_price):
    ref = _safe_float(ref_price)
    fill = _safe_float(fill_price)
    if ref is None or fill is None or ref <= 0:
        return None

    side = 1.0 if str(action or "").strip().upper() == "BUY" else -1.0
    raw_bps = (fill - ref) / ref * 10000.0
    return float(raw_bps * side)


def _estimate_quantity(amount_quote, price, lot_size: int = 100):
    amt = _safe_float(amount_quote)
    px = _safe_float(price)
    if amt is None or px is None or px <= 0:
        return None
    lots = int(amt // (px * lot_size))
    return int(max(0, lots) * lot_size)


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
    market = str(trades.get("market", "stock")).strip().lower()
    if market not in {"", "stock"}:
        logger.error(f"❌ 非股票交易文件: market={market}")
        return False

    try:
        config = load_config(force_paper=dry_run)
    except Exception as e:
        logger.error(f"❌ 加载交易配置失败: {e}")
        return False
    if dry_run:
        config["env"] = "paper"

    env = config.get("env", "paper")
    broker_name = config.get("broker", "unknown").upper()

    logger.info("=" * 60)
    logger.info("开始执行股票交易")
    logger.info("环境: %s", env.upper())
    logger.info("券商: %s", broker_name)
    logger.info("券商来源: %s", config.get("_runtime_broker_source", "unknown"))
    logger.info("账户来源: %s", config.get("_runtime_account_source", "unknown"))
    logger.info("时区: %s", config.get("_runtime_timezone", "Asia/Shanghai"))
    logger.info("=" * 60)

    if env == "live":
        now = _now_in_timezone(config.get("_runtime_timezone", "Asia/Shanghai"))
        current_time = now.time()
        is_trading_hours = (
            (current_time.hour == 9 and current_time.minute >= 30)
            or (current_time.hour == 10)
            or (current_time.hour == 11 and current_time.minute <= 30)
            or (current_time.hour == 13)
            or (current_time.hour == 14)
        )
        if not is_trading_hours:
            logger.warning("⚠️ 当前不在A股交易时间内（%s），实盘订单可能无法成交", now.strftime("%H:%M:%S"))

    try:
        trader = create_stock_trader(config)
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
        before_positions = _load_positions_snapshot("./outputs/state/stock_positions.json")
        before_weight_map = _positions_to_weight_map(before_positions)
        try:
            account = trader.get_account_info()
            logger.info(f"💰 账户可用资金: ¥{account.get('available_cash', 0):.2f}")
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
            delta_weight = float(trade.get("delta_weight", 0) or 0)

            logger.info(f"\n[{i}/{len(orders)}] 处理交易: {action} {symbol} ¥{amount:.2f}")
            requested_at = datetime.now().isoformat()
            ref_px = _reference_price(config, symbol, ref_price_cache)

            try:
                order = build_stock_order(trade)
            except Exception as e:
                failed_orders.append(
                    {"symbol": symbol, "action": action, "amount": amount, "status": "error", "error": str(e)}
                )
                order_results.append(
                    {
                        "symbol": symbol,
                        "action": action,
                        "amount_quote": amount,
                        "delta_weight": delta_weight,
                        "status": "error",
                        "order_id": None,
                        "requested_at": requested_at,
                        "finished_at": datetime.now().isoformat(),
                        "latency_ms": None,
                        "reference_price": ref_px,
                        "order_price": ref_px,
                        "filled_price": None,
                        "quantity": _estimate_quantity(amount, ref_px, lot_size=100),
                        "lot_size": 100,
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
                        "delta_weight": delta_weight,
                        "status": "error",
                        "order_id": None,
                        "requested_at": requested_at,
                        "finished_at": datetime.now().isoformat(),
                        "latency_ms": round(float(latency_ms), 3),
                        "reference_price": ref_px,
                        "order_price": ref_px,
                        "filled_price": None,
                        "quantity": _estimate_quantity(amount, ref_px, lot_size=100),
                        "lot_size": 100,
                        "slippage_bps": None,
                        "error_msg": str(e),
                    }
                )
                logger.error(f"  ❌ 失败 - 下单异常: {e}")
                continue

            latency_ms = (time.time() - place_start) * 1000.0
            status = _status_value(result)
            fill_px = _safe_float(getattr(result, "price", None))
            order_price = fill_px if fill_px is not None else ref_px
            est_qty = _estimate_quantity(amount, order_price, lot_size=100)
            slippage_bps = _calc_slippage_bps(action, ref_px, fill_px)
            row = {
                "symbol": symbol,
                "action": action,
                "amount_quote": amount,
                "delta_weight": delta_weight,
                "status": status,
                "order_id": getattr(result, "order_id", None),
                "requested_at": requested_at,
                "finished_at": datetime.now().isoformat(),
                "latency_ms": round(float(latency_ms), 3),
                "reference_price": ref_px,
                "order_price": order_price,
                "filled_price": fill_px,
                "quantity": est_qty,
                "lot_size": 100,
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
                        "delta_weight": delta_weight,
                        "quantity": row.get("quantity"),
                        "order_price": row.get("order_price"),
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

        pos_file = "./outputs/state/stock_positions.json"
        try:
            if dry_run:
                # 需求：dry-run 完成后视为交易成功并更新仓位
                after_positions = _apply_filled_orders_to_positions(before_positions, order_results)
                with open(pos_file, "w", encoding="utf-8") as f:
                    json.dump({"positions": after_positions}, f, ensure_ascii=False, indent=2)
                logger.info(f"\n💾 [DRY-RUN] 已按成交结果更新仓位到: {pos_file}")
            else:
                trader.sync_positions(pos_file)
                logger.info(f"\n💾 持仓已同步到: {pos_file}")
                after_positions = _load_positions_snapshot(pos_file)
        except Exception as e:
            logger.warning(f"⚠️ 同步持仓失败: {e}")
            after_positions = _apply_filled_orders_to_positions(before_positions, order_results)

        logger.info("\n" + "=" * 60)
        logger.info("执行结果统计")
        logger.info("=" * 60)
        logger.info(f"✅ 成功: {len(executed_orders)} 笔")
        logger.info(f"❌ 失败: {len(failed_orders)} 笔")

        execution_record = {
            "ts": datetime.now().isoformat(),
            "env": env,
            "market": "stock",
            "broker": broker_name,
            "trades_file": trades_file,
            "total_orders": len(orders),
            "success": len(executed_orders),
            "failed": len(failed_orders),
            "positions_before": before_positions,
            "positions_after": after_positions,
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
    parser = argparse.ArgumentParser(description="执行股票交易指令")
    parser.add_argument("--file", "-f", default="./outputs/orders/stock_trades.json", help="交易指令文件路径")
    parser.add_argument("--dry-run", "-d", action="store_true", help="模拟执行（不实际下单）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认，直接执行")
    args = parser.parse_args()

    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            trades = json.load(f)

        print("\n" + "=" * 60)
        print("股票交易指令预览")
        print("=" * 60)
        print(f"文件: {args.file}")
        print(f"总资金: ¥{trades.get('capital_total', 0):.2f}")

        before_positions = _load_positions_snapshot("./outputs/state/stock_positions.json")
        print("交易前仓位:")
        if not before_positions:
            print("  (空仓)")
        else:
            for p in before_positions:
                sym = p.get("symbol", "")
                w = p.get("weight", None)
                qty = p.get("quantity", None)
                if w is not None:
                    print(f"  - {sym} weight={float(w):.4f}" + (f" qty={qty}" if qty is not None else ""))
                else:
                    print(f"  - {sym}" + (f" qty={qty}" if qty is not None else ""))

        orders = trades.get("orders", [])
        print("交易指令:")
        if not orders:
            print("  (无)")
        else:
            data_dir = "./data/stock"
            for i, order in enumerate(orders, 1):
                symbol = order.get("symbol", "")
                amount = float(order.get("amount_quote", 0) or 0)
                action = str(order.get("action", "")).upper()
                px = _load_latest_close(os.path.join(data_dir, f"{symbol}.csv"))
                qty = _estimate_quantity(amount, px, lot_size=100)
                px_str = f"¥{px:.4f}" if px is not None else "N/A"
                qty_str = str(qty) if qty is not None else "N/A"
                print(
                    f"  {i}. {action:4} {symbol} 金额: ¥{amount:,.2f} 数量(估): {qty_str} 价格(参考): {px_str}"
                )

        # 预估交易后仓位（按 delta_weight 推演）
        before_map = _positions_to_weight_map(before_positions)
        after_map = dict(before_map)
        for order in orders:
            sym = str(order.get("symbol", "")).strip()
            if not sym:
                continue
            dw = float(order.get("delta_weight", 0.0) or 0.0)
            after_map[sym] = after_map.get(sym, 0.0) + dw
            if abs(after_map[sym]) < 1e-8:
                after_map.pop(sym, None)
        print("交易后仓位(预估):")
        if not after_map:
            print("  (空仓)")
        else:
            for sym, w in sorted(after_map.items()):
                print(f"  - {sym} weight={float(w):.4f}")

        print("=" * 60)

        if not args.yes and not args.dry_run:
            try:
                config = load_config()
            except Exception as e:
                print(f"\n❌ 配置检查失败: {e}")
                return
            env = config.get("env", "paper")
            broker = config.get("broker", "guotou")
            if env == "live":
                print(f"\n⚠️ 警告: 当前配置为 LIVE 实盘模式！券商: {broker}")
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
