#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import (
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
    EXIT_SIGNAL_ERROR,
)

SUCCESS_STATUSES = {"filled", "submitted", "partial_filled"}
FILL_STATUSES = {"filled", "partial_filled"}
FAIL_STATUSES = {"rejected", "error", "cancelled"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_date(ts: Any) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).date().isoformat()
    except Exception:
        return None


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    vals = sorted(float(v) for v in values)
    rank = (len(vals) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    w = rank - lo
    return float(vals[lo] * (1.0 - w) + vals[hi] * w)


def _as_status(x: Any) -> str:
    s = str(x or "").strip().lower()
    return s if s else "unknown"


def _iter_rows_from_record(rec: Dict[str, Any]) -> list[Dict[str, Any]]:
    market = str(rec.get("market", "unknown")).strip().lower()
    out = []
    if isinstance(rec.get("order_results"), list):
        for r in rec["order_results"]:
            status = _as_status(r.get("status"))
            out.append(
                {
                    "market": market,
                    "symbol": str(r.get("symbol", "")).strip(),
                    "status": status,
                    "latency_ms": _safe_float(r.get("latency_ms")),
                    "slippage_bps": _safe_float(r.get("slippage_bps")),
                    "error": str(r.get("error_msg", "")).strip(),
                }
            )
        return out

    # Backward compatibility with legacy execution_record format.
    for r in rec.get("executed", []) or []:
        status = _as_status(r.get("status", "submitted"))
        out.append(
            {
                "market": market,
                "symbol": str(r.get("symbol", "")).strip(),
                "status": status,
                "latency_ms": None,
                "slippage_bps": None,
                "error": "",
            }
        )
    for r in rec.get("failed_details", []) or []:
        status = _as_status(r.get("status", "error"))
        out.append(
            {
                "market": market,
                "symbol": str(r.get("symbol", "")).strip(),
                "status": status,
                "latency_ms": None,
                "slippage_bps": None,
                "error": str(r.get("error", "")).strip(),
            }
        )
    return out


def _build_summary(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    status_count: Dict[str, int] = {}
    for r in rows:
        s = _as_status(r.get("status"))
        status_count[s] = status_count.get(s, 0) + 1

    success = sum(status_count.get(x, 0) for x in SUCCESS_STATUSES)
    filled = sum(status_count.get(x, 0) for x in FILL_STATUSES)
    failed = total - success
    rejected = sum(status_count.get(x, 0) for x in FAIL_STATUSES)

    latency = [float(x["latency_ms"]) for x in rows if x.get("latency_ms") is not None]
    slippage = [float(x["slippage_bps"]) for x in rows if x.get("slippage_bps") is not None]
    abs_slippage = [abs(x) for x in slippage]

    return {
        "orders_total": int(total),
        "success_orders": int(success),
        "failed_orders": int(failed),
        "filled_orders": int(filled),
        "success_rate": float(success / total) if total else 0.0,
        "fill_rate": float(filled / total) if total else 0.0,
        "reject_rate": float(rejected / total) if total else 0.0,
        "avg_execution_latency_ms": float(sum(latency) / len(latency)) if latency else None,
        "p50_execution_latency_ms": _percentile(latency, 0.50),
        "p95_execution_latency_ms": _percentile(latency, 0.95),
        "latency_samples": int(len(latency)),
        "avg_slippage_bps": float(sum(slippage) / len(slippage)) if slippage else None,
        "avg_abs_slippage_bps": float(sum(abs_slippage) / len(abs_slippage)) if abs_slippage else None,
        "p50_abs_slippage_bps": _percentile(abs_slippage, 0.50),
        "p95_abs_slippage_bps": _percentile(abs_slippage, 0.95),
        "slippage_samples": int(len(slippage)),
        "status_count": status_count,
    }


def generate_execution_quality_report(
    report_date: Optional[str] = None,
    output_file: str = "./outputs/reports/execution_quality_daily.json",
    orders_dir: str = "./outputs/orders",
) -> Dict[str, Any]:
    if report_date is None:
        report_date = datetime.now().date().isoformat()

    pattern = os.path.join(orders_dir, "execution_record_*.json")
    files = sorted(glob.glob(pattern))

    rows: list[Dict[str, Any]] = []
    used_files: list[str] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        rec_date = _parse_date(rec.get("ts"))
        if rec_date != report_date:
            continue
        used_files.append(fp)
        rows.extend(_iter_rows_from_record(rec))

    by_market: Dict[str, Dict[str, Any]] = {}
    for m in sorted(set(r["market"] for r in rows)):
        m_rows = [r for r in rows if r["market"] == m]
        by_market[m] = _build_summary(m_rows)

    symbol_rows = []
    for s in sorted(set(r["symbol"] for r in rows if r["symbol"])):
        s_rows = [r for r in rows if r["symbol"] == s]
        x = _build_summary(s_rows)
        x["symbol"] = s
        symbol_rows.append(x)
    symbol_rows = sorted(
        symbol_rows,
        key=lambda x: (int(x["orders_total"]), float(x["success_rate"])),
        reverse=True,
    )

    out = {
        "ts": datetime.now().isoformat(),
        "date": report_date,
        "window": "daily_local",
        "orders_dir": orders_dir,
        "records_scanned": int(len(used_files)),
        "record_files": [os.path.basename(x) for x in used_files],
        "summary": _build_summary(rows),
        "by_market": by_market,
        "by_symbol": symbol_rows,
    }

    ensure_dir(os.path.dirname(output_file) or ".")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build daily execution quality report from execution records.")
    parser.add_argument("--date", default=None, help="report date in YYYY-MM-DD (default: today)")
    parser.add_argument("--orders-dir", default="./outputs/orders", help="execution record directory")
    parser.add_argument(
        "--output",
        default="./outputs/reports/execution_quality_daily.json",
        help="output report file",
    )
    args = parser.parse_args()

    try:
        report = generate_execution_quality_report(
            report_date=args.date,
            output_file=args.output,
            orders_dir=args.orders_dir,
        )
    except OSError as e:
        print(f"[exec-quality] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[exec-quality] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[exec-quality] report -> {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
