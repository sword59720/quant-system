#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_FORMAT_ERROR,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def comp_ret(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def excess_return(strategy_ret: pd.Series, benchmark_ret: pd.Series) -> float:
    return float(comp_ret(strategy_ret) - comp_ret(benchmark_ret))


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def build_weekly_summary(history_file: str):
    if not os.path.exists(history_file):
        raise RuntimeError(f"history file not found: {history_file}")

    df = pd.read_csv(history_file)
    if df.empty:
        raise RuntimeError("stock_paper_forward_history.csv is empty")

    for c in ["date", "next_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["strategy_ret", "benchmark_ret_alloc", "executed_turnover"]:
        if c not in df.columns:
            raise RuntimeError(f"missing required column in history file: {c}")

    df = df.dropna(subset=["date", "next_date"]).copy()
    if df.empty:
        raise RuntimeError("history file has no valid date rows")

    # Use next_date's week ending Friday as weekly bucket.
    df["week"] = df["next_date"].dt.to_period("W-FRI")

    rows = []
    for wk, g in df.groupby("week"):
        rows.append(
            {
                "week": str(wk),
                "week_start": g["date"].min().date().isoformat(),
                "week_end": g["next_date"].max().date().isoformat(),
                "days": int(len(g)),
                "strategy_ret": comp_ret(g["strategy_ret"]),
                "benchmark_ret_alloc": comp_ret(g["benchmark_ret_alloc"]),
                "excess_ret_vs_alloc": excess_return(g["strategy_ret"], g["benchmark_ret_alloc"]),
                "turnover": float(g["executed_turnover"].sum()),
                "trades": int((g["executed_turnover"] > 1e-9).sum()),
                "hold_days": int((g["action"] == "hold").sum()) if "action" in g.columns else None,
                "rebalance_days": int((g["action"] == "rebalance").sum()) if "action" in g.columns else None,
            }
        )
    weekly = pd.DataFrame(rows)

    weekly = weekly.sort_values("week_end").reset_index(drop=True)
    trailing_4 = weekly.tail(4)
    trailing_8 = weekly.tail(8)

    summary = {
        "ts": datetime.now().isoformat(),
        "history_file": history_file,
        "weeks_total": int(len(weekly)),
        "latest_week": weekly.iloc[-1].to_dict(),
        "trailing_4w": {
            "strategy_ret": comp_ret(trailing_4["strategy_ret"]),
            "benchmark_ret_alloc": comp_ret(trailing_4["benchmark_ret_alloc"]),
            "excess_ret_vs_alloc": excess_return(trailing_4["strategy_ret"], trailing_4["benchmark_ret_alloc"]),
            "turnover": float(trailing_4["turnover"].sum()),
            "trades": int(trailing_4["trades"].sum()),
            "win_weeks": int((trailing_4["excess_ret_vs_alloc"] > 0).sum()),
        },
        "trailing_8w": {
            "strategy_ret": comp_ret(trailing_8["strategy_ret"]),
            "benchmark_ret_alloc": comp_ret(trailing_8["benchmark_ret_alloc"]),
            "excess_ret_vs_alloc": excess_return(trailing_8["strategy_ret"], trailing_8["benchmark_ret_alloc"]),
            "turnover": float(trailing_8["turnover"].sum()),
            "trades": int(trailing_8["trades"].sum()),
            "win_weeks": int((trailing_8["excess_ret_vs_alloc"] > 0).sum()),
        },
    }
    return summary, weekly


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
    except Exception as e:
        print(f"[weekly-report] config error: {e}")
        return EXIT_CONFIG_ERROR

    report_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(report_dir)
    history_file = os.path.join(report_dir, "stock_paper_forward_history.csv")

    try:
        summary, weekly = build_weekly_summary(history_file)
    except Exception as e:
        print(f"[weekly-report] data error: {e}")
        return EXIT_DATA_FORMAT_ERROR

    try:
        json_file = os.path.join(report_dir, "stock_weekly_report.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[weekly-report] output error: {e}")
        return EXIT_OUTPUT_ERROR

    json_file = os.path.join(report_dir, "stock_weekly_report.json")
    # Markdown report for quick reading.
    md_lines = []
    md_lines.append("# Stock Weekly Report")
    md_lines.append("")
    md_lines.append(f"- Generated at: `{summary['ts']}`")
    md_lines.append(f"- Source: `{history_file}`")
    md_lines.append(f"- Total weeks: `{summary['weeks_total']}`")
    md_lines.append("")
    md_lines.append("## Latest Week")
    lw = summary["latest_week"]
    md_lines.append(f"- Week: `{lw['week_start']} -> {lw['week_end']}`")
    md_lines.append(f"- Strategy: `{fmt_pct(lw['strategy_ret'])}`")
    md_lines.append(f"- Benchmark (alloc): `{fmt_pct(lw['benchmark_ret_alloc'])}`")
    md_lines.append(f"- Excess (alloc): `{fmt_pct(lw['excess_ret_vs_alloc'])}`")
    md_lines.append(f"- Turnover: `{lw['turnover']:.4f}`")
    md_lines.append(f"- Trades: `{lw['trades']}`")
    md_lines.append("")
    md_lines.append("## Trailing 4 Weeks")
    t4 = summary["trailing_4w"]
    md_lines.append(f"- Strategy: `{fmt_pct(t4['strategy_ret'])}`")
    md_lines.append(f"- Benchmark (alloc): `{fmt_pct(t4['benchmark_ret_alloc'])}`")
    md_lines.append(f"- Excess (alloc): `{fmt_pct(t4['excess_ret_vs_alloc'])}`")
    md_lines.append(f"- Turnover: `{t4['turnover']:.4f}`")
    md_lines.append(f"- Trades: `{t4['trades']}`")
    md_lines.append(f"- Win weeks: `{t4['win_weeks']}`")
    md_lines.append("")
    md_lines.append("## Trailing 8 Weeks")
    t8 = summary["trailing_8w"]
    md_lines.append(f"- Strategy: `{fmt_pct(t8['strategy_ret'])}`")
    md_lines.append(f"- Benchmark (alloc): `{fmt_pct(t8['benchmark_ret_alloc'])}`")
    md_lines.append(f"- Excess (alloc): `{fmt_pct(t8['excess_ret_vs_alloc'])}`")
    md_lines.append(f"- Turnover: `{t8['turnover']:.4f}`")
    md_lines.append(f"- Trades: `{t8['trades']}`")
    md_lines.append(f"- Win weeks: `{t8['win_weeks']}`")
    md_lines.append("")
    md_lines.append("## Recent 8 Weeks Detail")
    md_lines.append("| Week | Strategy | Benchmark(alloc) | Excess(alloc) | Turnover | Trades |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in weekly.tail(8).iterrows():
        md_lines.append(
            f"| {row['week_start']} -> {row['week_end']} | {fmt_pct(float(row['strategy_ret']))} | "
            f"{fmt_pct(float(row['benchmark_ret_alloc']))} | {fmt_pct(float(row['excess_ret_vs_alloc']))} | "
            f"{float(row['turnover']):.4f} | {int(row['trades'])} |"
        )

    md_file = os.path.join(report_dir, "stock_weekly_report.md")
    try:
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
    except OSError as e:
        print(f"[weekly-report] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[weekly-report] json -> {json_file}")
    print(f"[weekly-report] md   -> {md_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
