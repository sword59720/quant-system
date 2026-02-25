#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.notify_wecom import send_wecom_message
from core.exit_codes import EXIT_OK, EXIT_OUTPUT_ERROR, EXIT_CONFIG_ERROR


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def comp_ret(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def sharpe(ret: pd.Series, ppy: int = 252) -> float:
    if len(ret) < 2:
        return 0.0
    sd = float(ret.std())
    if sd <= 1e-12:
        return 0.0
    return float((float(ret.mean()) / sd) * math.sqrt(ppy))


def max_drawdown_from_ret(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    nav = (1.0 + ret).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main():
    try:
        runtime = load_yaml('config/runtime.yaml')
    except Exception as e:
        print(f'[weekly-notify] config error: {e}')
        return EXIT_CONFIG_ERROR

    history_file = os.path.join(runtime['paths']['output_dir'], 'reports', 'stock_paper_forward_history.csv')
    if not os.path.exists(history_file):
        print(f'[weekly-notify] history missing: {history_file}')
        return EXIT_OUTPUT_ERROR

    df = pd.read_csv(history_file)
    if df.empty or ('next_date' not in df.columns) or ('strategy_ret' not in df.columns) or ('benchmark_ret_alloc' not in df.columns):
        print('[weekly-notify] history format invalid')
        return EXIT_OUTPUT_ERROR

    df['next_date'] = pd.to_datetime(df['next_date'], errors='coerce')
    df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
    df = df.dropna(subset=['next_date']).copy()
    if df.empty:
        print('[weekly-notify] no valid rows')
        return EXIT_OUTPUT_ERROR

    df['week'] = df['next_date'].dt.to_period('W-FRI')
    latest_week = str(df['week'].max())
    w = df[df['week'].astype(str) == latest_week].copy()
    if w.empty:
        print('[weekly-notify] latest week empty')
        return EXIT_OUTPUT_ERROR

    sret = pd.to_numeric(w['strategy_ret'], errors='coerce').dropna()
    bret = pd.to_numeric(w['benchmark_ret_alloc'], errors='coerce').dropna()
    n = min(len(sret), len(bret))
    sret = sret.tail(n)
    bret = bret.tail(n)

    strategy_r = comp_ret(sret)
    benchmark_r = comp_ret(bret)
    excess_r = strategy_r - benchmark_r
    s_sharpe = sharpe(sret)
    s_mdd = max_drawdown_from_ret(sret)

    week_start = w['date'].min()
    week_end = w['next_date'].max()
    span = f"{week_start.date().isoformat()} ~ {week_end.date().isoformat()}" if pd.notna(week_start) and pd.notna(week_end) else latest_week

    lines = [
        f"时间: {datetime.now().isoformat()}",
        f"周期: {span}",
        f"收益率: {fmt_pct(strategy_r)}",
        f"Sharpe: {s_sharpe:.2f}",
        f"最大回撤: {fmt_pct(s_mdd)}",
        f"超额收益率(相对基准): {fmt_pct(excess_r)}",
    ]

    ok, msg = send_wecom_message('\n'.join(lines), title='stock_etf 每周交易总结', dedup_key=f'stock_weekly_summary:{latest_week}', dedup_hours=72)
    print(f"[weekly-notify] wecom {'ok' if ok else 'fail'}: {msg}")
    return EXIT_OK if ok else EXIT_OUTPUT_ERROR


if __name__ == '__main__':
    raise SystemExit(main())
