#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def momentum_score(close: pd.Series, lb_short: int, lb_long: int, w_short: float, w_long: float):
    if len(close) < max(lb_short, lb_long) + 2:
        return None
    s_ret = close.iloc[-1] / close.iloc[-lb_short] - 1
    l_ret = close.iloc[-1] / close.iloc[-lb_long] - 1
    return float(s_ret * w_short + l_ret * w_long)


def volatility_score(close: pd.Series, lb: int = 20):
    if len(close) < lb + 2:
        return None
    ret = close.pct_change().dropna().tail(lb)
    if ret.empty:
        return None
    return float(ret.std())


def max_drawdown_score(close: pd.Series, lb: int = 60):
    if len(close) < lb + 2:
        return None
    s = close.tail(lb)
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return float(dd.min())  # negative number


def liquidity_score(amount: pd.Series, lb: int = 20):
    if len(amount) < lb:
        return None
    s = pd.to_numeric(amount, errors="coerce").dropna().tail(lb)
    if s.empty:
        return None
    return float(s.mean())


def normalize_rank(values: dict, ascending: bool = False):
    # returns 0~1 rank score
    if not values:
        return {}
    items = sorted(values.items(), key=lambda x: x[1], reverse=not ascending)
    n = len(items)
    if n == 1:
        return {items[0][0]: 1.0}
    out = {}
    for i, (k, _) in enumerate(items):
        out[k] = 1.0 - i / (n - 1)
    return out


