#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMT native all-in-one strategy for stock_single.

This version embeds the full pipeline in one script:
1) Build stock pool (daily)
2) Fast risk check (every 5 minutes)
3) Hourly scoring + signal generation + order execution

How to use in QMT strategy editor:
1) Paste this file into a QMT Python strategy.
2) Edit PROJECT_ROOT / ACCOUNT_ID / DRY_RUN.
3) Keep local data files synced under PROJECT_ROOT/data/stock_single.
4) Run strategy on 1-minute bar.
"""

import csv
import json
import math
import os
import re
import time
import traceback
from datetime import datetime


# ----------------------------
# User config (must edit)
# ----------------------------
PROJECT_ROOT = r"D:\quant-system"
ACCOUNT_ID = "YOUR_QMT_STOCK_ACCOUNT"
ACCOUNT_TYPE = "stock"
STRATEGY_NAME = "stock_single_native"

DRY_RUN = True

# schedule
POOL_REFRESH_TIME = "09:35"  # once per day after this time
RUN_TIMES = {"09:45", "10:45", "11:15", "13:45", "14:45"}
RISK_CHECK_INTERVAL_MINUTES = 5

# universe / pool
MAX_SYMBOLS_IN_POOL = 300
TARGET_POOL_SIZE = 60
ENTRY_QUANTILE = 0.30
EXIT_QUANTILE = 0.50
MIN_POOL_HOLD_DAYS = 5
MIN_HISTORY_DAYS = 120
ST_FILTER = True
SUSPENSION_LOOKBACK_DAYS = 20
MAX_ZERO_VOLUME_RATIO = 0.20
RECENT_LIMIT_LOOKBACK_DAYS = 5
LIMIT_MOVE_THRESHOLD_PCT = 9.7
MAX_RECENT_LIMIT_HITS = 2

# liquidity
LIQUIDITY_ENABLED = True
LIQ_LOOKBACK_DAYS = 20
MIN_AVG_TURNOVER = 100000000.0
MIN_AVG_VOLUME = 1000000.0

# industry diversification
INDUSTRY_DIVERSIFICATION_ENABLED = True
MAX_INDUSTRY_WEIGHT = 0.30
MIN_INDUSTRY_COUNT = 5

# signal / position
CAPITAL_ALLOC_PCT = 1.00
MAX_POSITIONS = 10
PER_SIGNAL_TARGET_WEIGHT = 0.10
SINGLE_MAX_PCT = 0.15
TRIGGER_MODE = "topk"  # threshold | quantile | topk
BUY_THRESHOLD_STATIC = 1.50
SELL_THRESHOLD_STATIC = -0.80
BUY_SCORE_QUANTILE = 0.80
SELL_SCORE_QUANTILE = 0.30
BUY_TOP_K = 15
HOLD_TOP_K = 25
BUY_BUFFER_BPS = 10.0
SELL_BUFFER_BPS = 10.0
MIN_DELTA_WEIGHT = 0.0025
MIN_ORDER_VALUE = 1000.0
MAX_ORDERS_PER_RUN = 40
LOT_SIZE = 100

# score adjustments
TIME_SERIES_MOMENTUM_WINDOW = 20
VOLATILITY_ADJUSTED_SIGNAL = True

# fast risk
FAST_RISK_ENABLED = True
TRIGGER_PORTFOLIO_RET_5M = -0.012
TRIGGER_SINGLE_RET_5M = -0.025
TRIGGER_VOL_ZSCORE = 3.5
BLOCK_NEW_BUYS_ON_TRIGGER = True
FORCE_SELL_ON_SINGLE_CRASH = True

# factor weights (BOLL kept at 0.0 baseline)
SCORE_WEIGHTS = {
    "mom20": 3.00,
    "mom60": 2.50,
    "rev5": -0.50,
    "vol20": -0.30,
    "vol_spike": 0.60,
    "ep": 0.10,
    "bp": 0.10,
    "main_flow": 3.00,
    "vol_change": 1.00,
    "volume_change": 1.00,
    "sentiment": 1.20,
    "boll_percent": 0.00,
    "boll_width": 0.00,
    "boll_break": 0.00,
}

# qmt order constants (may need adjustment by terminal version)
ORDER_TYPE_STOCK = 1101
OP_BUY = 23
OP_SELL = 24
PRICE_TYPE_LIMIT = 11


# ----------------------------
# Runtime state
# ----------------------------
STATE = {
    "last_hourly_key": "",
    "last_risk_key": "",
    "last_pool_date": "",
}


# ----------------------------
# Paths
# ----------------------------
def _path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def _daily_dir():
    return _path("data", "stock_single", "daily")


def _valuation_dir():
    return _path("data", "stock_single", "valuation")


def _flow_factor_dir():
    # matches current baseline config: fund_flow_real
    return _path("data", "stock_single", "fund_flow_real")


def _universe_file():
    return _path("data", "stock_single", "universe.csv")


def _industry_file():
    return _path("data", "stock_single", "industry_map.csv")


def _pool_file():
    return _path("outputs", "orders", "stock_single_pool.json")


def _signal_file():
    return _path("outputs", "orders", "stock_single_signals.json")


def _score_snapshot_file():
    return _path("data", "stock_single", "hourly_scores_latest.csv")


def _risk_snapshot_file():
    return _path("data", "stock_single", "intraday_risk_latest.csv")


def _risk_state_file():
    return _path("outputs", "orders", "stock_single_risk_state.json")


def _risk_alert_file():
    return _path("outputs", "orders", "stock_single_risk_alerts.json")


# ----------------------------
# Generic helpers
# ----------------------------
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print("[stock_single_qmt_native]", msg)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return default
            return float(s)
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return default
            return int(float(s))
        return int(x)
    except Exception:
        return default


def _mean(vals):
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _std_population(vals):
    if not vals:
        return None
    mu = _mean(vals)
    if mu is None:
        return None
    var = _mean([(v - mu) * (v - mu) for v in vals])
    if var is None:
        return None
    return math.sqrt(max(var, 0.0))


def _clip_lot(shares):
    shares = max(0, int(shares))
    return (shares // LOT_SIZE) * LOT_SIZE


def _is_finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _write_json(path, payload):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def _write_csv(path, fieldnames, rows):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ----------------------------
# Symbol helpers
# ----------------------------
def _normalize_symbol(sym):
    s = str(sym or "").strip().upper()
    if not s:
        return "", ""
    if "." in s:
        code = s.split(".")[0]
        if len(code) == 6 and code.isdigit():
            return code, code + "." + s.split(".")[-1]
        return "", ""
    if len(s) == 6 and s.isdigit():
        if s[0] in {"5", "6", "9"}:
            return s, s + ".SH"
        return s, s + ".SZ"
    return "", ""


def _is_a_share_stock(code, canonical):
    if len(code) != 6 or (not code.isdigit()):
        return False
    if canonical.endswith(".SH"):
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if canonical.endswith(".SZ"):
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    return False


# ----------------------------
# Daily data load
# ----------------------------
def _load_daily_series(symbol, code):
    fp = os.path.join(_daily_dir(), symbol + ".csv")
    if not os.path.exists(fp):
        fp = os.path.join(_daily_dir(), code + ".csv")
    if not os.path.exists(fp):
        return None

    rows = _read_csv_rows(fp)
    if not rows:
        return None

    parsed = []
    for r in rows:
        d = str(r.get("date", "")).strip()
        if not d:
            continue
        close = _safe_float(r.get("close"), float("nan"))
        if not _is_finite(close) or close <= 0:
            continue
        parsed.append(
            {
                "date": d,
                "open": _safe_float(r.get("open"), close),
                "high": _safe_float(r.get("high"), close),
                "low": _safe_float(r.get("low"), close),
                "close": close,
                "volume": _safe_float(r.get("volume"), 0.0),
                "amount": _safe_float(r.get("amount"), 0.0),
                "pct_change": _safe_float(r.get("pct_change"), float("nan")),
            }
        )

    if len(parsed) < 2:
        return None

    parsed = sorted(parsed, key=lambda x: x["date"])
    dates = [x["date"] for x in parsed]
    close = [x["close"] for x in parsed]
    high = [x["high"] for x in parsed]
    low = [x["low"] for x in parsed]
    volume = [x["volume"] for x in parsed]
    amount = [x["amount"] for x in parsed]

    ret = [None]
    for i in range(1, len(close)):
        prev = close[i - 1]
        cur = close[i]
        if prev > 0:
            ret.append(cur / prev - 1.0)
        else:
            ret.append(None)

    # pct_change is used by hard filter as percentage points (e.g., 3.0 for +3%)
    pct_pct = []
    for i, x in enumerate(parsed):
        p = x["pct_change"]
        if _is_finite(p):
            pct_pct.append(float(p))
            continue
        r = ret[i]
        pct_pct.append(float(r * 100.0) if _is_finite(r) else 0.0)

    return {
        "dates": dates,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
        "ret": ret,
        "pct_pct": pct_pct,
    }


def _rolling_std_at(vals, end_idx, window, min_periods):
    if end_idx < 0 or end_idx >= len(vals):
        return None
    start = max(0, end_idx - window + 1)
    x = [v for v in vals[start : end_idx + 1] if _is_finite(v)]
    if len(x) < min_periods:
        return None
    return _std_population(x)


def _ratio_change(series, lag):
    if len(series) <= lag:
        return None
    a = series[-1]
    b = series[-1 - lag]
    if not (_is_finite(a) and _is_finite(b)) or b <= 0:
        return None
    return a / b - 1.0


def _safe_window_mean(series, n):
    if len(series) < n:
        return None
    x = [v for v in series[-n:] if _is_finite(v)]
    if not x:
        return None
    return _mean(x)


def _calc_latest_atr14(daily):
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    n = len(close)
    if n == 0:
        return 0.01

    tr = []
    for i in range(n):
        prev_close = close[i - 1] if i > 0 else close[i]
        h = high[i]
        l = low[i]
        tr.append(max(abs(h - l), abs(h - prev_close), abs(l - prev_close)))

    x = [v for v in tr[-14:] if _is_finite(v)]
    if len(x) >= 5:
        atr = _mean(x)
        if _is_finite(atr) and atr > 0:
            return float(atr)

    rv = _rolling_std_at(daily["ret"], len(daily["ret"]) - 1, 14, 5)
    last = close[-1]
    if _is_finite(rv) and _is_finite(last):
        est = abs(rv * last)
        if est > 0:
            return float(est)

    return max(last * 0.02, 0.01)


def _load_latest_valuation(symbol, code, asof_date):
    fp = os.path.join(_valuation_dir(), symbol + ".csv")
    if not os.path.exists(fp):
        fp = os.path.join(_valuation_dir(), code + ".csv")
    if not os.path.exists(fp):
        return None, None

    rows = _read_csv_rows(fp)
    if not rows:
        return None, None

    best_date = ""
    pe = None
    pb = None
    for r in rows:
        d = str(r.get("date", "")).strip()
        if not d:
            continue
        if d > asof_date:
            continue
        if d >= best_date:
            best_date = d
            pe = _safe_float(r.get("pe_ttm"), float("nan"))
            pb = _safe_float(r.get("pb"), float("nan"))

    pe = pe if _is_finite(pe) and pe > 0 else None
    pb = pb if _is_finite(pb) and pb > 0 else None
    return pe, pb


def _load_flow_ratio_latest_5d(symbol, code, daily_dates, daily_amount):
    fp = os.path.join(_flow_factor_dir(), symbol + ".csv")
    if not os.path.exists(fp):
        fp = os.path.join(_flow_factor_dir(), code + ".csv")
    if not os.path.exists(fp):
        return None

    rows = _read_csv_rows(fp)
    if not rows:
        return None

    ratio_by_date = {}
    amount_by_date = {d: a for d, a in zip(daily_dates, daily_amount)}

    for r in rows:
        d = str(r.get("date", "")).strip()
        if not d:
            continue

        ratio = None
        if "main_net_inflow_ratio" in r and str(r.get("main_net_inflow_ratio", "")).strip() != "":
            ratio = _safe_float(r.get("main_net_inflow_ratio"), float("nan"))
            if _is_finite(ratio):
                ratio = float(ratio)
        if ratio is None or (not _is_finite(ratio)):
            main_flow = _safe_float(r.get("main_net_inflow"), float("nan"))
            amt = _safe_float(amount_by_date.get(d), float("nan"))
            if _is_finite(main_flow) and _is_finite(amt) and amt > 0:
                ratio = float(main_flow / amt)

        if ratio is not None and _is_finite(ratio):
            ratio_by_date[d] = float(ratio)

    aligned = [ratio_by_date.get(d) for d in daily_dates]
    tail = aligned[-5:]
    vals = [x for x in tail if _is_finite(x)]
    if len(vals) < 3:
        return None
    return float(sum(vals))


def _calc_boll_features(close, vol_base):
    if len(close) < 21 or len(vol_base) < 21:
        return None, None, 0.0, 0.0

    win_cur = close[-20:]
    mu = _mean(win_cur)
    sd = _std_population(win_cur)
    if not _is_finite(mu) or not _is_finite(sd) or mu <= 0:
        return None, None, 0.0, 0.0

    upper = mu + 2.0 * sd
    lower = mu - 2.0 * sd
    denom = upper - lower
    px = close[-1]

    boll_percent = None
    if denom > 1e-12:
        boll_percent = (px - lower) / denom

    boll_width = denom / mu if mu > 1e-12 else None

    win_prev = close[-21:-1]
    mu_prev = _mean(win_prev)
    sd_prev = _std_population(win_prev)
    boll_width_prev = None
    if _is_finite(mu_prev) and _is_finite(sd_prev) and mu_prev > 1e-12:
        boll_width_prev = (4.0 * sd_prev) / mu_prev

    boll_open_up = bool(
        _is_finite(boll_width)
        and _is_finite(boll_width_prev)
        and (boll_width - boll_width_prev) > 0
    )
    boll_open_down = bool(
        _is_finite(boll_width)
        and _is_finite(boll_width_prev)
        and (boll_width - boll_width_prev) < 0
    )

    upper_break = px > upper
    lower_break = px < lower

    vol_mu = _mean([v for v in vol_base[-20:] if _is_finite(v)])
    volume_shrink = False
    if _is_finite(vol_mu) and vol_mu > 0 and _is_finite(vol_base[-1]):
        volume_change = vol_base[-1] / vol_mu - 1.0
        volume_shrink = volume_change < -0.2

    buy_signal = 1.0 if (lower_break and volume_shrink and boll_open_up) else 0.0
    sell_signal = 1.0 if (upper_break and volume_shrink and boll_open_down) else 0.0

    return boll_percent, boll_width, buy_signal, sell_signal


def _calc_symbol_metrics(daily, symbol, code):
    close = daily["close"]
    ret = daily["ret"]
    volume = daily["volume"]
    amount = daily["amount"]
    dates = daily["dates"]

    n = len(close)
    if n < 61:
        return None

    mom20 = _ratio_change(close, 20)
    mom60 = _ratio_change(close, 60)
    rev5 = _ratio_change(close, 5)

    vol20 = _rolling_std_at(ret, len(ret) - 1, 20, 10)
    vol20_prev = _rolling_std_at(ret, len(ret) - 11, 20, 10)
    vol_change = None
    if _is_finite(vol20) and _is_finite(vol20_prev) and vol20_prev > 1e-12:
        vol_change = vol20 / vol20_prev - 1.0

    vol_base = volume if any(_is_finite(v) and v > 0 for v in volume[-20:]) else amount
    if not any(_is_finite(v) and v > 0 for v in vol_base[-20:]):
        vol_base = close

    vb_mu20 = _safe_window_mean(vol_base, 20)
    vol_spike = None
    if _is_finite(vb_mu20) and vb_mu20 > 0 and _is_finite(vol_base[-1]):
        vol_spike = vol_base[-1] / vb_mu20 - 1.0

    volume_change = _ratio_change(vol_base, 10)

    sentiment = None
    if len(close) > 5 and len(vol_base) > 5:
        pchg5 = _ratio_change(close, 5)
        vchg5 = _ratio_change(vol_base, 5)
        if _is_finite(pchg5) and _is_finite(vchg5):
            sentiment = pchg5 * vchg5

    pe_ttm, pb = _load_latest_valuation(symbol, code, dates[-1])
    ep = (1.0 / pe_ttm) if (pe_ttm and pe_ttm > 0) else None
    bp = (1.0 / pb) if (pb and pb > 0) else None

    flow_5d = _load_flow_ratio_latest_5d(symbol, code, dates, amount)

    boll_percent, boll_width, buy_sig, sell_sig = _calc_boll_features(close, vol_base)
    improved_boll_signal = None
    if _is_finite(boll_percent):
        improved_boll_signal = boll_percent - buy_sig * 3.0 + sell_sig * 3.0

    atr14 = _calc_latest_atr14(daily)

    ts_momentum = _ratio_change(close, TIME_SERIES_MOMENTUM_WINDOW)

    return {
        "symbol": symbol,
        "last_price": close[-1],
        "atr14": atr14,
        "mom20": mom20,
        "mom60": mom60,
        "rev5": rev5,
        "vol20": vol20,
        "vol_spike": vol_spike,
        "ep": ep,
        "bp": bp,
        "main_flow": flow_5d,
        "vol_change": vol_change,
        "volume_change": volume_change,
        "sentiment": sentiment,
        "boll_percent": boll_percent,
        "boll_width": boll_width,
        "boll_break": improved_boll_signal,
        "ts_momentum": ts_momentum,
    }


# ----------------------------
# Pool / scoring core
# ----------------------------
def _load_universe_rows():
    fp = _universe_file()
    rows = _read_csv_rows(fp)
    out = []
    seen = set()

    for r in rows:
        raw_symbol = str(r.get("symbol", "")).strip()
        code, canonical = _normalize_symbol(raw_symbol)
        if not canonical or canonical in seen:
            continue
        if not _is_a_share_stock(code, canonical):
            continue
        seen.add(canonical)
        out.append(
            {
                "symbol": canonical,
                "code": code,
                "name": str(r.get("name", "")).strip(),
                "industry": str(r.get("industry", "")).strip(),
            }
        )

    return out[:MAX_SYMBOLS_IN_POOL]


def _load_industry_map():
    fp = _industry_file()
    rows = _read_csv_rows(fp)
    if not rows:
        return {}

    out = {}
    for r in rows:
        raw_symbol = str(r.get("symbol", r.get("代码", ""))).strip()
        industry = str(r.get("industry", r.get("所属行业", r.get("行业", "")))).strip()
        code, canonical = _normalize_symbol(raw_symbol)
        if canonical and industry:
            out[canonical] = industry
    return out


def _infer_industry(row, industry_map):
    s = row.get("symbol", "")
    if s in industry_map and industry_map[s]:
        return industry_map[s]
    if row.get("industry"):
        return row.get("industry")
    if s.endswith(".SH"):
        return "SH"
    if s.endswith(".SZ"):
        return "SZ"
    return "UNKNOWN"


def _passes_hard_filters(row, daily):
    if daily is None:
        return False, "missing_daily"
    if len(daily["close"]) < MIN_HISTORY_DAYS:
        return False, "short_history"

    if ST_FILTER and row.get("name") and ("ST" in str(row.get("name", "")).upper()):
        return False, "st_name"

    tail_vol = daily["volume"][-max(1, SUSPENSION_LOOKBACK_DAYS) :]
    if tail_vol:
        zero_ratio = sum(1 for v in tail_vol if (not _is_finite(v)) or v <= 0) / float(len(tail_vol))
        if zero_ratio > MAX_ZERO_VOLUME_RATIO:
            return False, "suspension_like"

    tail_pct = daily["pct_pct"][-max(1, RECENT_LIMIT_LOOKBACK_DAYS) :]
    limit_hits = sum(1 for p in tail_pct if _is_finite(p) and abs(p) >= LIMIT_MOVE_THRESHOLD_PCT)
    if limit_hits > MAX_RECENT_LIMIT_HITS:
        return False, "recent_limit_hits"

    return True, "ok"


def _passes_liquidity(daily):
    if not LIQUIDITY_ENABLED:
        return True, "disabled"

    tail_amt = daily["amount"][-max(1, LIQ_LOOKBACK_DAYS) :]
    tail_vol = daily["volume"][-max(1, LIQ_LOOKBACK_DAYS) :]

    amt_vals = [x for x in tail_amt if _is_finite(x)]
    vol_vals = [x for x in tail_vol if _is_finite(x)]
    avg_amt = _mean(amt_vals) if amt_vals else 0.0
    avg_vol = _mean(vol_vals) if vol_vals else 0.0

    if avg_amt < MIN_AVG_TURNOVER:
        return False, "low_turnover"
    if avg_vol < MIN_AVG_VOLUME:
        return False, "low_volume"
    return True, "ok"


def _cross_section_zscore(metric_by_sym):
    vals = [v for v in metric_by_sym.values() if _is_finite(v)]
    if not vals:
        return {k: 0.0 for k in metric_by_sym.keys()}

    mu = _mean(vals)
    sd = _std_population(vals)
    if (not _is_finite(sd)) or sd <= 1e-12:
        return {k: 0.0 for k in metric_by_sym.keys()}

    out = {}
    for sym, v in metric_by_sym.items():
        if _is_finite(v):
            out[sym] = (float(v) - mu) / sd
        else:
            out[sym] = 0.0
    return out


def _apply_realtime_adjustments(score, metrics):
    out = float(score)
    if VOLATILITY_ADJUSTED_SIGNAL:
        vol20 = metrics.get("vol20")
        if _is_finite(vol20) and float(vol20) > 1e-12:
            out = out / float(vol20)

    ts_mom = metrics.get("ts_momentum")
    if _is_finite(ts_mom) and float(ts_mom) < 0:
        out = out * 0.5

    return out


def _compute_ranked_scores(universe_rows):
    metrics_by_sym = {}
    reject_reason = {}

    for row in universe_rows:
        sym = row["symbol"]
        code = row["code"]

        daily = _load_daily_series(sym, code)
        ok, reason = _passes_hard_filters(row, daily)
        if not ok:
            reject_reason[sym] = reason
            continue

        ok, reason = _passes_liquidity(daily)
        if not ok:
            reject_reason[sym] = reason
            continue

        m = _calc_symbol_metrics(daily, sym, code)
        if not m:
            reject_reason[sym] = "metric_unavailable"
            continue

        m["name"] = row.get("name", "")
        m["industry"] = row.get("industry", "")
        metrics_by_sym[sym] = m

    if not metrics_by_sym:
        return [], {
            "candidate_count_raw": len(universe_rows),
            "candidate_count_scored": 0,
            "reject_reason_count": {},
        }

    factor_names = [
        "mom20",
        "mom60",
        "rev5",
        "vol20",
        "vol_spike",
        "ep",
        "bp",
        "main_flow",
        "vol_change",
        "volume_change",
        "sentiment",
        "boll_percent",
        "boll_width",
        "boll_break",
    ]

    z_by_factor = {}
    for f in factor_names:
        raw = {sym: metrics_by_sym[sym].get(f) for sym in metrics_by_sym.keys()}
        z_by_factor[f] = _cross_section_zscore(raw)

    out_rows = []
    for sym, m in metrics_by_sym.items():
        score = 0.0
        for f in factor_names:
            w = float(SCORE_WEIGHTS.get(f, 0.0))
            score += w * float(z_by_factor[f].get(sym, 0.0))

        score = _apply_realtime_adjustments(score, m)

        out_rows.append(
            {
                "symbol": sym,
                "score": float(score),
                "last_price": float(m.get("last_price", 0.0)),
                "atr14": float(max(m.get("atr14", 0.01), 0.01)),
                "name": m.get("name", ""),
                "industry": m.get("industry", ""),
            }
        )

    out_rows = [r for r in out_rows if _is_finite(r["score"]) and r["last_price"] > 0]
    out_rows = sorted(out_rows, key=lambda x: x["score"], reverse=True)

    rc = {}
    for reason in reject_reason.values():
        rc[reason] = rc.get(reason, 0) + 1

    meta = {
        "candidate_count_raw": len(universe_rows),
        "candidate_count_scored": len(out_rows),
        "reject_reason_count": rc,
    }
    return out_rows, meta


def _load_previous_pool(pool_path):
    payload = _load_json(pool_path, {"symbols": [], "hold_days": {}})
    prev_symbols = [str(x) for x in payload.get("symbols", []) if str(x)]
    hold_raw = payload.get("hold_days", {}) or {}
    hold_days = {}
    for k, v in hold_raw.items():
        hold_days[str(k)] = _safe_int(v, 0)
    return prev_symbols, hold_days


def _apply_industry_cap(ranked_symbols, industry_of, target_pool_size):
    if not INDUSTRY_DIVERSIFICATION_ENABLED:
        return ranked_symbols[:target_pool_size]

    max_per_industry = max(1, int(target_pool_size * MAX_INDUSTRY_WEIGHT + 1e-9))

    out = []
    ind_count = {}
    for s in ranked_symbols:
        if len(out) >= target_pool_size:
            break
        ind = industry_of.get(s, "UNKNOWN")
        c = ind_count.get(ind, 0)
        if c >= max_per_industry:
            continue
        out.append(s)
        ind_count[ind] = c + 1

    if len(out) < target_pool_size:
        used = set(out)
        for s in ranked_symbols:
            if len(out) >= target_pool_size:
                break
            if s in used:
                continue
            out.append(s)
            used.add(s)

    if MIN_INDUSTRY_COUNT > 1 and out:
        distinct = set(industry_of.get(s, "UNKNOWN") for s in out)
        if len(distinct) < MIN_INDUSTRY_COUNT:
            best_by_ind = {}
            for s in ranked_symbols:
                ind = industry_of.get(s, "UNKNOWN")
                if ind not in best_by_ind:
                    best_by_ind[ind] = s
            for ind, cand in best_by_ind.items():
                if len(distinct) >= MIN_INDUSTRY_COUNT:
                    break
                if ind in distinct or cand in out:
                    continue

                counts = {}
                for x in out:
                    ix = industry_of.get(x, "UNKNOWN")
                    counts[ix] = counts.get(ix, 0) + 1

                replace_idx = None
                for i in range(len(out) - 1, -1, -1):
                    old = out[i]
                    old_ind = industry_of.get(old, "UNKNOWN")
                    if counts.get(old_ind, 0) > 1:
                        replace_idx = i
                        break

                if replace_idx is None:
                    break
                out[replace_idx] = cand
                distinct = set(industry_of.get(s, "UNKNOWN") for s in out)

    return out


def _build_pool_now():
    universe_rows = _load_universe_rows()
    if not universe_rows:
        raise RuntimeError("universe empty: " + _universe_file())

    score_rows, meta = _compute_ranked_scores(universe_rows)
    if not score_rows:
        raise RuntimeError("no symbols after filters/scoring")

    n = len(score_rows)
    target_pool_size = max(10, min(MAX_SYMBOLS_IN_POOL, TARGET_POOL_SIZE, n))

    entry_q = max(0.01, min(1.0, float(ENTRY_QUANTILE)))
    exit_q = max(entry_q, min(1.0, float(EXIT_QUANTILE)))

    entry_n = max(1, min(n, max(target_pool_size, int(round(n * entry_q)))))
    keep_n = max(entry_n, min(n, int(round(n * exit_q))))

    ranked_symbols = [x["symbol"] for x in score_rows]
    rank_map = {s: i + 1 for i, s in enumerate(ranked_symbols)}
    entry_set = set(ranked_symbols[:entry_n])

    prev_symbols, prev_hold_days = _load_previous_pool(_pool_file())
    selected = set(entry_set)

    for s in prev_symbols:
        if s not in rank_map:
            continue
        prev_days = int(prev_hold_days.get(s, 0))
        if s in entry_set:
            selected.add(s)
            continue
        if prev_days < MIN_POOL_HOLD_DAYS:
            selected.add(s)
            continue
        if rank_map[s] <= keep_n:
            selected.add(s)

    ordered_selected = sorted(list(selected), key=lambda x: rank_map.get(x, 10**9))
    for s in ranked_symbols:
        if len(ordered_selected) >= target_pool_size:
            break
        if s in selected:
            continue
        ordered_selected.append(s)
        selected.add(s)

    industry_map = _load_industry_map()
    row_map = {x["symbol"]: x for x in universe_rows}
    industry_of = {}
    for s in ordered_selected:
        industry_of[s] = _infer_industry(row_map.get(s, {"symbol": s}), industry_map)

    final_symbols = _apply_industry_cap(ordered_selected, industry_of, target_pool_size)

    hold_days = {}
    for s in final_symbols:
        hold_days[s] = int(prev_hold_days.get(s, 0)) + 1

    preview = []
    score_by_sym = {x["symbol"]: x["score"] for x in score_rows}
    for s in final_symbols[:20]:
        preview.append(
            {
                "symbol": s,
                "rank": int(rank_map.get(s, 10**9)),
                "score": round(float(score_by_sym.get(s, 0.0)), 6),
                "industry": industry_of.get(s, "UNKNOWN"),
                "hold_days": int(hold_days.get(s, 1)),
            }
        )

    payload = {
        "ts": datetime.now().isoformat(),
        "mode": "single_stock_hourly",
        "symbol_count": len(final_symbols),
        "symbols": final_symbols,
        "hold_days": hold_days,
        "selection": {
            "candidate_count_raw": int(meta.get("candidate_count_raw", 0)),
            "candidate_count_scored": int(meta.get("candidate_count_scored", 0)),
            "target_pool_size": int(target_pool_size),
            "entry_quantile": float(entry_q),
            "exit_quantile": float(exit_q),
            "entry_n": int(entry_n),
            "keep_n": int(keep_n),
            "min_pool_hold_days": int(MIN_POOL_HOLD_DAYS),
            "reject_reason_count": meta.get("reject_reason_count", {}),
            "rank_preview_top20": preview,
        },
    }

    _write_json(_pool_file(), payload)
    STATE["last_pool_date"] = datetime.now().strftime("%Y%m%d")
    return final_symbols, payload


# ----------------------------
# Fast risk (5m)
# ----------------------------
def _load_pool_symbols():
    payload = _load_json(_pool_file(), {"symbols": []})
    return [str(x).strip() for x in payload.get("symbols", []) if str(x).strip()]


def _load_risk_snapshot_rows():
    fp = _risk_snapshot_file()
    rows = _read_csv_rows(fp)
    out = []
    for r in rows:
        sym = str(r.get("symbol", "")).strip()
        code, canonical = _normalize_symbol(sym)
        if not canonical:
            continue
        ret_5m = _safe_float(r.get("ret_5m"), float("nan"))
        if not _is_finite(ret_5m):
            continue
        out.append(
            {
                "symbol": canonical,
                "ret_5m": float(ret_5m),
                "weight": _safe_float(r.get("weight"), float("nan")),
                "vol_zscore": _safe_float(r.get("vol_zscore"), float("nan")),
            }
        )
    return out


def _calc_portfolio_ret_5m(rows):
    if not rows:
        return 0.0

    w_vals = [x for x in [r.get("weight") for r in rows] if _is_finite(x) and x > 0]
    if not w_vals:
        return _mean([r["ret_5m"] for r in rows]) or 0.0

    wsum = 0.0
    vsum = 0.0
    for r in rows:
        w = r.get("weight")
        if not _is_finite(w) or w <= 0:
            continue
        wsum += float(w)
        vsum += float(w) * float(r["ret_5m"])
    if wsum <= 1e-12:
        return _mean([r["ret_5m"] for r in rows]) or 0.0
    return vsum / wsum


def _run_fast_risk_check_now():
    pool_symbols = set(_load_pool_symbols())
    snapshot_rows = _load_risk_snapshot_rows()
    if pool_symbols:
        snapshot_rows = [r for r in snapshot_rows if r["symbol"] in pool_symbols]

    portfolio_ret_5m = _calc_portfolio_ret_5m(snapshot_rows)

    single_crash_symbols = sorted(
        set(r["symbol"] for r in snapshot_rows if r["ret_5m"] <= TRIGGER_SINGLE_RET_5M)
    )

    vol_spike_symbols = sorted(
        set(
            r["symbol"]
            for r in snapshot_rows
            if _is_finite(r.get("vol_zscore")) and float(r.get("vol_zscore")) >= TRIGGER_VOL_ZSCORE
        )
    )

    portfolio_trigger = bool(portfolio_ret_5m <= TRIGGER_PORTFOLIO_RET_5M)
    single_trigger = bool(len(single_crash_symbols) > 0)
    vol_trigger = bool(len(vol_spike_symbols) > 0)
    triggered = bool(portfolio_trigger or single_trigger or vol_trigger)

    block_new_buys = bool(triggered and BLOCK_NEW_BUYS_ON_TRIGGER)
    force_sell_symbols = single_crash_symbols if FORCE_SELL_ON_SINGLE_CRASH else []

    state = {
        "ts": datetime.now().isoformat(),
        "market": "CN_STOCK_SINGLE",
        "mode": "fast_risk_check_5m",
        "snapshot_file": _risk_snapshot_file(),
        "snapshot_status": "ok" if os.path.exists(_risk_snapshot_file()) else "missing_snapshot",
        "universe_count": int(len(snapshot_rows)),
        "stage": "risk" if triggered else "normal",
        "triggered": triggered,
        "metrics": {
            "portfolio_ret_5m": round(float(portfolio_ret_5m), 8),
            "single_crash_count": int(len(single_crash_symbols)),
            "vol_spike_count": int(len(vol_spike_symbols)),
        },
        "trigger_flags": {
            "portfolio_drop": portfolio_trigger,
            "single_crash": single_trigger,
            "vol_spike": vol_trigger,
        },
        "thresholds": {
            "trigger_portfolio_ret_5m": TRIGGER_PORTFOLIO_RET_5M,
            "trigger_single_ret_5m": TRIGGER_SINGLE_RET_5M,
            "trigger_vol_zscore": TRIGGER_VOL_ZSCORE,
        },
        "actions": {
            "block_new_buys": block_new_buys,
            "force_sell_symbols": force_sell_symbols,
        },
    }

    alerts = {
        "ts": state["ts"],
        "triggered": triggered,
        "alerts": [],
    }
    if portfolio_trigger:
        alerts["alerts"].append(
            {
                "level": "risk",
                "type": "portfolio_drop_5m",
                "detail": "portfolio_ret_5m=%.4f <= %.4f" % (portfolio_ret_5m, TRIGGER_PORTFOLIO_RET_5M),
            }
        )
    if single_trigger:
        alerts["alerts"].append(
            {
                "level": "risk",
                "type": "single_crash_5m",
                "detail": ",".join(single_crash_symbols),
            }
        )
    if vol_trigger:
        alerts["alerts"].append(
            {
                "level": "warn",
                "type": "vol_spike_zscore",
                "detail": ",".join(vol_spike_symbols),
            }
        )

    _write_json(_risk_state_file(), state)
    _write_json(_risk_alert_file(), alerts)
    return state


def _load_risk_state_now():
    payload = _load_json(
        _risk_state_file(),
        {
            "stage": "unknown",
            "triggered": False,
            "actions": {"block_new_buys": False, "force_sell_symbols": []},
        },
    )
    actions = payload.get("actions", {}) if isinstance(payload, dict) else {}
    return {
        "stage": str(payload.get("stage", "unknown")),
        "triggered": bool(payload.get("triggered", False)),
        "block_new_buys": bool(actions.get("block_new_buys", False)),
        "force_sell_symbols": set(_normalize_symbol(x)[1] for x in (actions.get("force_sell_symbols", []) or [])),
    }


# ----------------------------
# Hourly scoring / signals
# ----------------------------
def _resolve_thresholds_topk(score_rows):
    if not score_rows:
        return BUY_THRESHOLD_STATIC, SELL_THRESHOLD_STATIC, {"mode": "threshold", "fallback": "no_scores"}

    valid = sorted([x["score"] for x in score_rows if _is_finite(x.get("score"))], reverse=True)
    if not valid:
        return BUY_THRESHOLD_STATIC, SELL_THRESHOLD_STATIC, {"mode": "threshold", "fallback": "no_valid_scores"}

    mode = str(TRIGGER_MODE or "topk").strip().lower()
    if mode == "threshold":
        return BUY_THRESHOLD_STATIC, SELL_THRESHOLD_STATIC, {
            "mode": "threshold",
            "buy_threshold": BUY_THRESHOLD_STATIC,
            "sell_threshold": SELL_THRESHOLD_STATIC,
        }

    if mode == "quantile":
        q_buy = max(0.0, min(1.0, float(BUY_SCORE_QUANTILE)))
        q_sell = max(0.0, min(1.0, float(SELL_SCORE_QUANTILE)))
        if q_sell >= q_buy:
            q_sell = max(0.0, q_buy - 0.10)
        i_buy = min(len(valid) - 1, max(0, int(round((len(valid) - 1) * q_buy))))
        i_sell = min(len(valid) - 1, max(0, int(round((len(valid) - 1) * q_sell))))
        buy_th = float(valid[i_buy])
        sell_th = float(valid[i_sell])
        if sell_th >= buy_th:
            sell_th = buy_th - 1e-6
        return buy_th, sell_th, {
            "mode": "quantile",
            "buy_score_quantile": q_buy,
            "sell_score_quantile": q_sell,
            "buy_threshold": buy_th,
            "sell_threshold": sell_th,
        }

    # topk
    buy_k = max(1, min(int(BUY_TOP_K), len(valid)))
    hold_k = max(buy_k, min(int(HOLD_TOP_K), len(valid)))
    buy_th = float(valid[buy_k - 1])
    sell_th = float(valid[hold_k - 1])
    if sell_th >= buy_th:
        sell_th = buy_th - 1e-6
    return buy_th, sell_th, {
        "mode": "topk",
        "buy_top_k": buy_k,
        "hold_top_k": hold_k,
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
    }


def _build_hourly_score_rows(pool_symbols):
    universe_rows = []
    for raw in pool_symbols:
        code, canonical = _normalize_symbol(raw)
        if not canonical:
            continue
        universe_rows.append({"symbol": canonical, "code": code, "name": "", "industry": ""})

    score_rows, meta = _compute_ranked_scores(universe_rows)
    score_rows = [x for x in score_rows if x["symbol"] in set(pool_symbols)]

    csv_rows = []
    for r in score_rows:
        csv_rows.append(
            {
                "symbol": r["symbol"],
                "score": round(float(r["score"]), 6),
                "last_price": round(float(r["last_price"]), 6),
                "atr14": round(float(r["atr14"]), 6),
            }
        )
    _write_csv(_score_snapshot_file(), ["symbol", "score", "last_price", "atr14"], csv_rows)
    return score_rows, meta


def _build_hourly_signals_now(score_rows, pool_symbols, risk):
    score_map = {x["symbol"]: x for x in score_rows}
    buy_th, sell_th, threshold_meta = _resolve_thresholds_topk(score_rows)

    buy_buffer = BUY_BUFFER_BPS * 1e-4
    sell_buffer = SELL_BUFFER_BPS * 1e-4
    per_weight = min(PER_SIGNAL_TARGET_WEIGHT, SINGLE_MAX_PCT)

    signals = []
    for symbol in pool_symbols:
        row = score_map.get(symbol)
        if not row:
            signals.append({"symbol": symbol, "action": "HOLD", "reason": "missing_score_snapshot"})
            continue

        score = float(row["score"])
        last_price = float(row["last_price"])
        atr = max(float(row["atr14"]), 0.01)

        action = "HOLD"
        reason = "score_in_neutral_range"
        entry_price = None
        exit_price = None
        stop_price = None
        target_weight = None  # None means keep current position unchanged.

        if symbol in risk["force_sell_symbols"]:
            action = "SELL"
            reason = "fast_risk_force_sell"
            exit_price = round(last_price * (1.0 - sell_buffer), 3)
            target_weight = 0.0
        elif score >= buy_th:
            action = "BUY"
            reason = "score=%.3f>=buy_threshold" % score
            entry_price = round(last_price * (1.0 + buy_buffer), 3)
            stop_price = round(entry_price - 1.2 * atr, 3)
            exit_price = round(entry_price + 2.2 * atr, 3)
            target_weight = per_weight
        elif score <= sell_th:
            action = "SELL"
            reason = "score=%.3f<=sell_threshold" % score
            exit_price = round(last_price * (1.0 - sell_buffer), 3)
            target_weight = 0.0

        if action == "BUY" and risk["block_new_buys"]:
            action = "HOLD"
            reason = reason + ";blocked_by_fast_risk"
            entry_price = None
            stop_price = None
            target_weight = None

        signals.append(
            {
                "symbol": symbol,
                "action": action,
                "score": round(score, 6),
                "last_price": round(last_price, 3),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_weight": (round(float(target_weight), 4) if target_weight is not None else None),
                "reason": reason,
                "risk_stage": risk["stage"],
            }
        )

    payload = {
        "ts": datetime.now().isoformat(),
        "market": "CN_STOCK_SINGLE",
        "mode": "single_stock_hourly",
        "pool_size": len(pool_symbols),
        "risk_overlay": {
            "source": _risk_state_file(),
            "stage": risk["stage"],
            "triggered": risk["triggered"],
            "block_new_buys": risk["block_new_buys"],
            "force_sell_count": len(risk["force_sell_symbols"]),
        },
        "signal_policy": {
            "mode": str(threshold_meta.get("mode", "threshold")),
            "buy_threshold_effective": float(buy_th),
            "sell_threshold_effective": float(sell_th),
        },
        "score_snapshot": {
            "status": "ok",
            "snapshot_file": _score_snapshot_file(),
            "symbol_count": len(score_rows),
        },
        "signals": signals,
    }

    _write_json(_signal_file(), payload)
    return payload


# ----------------------------
# Signal -> order target
# ----------------------------
def _build_target_weights_from_signals(signal_payload):
    signals = signal_payload.get("signals", []) or []

    buy_candidates = []
    target = {}
    for x in signals:
        sym = _normalize_symbol(x.get("symbol", ""))[1]
        if not sym:
            continue
        action = str(x.get("action", "HOLD")).upper().strip()
        if action == "BUY":
            buy_candidates.append(
                {
                    "symbol": sym,
                    "score": _safe_float(x.get("score"), -1e12),
                    "target_weight": max(0.0, _safe_float(x.get("target_weight"), 0.0)),
                }
            )
        elif action == "SELL":
            target[sym] = 0.0
        else:
            # HOLD: keep as is (None means no rebalance command for this symbol)
            target[sym] = None

    buy_candidates = sorted(buy_candidates, key=lambda z: z["score"], reverse=True)
    if MAX_POSITIONS > 0:
        buy_candidates = buy_candidates[:MAX_POSITIONS]

    for item in buy_candidates:
        target[item["symbol"]] = float(item["target_weight"])

    # cap total buy target
    buy_syms = [k for k, v in target.items() if (v is not None and v > 0)]
    total_buy = sum(target[s] for s in buy_syms)
    cap = max(0.0, float(CAPITAL_ALLOC_PCT))
    if total_buy > cap and total_buy > 1e-9:
        scale = cap / total_buy
        for s in buy_syms:
            target[s] = float(target[s]) * scale

    return target


# ----------------------------
# QMT account / order utils
# ----------------------------
def _query_trade_detail(account_id, market, dtype):
    g = globals().get("get_trade_detail_data", None)
    if callable(g):
        try:
            return g(account_id, market, dtype)
        except Exception:
            pass
        try:
            return g(account_id, market.upper(), dtype.upper())
        except Exception:
            pass
    return []


def _get_positions(last_price_map):
    rows = _query_trade_detail(ACCOUNT_ID, ACCOUNT_TYPE, "position")
    out = {}
    for r in rows or []:
        sym = _normalize_symbol(
            getattr(r, "m_strInstrumentID", "")
            or getattr(r, "stock_code", "")
            or getattr(r, "symbol", "")
        )[1]
        if not sym:
            continue

        vol = _safe_int(
            getattr(r, "m_nVolume", None)
            or getattr(r, "volume", None)
            or getattr(r, "total_volume", None),
            0,
        )
        can_use = _safe_int(
            getattr(r, "m_nCanUseVolume", None)
            or getattr(r, "can_use_volume", None)
            or getattr(r, "enable_volume", None),
            vol,
        )
        mkt_val = _safe_float(
            getattr(r, "m_dMarketValue", None)
            or getattr(r, "market_value", None),
            0.0,
        )

        if mkt_val <= 0 and vol > 0:
            px = _safe_float(last_price_map.get(sym), 0.0)
            if px > 0:
                mkt_val = vol * px

        out[sym] = {
            "symbol": sym,
            "volume": vol,
            "can_use_volume": can_use,
            "market_value": mkt_val,
        }
    return out


def _get_total_asset(positions):
    rows = _query_trade_detail(ACCOUNT_ID, ACCOUNT_TYPE, "account")
    for r in rows or []:
        for attr in ["m_dTotalAsset", "total_asset", "asset", "m_dBalance"]:
            total_asset = _safe_float(getattr(r, attr, None), 0.0)
            if total_asset > 0:
                return total_asset

    mkt = sum(_safe_float(x.get("market_value"), 0.0) for x in positions.values())
    return max(mkt, 100000.0)


def _build_orders(target_weights, positions, total_asset, score_rows):
    by_sym = {x["symbol"]: x for x in score_rows}
    current_weights = {}
    for sym, pos in positions.items():
        current_weights[sym] = _safe_float(pos.get("market_value"), 0.0) / total_asset if total_asset > 0 else 0.0

    symbols = sorted(set(current_weights.keys()) | set(target_weights.keys()))
    buy_buffer = BUY_BUFFER_BPS * 1e-4
    sell_buffer = SELL_BUFFER_BPS * 1e-4

    orders = []
    dropped = []

    for sym in symbols:
        tgt_w = target_weights.get(sym, None)
        if tgt_w is None:
            continue  # HOLD => unchanged

        cur_w = _safe_float(current_weights.get(sym), 0.0)
        delta_w = float(tgt_w) - cur_w
        if abs(delta_w) < MIN_DELTA_WEIGHT:
            continue

        side = "BUY" if delta_w > 0 else "SELL"
        row = by_sym.get(sym)
        if not row:
            dropped.append({"symbol": sym, "reason": "missing_score_row"})
            continue

        last_price = _safe_float(row.get("last_price"), 0.0)
        if last_price <= 0:
            dropped.append({"symbol": sym, "reason": "invalid_price"})
            continue

        price = last_price * (1.0 + buy_buffer if side == "BUY" else 1.0 - sell_buffer)
        value = abs(delta_w) * total_asset
        if value < MIN_ORDER_VALUE:
            dropped.append({"symbol": sym, "reason": "below_min_order_value", "value": value})
            continue

        shares = _clip_lot(int(value / price))
        if side == "SELL":
            can_use = _clip_lot(_safe_int(positions.get(sym, {}).get("can_use_volume"), 0))
            shares = min(shares, can_use)

        if shares < LOT_SIZE:
            dropped.append({"symbol": sym, "reason": "shares_below_1lot_or_unavailable"})
            continue

        orders.append(
            {
                "symbol": sym,
                "side": side,
                "price": round(price, 3),
                "shares": int(shares),
                "delta_weight": round(delta_w, 6),
                "amount_quote": round(price * shares, 2),
            }
        )

    orders = sorted(orders, key=lambda x: (0 if x["side"] == "SELL" else 1, -abs(x["delta_weight"])))
    if len(orders) > MAX_ORDERS_PER_RUN:
        dropped.extend([dict(x, reason="cut_by_max_orders") for x in orders[MAX_ORDERS_PER_RUN:]])
        orders = orders[:MAX_ORDERS_PER_RUN]

    return orders, dropped


def _passorder_try(ContextInfo, op_type, symbol, price, volume, remark):
    funcs = []
    if callable(globals().get("passorder", None)):
        funcs.append(globals().get("passorder"))
    if hasattr(ContextInfo, "passorder") and callable(getattr(ContextInfo, "passorder")):
        funcs.append(getattr(ContextInfo, "passorder"))
    if not funcs:
        raise RuntimeError("passorder not found in runtime")

    last_err = None
    for f in funcs:
        try:
            return f(op_type, ORDER_TYPE_STOCK, ACCOUNT_ID, symbol, PRICE_TYPE_LIMIT, price, volume, remark, 1, "", ContextInfo)
        except Exception as e:
            last_err = e
        try:
            return f(op_type, ORDER_TYPE_STOCK, ACCOUNT_ID, symbol, PRICE_TYPE_LIMIT, price, volume, remark, 1, ContextInfo)
        except Exception as e:
            last_err = e
        try:
            return f(op_type, ORDER_TYPE_STOCK, ACCOUNT_ID, symbol, PRICE_TYPE_LIMIT, price, volume, remark)
        except Exception as e:
            last_err = e

    if last_err:
        raise last_err
    raise RuntimeError("passorder call failed")


def _execute_orders(ContextInfo, orders):
    results = []
    for od in orders:
        op = OP_BUY if od["side"] == "BUY" else OP_SELL
        status = "dry_run" if DRY_RUN else "submitted"
        err = None
        ret = None
        t0 = time.time()

        if not DRY_RUN:
            try:
                ret = _passorder_try(
                    ContextInfo,
                    op_type=op,
                    symbol=od["symbol"],
                    price=od["price"],
                    volume=od["shares"],
                    remark=STRATEGY_NAME,
                )
            except Exception as e:
                status = "error"
                err = str(e)

        results.append(
            {
                **od,
                "status": status,
                "error": err,
                "ret": ret,
                "latency_ms": round((time.time() - t0) * 1000.0, 2),
            }
        )
    return results


def _save_run_files(plan, exec_results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _path("outputs", "orders")
    _ensure_dir(out_dir)

    plan_file = os.path.join(out_dir, "stock_single_qmt_native_plan_" + ts + ".json")
    exec_file = os.path.join(out_dir, "stock_single_qmt_native_exec_" + ts + ".json")

    _write_json(plan_file, plan)
    _write_json(exec_file, exec_results)
    return plan_file, exec_file


# ----------------------------
# Orchestrator
# ----------------------------
def _ensure_pool_ready(now):
    pool_exists = os.path.exists(_pool_file())
    today = now.strftime("%Y%m%d")
    if STATE.get("last_pool_date") == today and pool_exists:
        return

    hhmm = now.strftime("%H:%M")
    if pool_exists and hhmm < POOL_REFRESH_TIME:
        return

    symbols, payload = _build_pool_now()
    _log("pool refreshed: symbols=%s file=%s" % (len(symbols), _pool_file()))


def _maybe_run_risk(now):
    if not FAST_RISK_ENABLED:
        return

    minute = now.minute
    if minute % max(1, int(RISK_CHECK_INTERVAL_MINUTES)) != 0:
        return

    run_key = now.strftime("%Y%m%d%H%M")
    if STATE.get("last_risk_key") == run_key:
        return
    STATE["last_risk_key"] = run_key

    state = _run_fast_risk_check_now()
    _log(
        "risk check: stage=%s triggered=%s universe=%s portfolio_ret_5m=%.4f"
        % (
            state.get("stage"),
            state.get("triggered"),
            state.get("universe_count"),
            _safe_float(state.get("metrics", {}).get("portfolio_ret_5m"), 0.0),
        )
    )


def _should_run_hourly(now):
    return now.strftime("%H:%M") in RUN_TIMES


def _run_hourly_once(ContextInfo):
    pool_symbols = _load_pool_symbols()
    if not pool_symbols:
        # force rebuild if pool missing/empty
        pool_symbols, _ = _build_pool_now()

    score_rows, _ = _build_hourly_score_rows(pool_symbols)
    risk = _load_risk_state_now()
    signal_payload = _build_hourly_signals_now(score_rows, pool_symbols, risk)

    target_weights = _build_target_weights_from_signals(signal_payload)

    last_price_map = {x["symbol"]: x["last_price"] for x in score_rows}
    positions = _get_positions(last_price_map)
    total_asset = _get_total_asset(positions)

    orders, dropped = _build_orders(target_weights, positions, total_asset, score_rows)
    exec_rows = _execute_orders(ContextInfo, orders)

    plan = {
        "ts": datetime.now().isoformat(),
        "strategy": STRATEGY_NAME,
        "dry_run": DRY_RUN,
        "account_id": ACCOUNT_ID,
        "total_asset": total_asset,
        "pool_size": len(pool_symbols),
        "signal_policy": signal_payload.get("signal_policy", {}),
        "risk_overlay": signal_payload.get("risk_overlay", {}),
        "signals": signal_payload.get("signals", []),
        "orders": orders,
        "dropped": dropped,
    }

    exec_result = {
        "ts": datetime.now().isoformat(),
        "strategy": STRATEGY_NAME,
        "dry_run": DRY_RUN,
        "summary": {
            "orders_total": len(exec_rows),
            "submitted": sum(1 for x in exec_rows if x["status"] == "submitted"),
            "errors": sum(1 for x in exec_rows if x["status"] == "error"),
            "dry_run": sum(1 for x in exec_rows if x["status"] == "dry_run"),
        },
        "results": exec_rows,
    }

    plan_file, exec_file = _save_run_files(plan, exec_result)
    _log(
        "hourly done: pool=%s scores=%s orders=%s submitted=%s errors=%s dry_run=%s plan=%s exec=%s"
        % (
            len(pool_symbols),
            len(score_rows),
            exec_result["summary"]["orders_total"],
            exec_result["summary"]["submitted"],
            exec_result["summary"]["errors"],
            exec_result["summary"]["dry_run"],
            plan_file,
            exec_file,
        )
    )


def init(ContextInfo):
    _log("init at %s" % _now_str())
    _log("project_root=%s" % PROJECT_ROOT)
    _log("account_id=%s dry_run=%s" % (ACCOUNT_ID, DRY_RUN))


def handlebar(ContextInfo):
    now = datetime.now()

    try:
        _ensure_pool_ready(now)
        _maybe_run_risk(now)

        if not _should_run_hourly(now):
            return

        run_key = now.strftime("%Y%m%d%H%M")
        if STATE.get("last_hourly_key") == run_key:
            return
        STATE["last_hourly_key"] = run_key

        _run_hourly_once(ContextInfo)
    except Exception:
        _log("ERROR at %s\n%s" % (_now_str(), traceback.format_exc()))


def stop(ContextInfo):
    _log("stop at %s" % _now_str())
