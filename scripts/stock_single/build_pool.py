#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_DISABLED, EXIT_OK, EXIT_OUTPUT_ERROR
from scripts.stock_single import scoring_core
from scripts.stock_single.fetch_stock_single_data import is_a_share_stock_code, normalize_symbol


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_symbols(raw_symbols: List[str]) -> Tuple[List[str], List[str], Dict[str, str]]:
    out = []
    seen = set()
    skipped_non_stock = []
    canonical_to_code: Dict[str, str] = {}
    for symbol in raw_symbols:
        norm = normalize_symbol(symbol)
        if norm is None:
            continue
        code, canonical = norm
        if not is_a_share_stock_code(code, canonical):
            skipped_non_stock.append(canonical)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
        canonical_to_code[canonical] = code
    return out, skipped_non_stock, canonical_to_code


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return ""


def load_candidate_table(pool_cfg: dict) -> pd.DataFrame:
    source_file = pool_cfg.get("source_file", "./data/stock_single/universe.csv")
    symbol_column = pool_cfg.get("symbol_column", "symbol")
    if os.path.exists(source_file):
        df = pd.read_csv(source_file)
        if symbol_column not in df.columns:
            raise RuntimeError(f"pool source missing column: {symbol_column}")
        out = pd.DataFrame({"raw_symbol": df[symbol_column].astype(str)})
        name_col = _first_existing_col(df, ["name", "名称", "证券简称"])
        industry_col = _first_existing_col(df, ["industry", "所属行业", "行业"])
        if name_col:
            out["name"] = df[name_col].astype(str)
        if industry_col:
            out["industry"] = df[industry_col].astype(str)
        return out

    static = [str(x) for x in pool_cfg.get("static_fallback", [])]
    return pd.DataFrame({"raw_symbol": static})


def load_previous_pool(pool_file: str) -> Tuple[List[str], Dict[str, int]]:
    if not os.path.exists(pool_file):
        return [], {}
    try:
        with open(pool_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return [], {}
    prev_symbols = [str(x) for x in payload.get("symbols", []) if str(x)]
    prev_hold_days = payload.get("hold_days", {}) or {}
    hold_days = {}
    for k, v in prev_hold_days.items():
        try:
            hold_days[str(k)] = int(v)
        except Exception:
            continue
    return prev_symbols, hold_days


def load_single_daily_frame(daily_dir: str, canonical: str, code: str) -> pd.DataFrame:
    fp = os.path.join(daily_dir, f"{canonical}.csv")
    if not os.path.exists(fp):
        fp = os.path.join(daily_dir, f"{code}.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date").set_index("date")
    return x


def infer_industry(symbol: str, source_row: pd.Series, industry_map: Dict[str, str]) -> str:
    if symbol in industry_map and str(industry_map[symbol]).strip():
        return str(industry_map[symbol]).strip()
    raw = ""
    if "industry" in source_row and pd.notna(source_row["industry"]):
        raw = str(source_row["industry"]).strip()
    if raw:
        return raw
    if "." in symbol:
        return symbol.split(".")[-1]
    return "UNKNOWN"


def _load_industry_map_file(path: str) -> Dict[str, str]:
    if not path or (not os.path.exists(path)):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    sym_col = _first_existing_col(df, ["symbol", "代码", "证券代码"])
    ind_col = _first_existing_col(df, ["industry", "所属行业", "行业"])
    if not sym_col or not ind_col:
        return {}
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        sym = str(r[sym_col]).strip()
        ind = str(r[ind_col]).strip()
        norm = normalize_symbol(sym)
        if not norm or not ind:
            continue
        _, canonical = norm
        out[canonical] = ind
    return out


def passes_hard_filters(symbol: str, name: str, df: pd.DataFrame, sel_cfg: dict) -> Tuple[bool, str]:
    min_history_days = int(sel_cfg.get("min_history_days", 120))
    st_filter = bool(sel_cfg.get("st_filter", True))
    suspension_lookback_days = int(sel_cfg.get("suspension_lookback_days", 20))
    max_zero_volume_ratio = float(sel_cfg.get("max_zero_volume_ratio", 0.20))
    recent_limit_lookback_days = int(sel_cfg.get("recent_limit_lookback_days", 5))
    limit_move_threshold_pct = float(sel_cfg.get("limit_move_threshold_pct", 9.7))
    max_recent_limit_hits = int(sel_cfg.get("max_recent_limit_hits", 2))

    if df.empty:
        return False, "missing_daily"
    if len(df) < min_history_days:
        return False, "short_history"
    if st_filter and name:
        if "ST" in str(name).upper():
            return False, "st_name"

    tail = df.tail(max(1, suspension_lookback_days))
    vol = pd.to_numeric(tail.get("volume"), errors="coerce")
    zero_ratio = float((vol.fillna(0.0) <= 0).mean()) if len(tail) > 0 else 1.0
    if zero_ratio > max_zero_volume_ratio:
        return False, "suspension_like"

    pct = pd.to_numeric(df.get("pct_change"), errors="coerce").tail(max(1, recent_limit_lookback_days))
    limit_hits = int((pct.abs() >= limit_move_threshold_pct).sum()) if len(pct) > 0 else 0
    if limit_hits > max_recent_limit_hits:
        return False, "recent_limit_hits"
    return True, "ok"


def passes_liquidity(df: pd.DataFrame, liq_cfg: dict) -> Tuple[bool, str]:
    if not bool(liq_cfg.get("enabled", True)):
        return True, "disabled"
    lookback_days = int(liq_cfg.get("lookback_days", 20))
    min_avg_turnover = float(liq_cfg.get("min_avg_turnover", 100000000))
    min_avg_volume = float(liq_cfg.get("min_avg_volume", 1000000))
    tail = df.tail(max(1, lookback_days))
    amt = pd.to_numeric(tail.get("amount"), errors="coerce")
    vol = pd.to_numeric(tail.get("volume"), errors="coerce")
    avg_amt = float(amt.mean()) if len(amt) > 0 else 0.0
    avg_vol = float(vol.mean()) if len(vol) > 0 else 0.0
    if avg_amt < min_avg_turnover:
        return False, "low_turnover"
    if avg_vol < min_avg_volume:
        return False, "low_volume"
    return True, "ok"


def apply_industry_cap(
    ranked_symbols: List[str],
    industry_of: Dict[str, str],
    target_pool_size: int,
    industry_cfg: dict,
) -> List[str]:
    if not bool(industry_cfg.get("enabled", True)):
        return ranked_symbols[:target_pool_size]
    max_industry_weight = float(industry_cfg.get("max_industry_weight", 0.30))
    max_per_industry = max(1, int(target_pool_size * max_industry_weight + 1e-9))

    min_industry_count = int(industry_cfg.get("min_industry_count", 0))
    out: List[str] = []
    ind_count: Dict[str, int] = {}
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

    if min_industry_count > 1 and out:
        distinct = set([industry_of.get(s, "UNKNOWN") for s in out])
        if len(distinct) < min_industry_count:
            best_by_ind = {}
            for s in ranked_symbols:
                ind = industry_of.get(s, "UNKNOWN")
                if ind not in best_by_ind:
                    best_by_ind[ind] = s
            for ind, cand in best_by_ind.items():
                if len(distinct) >= min_industry_count:
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
                distinct = set([industry_of.get(s, "UNKNOWN") for s in out])
    return out


def build_pool(runtime: dict, stock_single: dict) -> tuple[list[str], str]:
    pool_cfg = stock_single.get("pool", {})
    max_symbols = int(stock_single.get("max_symbols_in_pool", 150))
    liq_cfg = stock_single.get("liquidity_filter", {})
    industry_cfg = stock_single.get("industry_diversification", {})
    selection_cfg = pool_cfg.get("selection", {})

    candidate_df = load_candidate_table(pool_cfg)
    raw_symbols = candidate_df.get("raw_symbol", pd.Series(dtype=str)).astype(str).tolist()
    symbols, skipped_non_stock, canonical_to_code = normalize_symbols(raw_symbols)
    if not symbols:
        symbols = [str(x) for x in pool_cfg.get("static_fallback", [])]
        symbols, skipped_extra, canonical_to_code = normalize_symbols(symbols)
        skipped_non_stock.extend(skipped_extra)

    source_map = {}
    for _, r in candidate_df.iterrows():
        norm = normalize_symbol(str(r.get("raw_symbol", "")).strip())
        if not norm:
            continue
        _, canonical = norm
        source_map[canonical] = r

    data_dir = runtime.get("paths", {}).get("data_dir", "./data")
    daily_dir = stock_single.get("data", {}).get("daily_output_dir", os.path.join(data_dir, "stock_single", "daily"))
    pool_out = stock_single.get("paths", {}).get("pool_file", "./outputs/orders/stock_single_pool.json")
    prev_symbols, prev_hold_days = load_previous_pool(pool_out)

    industry_map_file = str(selection_cfg.get("industry_file", "./data/stock_single/industry_map.csv"))
    industry_map = _load_industry_map_file(industry_map_file)

    hard_pass: List[str] = []
    frames: Dict[str, pd.DataFrame] = {}
    reject_reason = {}

    for s in symbols:
        code = canonical_to_code.get(s, re.sub(r"\D", "", s))
        row = source_map.get(s, pd.Series(dtype=object))
        name = str(row.get("name", "")).strip() if "name" in row else ""
        df = load_single_daily_frame(daily_dir, s, code)
        ok_hard, reason_hard = passes_hard_filters(symbol=s, name=name, df=df, sel_cfg=selection_cfg)
        if not ok_hard:
            reject_reason[s] = reason_hard
            continue
        ok_liq, reason_liq = passes_liquidity(df=df, liq_cfg=liq_cfg)
        if not ok_liq:
            reject_reason[s] = reason_liq
            continue
        hard_pass.append(s)
        frames[s] = df

    if not hard_pass:
        raise RuntimeError("no symbols left after hard/liquidity filters")

    close_df = scoring_core.build_panel(frames, "close")
    vol_df = scoring_core.build_panel(frames, "volume")
    amt_df = scoring_core.build_panel(frames, "amount")
    if close_df.empty:
        raise RuntimeError("daily close panel empty after filtering")

    score_bundle = scoring_core.compute_score_bundle(
        close_df=close_df,
        vol_df=vol_df,
        amt_df=amt_df,
        data_cfg=stock_single.get("data", {}),
        backtest_cfg=stock_single.get("backtest", {}),
    )
    score_df = score_bundle["score_df"]
    vol20 = score_bundle["vol20"]
    idx_signal = len(score_df.index) - 1
    if idx_signal < 0:
        raise RuntimeError("score panel empty")

    tradable_symbols = set([s for s in close_df.columns if pd.notna(close_df.iloc[idx_signal].get(s))])
    score_row = scoring_core.apply_realtime_adjustments(
        score_row=score_df.iloc[idx_signal],
        tradable_symbols=tradable_symbols,
        close_df=close_df,
        vol20=vol20,
        idx_signal=idx_signal,
        signal_cfg=stock_single.get("signal", {}),
    )

    rank_df = pd.DataFrame({"symbol": list(tradable_symbols)})
    rank_df["score"] = rank_df["symbol"].map(lambda x: pd.to_numeric(pd.Series([score_row.get(x)]), errors="coerce").iloc[0])
    rank_df = rank_df.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)
    if rank_df.empty:
        raise RuntimeError("no tradable symbols with valid score")

    n = len(rank_df)
    default_target = max(int(stock_single.get("max_positions", 10)) * 6, 30)
    target_pool_size = int(selection_cfg.get("target_pool_size", default_target))
    target_pool_size = max(10, min(max_symbols, target_pool_size, n))

    entry_q = float(selection_cfg.get("entry_quantile", 0.30))
    exit_q = float(selection_cfg.get("exit_quantile", 0.50))
    entry_q = max(0.01, min(1.0, entry_q))
    exit_q = max(entry_q, min(1.0, exit_q))
    entry_n = max(1, min(n, max(target_pool_size, int(round(n * entry_q)))))
    keep_n = max(entry_n, min(n, int(round(n * exit_q))))
    min_pool_hold_days = int(selection_cfg.get("min_pool_hold_days", 5))

    ranked_symbols = rank_df["symbol"].tolist()
    rank_map = {s: i + 1 for i, s in enumerate(ranked_symbols)}
    entry_set = set(ranked_symbols[:entry_n])

    selected = set(entry_set)
    for s in prev_symbols:
        if s not in rank_map:
            continue
        prev_days = int(prev_hold_days.get(s, 0))
        if s in entry_set:
            selected.add(s)
            continue
        if prev_days < min_pool_hold_days:
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

    industry_of: Dict[str, str] = {}
    for s in ordered_selected:
        row = source_map.get(s, pd.Series(dtype=object))
        industry_of[s] = infer_industry(s, row, industry_map)
    final_symbols = apply_industry_cap(
        ranked_symbols=ordered_selected,
        industry_of=industry_of,
        target_pool_size=target_pool_size,
        industry_cfg=industry_cfg,
    )

    hold_days = {}
    for s in final_symbols:
        hold_days[s] = int(prev_hold_days.get(s, 0)) + 1

    rank_preview = []
    for s in final_symbols[:20]:
        rank_preview.append(
            {
                "symbol": s,
                "rank": int(rank_map.get(s, 10**9)),
                "score": float(rank_df.loc[rank_df["symbol"] == s, "score"].iloc[0]),
                "industry": industry_of.get(s, "UNKNOWN"),
                "hold_days": int(hold_days.get(s, 1)),
            }
        )

    reason_count = {}
    for r in reject_reason.values():
        reason_count[r] = int(reason_count.get(r, 0)) + 1

    ensure_dir(os.path.dirname(pool_out) or ".")
    payload = {
        "ts": datetime.now().isoformat(),
        "mode": "single_stock_hourly",
        "symbol_count": len(final_symbols),
        "symbols": final_symbols,
        "hold_days": hold_days,
        "selection": {
            "candidate_count_raw": len(raw_symbols),
            "candidate_count_normalized": len(symbols),
            "candidate_count_after_filters": len(hard_pass),
            "candidate_count_scored": int(n),
            "target_pool_size": int(target_pool_size),
            "entry_quantile": float(entry_q),
            "exit_quantile": float(exit_q),
            "entry_n": int(entry_n),
            "keep_n": int(keep_n),
            "min_pool_hold_days": int(min_pool_hold_days),
            "reject_reason_count": reason_count,
            "skipped_non_stock_count": len(skipped_non_stock),
            "liquidity_filter_enabled": bool(liq_cfg.get("enabled", True)),
            "industry_constraint_enabled": bool(industry_cfg.get("enabled", True)),
            "score_date": str(score_df.index[idx_signal].date()),
            "feature_coverage": score_bundle.get("coverage", {}),
            "rank_preview_top20": rank_preview,
        },
    }
    with open(pool_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return final_symbols, pool_out


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[stock-single-pool] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_single.get("enabled", False):
        print("[stock-single] disabled")
        return EXIT_DISABLED

    try:
        symbols, pool_out = build_pool(runtime, stock_single)
    except OSError as e:
        print(f"[stock-single-pool] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[stock-single-pool] failed: {e}")
        return EXIT_CONFIG_ERROR

    print(f"[stock-single-pool] symbols={len(symbols)} -> {pool_out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
