#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any


def normalize_market_type(raw: Any) -> str:
    v = str(raw or "spot").strip().lower()
    if v in {"spot", "crypto_spot"}:
        return "spot"
    if v in {"future", "futures", "crypto_futures"}:
        return "future"
    if v in {"swap", "perp", "perpetual", "crypto_swap"}:
        return "swap"
    if "future" in v:
        return "future"
    if "swap" in v or "perp" in v:
        return "swap"
    return "spot"


def resolve_crypto_model_cfg(crypto_cfg: Dict[str, Any]) -> Dict[str, Any]:
    trade_cfg = crypto_cfg.get("trade", {}) or {}
    market_type = normalize_market_type(trade_cfg.get("market_type", crypto_cfg.get("market", "spot")))
    contract_mode = market_type in {"swap", "future"}

    profile_key = "contract_model" if contract_mode else "spot_model"
    profile_cfg = crypto_cfg.get(profile_key, {}) or {}

    signal = dict(crypto_cfg.get("signal", {}) or {})
    defense = dict(crypto_cfg.get("defense", {}) or {})
    signal.update(profile_cfg.get("signal", {}) or {})
    defense.update(profile_cfg.get("defense", {}) or {})

    profile_name = str(profile_cfg.get("name", profile_key))
    return {
        "market_type": market_type,
        "contract_mode": contract_mode,
        "profile_key": profile_key,
        "profile_name": profile_name,
        "signal": signal,
        "defense": defense,
    }
