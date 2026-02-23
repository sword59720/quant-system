#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for resolving stock broker config from runtime.yaml."""

from __future__ import annotations

import copy

SUPPORTED_STOCK_BROKERS = {"guotou", "myquant"}


def _normalize_broker(value) -> str:
    return str(value or "").strip().lower()


def resolve_runtime_stock_broker(runtime: dict, *, strategy: str) -> tuple[str, str]:
    """
    Resolve broker for a stock strategy from runtime config.

    Priority:
    1) stock_brokers.<strategy>
    2) broker.<strategy> (when broker is a dict)
    3) broker.default (when broker is a dict)
    4) broker (legacy scalar)
    """
    if not isinstance(runtime, dict):
        return "", ""

    stock_brokers = runtime.get("stock_brokers")
    if isinstance(stock_brokers, dict):
        broker = _normalize_broker(stock_brokers.get(strategy))
        if broker:
            return broker, f"stock_brokers.{strategy}"

    broker_cfg = runtime.get("broker")
    if isinstance(broker_cfg, dict):
        broker = _normalize_broker(broker_cfg.get(strategy))
        if broker:
            return broker, f"broker.{strategy}"
        broker = _normalize_broker(broker_cfg.get("default"))
        if broker:
            return broker, "broker.default"

    broker = _normalize_broker(broker_cfg)
    if broker:
        return broker, "broker"
    return "", ""


def _merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key == "accounts":
            continue
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            out[key] = _merge_dict(base_value, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_strategy_account_config(
    broker_full_config: dict,
    *,
    broker: str,
    strategy: str,
) -> tuple[dict, str]:
    """
    Resolve broker account config by strategy, with backward compatibility.

    Priority under broker node:
    1) accounts.<strategy>
    2) accounts.default
    3) root broker config (legacy)
    """
    if not isinstance(broker_full_config, dict):
        return {}, ""

    root = broker_full_config.get(broker)
    if not isinstance(root, dict):
        return {}, ""

    base = {k: copy.deepcopy(v) for k, v in root.items() if k != "accounts"}
    accounts = root.get("accounts")
    if isinstance(accounts, dict):
        strategy_cfg = accounts.get(strategy)
        if isinstance(strategy_cfg, dict):
            return _merge_dict(base, strategy_cfg), f"{broker}.accounts.{strategy}"
        default_cfg = accounts.get("default")
        if isinstance(default_cfg, dict):
            return _merge_dict(base, default_cfg), f"{broker}.accounts.default"

    return base, broker
