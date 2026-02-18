#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def resolve_min_turnover(base_min_turnover: float, alloc_pct: float, guard_cfg: dict) -> tuple[float, dict]:
    base = float(base_min_turnover)
    dynamic_cfg = guard_cfg.get("dynamic_min_turnover", {}) if isinstance(guard_cfg, dict) else {}
    enabled = bool(dynamic_cfg.get("enabled", False))
    meta = {
        "enabled": enabled,
        "base_min_turnover": base,
    }
    if not enabled:
        meta["effective_min_turnover"] = base
        return base, meta

    alloc_ref = float(dynamic_cfg.get("alloc_ref", 0.70))
    alloc_ref = max(alloc_ref, 1e-6)
    ratio = float(alloc_pct) / alloc_ref

    min_multiplier = float(dynamic_cfg.get("min_multiplier", 0.60))
    max_multiplier = float(dynamic_cfg.get("max_multiplier", 1.40))
    if min_multiplier > max_multiplier:
        min_multiplier, max_multiplier = max_multiplier, min_multiplier

    multiplier = _clamp(ratio, min_multiplier, max_multiplier)
    effective = base * multiplier

    floor_v = dynamic_cfg.get("floor")
    ceil_v = dynamic_cfg.get("ceil")
    if floor_v is not None:
        effective = max(effective, float(floor_v))
    if ceil_v is not None:
        effective = min(effective, float(ceil_v))
    effective = max(effective, 0.0)

    meta.update(
        {
            "alloc_pct": float(alloc_pct),
            "alloc_ref": alloc_ref,
            "ratio": ratio,
            "multiplier": multiplier,
            "min_multiplier": min_multiplier,
            "max_multiplier": max_multiplier,
            "floor": None if floor_v is None else float(floor_v),
            "ceil": None if ceil_v is None else float(ceil_v),
            "effective_min_turnover": effective,
        }
    )
    return effective, meta
