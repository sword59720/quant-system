#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import hmac
import json
import hashlib
import secrets
import requests
import yaml
from datetime import datetime
from typing import Optional


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _project_root():
    return os.path.dirname(os.path.dirname(__file__))


def _load_dedup_state(path):
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return {}


def _save_dedup_state(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def send_wecom_message(
    content: str,
    title: str = "量化系统通知",
    dedup_key: Optional[str] = None,
    dedup_hours: int = 0,
):
    try:
        cfg = _load_yaml(os.path.join(_project_root(), "config", "notify.yaml"))
    except Exception as e:
        return False, f"notify config load failed: {e}"

    if not cfg or not cfg.get("enabled", False):
        return False, "notify disabled"

    bridge = cfg.get("wecom_bridge", {})
    endpoint = bridge.get("endpoint", "").strip()
    to_user = bridge.get("to_user", "").strip()
    api_token = bridge.get("api_token", "").strip()
    sign_secret = bridge.get("sign_secret", "").strip()
    timeout_sec = int(bridge.get("timeout_sec", 10))
    use_env_proxy = bool(bridge.get("use_env_proxy", False))

    if not all([endpoint, to_user, api_token, sign_secret]):
        return False, "notify config incomplete"

    # dedup: same key only once within dedup_hours
    if dedup_key and dedup_hours > 0:
        dedup_file = os.path.join(_project_root(), "outputs", "state", "wecom_dedup_state.json")
        state = _load_dedup_state(dedup_file)
        now_ts = int(time.time())
        last_ts = int(state.get(dedup_key, 0)) if str(state.get(dedup_key, 0)).isdigit() else 0
        if last_ts > 0 and (now_ts - last_ts) < int(dedup_hours * 3600):
            remain = int(dedup_hours * 3600) - (now_ts - last_ts)
            return False, f"dedup_skipped({remain}s_left)"

    body_obj = {
        "token": api_token,
        "toUser": to_user,
        "content": f"【{title}】\n{content}",
    }
    body = json.dumps(body_obj, ensure_ascii=False, separators=(",", ":"))

    ts = str(int(time.time() * 1000))
    nonce = secrets.token_hex(8)
    payload = f"{ts}.{nonce}.{body}".encode("utf-8")
    sig = hmac.new(sign_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "x-bridge-ts": ts,
        "x-bridge-nonce": nonce,
        "x-bridge-signature": sig,
    }

    try:
        with requests.Session() as s:
            s.trust_env = use_env_proxy
            r = s.post(endpoint, data=body.encode("utf-8"), headers=headers, timeout=timeout_sec)
        if r.status_code != 200:
            return False, f"http {r.status_code}: {r.text[:200]}"
        obj = r.json()
        if not obj.get("ok", False):
            return False, f"bridge error: {obj}"

        if dedup_key and dedup_hours > 0:
            dedup_file = os.path.join(_project_root(), "outputs", "state", "wecom_dedup_state.json")
            state = _load_dedup_state(dedup_file)
            state[dedup_key] = int(time.time())
            state[f"{dedup_key}__updated_at"] = datetime.now().isoformat()
            _save_dedup_state(dedup_file, state)

        return True, "ok"
    except Exception as e:
        return False, str(e)
