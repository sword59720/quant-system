#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import hmac
import json
import socket
import hashlib
import secrets
import requests
import yaml
from urllib.parse import urlparse
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


def _check_proxy_reachable(timeout_sec: int = 3):
    proxy = (
        os.environ.get("ALL_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("all_proxy")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    if not proxy:
        return False, "proxy env missing"
    try:
        u = urlparse(proxy)
        host = u.hostname
        port = int(u.port or (443 if (u.scheme or "").endswith("s") else 80))
        if not host:
            return False, f"invalid proxy url: {proxy}"
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True, f"proxy reachable: {host}:{port}"
    except Exception as e:
        return False, f"proxy unreachable: {e}"


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

    def _post_once(trust_env_flag: bool):
        with requests.Session() as s:
            s.trust_env = trust_env_flag
            return s.post(endpoint, data=body.encode("utf-8"), headers=headers, timeout=timeout_sec)

    def _save_dedup_if_needed():
        if dedup_key and dedup_hours > 0:
            dedup_file = os.path.join(_project_root(), "outputs", "state", "wecom_dedup_state.json")
            state = _load_dedup_state(dedup_file)
            state[dedup_key] = int(time.time())
            state[f"{dedup_key}__updated_at"] = datetime.now().isoformat()
            _save_dedup_state(dedup_file, state)

    def _attempt_send(trust_env_flag: bool):
        try:
            r = _post_once(trust_env_flag)
            if r.status_code != 200:
                return False, f"http {r.status_code}: {r.text[:200]}"
            obj = r.json()
            if not obj.get("ok", False):
                return False, f"bridge error: {obj}"
            return True, "ok"
        except Exception as e:
            return False, str(e)

    # 第1次发送：按配置执行（要求 use_env_proxy=true）
    ok, detail = _attempt_send(use_env_proxy)
    if ok:
        _save_dedup_if_needed()
        return True, detail

    # 失败后：先检查代理，再进行第2/第3次重试（强制使用环境代理）
    ok_proxy, proxy_msg = _check_proxy_reachable(timeout_sec=min(timeout_sec, 3))
    if not ok_proxy:
        return False, f"first_fail={detail}; proxy_check={proxy_msg}"

    last_detail = detail
    for idx in [2, 3]:
        ok_retry, detail_retry = _attempt_send(True)
        if ok_retry:
            _save_dedup_if_needed()
            return True, f"ok(retry#{idx}; {proxy_msg})"
        last_detail = detail_retry

    return False, f"first_fail={detail}; retry_fail={last_detail}; {proxy_msg}"
