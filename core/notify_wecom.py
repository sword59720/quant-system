#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import time
import hmac
import hashlib
import secrets
import socket
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlparse

import yaml


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


def _dedup_file_path() -> str:
    return os.path.join(_project_root(), "outputs", "state", "wecom_dedup_state.json")


def _should_skip_dedup(dedup_key: Optional[str], dedup_hours: int) -> Tuple[bool, str]:
    if not dedup_key or dedup_hours <= 0:
        return False, ""
    dedup_file = _dedup_file_path()
    state = _load_dedup_state(dedup_file)
    now_ts = int(time.time())
    last_ts = int(state.get(dedup_key, 0)) if str(state.get(dedup_key, 0)).isdigit() else 0
    if last_ts > 0 and (now_ts - last_ts) < int(dedup_hours * 3600):
        remain = int(dedup_hours * 3600) - (now_ts - last_ts)
        return True, f"dedup_skipped({remain}s_left)"
    return False, ""


def _save_dedup_if_needed(dedup_key: Optional[str], dedup_hours: int):
    if not dedup_key or dedup_hours <= 0:
        return
    dedup_file = _dedup_file_path()
    state = _load_dedup_state(dedup_file)
    state[dedup_key] = int(time.time())
    state[f"{dedup_key}__updated_at"] = datetime.now().isoformat()
    _save_dedup_state(dedup_file, state)


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


def _send_via_openclaw(content: str, title: str, cfg: dict) -> Tuple[bool, str]:
    channel = str(cfg.get("channel", "feishu")).strip() or "feishu"
    target = str(cfg.get("target") or cfg.get("to_user") or "").strip()
    account = str(cfg.get("account") or "").strip()
    binary = str(cfg.get("binary", "openclaw")).strip() or "openclaw"
    timeout_sec = int(cfg.get("timeout_sec", 60))
    dry_run = bool(cfg.get("dry_run", False))
    extra_args = cfg.get("extra_args", []) or []

    if not target:
        return False, "openclaw target missing"

    message = f"【{title}】\n{content}"
    cmd = [binary, "message", "send", "--channel", channel, "--target", target, "--message", message, "--json"]
    if account:
        cmd.extend(["--account", account])
    if dry_run:
        cmd.append("--dry-run")
    for arg in extra_args:
        cmd.append(str(arg))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except Exception as e:
        return False, f"openclaw exec failed: {e}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit={proc.returncode}"
        return False, f"openclaw send failed: {detail[:400]}"
    return True, "ok"


def _send_via_bridge(content: str, title: str, cfg: dict) -> Tuple[bool, str]:
    try:
        import requests
    except Exception as e:
        return False, f"requests unavailable: {e}"

    endpoint = str(cfg.get("endpoint", "")).strip()
    to_user = str(cfg.get("to_user", "")).strip()
    api_token = str(cfg.get("api_token", "")).strip()
    sign_secret = str(cfg.get("sign_secret", "")).strip()
    timeout_sec = int(cfg.get("timeout_sec", 10))
    use_env_proxy = bool(cfg.get("use_env_proxy", False))

    if not all([endpoint, to_user, api_token, sign_secret]):
        return False, "bridge config incomplete"

    body_obj = {"token": api_token, "toUser": to_user, "content": f"【{title}】\n{content}"}
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

    ok, detail = _attempt_send(use_env_proxy)
    if ok:
        return True, detail
    ok_proxy, proxy_msg = _check_proxy_reachable(timeout_sec=min(timeout_sec, 3))
    if not ok_proxy:
        return False, f"first_fail={detail}; proxy_check={proxy_msg}"
    last_detail = detail
    for idx in [2, 3]:
        ok_retry, detail_retry = _attempt_send(True)
        if ok_retry:
            return True, f"ok(retry#{idx}; {proxy_msg})"
        last_detail = detail_retry
    return False, f"first_fail={detail}; retry_fail={last_detail}; {proxy_msg}"


def send_wecom_message(content: str, title: str = "量化系统通知", dedup_key: Optional[str] = None, dedup_hours: int = 0):
    try:
        cfg = _load_yaml(os.path.join(_project_root(), "config", "notify.yaml"))
    except Exception as e:
        return False, f"notify config load failed: {e}"

    if not cfg or not cfg.get("enabled", False):
        return False, "notify disabled"

    should_skip, skip_msg = _should_skip_dedup(dedup_key, dedup_hours)
    if should_skip:
        return False, skip_msg

    provider = str(cfg.get("provider", "openclaw")).strip().lower() or "openclaw"
    if provider == "openclaw":
        ok, detail = _send_via_openclaw(content, title, cfg.get("openclaw_message", {}) or {})
        if ok:
            _save_dedup_if_needed(dedup_key, dedup_hours)
            return True, detail
        fallback = str(cfg.get("fallback_provider", "bridge")).strip().lower()
        if fallback == "bridge":
            ok2, detail2 = _send_via_bridge(content, title, cfg.get("wecom_bridge", {}) or {})
            if ok2:
                _save_dedup_if_needed(dedup_key, dedup_hours)
                return True, f"fallback_bridge={detail2}"
            return False, f"openclaw={detail}; bridge={detail2}"
        return False, detail

    if provider == "bridge":
        ok, detail = _send_via_bridge(content, title, cfg.get("wecom_bridge", {}) or {})
        if ok:
            _save_dedup_if_needed(dedup_key, dedup_hours)
            return True, detail
        return False, detail

    return False, f"unsupported notify provider: {provider}"
