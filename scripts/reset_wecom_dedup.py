#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "outputs", "state", "wecom_dedup_state.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    print(f"[dedup] reset ok: {path}")


if __name__ == "__main__":
    main()
