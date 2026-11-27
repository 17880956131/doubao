#!/usr/bin/env python3
"""从记忆/主会话/修正里导出 SFT 训练集（单主体：Agent 的话术 + 自我内核）"""
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "sft.jsonl"


def self_core():
    try:
        return json.load(open(DATA / "self_core.json", encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def system_prompt(core):
    parts = ["你是豆宝，属于智命（ZhiMing）。"]
    if core.get("identity"):
        parts.append(core["identity"])
    if core.get("values"):
        parts.append("原则：" + "；".join(core["values"]))
    if core.get("relationship"):
        parts.append(core["relationship"])
    return "\n".join(parts)


def load_chat_pairs():
    pairs = []
    try:
        conn = sqlite3.connect(DATA / "memory.db", timeout=5)
        rows = conn.execute("SELECT ts, text FROM memories WHERE kind='chat' ORDER BY ts").fetchall()
        conn.close()
    except sqlite3.Error:
        return pairs
    last_user = None
    for ts, text in rows:
        text = str(text or "")
        if text.startswith("用户在对话里说：") or text.startswith("我说：") or text.startswith("用户："):
            cleaned = text.split("：", 1)[-1].strip()
            if cleaned:
                last_user = cleaned
    # 主会话历史里成对提取
    ctx = DATA.parent / "data" / "orchestrate_history.json"
    try:
        sessions = json.load(open(ctx, encoding="utf-8"))
    except (OSError, ValueError):
        sessions = {}
    for history in sessions.values():
        user_msg = None
        for item in history:
            if item.get("role") == "user":
                user_msg = item.get("content")
            elif item.get("role") == "assistant" and user_msg:
                pairs.append((user_msg, item.get("content")))
                user_msg = None
    return pairs


def main():
    core = self_core()
    system = system_prompt(core)
    pairs = load_chat_pairs()
    seen = set()
    count = 0
    with open(OUT, "w", encoding="utf-8") as handle:
        for user, assistant in pairs:
            user = str(user or "").strip()[:800]
            assistant = str(assistant or "").strip()[:800]
            if len(user) < 2 or len(assistant) < 2:
                continue
            key = (user, assistant)
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"ok": True, "samples": count, "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
