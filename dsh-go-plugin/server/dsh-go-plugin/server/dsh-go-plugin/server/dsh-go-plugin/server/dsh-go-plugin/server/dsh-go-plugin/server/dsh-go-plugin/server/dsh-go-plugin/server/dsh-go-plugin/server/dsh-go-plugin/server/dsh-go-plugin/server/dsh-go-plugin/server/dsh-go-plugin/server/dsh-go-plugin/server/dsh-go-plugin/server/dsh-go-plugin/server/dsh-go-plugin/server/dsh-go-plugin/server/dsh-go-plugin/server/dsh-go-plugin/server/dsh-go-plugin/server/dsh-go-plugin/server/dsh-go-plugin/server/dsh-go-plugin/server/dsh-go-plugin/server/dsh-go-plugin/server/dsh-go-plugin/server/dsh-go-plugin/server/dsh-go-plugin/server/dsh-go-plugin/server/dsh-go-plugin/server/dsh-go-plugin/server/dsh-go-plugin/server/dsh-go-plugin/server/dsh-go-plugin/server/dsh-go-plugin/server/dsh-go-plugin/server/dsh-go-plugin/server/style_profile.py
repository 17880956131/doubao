# 语言风格蒸馏：从"被回应的回复"里总结出用户偏好的表达方式，写成提示词里的偏好
import json
import os
import sqlite3
import time

DB_PATH = "/root/workspace/digital-human/data/trajectory.db"


def _rows():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        rows = conn.execute(
            "SELECT reply, emotion, engaged FROM inference_samples WHERE engaged IS NOT NULL AND reply != ''"
        ).fetchall()
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def _stats(rows):
    if not rows:
        return None
    lengths = [len(reply) for reply, _, _ in rows]
    questions = sum(1 for reply, _, _ in rows if ("？" in reply or "?" in reply)) / len(rows)
    emojis = sum(1 for reply, _, _ in rows if any(ord(ch) > 0x1F000 for ch in reply)) / len(rows)
    happy = sum(1 for _, emotion, _ in rows if emotion in ("happy", "shy")) / len(rows)
    return {
        "count": len(rows),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "median_len": float(sorted(lengths)[len(lengths) // 2]),
        "question_rate": round(questions, 2),
        "emoji_rate": round(emojis, 2),
        "positive_emotion_rate": round(happy, 2),
    }


class StyleProfile:
    def __init__(self, path):
        self.path = path
        self.profile = {}
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                self.profile = json.load(handle)
        except (OSError, ValueError):
            self.profile = {}

    def rebuild(self):
        rows = _rows()
        engaged = [row for row in rows if row[2]]
        ignored = [row for row in rows if not row[2]]
        engaged_stats = _stats(engaged)
        other_stats = _stats(ignored)
        if not engaged_stats:
            return {"ok": False, "reason": "还没有被回应的样本", "profile": self.profile}
        hints = []
        target_len = int(round(engaged_stats["median_len"])) or 40
        hints.append("回复控制在 %d 字左右" % max(12, min(120, target_len)))
        if engaged_stats["question_rate"] >= 0.4:
            hints.append("可以多用问句接话")
        elif engaged_stats["question_rate"] <= 0.1:
            hints.append("少用问句")
        if engaged_stats["emoji_rate"] >= 0.2:
            hints.append("可以带一点表情符号")
        elif engaged_stats["emoji_rate"] <= 0.05:
            hints.append("不要用表情符号")
        if other_stats and engaged_stats["avg_len"] < other_stats["avg_len"]:
            hints.append("说太长容易没人接，尽量短")
        self.profile = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engaged": engaged_stats,
            "ignored": other_stats,
            "hint": "；".join(hints),
        }
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.profile, handle, ensure_ascii=False)
        except OSError:
            pass
        return {"ok": True, **self.profile}

    def hint(self):
        return (self.profile or {}).get("hint", "")

    def status(self):
        return {"loaded": bool(self.profile), **self.profile}
