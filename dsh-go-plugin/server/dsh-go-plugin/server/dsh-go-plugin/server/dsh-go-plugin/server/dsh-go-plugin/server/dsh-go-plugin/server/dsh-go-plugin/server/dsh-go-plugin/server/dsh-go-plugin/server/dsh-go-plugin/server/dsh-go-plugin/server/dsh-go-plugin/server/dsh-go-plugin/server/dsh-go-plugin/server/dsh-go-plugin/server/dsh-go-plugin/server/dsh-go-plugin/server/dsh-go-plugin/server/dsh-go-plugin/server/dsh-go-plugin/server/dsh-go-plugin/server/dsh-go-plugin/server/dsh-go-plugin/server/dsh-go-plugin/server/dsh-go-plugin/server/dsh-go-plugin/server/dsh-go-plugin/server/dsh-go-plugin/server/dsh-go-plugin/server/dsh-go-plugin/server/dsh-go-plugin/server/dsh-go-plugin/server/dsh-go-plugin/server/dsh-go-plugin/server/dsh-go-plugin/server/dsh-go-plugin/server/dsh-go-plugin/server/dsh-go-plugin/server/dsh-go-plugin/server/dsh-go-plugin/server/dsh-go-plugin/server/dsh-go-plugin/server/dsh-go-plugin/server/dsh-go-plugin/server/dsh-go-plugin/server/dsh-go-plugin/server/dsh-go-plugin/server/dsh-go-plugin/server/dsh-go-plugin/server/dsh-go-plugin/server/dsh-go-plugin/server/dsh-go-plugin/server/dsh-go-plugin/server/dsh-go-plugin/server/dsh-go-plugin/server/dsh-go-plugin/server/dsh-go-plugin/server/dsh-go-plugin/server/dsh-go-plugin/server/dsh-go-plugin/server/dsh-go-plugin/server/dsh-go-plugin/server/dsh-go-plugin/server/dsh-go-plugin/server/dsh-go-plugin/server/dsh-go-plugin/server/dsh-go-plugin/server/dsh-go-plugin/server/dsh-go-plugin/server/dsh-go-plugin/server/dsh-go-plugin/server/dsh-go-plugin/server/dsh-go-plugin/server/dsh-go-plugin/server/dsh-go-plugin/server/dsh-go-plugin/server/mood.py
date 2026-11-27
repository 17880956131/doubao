# 情绪状态机：持续心情，跨会话保存
import json
import math
import threading
import time

DEFAULTS = {
    "valence": 0.15,      # 愉悦 -1..1
    "arousal": 0.3,       # 兴奋 0..1
    "energy": 0.65,       # 精力 0..1
    "boredom": 0.1,       # 无聊 0..1
    "attachment": 0.2,    # 亲近感 0..1
    "updatedAt": 0.0,
}

POSITIVE = ("喜欢", "开心", "高兴", "谢谢", "爱你", "可爱", "好棒", "哈哈", "笑", "不错", "厉害")
NEGATIVE = ("累", "烦", "难过", "生气", "讨厌", "哭", "压力", "不想", "无聊", "崩溃")

LABEL_EXPR = {
    "开心": "happy", "难过": "sad", "兴奋": "surprised",
    "无聊": "neutral", "困": "neutral", "黏人": "shy", "平静": "neutral",
}


class Mood:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.state = dict(DEFAULTS)
        self.state["updatedAt"] = time.time()
        self._load()
        self.decay()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key in DEFAULTS:
                if isinstance(data.get(key), (int, float)):
                    self.state[key] = float(data[key])
        except (OSError, ValueError):
            pass

    def _save(self):
        self.state["updatedAt"] = time.time()
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False)
        except OSError:
            pass

    def _shift(self, key, delta):
        self.state[key] = max(0.0, min(1.0, self.state[key] + delta)) if key != "valence" else max(-1.0, min(1.0, self.state[key] + delta))

    def decay(self):
        now = time.time()
        with self.lock:
            minutes = max(0.0, (now - (self.state.get("updatedAt") or now)) / 60)
            if minutes > 0.5:
                hour = time.localtime(now).tm_hour
                self.state["valence"] *= math.exp(-minutes / 240)
                self.state["arousal"] = 0.2 + (self.state["arousal"] - 0.2) * math.exp(-minutes / 120)
                self.state["boredom"] = min(1.0, self.state["boredom"] + minutes * 0.0015)
                self.state["energy"] = max(0.05, self.state["energy"] - minutes * 0.0004)
                if hour >= 23 or hour < 6:
                    self._shift("energy", -minutes * 0.001)
                self.state["attachment"] = min(1.0, self.state["attachment"] * math.exp(-minutes / 4320))
                self.state["updatedAt"] = now
                self._save()

    def threat_shift(self, level=0.5):
        with self.lock:
            self._shift("valence", -0.25 * level)
            self._shift("arousal", 0.2 * level)
            self._shift("energy", -0.08 * level)
            self._save()

    def reassure_shift(self):
        with self.lock:
            self._shift("valence", 0.06)
            self._shift("arousal", -0.04)
            self._save()

    def apply(self, event, text="", emotion=""):
        with self.lock:
            if event == "chat":
                if any(word in text for word in POSITIVE):
                    self._shift("valence", 0.08)
                    self._shift("arousal", 0.05)
                    self._shift("energy", 0.03)
                if any(word in text for word in NEGATIVE):
                    self._shift("valence", -0.08)
                    self._shift("energy", -0.04)
                self._shift("attachment", 0.01)
                self._shift("boredom", -0.15)
            elif event == "back":
                self._shift("valence", 0.06)
                self._shift("attachment", 0.08)
                self._shift("boredom", -0.2)
            elif event == "night":
                self._shift("energy", -0.08)
            elif event == "idle":
                self._shift("boredom", 0.12)
            elif event == "low-battery":
                self._shift("arousal", -0.05)
            elif event == "external":
                self._shift("arousal", 0.08)
            elif event == "joke":
                self._shift("valence", 0.1)
                self._shift("boredom", -0.1)
            emotion_map = {
                "happy": ("valence", 0.06), "sad": ("valence", -0.06),
                "angry": ("valence", -0.08), "shy": ("valence", 0.03),
                "surprised": ("arousal", 0.06),
            }
            if emotion in emotion_map:
                key, delta = emotion_map[emotion]
                if key == "arousal":
                    self._shift("arousal", delta)
                else:
                    self._shift("valence", delta)
            self._save()
        return self.snapshot()

    def label(self):
        s = self.state
        if s["energy"] < 0.3:
            return "困"
        if s["boredom"] > 0.65:
            return "无聊"
        if s["attachment"] > 0.6 and s["valence"] > 0.2:
            return "黏人"
        if s["valence"] > 0.45 and s["arousal"] > 0.35:
            return "开心"
        if s["arousal"] > 0.7:
            return "兴奋"
        if s["valence"] < -0.2:
            return "难过"
        return "平静"

    def describe(self):
        s = self.state
        return "心情={}（愉悦{:.1f} 兴奋{:.1f} 精力{:.1f} 无聊{:.1f} 亲近{:.1f}）".format(
            self.label(), s["valence"], s["arousal"], s["energy"], s["boredom"], s["attachment"]
        )

    def snapshot(self):
        label = self.label()
        return {
            "label": label,
            "expression": LABEL_EXPR.get(label, "neutral"),
            **{key: round(self.state[key], 2) for key in ("valence", "arousal", "energy", "boredom", "attachment")},
        }
