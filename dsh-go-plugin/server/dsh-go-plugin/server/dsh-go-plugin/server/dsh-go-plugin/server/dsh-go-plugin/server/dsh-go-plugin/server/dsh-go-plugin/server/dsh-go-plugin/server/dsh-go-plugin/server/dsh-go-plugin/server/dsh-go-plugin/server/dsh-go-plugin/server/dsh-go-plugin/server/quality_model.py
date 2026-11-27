# 回复质量模型：用"用户是否接话"训练一个判断"这次回复好不好"的小模型
import json
import os
import sqlite3
import threading
import time

import numpy as np

DB_PATH = "/root/workspace/digital-human/data/trajectory.db"
SOURCES = ["dsh-account", "local", "deepseek", "other"]
EMOTIONS = ["neutral", "happy", "sad", "surprised", "angry", "shy"]
MIN_SAMPLES = 12


def onehot(value, labels):
    vec = [0.0] * len(labels)
    if value in labels:
        vec[labels.index(value)] = 1.0
    return vec


def has_emoji(text):
    for ch in str(text or ""):
        if ord(ch) > 0x1F000:
            return 1.0
    return 0.0


def encode(sample):
    reply = str(sample.get("reply") or "")
    user = str(sample.get("user") or "")
    latency = float(sample.get("latency_ms") or 0)
    hour = time.localtime(float(sample.get("ts") or time.time())).tm_hour
    vec = [
        min(len(reply), 160) / 160.0,
        min(len(user), 80) / 80.0,
        min(latency, 4000.0) / 4000.0,
        1.0 if ("？" in reply or "?" in reply) else 0.0,
        1.0 if ("！" in reply or "!" in reply) else 0.0,
        has_emoji(reply),
        float(np.sin(2 * np.pi * hour / 24)),
        float(np.cos(2 * np.pi * hour / 24)),
    ]
    vec += onehot(str(sample.get("source") or "other"), SOURCES)
    vec += onehot(str(sample.get("emotion") or "neutral"), EMOTIONS)
    return np.asarray(vec, dtype=np.float32)


class QualityModel:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.weights = None
        self.meta = {}
        self.load()

    def dataset(self):
        rows = []
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5)
            rows = conn.execute(
                "SELECT ts,source,user_text,reply,emotion,latency_ms,engaged "
                "FROM inference_samples WHERE engaged IS NOT NULL"
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        if not rows:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        X = np.stack([encode({"ts": ts, "source": source, "user": user, "reply": reply,
                              "emotion": emotion, "latency_ms": latency}) for ts, source, user, reply, emotion, latency, _ in rows])
        y = np.asarray([1.0 if engaged else 0.0 for *_, engaged in rows], dtype=np.float32)
        return X, y

    def train(self, epochs=500, lr=0.3, l2=1e-3):
        X, y = self.dataset()
        if X.shape[0] < MIN_SAMPLES:
            return {"ok": False, "reason": "样本不足（至少 %d 条有回应的样本）" % MIN_SAMPLES, "samples": int(X.shape[0])}
        rng = np.random.default_rng(5)
        w = np.zeros(X.shape[1], dtype=np.float32)
        b = 0.0
        n = X.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, 16):
                idx = order[start:start + 16]
                xb, yb = X[idx], y[idx]
                logits = xb @ w + b
                probs = 1.0 / (1.0 + np.exp(-logits))
                grad = (probs - yb) / len(yb)
                w -= lr * (xb.T @ grad + l2 * w)
                b -= lr * grad.sum()
        probs = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        accuracy = float(((probs >= 0.5) == (y >= 0.5)).mean())
        with self.lock:
            self.weights = {"w": w, "b": np.float32(b)}
            self.meta = {
                "samples": int(n),
                "engaged_rate": round(float(y.mean()), 3),
                "accuracy": round(accuracy, 3),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": int(self.meta.get("version", 0)) + 1,
            }
            self._save()
        return {"ok": True, **self.meta}

    def _save(self):
        try:
            tmp = self.path + ".tmp"
            np.savez(tmp, **self.weights)
            os.replace(tmp + ".npz", self.path + ".npz")
            with open(self.path + ".tmp.json", "w", encoding="utf-8") as handle:
                json.dump(self.meta, handle, ensure_ascii=False)
            os.replace(self.path + ".tmp.json", self.path + ".json")
        except OSError:
            pass

    def load(self):
        try:
            data = np.load(self.path + ".npz")
            self.weights = {key: data[key] for key in data.files}
            with open(self.path + ".json", "r", encoding="utf-8") as handle:
                self.meta = json.load(handle)
        except (OSError, ValueError):
            self.weights = None
            self.meta = {}

    def score(self, sample):
        with self.lock:
            if not self.weights:
                return None
            x = encode(sample)
            logit = float(x @ self.weights["w"] + self.weights["b"])
            return 1.0 / (1.0 + np.exp(-logit))

    def status(self):
        X, _ = self.dataset()
        return {"loaded": self.weights is not None, "labeled_samples": int(X.shape[0]), **self.meta}
