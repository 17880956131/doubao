# 行为效果模型：学「什么时候该行动」而不是写死阈值
# 输入：行为类型、闲置时长、当前表情、预测表情/置信度、心情
# 输出：你大概率会回应的概率
import json
import os
import shutil
import threading
import time

import numpy as np

ACTIONS = ["none", "quiet", "gesture", "joke", "care", "greet"]
EXPRESSIONS = ["neutral", "happy", "sad", "surprised", "angry", "shy"]
MOODS = ["平静", "开心", "难过", "困", "无聊", "黏人", "兴奋"]
DIM = len(ACTIONS) + 1 + len(EXPRESSIONS) * 2 + 1 + len(MOODS)
MIN_SAMPLES = 24
RETRAIN_GAP = 15


def onehot(value, labels):
    vec = [0.0] * len(labels)
    if value in labels:
        vec[labels.index(value)] = 1.0
    return vec


def encode(entry):
    idle = entry.get("idle") or 0
    try:
        idle = float(idle)
    except (TypeError, ValueError):
        idle = 0.0
    vec = []
    vec += onehot(entry.get("action") or "quiet", ACTIONS)
    vec.append(min(idle, 1800.0) / 1800.0)
    vec += onehot(entry.get("expression") or "neutral", EXPRESSIONS)
    vec += onehot(entry.get("predict_expression") or "neutral", EXPRESSIONS)
    confidence = entry.get("confidence") or 0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    vec.append(max(0.0, min(1.0, confidence)))
    vec += onehot(entry.get("mood") or "平静", MOODS)
    return np.asarray(vec, dtype=np.float32)


class BehaviorModel:
    def __init__(self, log_path, model_path):
        self.log_path = log_path
        self.model_path = model_path
        self.lock = threading.Lock()
        self.weights = None
        self.meta = {}
        self.last_count = 0
        self.load()

    def dataset(self):
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if "engaged" in entry and entry.get("action"):
                        entries.append(entry)
        except OSError:
            return np.zeros((0, DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
        if not entries:
            return np.zeros((0, DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
        X = np.stack([encode(e) for e in entries])
        y = np.asarray([1.0 if e.get("engaged") else 0.0 for e in entries], dtype=np.float32)
        return X, y

    def train(self, epochs=400, lr=0.35, l2=1e-3):
        X, y = self.dataset()
        if X.shape[0] < MIN_SAMPLES:
            return {"ok": False, "reason": "反馈样本太少（至少 %d 条）" % MIN_SAMPLES, "samples": int(X.shape[0])}
        rng = np.random.default_rng(11)
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
        preds = (1.0 / (1.0 + np.exp(-(X @ w + b))) >= 0.5).astype(np.float32)
        accuracy = float((preds == y).mean())
        with self.lock:
            self.weights = {"w": w, "b": np.float32(b)}
            self.last_count = int(n)
            previous = self.meta.get("accuracy")
            if self.weights is not None and isinstance(previous, (int, float)) and accuracy < previous - 0.05:
                return {"ok": False, "reason": "新模型比旧模型差，保留旧权重",
                        "candidate_accuracy": round(accuracy, 3), "previous_accuracy": previous}
            self.meta = {
                "samples": int(n),
                "positive_rate": round(float(y.mean()), 3),
                "accuracy": round(accuracy, 3),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": int(self.meta.get("version", 0)) + 1,
            }
            self._save()
        return {"ok": True, **self.meta}

    def _save(self):
        if not self.weights:
            return
        try:
            np.savez(self.model_path, **self.weights)
            with open(self.model_path + ".json", "w", encoding="utf-8") as handle:
                json.dump(self.meta, handle, ensure_ascii=False)
        except OSError:
            pass

    def load(self):
        try:
            data = np.load(self.model_path + ".npz")
            self.weights = {key: data[key] for key in data.files}
        except (OSError, ValueError):
            self.weights = None
        try:
            with open(self.model_path + ".json", "r", encoding="utf-8") as handle:
                self.meta = json.load(handle)
            self.last_count = int(self.meta.get("samples", 0))
        except (OSError, ValueError):
            self.meta = {}

    def predict(self, entry):
        with self.lock:
            if not self.weights:
                return None
            x = encode(entry)
            logit = float(x @ self.weights["w"] + self.weights["b"])
            return 1.0 / (1.0 + np.exp(-logit))

    def status(self):
        X, _ = self.dataset()
        return {"loaded": self.weights is not None, "feedback_samples": int(X.shape[0]),
                "last_trained_count": self.last_count, **self.meta}

    def should_retrain(self):
        X, _ = self.dataset()
        count = int(X.shape[0])
        if count < MIN_SAMPLES:
            return False
        return self.weights is None or count - self.last_count >= RETRAIN_GAP
