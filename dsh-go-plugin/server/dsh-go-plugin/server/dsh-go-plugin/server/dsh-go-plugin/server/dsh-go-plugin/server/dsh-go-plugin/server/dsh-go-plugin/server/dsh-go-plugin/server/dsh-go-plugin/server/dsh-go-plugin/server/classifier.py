# 威胁语义分类器：用反馈标签训练，替代关键词硬匹配（关键词退为先验）
import hashlib
import json
import os
import threading
import time

import numpy as np

FEATURE_DIM = 768
MIN_SAMPLES = 8


def _hash_features(text, dim=256):
    vec = np.zeros(dim, dtype=np.float32)
    text = str(text or "")
    for i in range(len(text) - 1):
        token = text[i:i + 2]
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[index] += 1.0
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm


class ThreatClassifier:
    def __init__(self, model_path, feedback_path):
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.lock = threading.Lock()
        self.weights = None
        self.meta = {}
        self._load()

    def _load(self):
        try:
            data = np.load(self.model_path + ".npz")
            self.weights = {"w": data["w"], "b": float(data["b"])}
            with open(self.model_path + ".json", "r", encoding="utf-8") as handle:
                self.meta = json.load(handle)
        except (OSError, ValueError, KeyError):
            self.weights = None
            self.meta = {}

    def encode(self, text):
        try:
            from trajectory_store import embed
            vector = embed(text)
            if vector and len(vector) == FEATURE_DIM * 4:
                return np.frombuffer(vector, dtype=np.float32).copy()
        except Exception:  # noqa: BLE001
            pass
        return _hash_features(text)

    def dataset(self):
        samples = []
        try:
            with open(self.feedback_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    text = str(entry.get("text") or "")
                    verdict = entry.get("verdict")
                    if not text:
                        continue
                    if verdict == "overreact":
                        label = 0.0
                    elif verdict in ("correct", "missed"):
                        label = 1.0
                    else:
                        continue
                    samples.append((text, label, bool(entry.get("seed"))))
        except OSError:
            pass
        return samples

    def train(self, epochs=350, lr=0.5, l2=1e-3):
        samples = self.dataset()
        if len(samples) < MIN_SAMPLES:
            return {"ok": False, "reason": "反馈样本不足（至少 %d 条）" % MIN_SAMPLES,
                    "samples": len(samples)}
        X = np.stack([self.encode(text) for text, _, _ in samples])
        y = np.asarray([label for _, label, _ in samples], dtype=np.float32)
        rng = np.random.default_rng(3)
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
                b -= lr * float(grad.sum())
        probs = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        accuracy = float(((probs >= 0.5) == (y >= 0.5)).mean())
        with self.lock:
            self.weights = {"w": w, "b": b}
            self.meta = {
                "hyper": {"epochs": epochs, "lr": lr, "l2": l2},
                "samples": int(n),
                "seed_samples": int(sum(1 for _, _, seed in samples if seed)),
                "positive_rate": round(float(y.mean()), 3),
                "accuracy": round(accuracy, 3),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": int(self.meta.get("version", 0)) + 1,
            }
            self._save()
        return {"ok": True, **self.meta}

    def _save(self):
        try:
            tmp = self.model_path + ".tmp"
            np.savez(tmp, **self.weights)
            os.replace(tmp + ".npz", self.model_path + ".npz")
            with open(self.model_path + ".tmp.json", "w", encoding="utf-8") as handle:
                json.dump(self.meta, handle, ensure_ascii=False)
            os.replace(self.model_path + ".tmp.json", self.model_path + ".json")
        except OSError:
            pass

    def predict(self, text):
        with self.lock:
            if not self.weights:
                return None
            x = self.encode(text)
            prob = 1.0 / (1.0 + np.exp(-(float(x @ self.weights["w"]) + self.weights["b"])))
            return round(float(prob), 3)

    def status(self):
        return {"loaded": self.weights is not None, **self.meta,
                "feedback_samples": len(self.dataset())}
