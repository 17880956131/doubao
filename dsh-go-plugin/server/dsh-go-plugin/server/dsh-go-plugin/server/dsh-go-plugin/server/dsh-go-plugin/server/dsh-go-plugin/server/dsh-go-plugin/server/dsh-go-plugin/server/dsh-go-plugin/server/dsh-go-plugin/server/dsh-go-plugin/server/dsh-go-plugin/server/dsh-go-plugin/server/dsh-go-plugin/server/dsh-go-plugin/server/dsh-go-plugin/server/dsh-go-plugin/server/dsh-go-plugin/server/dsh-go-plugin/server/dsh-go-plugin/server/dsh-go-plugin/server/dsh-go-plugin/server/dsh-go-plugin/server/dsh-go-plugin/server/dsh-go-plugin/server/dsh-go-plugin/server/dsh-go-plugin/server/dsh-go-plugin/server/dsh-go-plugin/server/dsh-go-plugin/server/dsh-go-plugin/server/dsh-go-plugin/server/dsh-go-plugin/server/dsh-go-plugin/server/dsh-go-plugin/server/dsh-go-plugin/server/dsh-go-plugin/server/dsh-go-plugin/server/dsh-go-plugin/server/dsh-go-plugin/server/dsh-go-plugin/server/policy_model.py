# 本地行为策略模型：训练 + 推理闭环
# 数据来自活动轨迹库，用 numpy 训练小型 softmax 网络（无需大模型、无需联网）
import json
import os
import shutil
import sqlite3
import threading
import time

import hashlib
import inspect

import numpy as np

EXPRESSIONS = ["neutral", "happy", "sad", "surprised", "angry", "shy"]
GESTURES = ["", "nod", "tilt", "sway", "look"]
MOODS = ["平静", "开心", "难过", "困", "无聊", "黏人", "兴奋"]
FEATURE_DIM = len(EXPRESSIONS) + len(GESTURES) + 1 + 4 + len(MOODS)
MIN_SAMPLES = 40
RETRAIN_GAP = 15


def onehot(value, labels):
    vec = [0.0] * len(labels)
    if value in labels:
        vec[labels.index(value)] = 1.0
    return vec


def encode(expression, gesture, speaking, params, mood):
    params = params or {}
    vec = []
    vec += onehot(expression or "neutral", EXPRESSIONS)
    vec += onehot(gesture or "", GESTURES)
    vec.append(1.0 if speaking else 0.0)
    for key in ("PARAM_MOUTH_OPEN_Y", "PARAM_ANGLE_X", "PARAM_ANGLE_Y", "PARAM_BODY_ANGLE_Z"):
        value = params.get(key)
        vec.append(float(value) / 30.0 if isinstance(value, (int, float)) else 0.0)
    vec += onehot(mood or "平静", MOODS)
    return np.asarray(vec, dtype=np.float32)


class PolicyModel:
    def __init__(self, db_path, model_path):
        self.db_path = db_path
        self.model_path = model_path
        self.lock = threading.Lock()
        self.params = None
        self.meta = {}
        self.last_count = 0
        self.load()

    # ---------- 数据 ----------
    def dataset(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            rows = conn.execute(
                "SELECT expression, gesture, speaking, mood, params, next_expression, next_gesture, ts "
                "FROM trajectories WHERE next_expression IS NOT NULL AND next_expression != ''"
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32), np.zeros(0, int), np.zeros(0, int)

        features, labels_expr, labels_gest, times = [], [], [], []
        for expression, gesture, speaking, mood, params_json, next_expr, next_gest, ts in rows:
            try:
                params = json.loads(params_json) if params_json else {}
            except ValueError:
                params = {}
            if not isinstance(params, dict):
                params = {}
            features.append(encode(expression, gesture, bool(speaking), params, mood))
            labels_expr.append(EXPRESSIONS.index(next_expr) if next_expr in EXPRESSIONS else 0)
            labels_gest.append(GESTURES.index(next_gest) if next_gest in GESTURES else 0)
            times.append(float(ts or 0))
        if not features:
            return (np.zeros((0, FEATURE_DIM), dtype=np.float32), np.zeros(0, int),
                    np.zeros(0, int), np.zeros(0, dtype=np.float64))
        return (np.stack(features), np.asarray(labels_expr, dtype=np.int64),
                np.asarray(labels_gest, dtype=np.int64), np.asarray(times, dtype=np.float64))

    # ---------- 训练 ----------
    def _train_head(self, X, y, classes, sample_weight=None, epochs=260, lr=0.25, l2=1e-3):
        rng = np.random.default_rng(7)
        weights = np.zeros((X.shape[1], classes), dtype=np.float32)
        bias = np.zeros(classes, dtype=np.float32)
        n = X.shape[0]
        for epoch in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, 16):
                idx = order[start:start + 16]
                xb, yb = X[idx], y[idx]
                logits = xb @ weights + bias
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                probs[np.arange(len(yb)), yb] -= 1.0
                if sample_weight is not None:
                    wb = sample_weight[idx][:, None]
                    grad = (xb.T @ (probs * wb)) / max(float(wb.sum()), 1e-6)
                    weights -= lr * (grad + l2 * weights)
                    bias -= lr * (probs * wb).sum(axis=0) / max(float(wb.sum()), 1e-6)
                else:
                    weights -= lr * (xb.T @ probs / len(yb) + l2 * weights)
                    bias -= lr * probs.mean(axis=0)
        logits = X @ weights + bias
        pred = logits.argmax(axis=1)
        accuracy = float((pred == y).mean()) if n else 0.0
        return weights, bias, accuracy

    def train(self, weight_fn=None):
        X, y_expr, y_gest, times = self.dataset()
        if X.shape[0] < 8:
            return {"ok": False, "reason": "样本太少", "samples": int(X.shape[0])}
        sample_weight = None
        if weight_fn is not None:
            try:
                sample_weight = np.asarray([float(weight_fn(ts)) for ts in times], dtype=np.float32)
                if sample_weight.sum() <= 0:
                    sample_weight = None
            except Exception:  # noqa: BLE001
                sample_weight = None
        with self.lock:
            w_expr, b_expr, acc_expr = self._train_head(X, y_expr, len(EXPRESSIONS), sample_weight)
            w_gest, b_gest, acc_gest = self._train_head(X, y_gest, len(GESTURES), sample_weight)
            self.params = {
                "w_expr": w_expr, "b_expr": b_expr,
                "w_gest": w_gest, "b_gest": b_gest,
            }
            self.last_count = int(X.shape[0])
            previous = self.meta.get("accuracy_expression")
            if self.params is not None and isinstance(previous, (int, float)) and acc_expr < previous - 0.05:
                return {"ok": False, "reason": "新模型比旧模型差，保留旧权重",
                        "candidate_accuracy": round(acc_expr, 3), "previous_accuracy": previous}
            self.meta = {
                "samples": int(X.shape[0]),
                "accuracy_expression": round(acc_expr, 3),
                "accuracy_gesture": round(acc_gest, 3),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": int(self.meta.get("version", 0)) + 1,
            }
            self._save()
        return {"ok": True, **self.meta}

    def _save(self):
        if not self.params:
            return
        try:
            tmp = self.model_path + ".tmp"
            np.savez(tmp, **self.params)
            os.replace(tmp + ".npz", self.model_path + ".npz")
            with open(self.model_path + ".tmp.json", "w", encoding="utf-8") as handle:
                json.dump(self.meta, handle, ensure_ascii=False)
            os.replace(self.model_path + ".tmp.json", self.model_path + ".json")
            self._snapshot()
        except OSError:
            pass

    def _snapshot(self):
        """保留最近几个版本，训练覆盖时旧权重可回滚。"""
        try:
            folder = os.path.join(os.path.dirname(self.model_path), "models")
            os.makedirs(folder, exist_ok=True)
            version = int(self.meta.get("version", 0))
            base = os.path.basename(self.model_path)
            shutil.copyfile(self.model_path + ".npz", os.path.join(folder, f"{base}-v{version}.npz"))
            with open(os.path.join(folder, f"{base}-v{version}.json"), "w", encoding="utf-8") as handle:
                json.dump(self.meta, handle, ensure_ascii=False)
            files = sorted(f for f in os.listdir(folder) if f.startswith(base + "-v") and f.endswith(".npz"))
            for old in files[:-5]:
                try:
                    os.remove(os.path.join(folder, old))
                    os.remove(os.path.join(folder, old[:-4] + ".json"))
                except OSError:
                    pass
        except OSError:
            pass

    def load(self):
        try:
            data = np.load(self.model_path + ".npz")
            self.params = {key: data[key] for key in data.files}
        except (OSError, ValueError):
            self.params = None
        try:
            with open(self.model_path + ".json", "r", encoding="utf-8") as handle:
                self.meta = json.load(handle)
            self.last_count = int(self.meta.get("samples", 0))
        except (OSError, ValueError):
            self.meta = {}

    # ---------- 推理 ----------
    def predict(self, expression, gesture, speaking, params, mood):
        with self.lock:
            if not self.params:
                return None
            x = encode(expression, gesture, speaking, params, mood).reshape(1, -1)
            logits_expr = x @ self.params["w_expr"] + self.params["b_expr"]
            logits_gest = x @ self.params["w_gest"] + self.params["b_gest"]
            probs_expr = np.exp(logits_expr - logits_expr.max())
            probs_expr /= probs_expr.sum()
            probs_gest = np.exp(logits_gest - logits_gest.max())
            probs_gest /= probs_gest.sum()
            expr_idx = int(probs_expr.argmax())
            gest_idx = int(probs_gest.argmax())
            return {
                "expression": EXPRESSIONS[expr_idx],
                "gesture": GESTURES[gest_idx],
                "confidence": round(float(probs_expr[0, expr_idx]), 3),
                "gesture_confidence": round(float(probs_gest[0, gest_idx]), 3),
                "source": "local-policy-model",
                **self.meta,
            }

    def status(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            labeled = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE next_expression IS NOT NULL AND next_expression != ''"
            ).fetchone()[0]
            conn.close()
        except sqlite3.Error:
            labeled = 0
        return {
            "loaded": self.params is not None,
            "labeled_samples": int(labeled),
            "last_trained_count": self.last_count,
            **self.meta,
        }

    def should_retrain(self):
        labeled = self.status()["labeled_samples"]
        if labeled < MIN_SAMPLES:
            return False
        return self.params is None or labeled - self.last_count >= RETRAIN_GAP

# 特征语义契约：代码改了 encode()，指纹就会变，用来识别"旧权重是旧语义"
ENCODE_VERSION = 2
FEATURE_SPEC_TEXT = "expression+gesture+speaking+mood+params(angles/30,mouth/1,eyes)"


def encode_fingerprint():
    try:
        source = inspect.getsource(encode)
    except (OSError, TypeError):
        source = FEATURE_SPEC_TEXT
    return hashlib.sha256((str(ENCODE_VERSION) + "|" + source).encode("utf-8")).hexdigest()[:16]


ENCODE_FINGERPRINT = encode_fingerprint()
FEATURE_SPEC = {"version": ENCODE_VERSION, "dim": FEATURE_DIM,
                "fingerprint": ENCODE_FINGERPRINT, "text": FEATURE_SPEC_TEXT}
