# 知识置信层：从"猜测 vs 事实"的对照里学"我什么时候猜得准"
import json
import math
import os
import time

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RECORDS = os.path.join(DATA, "hallucinations.jsonl")
MODEL = os.path.join(DATA, "knowledge_confidence.npz")


def load_records():
    items = []
    try:
        with open(RECORDS, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return items


def features(record):
    guess = str(record.get("guess") or "")
    fact = str(record.get("fact") or "")
    cosine = float(record.get("cosine") or 0.0)
    return [
        1.0,
        cosine,
        min(1.0, len(guess) / 40.0),
        min(1.0, len(fact) / 120.0),
        1.0 if record.get("verdict") == "部分一致" else 0.0,
        min(1.0, abs(len(guess) - len(fact)) / 120.0),
    ]


def train(min_samples=8):
    records = [item for item in load_records() if item.get("cosine") is not None]
    if len(records) < min_samples:
        return {"ok": False, "reason": "样本不足", "samples": len(records), "need": min_samples}
    X = np.asarray([features(item) for item in records], dtype=np.float64)
    y = np.asarray([1.0 if float(item.get("reward") or 0) > 0 else 0.0 for item in records], dtype=np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    for _ in range(400):
        logits = X @ w
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        grad = X.T @ (probs - y) / len(y) + 0.01 * w
        w -= 0.1 * grad
    pred = (1.0 / (1.0 + np.exp(-np.clip(X @ w, -20, 20))) >= 0.5).astype(np.float64)
    accuracy = float((pred == y).mean())
    np.savez(MODEL, weights=w, trained_at=time.time(), samples=len(records), accuracy=accuracy)
    return {"ok": True, "samples": len(records), "accuracy": round(accuracy, 3),
            "weights": [round(float(v), 3) for v in w]}


def status():
    try:
        with np.load(MODEL) as data:
            return {"ok": True, "trained": True, "samples": int(data["samples"]),
                    "accuracy": round(float(data["accuracy"]), 3),
                    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(data["trained_at"])))}
    except (OSError, KeyError, ValueError):
        return {"ok": True, "trained": False, "records": len(load_records()),
                "note": "样本够了调用 /api/knowledge/train 训练"}


def confidence_for(record):
    try:
        with np.load(MODEL) as data:
            w = np.asarray(data["weights"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None
    x = np.asarray(features(record), dtype=np.float64)
    p = 1.0 / (1.0 + math.exp(-float(np.clip(x @ w, -20, 20))))
    return round(p, 3)
