# 离线参数优化 pass：固定结构，只评估可逆参数，不自动改线上
import json
import math
import os
import tempfile
import time

import numpy as np

from layer_models import LayerModelManager
from policy_model import EXPRESSIONS, GESTURES, encode
from classifier import ThreatClassifier

REPORT = "data/param_report.json"


def load_layers():
    return LayerModelManager("data/trajectory.db", "data/layer_models")


def holdout_rows(manager, ratio=0.2, minimum=20):
    rows = sorted(manager._rows(), key=lambda row: row[0])
    hold = max(minimum, int(len(rows) * ratio))
    return rows[-hold:]


def predict_with_weights(manager, X, power, sample_power):
    buckets = [b for b in manager.models
               if not (manager.meta.get(b, {}).get("semantic") or {}).get("stale")
               and not manager.meta.get(b, {}).get("rejected")]
    if not buckets:
        return None
    probs = None
    total = 0.0
    for bucket in buckets:
        meta = manager.meta.get(bucket, {})
        weight = max(float(meta.get("adapted_weight") or meta.get("weight") or 0.2), 1e-6) ** power
        weight *= max(float(meta.get("samples") or 1), 1) ** sample_power
        model = manager.models[bucket]
        logits = X @ model["w_expr"] + model["b_expr"]
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        probs = p * weight if probs is None else probs + p * weight
        total += weight
    return (probs / total).argmax(axis=1)


def main():
    manager = load_layers()
    rows = holdout_rows(manager)
    X, y_expr, y_gest, _ = manager._features(rows)
    baseline = predict_with_weights(manager, X, 1.0, 0.5)
    baseline_acc = float((baseline == y_expr).mean()) if baseline is not None else 0.0
    results = []
    for power in (0.0, 0.5, 1.0, 1.5):
        for sample_power in (0.0, 0.25, 0.5, 0.75):
            pred = predict_with_weights(manager, X, power, sample_power)
            if pred is None:
                continue
            acc = float((pred == y_expr).mean())
            results.append({"power": power, "sample_power": sample_power, "accuracy": round(acc, 3)})
    results.sort(key=lambda item: -item["accuracy"])
    best = results[0] if results else None

    # 分类器超参（仅报告，模型写到临时目录，不动线上）
    classifier_rows = []
    try:
        temp_dir = tempfile.mkdtemp(prefix="param-clf-")
        clf = ThreatClassifier(os.path.join(temp_dir, "model"), "data/feedback.jsonl")
        samples = clf.dataset()
        Xc = np.stack([clf.encode(text) for text, _, _ in samples])
        yc = np.asarray([label for _, label, _ in samples], dtype=np.float32)
        for epochs in (150, 350, 700):
            for lr in (0.25, 0.5):
                w = np.zeros(Xc.shape[1], dtype=np.float32)
                b = 0.0
                rng = np.random.default_rng(3)
                for _ in range(epochs):
                    order = rng.permutation(len(yc))
                    for start in range(0, len(yc), 16):
                        idx = order[start:start + 16]
                        xb, yb = Xc[idx], yc[idx]
                        logits = xb @ w + b
                        probs = 1.0 / (1.0 + np.exp(-logits))
                        grad = (probs - yb) / len(yb)
                        w -= lr * (xb.T @ grad + 1e-3 * w)
                        b -= lr * float(grad.sum())
                probs = 1.0 / (1.0 + np.exp(-(Xc @ w + b)))
                acc = float(((probs >= 0.5) == (yc >= 0.5)).mean())
                classifier_rows.append({"epochs": epochs, "lr": lr, "accuracy": round(acc, 3),
                                        "samples": int(len(yc))})
        results_clf = classifier_rows
    except Exception as error:  # noqa: BLE001
        results_clf = [{"error": str(error)[:120]}]

    report = {
        "ts": time.time(),
        "holdout": len(rows),
        "baseline": {"power": 1.0, "sample_power": 0.5, "accuracy": round(baseline_acc, 3)},
        "best": best,
        "top": results[:6],
        "classifier": results_clf,
        "note": "只评估不应用；参数改动需要人工确认（避免小样本过拟合）",
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
