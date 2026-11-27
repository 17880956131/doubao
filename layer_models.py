# 分层权重：每个时代（地层）一个自己的小模型，加载全部层一起推理，只增量更新变化的层
import json
import math
import os
import sqlite3
import struct
import threading
import time

import numpy as np

import model_store
from policy_model import (ENCODE_FINGERPRINT, EXPRESSIONS, FEATURE_DIM, FEATURE_SPEC,
                          GESTURES, encode)
from trajectory_store import embed

MIN_LAYER_SAMPLES = 8
_TUNABLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tunables.json")
_tunables_cache = {"mtime": 0.0, "values": {}}
_TRANSMISSION_CACHE = {"mtime": 0.0, "value": 0.15}


def _layer_transmission():
    try:
        stat = os.stat(_TUNABLES_PATH)
        if stat.st_mtime != _TRANSMISSION_CACHE["mtime"]:
            import json as _json
            with open(_TUNABLES_PATH, "r", encoding="utf-8") as handle:
                data = _json.load(handle)
            _TRANSMISSION_CACHE["value"] = float((data.get("layer_transmission") or {}).get("value", 0.15))
            _TRANSMISSION_CACHE["mtime"] = stat.st_mtime
    except (OSError, ValueError):
        pass
    return float(_TRANSMISSION_CACHE["value"])


def _ensemble_params():
    try:
        stat = os.stat(_TUNABLES_PATH)
        if stat.st_mtime != _tunables_cache["mtime"]:
            import json as _json
            with open(_TUNABLES_PATH, "r", encoding="utf-8") as handle:
                data = _json.load(handle)
            _tunables_cache["values"] = {
                "weight_power": float((data.get("ensemble_weight_power") or {}).get("value", 1.0)),
                "sample_power": float((data.get("ensemble_sample_power") or {}).get("value", 0.5)),
            }
            _tunables_cache["mtime"] = stat.st_mtime
    except (OSError, ValueError):
        pass
    return _tunables_cache["values"] or {"weight_power": 1.0, "sample_power": 0.5}


def _bucket(ts, granularity):
    if granularity == "hour":
        return math.floor(ts / 3600) * 3600
    if granularity == "day":
        return time.mktime(time.localtime(ts)[:3] + (0, 0, 0, 0, 0, -1))
    return math.floor(ts / 600) * 600


def _label(ts, granularity):
    if granularity == "hour":
        return time.strftime("%m-%d %H时", time.localtime(ts))
    if granularity == "day":
        return time.strftime("%Y-%m-%d", time.localtime(ts))
    return time.strftime("%H:%M", time.localtime(ts))


def _softmax_train(X, y, classes, epochs=200, lr=0.3, l2=1e-3, sample_weight=None):
    rng = np.random.default_rng(3)
    w = np.zeros((X.shape[1], classes), dtype=np.float32)
    b = np.zeros(classes, dtype=np.float32)
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, 16):
            idx = order[start:start + 16]
            xb, yb = X[idx], y[idx]
            logits = xb @ w + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            if sample_weight is not None:
                wb = sample_weight[idx][:, None]
                denom = max(float(wb.sum()), 1e-6)
                w -= lr * (xb.T @ (probs * wb) / denom + l2 * w)
                b -= lr * (probs * wb).sum(axis=0) / denom
            else:
                w -= lr * (xb.T @ probs / len(yb) + l2 * w)
                b -= lr * probs.mean(axis=0)
    pred = (X @ w + b).argmax(axis=1)
    return w, b, float((pred == y).mean()) if n else 0.0


class LayerModelManager:
    def __init__(self, trajectory_db, layers_dir):
        self.trajectory_db = trajectory_db
        self.layers_dir = layers_dir
        os.makedirs(layers_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.models = {}
        self.meta = {}
        self.corrections = []
        self.corrections_path = os.path.join(layers_dir, "corrections.jsonl")
        self.load()
        self._load_corrections()

    # ---------- 数据 ----------
    @staticmethod
    def inference_rows(since=0.0):
        """API 推理沉淀下来的"专家示范"：状态 → 模型给出的表情/动作。"""
        try:
            conn = sqlite3.connect("/root/workspace/digital-human/data/trajectory.db", timeout=5)
            rows = conn.execute(
                "SELECT ts, state_json, emotion, gesture, quality FROM inference_samples WHERE ts > ? AND emotion IS NOT NULL",
                (since,),
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            return []
        result = []
        for ts, state_json, emotion, gesture, quality in rows:
            try:
                state = json.loads(state_json) if state_json else {}
            except ValueError:
                state = {}
            params = state.get("params") if isinstance(state.get("params"), dict) else {}
            result.append((ts, state.get("expression") or "neutral", state.get("gesture") or "",
                           False, state.get("mood") or "平静",
                           json.dumps(params, ensure_ascii=False), emotion, gesture or "",
                           quality if isinstance(quality, (int, float)) else 1.0))
        return result

    def _rows(self):
        try:
            conn = sqlite3.connect(self.trajectory_db, timeout=5)
            rows = conn.execute(
                "SELECT ts, expression, gesture, speaking, mood, params, next_expression, next_gesture, source "
                "FROM trajectories WHERE next_expression IS NOT NULL AND next_expression != '' ORDER BY ts"
            ).fetchall()
            conn.close()
            return rows + self.inference_rows(0)
        except sqlite3.Error:
            return self.inference_rows(0)

    @staticmethod
    def _features(rows):
        X, y_expr, y_gest, weights = [], [], [], []
        for row in rows:
            ts, expression, gesture, speaking, mood, params_json, next_expr, next_gest = row[:8]
            quality = None
            source = "stage"
            if len(row) > 8:
                if isinstance(row[8], str):
                    source = row[8]
                else:
                    quality = row[8]
                    source = row[9] if len(row) > 9 and isinstance(row[9], str) else "stage"
            try:
                params = json.loads(params_json) if params_json else {}
            except ValueError:
                params = {}
            X.append(encode(expression, gesture, bool(speaking), params, mood))
            y_expr.append(EXPRESSIONS.index(next_expr) if next_expr in EXPRESSIONS else 0)
            y_gest.append(GESTURES.index(next_gest) if next_gest in GESTURES else 0)
            weight = max(0.15, float(quality)) if isinstance(quality, (int, float)) else 1.0
            if source == "shadow":
                weight *= 0.5
            weights.append(weight)
        if not X:
            return (np.zeros((0, FEATURE_DIM), dtype=np.float32),
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float32))
        return (np.stack(X), np.asarray(y_expr, dtype=np.int64),
                np.asarray(y_gest, dtype=np.int64), np.asarray(weights, dtype=np.float32))

    # ---------- 按需增量更新 ----------
    def update(self, granularity="10min", weight_fn=None):
        rows = self._rows()
        grouped = {}
        for row in rows:
            bucket = _bucket(row[0], granularity)
            grouped.setdefault(bucket, []).append(row)
        updated, skipped = [], []
        now = time.time()
        bucket_seconds = 600 if granularity == "10min" else 3600 if granularity == "hour" else 86400
        for bucket, bucket_rows in sorted(grouped.items()):
            # 已成层的永久冻结，绝不重训
            if bucket in self.models:
                continue
            if bucket + bucket_seconds > now:
                skipped.append({"bucket": _label(bucket, granularity), "reason": "时代尚未结束，等它定型"})
                continue
            if len(bucket_rows) < MIN_LAYER_SAMPLES:
                skipped.append({"bucket": _label(bucket, granularity), "reason": "样本不足", "samples": len(bucket_rows)})
                continue
            X, y_expr, y_gest, sample_weight = self._features(bucket_rows)
            w_expr, b_expr, acc_expr = _softmax_train(X, y_expr, len(EXPRESSIONS), sample_weight=sample_weight)
            w_gest, b_gest, acc_gest = _softmax_train(X, y_gest, len(GESTURES), sample_weight=sample_weight)
            weight = weight_fn(bucket) if weight_fn else 0.2
            record = {
                "bucket": bucket,
                "label": _label(bucket, granularity),
                "granularity": granularity,
                "range": [bucket, bucket + (3600 if granularity == "hour" else 600 if granularity == "10min" else 86400)],
                "feature_centroid": np.round(X.mean(axis=0), 4).tolist(),
                "kind": "era",
                "frozen": True,
                "rows": len(bucket_rows),
                "samples": int(X.shape[0]),
                "weight": round(float(weight), 3),
                "accuracy_expression": round(acc_expr, 3),
                "accuracy_gesture": round(acc_gest, 3),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with self.lock:
                self.models[bucket] = {"w_expr": w_expr, "b_expr": b_expr, "w_gest": w_gest, "b_gest": b_gest}
                self.meta[bucket] = record
            self._save_layer(bucket, record)
            updated.append(record)
        return {"granularity": granularity, "updated": updated, "skipped": skipped,
                "total_layers": len(self.models)}

    @staticmethod
    def match_text(expression, gesture):
        return "表情=" + str(expression or "neutral") + "；动作=" + str(gesture or "")

    def add_consolidation_layer(self, rows, weight=0.5, label=None, kind="consolidation", min_samples=MIN_LAYER_SAMPLES):
        """把一批积累的推理数据归一化后训练成一层，叠到星球上。"""
        rows = list(rows)
        if len(rows) < min_samples:
            return None
        X, y_expr, y_gest, sample_weight = self._features(rows)
        if X.shape[0] < min_samples:
            return None
        w_expr, b_expr, acc_expr = _softmax_train(X, y_expr, len(EXPRESSIONS), sample_weight=sample_weight)
        w_gest, b_gest, acc_gest = _softmax_train(X, y_gest, len(GESTURES), sample_weight=sample_weight)
        bucket = int(time.time() * 1000)
        times = [float(row[0]) for row in rows if row and isinstance(row[0], (int, float)) and row[0] > 0]
        centroid = np.round(X.mean(axis=0), 4)
        unit = centroid / (np.linalg.norm(centroid) or 1.0)
        duplicate_of = None
        for existing, meta in self.meta.items():
            other = meta.get("feature_centroid")
            if not other or meta.get("rejected") or meta.get("fossil"):
                continue
            other_vec = np.asarray(other, dtype=np.float64)
            similarity = float(unit @ (other_vec / (np.linalg.norm(other_vec) or 1.0)))
            if similarity >= 0.97:
                duplicate_of = meta.get("label")
                break
        if duplicate_of:
            record = {
                "bucket": bucket,
                "range": [min(times), max(times)] if times else None,
                "feature_centroid": centroid.tolist(),
                "label": label or ("重复层 " + time.strftime("%H:%M:%S")),
                "granularity": "consolidation",
                "kind": "duplicate",
                "frozen": True,
                "rejected": True,
                "duplicate_of": duplicate_of,
                "rows": len(rows),
                "samples": int(X.shape[0]),
                "weight": round(float(weight), 3),
                "accuracy_expression": round(acc_expr, 3),
                "accuracy_gesture": round(acc_gest, 3),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with self.lock:
                self.meta[bucket] = record
            self._save_layer(bucket, record)
            return record
        record = {
            "bucket": bucket,
            "range": [min(times), max(times)] if times else None,
            "feature_centroid": centroid.tolist(),
            "label": label or ("巩固层 " + time.strftime("%H:%M:%S")),
            "granularity": "consolidation",
            "kind": kind,
            "frozen": True,
            "rows": len(rows),
            "samples": int(X.shape[0]),
            "weight": round(float(weight), 3),
            "accuracy_expression": round(acc_expr, 3),
            "accuracy_gesture": round(acc_gest, 3),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self.lock:
            self.models[bucket] = {"w_expr": w_expr, "b_expr": b_expr, "w_gest": w_gest, "b_gest": b_gest}
            self.meta[bucket] = record
        self._save_layer(bucket, record)
        return record

    def add_correction(self, snapshot, next_expression, next_gesture):
        """某层被判定错误时，不修改旧层，而是叠一个修正层覆盖它。"""
        from trajectory_store import TrajectoryStore
        text = TrajectoryStore.snapshot_text(snapshot)
        vector = embed(self.match_text(snapshot.get("expression"), snapshot.get("gesture")))
        entry = {
            "ts": time.time(),
            "text": text,
            "embedding": vector.hex() if vector else None,
            "vector": vector,
            "expression": snapshot.get("expression") or "neutral",
            "gesture": snapshot.get("gesture") or "",
            "next_expression": next_expression if next_expression in EXPRESSIONS else "neutral",
            "next_gesture": next_gesture if next_gesture in GESTURES else "",
        }
        # 记录这个修正层"压过"了哪些时代层（地层学里的不整合面）
        overrides = []
        if self.models:
            x = encode(snapshot.get("expression"), snapshot.get("gesture"),
                       bool(snapshot.get("speaking")), snapshot.get("params") or {},
                       snapshot.get("mood") or "平静").reshape(1, -1)
            for model_bucket, model in self.models.items():
                logits = x @ model["w_expr"] + model["b_expr"]
                predicted = EXPRESSIONS[int(logits.argmax())]
                if predicted != entry["next_expression"]:
                    overrides.append(self.meta.get(model_bucket, {}).get("label", str(model_bucket)))
        entry["overrides"] = overrides
        with self.lock:
            self.corrections.append(entry)
            bucket = int(time.time() * 1000)
            record = {
                "bucket": bucket,
                "label": "修正层 " + time.strftime("%H:%M:%S"),
                "granularity": "correction",
                "kind": "correction",
                "frozen": True,
                "rows": len(self.corrections),
                "samples": len(self.corrections),
                "weight": 1.0,
                "accuracy_expression": None,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overrides": overrides,
            }
            self.meta[bucket] = record
        try:
            serializable = {key: value for key, value in entry.items() if key != "vector"}
            with open(self.corrections_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
            path = os.path.join(self.layers_dir, f"layer-corr-{bucket}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
        except OSError:
            pass
        return record

    def _load_corrections(self):
        try:
            with open(self.corrections_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if entry.get("expression") is not None:
                        vector = self.match_text(entry.get("expression"), entry.get("gesture"))
                        entry["vector"] = embed(vector)
                    elif entry.get("embedding"):
                        entry["vector"] = bytes.fromhex(entry["embedding"])
                    if entry.get("vector"):
                        self.corrections.append(entry)
        except OSError:
            pass

    def _save_layer(self, bucket, record):
        model = self.models.get(bucket)
        if not model:
            return
        record["schema"] = {
            "format": model_store.WEIGHT_FORMAT,
            "schema_version": model_store.SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "labels": {"expressions": list(EXPRESSIONS), "gestures": list(GESTURES)},
            "encode_version": FEATURE_SPEC["version"],
            "encode_fingerprint": ENCODE_FINGERPRINT,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        record["schema_version"] = model_store.SCHEMA_VERSION
        base = os.path.join(self.layers_dir, f"layer-{int(bucket)}")
        try:
            np.savez(base, **model)
            with open(base + ".json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
            model_store.register_layer(
                self.layers_dir, bucket, record, base + ".npz", base + ".json",
                features=record["schema"]["labels"]["expressions"],  # 仅用于记录
                labels={"expressions": list(EXPRESSIONS), "gestures": list(GESTURES),
                        "encode_fingerprint": ENCODE_FINGERPRINT})
        except OSError:
            pass

    def load(self):
        try:
            names = [name for name in os.listdir(self.layers_dir) if name.endswith(".json")]
        except OSError:
            return
        model_store.ensure_manifest(self.layers_dir)
        for name in names:
            path = os.path.join(self.layers_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
                bucket = int(record.get("bucket") or 0)
                data = np.load(path[:-5] + ".npz")
                weights = {key: data[key] for key in data.files}
                adapted, notes = model_store.adapt_weights(weights, record, FEATURE_DIM,
                                                           EXPRESSIONS, GESTURES)
                if notes:
                    record["compat"] = {"migrated": True, "notes": notes,
                                        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                record.setdefault("schema", {
                    "format": model_store.WEIGHT_FORMAT,
                    "schema_version": record.get("schema_version") or 0,
                    "feature_dim": int(adapted.get("w_expr", np.zeros((0, 0))).shape[0]),
                    "labels": {"expressions": list(EXPRESSIONS), "gestures": list(GESTURES)},
                })
                stored_fp = (record.get("schema") or {}).get("encode_fingerprint")
                if stored_fp and stored_fp != ENCODE_FINGERPRINT:
                    record["semantic"] = {
                        "stale": True,
                        "stored": stored_fp,
                        "current": ENCODE_FINGERPRINT,
                        "note": "特征语义已变化，建议重训迁移（旧层保留为化石）",
                    }
                elif not stored_fp:
                    record.setdefault("semantic", {"stale": False, "adopted": True,
                                                   "current": ENCODE_FINGERPRINT})
                self.models[bucket] = adapted
                self.meta[bucket] = record
                # 元数据可以升级，权重文件保持原样
                if notes or "schema_version" not in record:
                    try:
                        with open(path, "w", encoding="utf-8") as handle:
                            record["schema_version"] = model_store.SCHEMA_VERSION
                            json.dump(record, handle, ensure_ascii=False)
                    except OSError:
                        pass
            except (OSError, ValueError):
                continue

    # ---------- 全部层一起推理 ----------
    def ensemble_probs(self, expression, gesture, speaking, params, mood, attention=None):
        """返回各表达式/动作的概率分布，供仲裁层使用。"""
        with self.lock:
            if not self.models:
                return None
            x = encode(expression, gesture, speaking, params, mood).reshape(1, -1)
            probs_expr = np.zeros(len(EXPRESSIONS), dtype=np.float64)
            probs_gest = np.zeros(len(GESTURES), dtype=np.float64)
            total_weight = 0.0
            per_layer = []
            cascade = []
            fresh = [bucket for bucket in self.models
                     if not (self.meta.get(bucket, {}).get("semantic") or {}).get("stale")
                     and not self.meta.get(bucket, {}).get("rejected")
                     and not self.meta.get(bucket, {}).get("fossil")]
            buckets = fresh or list(self.models)
            for bucket in buckets:
                model = self.models[bucket]
                meta = self.meta.get(bucket, {})
                params = _ensemble_params()
                base = max(float(meta.get("adapted_weight") or meta.get("weight") or 0.1), 1e-6)
                weight = (base ** params["weight_power"]) * (max(meta.get("samples", 1), 1) ** params["sample_power"])
                if attention is not None:
                    weight *= max(float(attention.get(bucket, 0.25)), 0.02)
                logits = x @ model["w_expr"] + model["b_expr"]
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                glog = x @ model["w_gest"] + model["b_gest"]
                gprobs = np.exp(glog - glog.max())
                gprobs = gprobs / gprobs.sum()
                cascade.append((bucket, probs[0], weight))
                total_weight += weight
                per_layer.append({
                    "layer": meta.get("label"),
                    "weight": meta.get("weight"),
                    "samples": meta.get("samples"),
                    "expression": EXPRESSIONS[int(probs.argmax())],
                    "gesture": GESTURES[int(gprobs.argmax())],
                    "confidence": round(float(probs.max()), 3),
                })
            # 向前传播：按时间顺序，把上一层梯度按传递系数带进下一层
            transmission = _layer_transmission()
            ordered = sorted(cascade, key=lambda item: int(item[0]))
            propagated = []
            previous = None
            for bucket, probs, weight in ordered:
                if previous is None or transmission <= 0:
                    current = probs
                else:
                    gradient = float(np.abs(probs - previous).sum()) / 2.0  # 认知梯度：层间分布差
                    alpha = max(0.0, min(1.0, transmission * gradient))
                    current = (1.0 - alpha) * probs + alpha * previous
                propagated.append((current, weight))
                previous = current
            for probs, weight in propagated:
                probs_expr += probs * weight
            probs_expr /= total_weight
            probs_gest /= total_weight
            votes = {}
            for item in per_layer:
                votes[item["expression"]] = votes.get(item["expression"], 0) + 1
            disagreement = 1.0 - (max(votes.values()) / len(per_layer)) if per_layer else 0.0
            return {
                "expr_probs": {EXPRESSIONS[i]: round(float(probs_expr[i]), 4) for i in range(len(EXPRESSIONS))},
                "gest_probs": {GESTURES[i]: round(float(probs_gest[i]), 4) for i in range(len(GESTURES))},
                "per_layer": per_layer,
                "disagreement": round(disagreement, 3),
            }

    def evaluate_layers(self, limit=200, holdout_ratio=0.2):
        """留出验证：只用最近一段没参与本层训练的样本回测，避免过拟合。"""
        rows = self._rows()
        rows = sorted(rows, key=lambda row: row[0])[-limit:] if limit else rows
        if not rows:
            return {}
        holdout = max(4, int(len(rows) * holdout_ratio))
        if len(rows) - holdout < 4:
            validation = rows
            used_holdout = False
        else:
            validation = rows[-holdout:]
            used_holdout = True
        X, y_expr, y_gest, _ = self._features(validation)
        result = {}
        with self.lock:
            for bucket, model in self.models.items():
                if X.shape[0] == 0:
                    continue
                logits = X @ model["w_expr"] + model["b_expr"]
                pred = logits.argmax(axis=1)
                accuracy = float((pred == y_expr).mean())
                glog = X @ model["w_gest"] + model["b_gest"]
                gp = glog.argmax(axis=1)
                result[bucket] = {
                    "accuracy": round(accuracy, 3),
                    "gesture_accuracy": round(float((gp == y_gest).mean()), 3),
                    "samples": int(X.shape[0]),
                    "holdout": used_holdout,
                }
        return result

    def adapt_weights(self, min_samples=12):
        """按最近回测准确率调整每层在整体中的份额（无梯度，可解释）。"""
        evaluation = self.evaluate_layers()
        if not evaluation:
            return {"ok": False, "reason": "没有可回测的数据"}
        adapted = {}
        with self.lock:
            for bucket, model in self.models.items():
                meta = self.meta.get(bucket, {})
                if meta.get("kind") == "correction":
                    continue
                ev = evaluation.get(bucket)
                base = float(meta.get("adapted_weight") or meta.get("weight") or 0.2)
                if not ev or ev["samples"] < min_samples:
                    meta["recent_accuracy"] = ev["accuracy"] if ev else None
                    continue
                acc = float(ev["accuracy"])
                new_weight = round(base * (0.4 + 1.2 * acc), 4)
                meta["adapted_weight"] = new_weight
                meta["recent_accuracy"] = ev["accuracy"]
                meta["gesture_accuracy"] = ev["gesture_accuracy"]
                meta["adapted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                adapted[str(bucket)] = {"label": meta.get("label"), "old": base,
                                        "new": new_weight, "accuracy": ev["accuracy"]}
                try:
                    path = os.path.join(self.layers_dir, f"layer-{int(bucket)}.json")
                    if os.path.exists(path):
                        with open(path, "w", encoding="utf-8") as handle:
                            json.dump(meta, handle, ensure_ascii=False)
                except OSError:
                    pass
        detail = {str(bucket): {"label": self.meta.get(bucket, {}).get("label"),
                                "accuracy": ev.get("accuracy"),
                                "samples": ev.get("samples"),
                                "holdout": ev.get("holdout")}
                  for bucket, ev in evaluation.items()}
        return {"ok": True, "adapted": adapted, "evaluated": len(evaluation),
                "evaluation": detail}

    def contested_rows(self, limit=200, top=12):
        """挑出各层意见最分裂的状态（认知景观交叉点）。"""
        rows = self._rows()[-limit:] if limit else self._rows()
        if not rows:
            return [], {"contested": 0, "picked": 0}
        X, y_expr, y_gest, _ = self._features(rows)
        if X.shape[0] == 0:
            return [], {"contested": 0, "picked": 0}
        predictions = {}
        for bucket, model in self.models.items():
            logits = X @ model["w_expr"] + model["b_expr"]
            predictions[bucket] = logits.argmax(axis=1)
        scored = []
        for idx in range(len(rows)):
            counts = {}
            for bucket in predictions:
                label = EXPRESSIONS[predictions[bucket][idx]]
                counts[label] = counts.get(label, 0) + 1
            if len(counts) < 2:
                continue
            ordered = sorted(counts.values(), reverse=True)
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
            scored.append((margin, idx, len(counts)))
        scored.sort(key=lambda item: (item[0], -item[2]))
        picked = [idx for _, idx, _ in scored[:top]]
        return [rows[i] for i in picked], {"contested": len(scored), "picked": len(picked)}

    def train_active_layer(self, top=12, weight=0.55):
        """用交叉点当"难题集"，训练一层主动层（只往上叠，不改旧层）。"""
        rows, stats = self.contested_rows(top=top)
        if len(rows) < 4:
            return {"ok": False, "reason": "交叉点样本不足", **stats}
        record = self.add_consolidation_layer(
            rows, weight=weight, kind="active",
            label="主动层 " + time.strftime("%H:%M:%S"), min_samples=4)
        return {"ok": bool(record), "layer": record, **stats}

    def retrain_layer(self, record):
        """语义迁移：用该层时间段的数据 + 当前特征语义重训一层（旧层保留）。"""
        import sqlite3
        time_range = record.get("range")
        if not time_range or len(time_range) != 2:
            return {"ok": False, "reason": "该层没有时间范围，无法重训"}
        try:
            conn = sqlite3.connect("/root/workspace/digital-human/data/trajectory.db", timeout=5)
            rows = conn.execute(
                "SELECT ts, expression, gesture, speaking, mood, params, next_expression, next_gesture "
                "FROM trajectories WHERE ts >= ? AND ts <= ? AND next_expression IS NOT NULL AND next_expression != '' "
                "ORDER BY ts LIMIT 500", (time_range[0], time_range[1])).fetchall()
            conn.close()
        except sqlite3.Error:
            rows = []
        if len(rows) < 4:
            return {"ok": False, "reason": "该时间段数据不足", "samples": len(rows)}
        weight = record.get("adapted_weight") or record.get("weight") or 0.5
        new_record = self.add_consolidation_layer(
            rows, weight=float(weight), kind="retrained",
            label="重锚 " + str(record.get("label") or ""), min_samples=4)
        if not new_record:
            return {"ok": False, "reason": "训练失败"}
        new_record["source_layer"] = record.get("label")
        return {"ok": True, "layer": new_record, "samples": len(rows),
                "old_label": record.get("label")}

    def rows_in_ranges(self, ranges, limit=500):
        """取若干时间范围内的标注轨迹，作为交叉育种的素材。"""
        import sqlite3
        rows = []
        try:
            conn = sqlite3.connect("/root/workspace/digital-human/data/trajectory.db", timeout=5)
            for time_range in ranges:
                if not time_range or len(time_range) != 2:
                    continue
                rows.extend(conn.execute(
                    "SELECT ts, expression, gesture, speaking, mood, params, next_expression, next_gesture "
                    "FROM trajectories WHERE ts >= ? AND ts <= ? AND next_expression IS NOT NULL "
                    "AND next_expression != '' ORDER BY ts LIMIT ?",
                    (time_range[0], time_range[1], limit)).fetchall())
            conn.close()
        except sqlite3.Error:
            return []
        seen = set()
        unique = []
        for row in rows:
            if row[0] in seen:
                continue
            seen.add(row[0])
            unique.append(row)
        return unique

    def credit_assignment(self, rows, apply=True, min_obs=20):
        """表面误差的信用分配：每层在真实标注上对/错，按准确率微调层权重。"""
        if not rows:
            return {"ok": False, "reason": "没有数据"}
        X, y_expr, y_gest, _ = self._features(rows)
        if X.shape[0] < min_obs:
            return {"ok": False, "reason": "样本不足", "samples": int(X.shape[0])}
        stats = {}
        with self.lock:
            for bucket, model in self.models.items():
                meta = self.meta.get(bucket, {})
                if meta.get("rejected") or meta.get("fossil") or (meta.get("semantic") or {}).get("stale"):
                    continue
                pred = (X @ model["w_expr"] + model["b_expr"]).argmax(axis=1)
                right = int((pred == y_expr).sum())
                credit = right / max(1, len(y_expr))
                old_weight = float(meta.get("adapted_weight") or meta.get("weight") or 0.4)
                new_weight = max(0.05, min(0.95, old_weight * (0.7 + 0.6 * credit)))
                stats[bucket] = {"label": meta.get("label"), "accuracy": round(credit, 3),
                                 "right": right, "wrong": len(y_expr) - right,
                                 "old_weight": old_weight, "new_weight": new_weight}
                if apply:
                    meta["credit"] = {"accuracy": round(credit, 3), "right": right,
                                      "wrong": len(y_expr) - right,
                                      "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                    meta["adapted_weight"] = round(new_weight, 3)
                    self.meta[bucket] = meta
        if apply and stats:
            old_mean = sum(item["old_weight"] for item in stats.values()) / len(stats)
            new_mean = sum(item["new_weight"] for item in stats.values()) / len(stats)
            scale = old_mean / new_mean if new_mean else 1.0
            for item in stats.values():
                item["new_weight"] = round(max(0.05, min(0.95, item["new_weight"] * scale)), 3)
            for bucket, item in stats.items():
                if bucket in self.meta:
                    self.meta[bucket]["adapted_weight"] = item["new_weight"]
            for bucket, item in stats.items():
                meta = self.meta.get(bucket, {})
                try:
                    path = os.path.join(self.layers_dir, f"layer-{int(bucket)}.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(meta, handle, ensure_ascii=False)
                except OSError:
                    pass
        return {"ok": True, "samples": int(X.shape[0]), "layers": stats}

    def evaluate_holdout(self, bucket, limit=60):
        """在最近一段数据上评估某层准确率（用于后代接受/拒绝）。"""
        rows = self._rows()[-limit:] if limit else self._rows()
        if not rows or bucket not in self.models:
            return None
        X, y_expr, y_gest, _ = self._features(rows)
        if X.shape[0] == 0:
            return None
        model = self.models[bucket]
        pred = (X @ model["w_expr"] + model["b_expr"]).argmax(axis=1)
        return round(float((pred == y_expr).mean()), 3)

    def landscape(self, limit=200, top=6):
        """认知景观：每层的损失/梯度/曲率 + 层间分歧矩阵与交叉点。"""
        rows = self._rows()[-limit:] if limit else self._rows()
        if not rows:
            return {"ok": False, "reason": "没有数据"}
        X, y_expr, y_gest, _ = self._features(rows)
        if X.shape[0] == 0:
            return {"ok": False, "reason": "特征为空"}
        n = X.shape[0]
        predictions = {}
        layer_stats = {}
        rng = np.random.default_rng(7)
        for bucket, model in self.models.items():
            meta = self.meta.get(bucket, {})
            logits = X @ model["w_expr"] + model["b_expr"]
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            onehot = np.zeros_like(probs)
            onehot[np.arange(n), y_expr] = 1.0
            loss = float(-np.mean(np.log(np.clip(probs[np.arange(n), y_expr], 1e-9, 1.0))))
            grad = X.T @ (probs - onehot) / n
            variance = probs * (1.0 - probs)
            direction = rng.normal(size=X.shape[1])
            for _ in range(6):
                inner = ((X @ direction)[:, None] * variance).sum(axis=1)
                direction = X.T @ inner / n
                norm = np.linalg.norm(direction) or 1e-9
                direction /= norm
            sharpness = float(norm)
            accuracy = float((probs.argmax(axis=1) == y_expr).mean())
            predictions[bucket] = probs.argmax(axis=1)
            layer_stats[str(bucket)] = {
                "label": meta.get("label"),
                "kind": meta.get("kind"),
                "loss": round(loss, 3),
                "gradient_norm": round(float(np.linalg.norm(grad)), 3),
                "sharpness": round(sharpness, 3),
                "accuracy": round(accuracy, 3),
                "samples": n,
            }
        buckets = list(self.models)
        disagreement = []
        for i, left in enumerate(buckets):
            for right in buckets[i + 1:]:
                rate = float((predictions[left] != predictions[right]).mean())
                if rate > 0:
                    disagreement.append({
                        "a": self.meta.get(left, {}).get("label"),
                        "b": self.meta.get(right, {}).get("label"),
                        "rate": round(rate, 3),
                    })
        disagreement.sort(key=lambda item: -item["rate"])
        contest = []
        for idx in range(n):
            votes = {bucket: EXPRESSIONS[predictions[bucket][idx]] for bucket in buckets}
            counts = {}
            for label in votes.values():
                counts[label] = counts.get(label, 0) + 1
            if len(counts) < 2:
                continue
            ordered = sorted(counts.values(), reverse=True)
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
            contest.append({
                "state": rows[idx][1] if len(rows[idx]) > 1 else "",
                "next": rows[idx][6] if len(rows[idx]) > 6 else "",
                "margin": margin,
                "votes": {self.meta.get(bucket, {}).get("label"): label for bucket, label in votes.items()},
            })
        contest.sort(key=lambda item: item["margin"])
        return {
            "ok": True,
            "layers": layer_stats,
            "disagreement": disagreement[:8],
            "contest_points": contest[:top],
        }

    def predict(self, expression, gesture, speaking, params, mood, attention=None):
        ensemble = self.ensemble_probs(expression, gesture, speaking, params, mood, attention=attention)
        if ensemble is None:
            return None
        with self.lock:
            expr_probs = ensemble["expr_probs"]
            gest_probs = ensemble["gest_probs"]
            expr_idx = max(range(len(EXPRESSIONS)), key=lambda i: expr_probs[EXPRESSIONS[i]])
            gest_idx = max(range(len(GESTURES)), key=lambda i: gest_probs[GESTURES[i]])
            per_layer = ensemble["per_layer"]
            result = {
                "expression": EXPRESSIONS[expr_idx],
                "gesture": GESTURES[gest_idx],
                "confidence": round(float(expr_probs[EXPRESSIONS[expr_idx]]), 3),
                "source": "layered-ensemble",
                "layers": per_layer,
                "disagreement": ensemble["disagreement"],
                "expr_probs": expr_probs,
                "gest_probs": gest_probs,
            }
            # 修正层优先级最高：相似状态直接覆盖
            from trajectory_store import TrajectoryStore
            query_vec = embed(self.match_text(expression, gesture))
            if query_vec:
                q = np.frombuffer(query_vec, dtype=np.float32).astype(np.float64)
                best = None
                for correction in self.corrections:
                    if not correction.get("vector"):
                        continue
                    v = np.frombuffer(correction["vector"], dtype=np.float32).astype(np.float64)
                    similarity = float(np.dot(q, v))
                    if similarity >= 0.9 and (best is None or similarity > best[0]):
                        best = (similarity, correction)
                if best:
                    similarity, correction = best
                    result.update({
                        "expression": correction["next_expression"],
                        "gesture": correction["next_gesture"],
                        "confidence": round(similarity, 3),
                        "source": "correction-layer",
                        "corrected": True,
                    })
            return result

    def status(self):
        migrated = [m.get("label") for m in self.meta.values() if (m.get("compat") or {}).get("migrated")]
        stale = [m.get("label") for m in self.meta.values() if (m.get("semantic") or {}).get("stale")]
        return {"layers": len(self.models), "frozen_layers": len(self.models) + len([m for m in self.meta.values() if m.get("kind") == "correction"]),
                "corrections": len(self.corrections), "migrated": migrated, "stale": stale,
                "updated": len([m for m in self.meta.values()]),
                "detail": sorted(self.meta.values(), key=lambda r: -r.get("bucket", 0))[:12]}
