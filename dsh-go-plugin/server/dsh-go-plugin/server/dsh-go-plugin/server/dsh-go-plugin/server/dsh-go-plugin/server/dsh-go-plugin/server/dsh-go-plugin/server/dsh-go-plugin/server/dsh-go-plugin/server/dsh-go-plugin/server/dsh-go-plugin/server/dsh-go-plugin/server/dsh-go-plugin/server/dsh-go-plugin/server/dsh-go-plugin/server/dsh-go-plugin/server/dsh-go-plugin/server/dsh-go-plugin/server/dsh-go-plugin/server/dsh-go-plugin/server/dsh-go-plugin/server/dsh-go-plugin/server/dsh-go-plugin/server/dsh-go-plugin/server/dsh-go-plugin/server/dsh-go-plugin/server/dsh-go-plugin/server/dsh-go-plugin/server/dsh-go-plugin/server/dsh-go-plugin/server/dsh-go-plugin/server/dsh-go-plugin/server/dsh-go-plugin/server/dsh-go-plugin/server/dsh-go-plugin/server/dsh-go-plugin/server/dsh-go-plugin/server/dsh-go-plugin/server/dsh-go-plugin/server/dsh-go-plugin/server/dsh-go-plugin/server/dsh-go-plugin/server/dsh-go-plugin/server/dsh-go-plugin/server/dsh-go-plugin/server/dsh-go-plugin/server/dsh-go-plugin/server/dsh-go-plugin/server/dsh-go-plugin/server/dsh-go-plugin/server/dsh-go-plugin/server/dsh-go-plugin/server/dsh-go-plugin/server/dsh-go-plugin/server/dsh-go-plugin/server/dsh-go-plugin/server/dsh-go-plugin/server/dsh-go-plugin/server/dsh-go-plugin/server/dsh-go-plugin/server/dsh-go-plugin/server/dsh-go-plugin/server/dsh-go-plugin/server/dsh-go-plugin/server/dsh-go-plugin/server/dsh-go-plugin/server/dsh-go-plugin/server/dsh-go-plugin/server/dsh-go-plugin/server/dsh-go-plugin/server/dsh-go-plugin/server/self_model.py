# 自主注意力 + 自我状态：她此刻关注哪些地层/记忆，并持续记录"我正处在什么状态"
import json
import math
import os
import sqlite3
import threading
import time

import numpy as np

from policy_model import encode
from trajectory_store import embed


class SelfModel:
    def __init__(self, state_path, trajectory_db, layers):
        self.state_path = state_path
        self.trajectory_db = trajectory_db
        self.layers = layers
        self.lock = threading.Lock()
        self.centroids = {}
        self.state = self._load()

    def _load(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {"updated_at": None, "context": "", "attention": {}, "applied": None, "history": []}

    def _save(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False)
        except OSError:
            pass

    def _centroid(self, meta):
        """用该层时间段内的轨迹 embedding 均值作为"这个时代的样子"。"""
        bucket = meta.get("bucket")
        if bucket in self.centroids:
            return self.centroids[bucket]
        centroid = None
        time_range = meta.get("range")
        if time_range and len(time_range) == 2:
            try:
                conn = sqlite3.connect(self.trajectory_db, timeout=5)
                rows = conn.execute(
                    "SELECT embedding FROM trajectories WHERE ts >= ? AND ts <= ? AND embedding IS NOT NULL LIMIT 200",
                    (time_range[0], time_range[1]),
                ).fetchall()
                conn.close()
                vectors = [np.frombuffer(blob, dtype=np.float32).astype(np.float64)
                           for (blob,) in rows if blob and len(blob) == 768 * 4]
                if vectors:
                    mean = np.mean(np.stack(vectors), axis=0)
                    norm = np.linalg.norm(mean) or 1.0
                    centroid = mean / norm
            except sqlite3.Error:
                centroid = None
        self.centroids[bucket] = centroid
        return centroid

    def attention(self, context_text, state=None, temperature=1.6):
        """自主注意力：当前状态/语境 × 各层"典型状态"向量 × 先验权重 → 关注分布。"""
        metas = [meta for meta in self.layers.meta.values() if meta.get("kind") != "correction"]
        if not metas:
            return {}
        query = None
        if state:
            try:
                query = encode(state.get("expression") or "neutral", state.get("gesture") or "",
                               bool(state.get("speaking")), state.get("params") or {},
                               state.get("mood") or "平静").astype(np.float64)
            except Exception:  # noqa: BLE001
                query = None
        vector = None
        if query is None:
            vector = embed(context_text or "平静")
            if vector:
                query = np.frombuffer(vector, dtype=np.float32).astype(np.float64)
        sims, priors = {}, {}
        for meta in metas:
            prior = float(meta.get("adapted_weight") or meta.get("weight") or 0.2)
            similarity = 0.3
            if query is not None:
                feature_centroid = meta.get("feature_centroid")
                if feature_centroid and len(feature_centroid) == len(query):
                    similarity = float(np.dot(query, np.asarray(feature_centroid, dtype=np.float64)))
                else:
                    centroid = self._centroid(meta)
                    if centroid is not None and len(centroid) == len(query):
                        similarity = float(np.dot(query, centroid))
            sims[meta["bucket"]] = similarity
            priors[meta["bucket"]] = prior
        values = np.asarray(list(sims.values()), dtype=np.float64)
        mean, std = float(values.mean()), float(values.std() or 1e-6)
        scores = {}
        for bucket in sims:
            z = (sims[bucket] - mean) / std
            scores[bucket] = temperature * z + 0.8 * math.log(max(priors[bucket], 1e-3))
        if not scores:
            return {}
        peak = max(scores.values())
        weights = {bucket: math.exp(score - peak) for bucket, score in scores.items()}
        total = sum(weights.values()) or 1.0
        attention = {bucket: round(value / total, 4) for bucket, value in weights.items()}
        self._last_attention = attention
        return attention

    def observe(self, context_text, attention, applied=None):
        """把"我此刻在关注什么"写进自我状态。"""
        labels = {str(meta.get("bucket")): meta.get("label") for meta in self.layers.meta.values()}
        top = sorted(attention.items(), key=lambda item: -item[1])[:3]
        with self.lock:
            self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.state["context"] = str(context_text or "")[:200]
            self.state["attention"] = [
                {"layer": labels.get(str(bucket)) or str(bucket), "weight": weight}
                for bucket, weight in top
            ]
            if applied:
                self.state["applied"] = applied
            self.state.setdefault("history", []).append({
                "ts": time.time(),
                "context": self.state["context"][:80],
                "attention": self.state["attention"],
                "applied": applied,
            })
            self.state["history"] = self.state["history"][-50:]
            self._save()

    def check_consistency(self, narrated, applied, mood_expr, local_expr=None):
        """自我一致性：口头表情、实际执行、当前心情、本地预测是否对得上。"""
        pairs = {
            "narrated_applied": narrated == applied,
            "applied_mood": applied == mood_expr,
        }
        if local_expr:
            pairs["narrated_local"] = narrated == local_expr
        score = sum(1 for ok in pairs.values() if ok) / max(len(pairs), 1)
        mismatches = [name for name, ok in pairs.items() if not ok]
        event = {
            "ts": time.time(),
            "narrated": narrated,
            "applied": applied,
            "mood": mood_expr,
            "local": local_expr,
            "score": round(score, 2),
            "mismatches": mismatches,
        }
        with self.lock:
            state = self.state
            state["consistency"] = {
                "score": round(score, 2),
                "last": event,
                "checks": int((state.get("consistency") or {}).get("checks", 0)) + 1,
            }
            history = state.setdefault("consistency_history", [])
            history.append(event)
            state["consistency_history"] = history[-100:]
            self._save()
        return event

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def focus(self):
        return (self.state or {}).get("attention", [])
