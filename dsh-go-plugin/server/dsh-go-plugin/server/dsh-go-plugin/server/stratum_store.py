# 地层式记忆：时间切片 → 每层权重 → 层间认知梯度
# 概念在每一层的表达也切片存储，可比较"一个时代"和"两个时代"的差异
import json
import math
import sqlite3
import struct
import threading
import time

import numpy as np

from trajectory_store import DIMS, embed

BEHAVIOR_ACTIONS = ["none", "quiet", "gesture", "joke", "care", "greet"]


def _day_start(ts):
    return time.mktime(time.localtime(ts)[:3] + (0, 0, 0, 0, 0, -1))


class StratumStore:
    def __init__(self, path, trajectory_db, behavior_log):
        self.path = path
        self.trajectory_db = trajectory_db
        self.behavior_log = behavior_log
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS strata (
                bucket_start REAL PRIMARY KEY, bucket_end REAL, granularity TEXT,
                samples INTEGER, weight REAL, centroid BLOB,
                expression_json TEXT, behavior_json TEXT, mood_json TEXT)"""
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS strata_boundaries (
                from_bucket REAL, to_bucket REAL,
                cosine_distance REAL, behavior_divergence REAL, mood_divergence REAL,
                concept_json TEXT, PRIMARY KEY (from_bucket, to_bucket))"""
        )
        self.conn.commit()

    # ---------- 构建地层 ----------
    def _bucket_of(self, ts, granularity):
        if granularity == "hour":
            return math.floor(ts / 3600) * 3600
        if granularity == "10min":
            return math.floor(ts / 600) * 600
        return _day_start(ts)

    def build(self, granularity="hour", tau_days=7, weight_base=0.1, weight_span=0.4):
        with self.lock:
            self.conn.execute("DELETE FROM strata")
            self.conn.execute("DELETE FROM strata_boundaries")
            self.conn.commit()

        rows = []
        try:
            conn = sqlite3.connect(self.trajectory_db, timeout=5)
            rows = conn.execute(
                "SELECT ts, embedding, expression, gesture, mood FROM trajectories"
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            rows = []

        layers = {}
        now = time.time()
        for ts, embedding, expression, gesture, mood in rows:
            if not ts:
                continue
            bucket = self._bucket_of(ts, granularity)
            layer = layers.setdefault(bucket, {"embeddings": [], "expressions": {}, "moods": {}, "samples": 0})
            layer["samples"] += 1
            if embedding and len(embedding) == DIMS * 4:
                vec = np.frombuffer(embedding, dtype=np.float32).astype(np.float64)
                norm = np.linalg.norm(vec) or 1.0
                layer["embeddings"].append(vec / norm)
            layer["expressions"][expression or "neutral"] = layer["expressions"].get(expression or "neutral", 0) + 1
            layer["moods"][mood or "平静"] = layer["moods"].get(mood or "平静", 0) + 1

        behaviors = {}
        try:
            with open(self.behavior_log, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    ts = entry.get("ts") or 0
                    bucket = self._bucket_of(ts, granularity)
                    bucket_data = behaviors.setdefault(bucket, {})
                    action = entry.get("action") or "?"
                    item = bucket_data.setdefault(action, {"count": 0, "engaged": 0})
                    item["count"] += 1
                    if entry.get("engaged"):
                        item["engaged"] += 1
        except OSError:
            pass

        with self.lock:
            for bucket in sorted(layers):
                layer = layers[bucket]
                age_days = max(0.0, (now - bucket) / 86400)
                weight = round(weight_base + weight_span * math.exp(-age_days / max(tau_days, 0.01)), 3)
                if layer["embeddings"]:
                    centroid = np.mean(np.stack(layer["embeddings"]), axis=0)
                    centroid /= (np.linalg.norm(centroid) or 1.0)
                    blob = struct.pack(f"<{DIMS}f", *centroid.astype(np.float32))
                else:
                    blob = None
                self.conn.execute(
                    "INSERT OR REPLACE INTO strata VALUES (?,?,?,?,?,?,?,?,?)",
                    (bucket, bucket + (3600 if granularity == "hour" else 86400), granularity,
                     layer["samples"], weight, blob,
                     json.dumps(layer["expressions"], ensure_ascii=False),
                     json.dumps(behaviors.get(bucket, {}), ensure_ascii=False),
                     json.dumps(layer["moods"], ensure_ascii=False)),
                )
            self.conn.commit()

        self._compute_boundaries(granularity)
        return self.list_layers()

    def _compute_boundaries(self, granularity):
        layers = self.list_layers()
        with self.lock:
            for left, right in zip(layers, layers[1:]):
                cos = None
                if left.get("centroid") and right.get("centroid"):
                    a = np.frombuffer(left["centroid"], dtype=np.float32).astype(np.float64)
                    b = np.frombuffer(right["centroid"], dtype=np.float32).astype(np.float64)
                    cos = round(1.0 - float(np.dot(a, b)), 4)
                div = self._distribution_divergence(left["behaviors"], right["behaviors"])
                mood_div = self._distribution_divergence(left["moods"], right["moods"])
                self.conn.execute(
                    "INSERT OR REPLACE INTO strata_boundaries VALUES (?,?,?,?,?,?)",
                    (left["bucket_start"], right["bucket_start"], cos, div, mood_div, None),
                )
            self.conn.commit()

    @staticmethod
    def _counts(dist):
        return {key: (value.get("count", 0) if isinstance(value, dict) else value)
                for key, value in (dist or {}).items()}

    @classmethod
    def _distribution_divergence(cls, a, b):
        a = cls._counts(a)
        b = cls._counts(b)
        keys = set(a) | set(b)
        if not keys:
            return None
        pa = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
        pb = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
        pa = pa / (pa.sum() or 1)
        pb = pb / (pb.sum() or 1)
        m = (pa + pb) / 2
        def kl(p, q):
            mask = (p > 0) & (q > 0)
            return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
        return round(0.5 * kl(pa, m) + 0.5 * kl(pb, m), 4)

    # ---------- 查询 ----------
    def list_layers(self, limit=24):
        rows = self.conn.execute(
            "SELECT bucket_start,bucket_end,granularity,samples,weight,centroid,expression_json,behavior_json,mood_json "
            "FROM strata ORDER BY bucket_start DESC LIMIT ?", (limit,)
        ).fetchall()
        items = []
        for bucket, end, gran, samples, weight, centroid, expressions, behaviors, moods in rows:
            items.append({
                "bucket_start": bucket,
                "bucket_end": end,
                "label": time.strftime("%Y-%m-%d %H时" if gran == "hour" else "%H:%M" if gran == "10min" else "%Y-%m-%d", time.localtime(bucket)),
                "granularity": gran,
                "samples": samples,
                "weight": weight,
                "centroid": centroid,
                "expressions": json.loads(expressions or "{}"),
                "behaviors": json.loads(behaviors or "{}"),
                "moods": json.loads(moods or "{}"),
            })
        return items

    def boundaries(self):
        rows = self.conn.execute(
            "SELECT from_bucket,to_bucket,cosine_distance,behavior_divergence,mood_divergence "
            "FROM strata_boundaries ORDER BY from_bucket DESC LIMIT 24"
        ).fetchall()
        return [
            {"from": time.strftime("%m-%d %H:%M", time.localtime(f)),
             "to": time.strftime("%m-%d %H:%M", time.localtime(t)),
             "cognitive_gradient": cos, "behavior_divergence": div, "mood_divergence": mood}
            for f, t, cos, div, mood in rows
        ]

    def concept_trajectory(self, query, limit=50):
        """一个概念在各层里的表达（按相似度），展示梯度变化。"""
        vector = embed(query)
        if vector is None:
            return {"query": query, "layers": []}
        q = np.frombuffer(vector, dtype=np.float32).astype(np.float64)
        result = []
        conn = sqlite3.connect(self.trajectory_db, timeout=5)
        try:
            for layer in self.list_layers(limit=24):
                rows = conn.execute(
                    "SELECT embedding FROM trajectories WHERE ts >= ? AND ts < ? AND embedding IS NOT NULL LIMIT ?",
                    (layer["bucket_start"], layer["bucket_end"], limit),
                ).fetchall()
                scores = []
                for (blob,) in rows:
                    if not blob or len(blob) != DIMS * 4:
                        continue
                    v = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
                    scores.append(float(np.dot(q, v)))
                if scores:
                    result.append({
                        "label": layer["label"],
                        "weight": layer["weight"],
                        "samples": layer["samples"],
                        "similarity": round(float(np.mean(scores)), 3),
                        "max": round(float(max(scores)), 3),
                    })
        finally:
            conn.close()
        return {"query": query, "layers": result}

    def conflicts(self, similarity_threshold=0.9):
        """检测层间冲突：同一状态在不同时代被学成了不同的下一步。"""
        try:
            conn = sqlite3.connect(self.trajectory_db, timeout=5)
            rows = conn.execute(
                "SELECT ts, embedding, expression, next_expression, text FROM trajectories "
                "WHERE next_expression IS NOT NULL AND next_expression != '' ORDER BY ts"
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            return {"pairs": 0, "conflicts": [], "rate": 0.0}
        layers = {}
        for ts, embedding, expression, next_expression, text in rows:
            if not embedding or len(embedding) != DIMS * 4:
                continue
            bucket = self._bucket_of(ts, "10min")
            layer = layers.setdefault(bucket, [])
            vec = np.frombuffer(embedding, dtype=np.float32).astype(np.float64)
            norm = np.linalg.norm(vec) or 1.0
            layer.append({"vec": vec / norm, "expression": expression or "neutral",
                          "next": next_expression or "neutral", "text": (text or "")[:60]})
        buckets = sorted(layers)
        found = []
        compared = 0
        for bucket_a, bucket_b in zip(buckets, buckets[1:]):
            for old in layers[bucket_a]:
                for new in layers[bucket_b]:
                    compared += 1
                    if old["next"] == new["next"]:
                        continue
                    sim = float(np.dot(old["vec"], new["vec"]))
                    if sim >= similarity_threshold:
                        found.append({
                            "similarity": round(sim, 3),
                            "old_layer": time.strftime("%H:%M", time.localtime(bucket_a)),
                            "new_layer": time.strftime("%H:%M", time.localtime(bucket_b)),
                            "same_state": old["text"],
                            "old_learned": old["next"],
                            "new_learned": new["next"],
                        })
        found.sort(key=lambda item: -item["similarity"])
        return {
            "pairs": compared,
            "conflict_count": len(found),
            "rate": round(len(found) / compared, 3) if compared else 0.0,
            "conflicts": found[:8],
        }

    def weight_at(self, ts, tau_days=7, base=0.1, span=0.4):
        """任意时刻所属地层的权重（按年龄衰减：旧层≈0.1，新层≈0.5）。"""
        age_days = max(0.0, (time.time() - float(ts)) / 86400)
        return round(base + span * math.exp(-age_days / max(tau_days, 0.01)), 4)

    def weight_map(self, granularity="hour", tau_days=7):
        """给训练用的按层权重：层越新权重越高。"""
        mapping = {}
        for layer in self.list_layers(limit=1000):
            if layer["granularity"] == granularity:
                mapping[int(layer["bucket_start"])] = layer["weight"]
        return mapping

    def planet(self):
        """把所有地层聚合成一颗"权重星球"：加权质心 + 加权分布 + 漂移指数。"""
        layers = self.list_layers(limit=1000)
        layers = [layer for layer in layers if layer.get("centroid")]
        if not layers:
            return {"layers": 0, "samples": 0}
        total = sum(layer["samples"] for layer in layers) or 1
        weights = {layer["bucket_start"]: max(layer["weight"], 1e-6) for layer in layers}
        total_weight = sum(weights.values()) or 1.0

        centroid = None
        for layer in layers:
            vec = np.frombuffer(layer["centroid"], dtype=np.float32).astype(np.float64)
            share = weights[layer["bucket_start"]] * layer["samples"] / total
            centroid = vec * share if centroid is None else centroid + vec * share
        norm = np.linalg.norm(centroid) or 1.0
        centroid = centroid / norm

        def blend(field):
            acc = {}
            for layer in layers:
                share = weights[layer["bucket_start"]] * layer["samples"] / total
                for key, value in (layer.get(field) or {}).items():
                    if isinstance(value, dict):
                        value = value.get("count", 0)
                    acc[key] = acc.get(key, 0.0) + float(value) * share
            sink = sum(acc.values()) or 1.0
            return {key: round(value / sink, 3) for key, value in sorted(acc.items(), key=lambda x: -x[1])}

        boundaries = self.boundaries()
        gradients = [b["cognitive_gradient"] for b in boundaries if b.get("cognitive_gradient") is not None]
        drift = round(float(np.mean(gradients)), 4) if gradients else 0.0

        expressions = blend("expressions")
        entropy = 0.0
        total_expr = sum(expressions.values()) or 1.0
        for value in expressions.values():
            p_ = value / total_expr
            if p_ > 0:
                entropy -= p_ * math.log(p_)
        span_days = round((max(l["bucket_start"] for l in layers) - min(l["bucket_start"] for l in layers)) / 86400, 3)

        return {
            "layers": len(layers),
            "samples": int(total),
            "span_days": span_days,
            "avg_weight": round(total_weight / len(layers), 3),
            "total_influence": round(sum(weights[k] * sum(l["samples"] for l in layers if l["bucket_start"] == k) for k in weights), 1),
            "centroid_preview": [round(float(x), 4) for x in centroid[:8]],
            "centroid_dims": int(len(centroid)),
            "expressions": expressions,
            "behaviors": blend("behaviors"),
            "moods": blend("moods"),
            "drift_index": drift,
            "conflict": self.conflicts(),
            "expression_entropy": round(entropy, 3),
            "layer_weights": {l["label"]: l["weight"] for l in layers[:12]},
        }

    def stats(self):
        rows = self.conn.execute("SELECT COUNT(*), SUM(samples), AVG(weight) FROM strata").fetchone()
        return {"layers": rows[0] or 0, "samples": rows[1] or 0, "avg_weight": round(rows[2] or 0, 3)}
