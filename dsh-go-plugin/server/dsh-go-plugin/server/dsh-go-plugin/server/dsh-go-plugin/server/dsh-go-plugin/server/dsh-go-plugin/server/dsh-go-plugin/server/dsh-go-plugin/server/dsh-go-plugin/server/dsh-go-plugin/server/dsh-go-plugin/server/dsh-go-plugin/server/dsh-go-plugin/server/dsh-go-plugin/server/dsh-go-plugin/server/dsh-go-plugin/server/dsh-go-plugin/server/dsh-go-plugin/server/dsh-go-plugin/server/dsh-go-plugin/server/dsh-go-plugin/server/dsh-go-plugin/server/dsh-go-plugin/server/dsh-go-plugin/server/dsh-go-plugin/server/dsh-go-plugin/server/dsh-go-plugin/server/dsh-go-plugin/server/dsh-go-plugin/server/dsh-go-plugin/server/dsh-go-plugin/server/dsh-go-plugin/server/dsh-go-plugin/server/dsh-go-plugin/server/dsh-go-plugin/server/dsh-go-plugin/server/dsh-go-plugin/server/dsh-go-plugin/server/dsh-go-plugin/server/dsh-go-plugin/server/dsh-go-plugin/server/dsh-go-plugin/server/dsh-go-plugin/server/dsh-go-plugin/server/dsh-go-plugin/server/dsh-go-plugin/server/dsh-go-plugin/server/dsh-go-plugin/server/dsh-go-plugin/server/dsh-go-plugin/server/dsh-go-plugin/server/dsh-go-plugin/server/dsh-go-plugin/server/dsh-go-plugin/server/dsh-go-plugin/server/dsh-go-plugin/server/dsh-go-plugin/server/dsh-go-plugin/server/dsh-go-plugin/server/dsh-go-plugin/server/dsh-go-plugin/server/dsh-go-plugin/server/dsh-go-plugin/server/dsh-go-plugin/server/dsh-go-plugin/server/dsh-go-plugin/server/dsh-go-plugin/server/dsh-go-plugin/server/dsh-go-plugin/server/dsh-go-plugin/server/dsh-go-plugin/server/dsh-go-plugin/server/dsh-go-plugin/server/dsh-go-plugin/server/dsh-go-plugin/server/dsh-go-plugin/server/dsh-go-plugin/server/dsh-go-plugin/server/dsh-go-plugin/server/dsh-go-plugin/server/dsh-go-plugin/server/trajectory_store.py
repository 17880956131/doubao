# 活动轨迹记忆：把数字人的表情/动作/口型/心情按时间编码成向量
# 用本机 embedding 神经网络（embeddinggemma-300m）做记忆检索与下一步预测
import json
import math
import sqlite3
import struct
import threading
import time
import urllib.request

EMBED_URL = "http://127.0.0.1:8182/v1/embeddings"
DIMS = 768


def embed(text):
    try:
        body = json.dumps({"input": [text[:500]]}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=6) as response:
            vector = json.loads(response.read().decode("utf-8"))["data"][0]["embedding"]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return struct.pack(f"<{DIMS}f", *[v / norm for v in vector])
    except Exception:  # noqa: BLE001
        return None


class TrajectoryStore:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, text TEXT, embedding BLOB,
                expression TEXT, gesture TEXT, speaking INTEGER,
                mood TEXT, scene TEXT,
                next_expression TEXT, next_gesture TEXT)"""
        )
        self.conn.commit()
        # 兼容旧库：补 params 列（给本地策略模型训练用）
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(trajectories)").fetchall()]
        if "source" not in columns:
            self.conn.execute("ALTER TABLE trajectories ADD COLUMN source TEXT DEFAULT 'stage'")
            self.conn.commit()
        if "context_text" not in columns:
            self.conn.execute("ALTER TABLE trajectories ADD COLUMN context_text TEXT DEFAULT ''")
            self.conn.commit()
        if "params" not in columns:
            self.conn.execute("ALTER TABLE trajectories ADD COLUMN params TEXT")
            self.conn.commit()

    @staticmethod
    def snapshot_text(snapshot):
        params = snapshot.get("params") or {}
        parts = [
            "表情=" + str(snapshot.get("expression") or "neutral"),
            "动作=" + str(snapshot.get("gesture") or "none"),
            "说话=" + ("是" if snapshot.get("speaking") else "否"),
        ]
        for key, label in (("PARAM_MOUTH_OPEN_Y", "嘴"), ("PARAM_ANGLE_X", "头左右"),
                           ("PARAM_ANGLE_Y", "头上下"), ("PARAM_BODY_ANGLE_Z", "身体")):
            value = params.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={round(value, 2)}")
        if snapshot.get("mood"):
            parts.append("心情=" + str(snapshot["mood"]))
        if snapshot.get("scene"):
            parts.append("场景=" + str(snapshot["scene"]))
        return "；".join(parts)

    def add(self, snapshot):
        text = self.snapshot_text(snapshot)
        vector = embed(text)
        expression = str(snapshot.get("expression") or "neutral")[:20]
        gesture = str(snapshot.get("gesture") or "")[:20]
        with self.lock:
            previous = self.conn.execute(
                "SELECT id, expression, gesture FROM trajectories ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if previous and (previous[1] != expression or previous[2] != gesture):
                self.conn.execute(
                    "UPDATE trajectories SET next_expression=?, next_gesture=? WHERE id=?",
                    (expression, gesture, previous[0]),
                )
            self.conn.execute(
                """INSERT INTO trajectories (ts,text,embedding,expression,gesture,speaking,mood,scene,params,source,context_text,next_expression,next_gesture)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)""",
                (time.time(), text, vector, expression, gesture, 1 if snapshot.get("speaking") else 0,
                 str(snapshot.get("mood") or "")[:20], str(snapshot.get("scene") or "")[:20],
                 json.dumps(snapshot.get("params") or {}, ensure_ascii=False),
                 str(snapshot.get("source") or "stage")[:20],
                 str(snapshot.get("context_text") or "")[:400]),
            )
            self.conn.commit()
            count = self.conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        return {"text": text, "count": count, "embedded": vector is not None}

    def recent(self, limit=10):
        rows = self.conn.execute(
            "SELECT ts,text,expression,gesture,next_expression,next_gesture,params FROM trajectories ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        items = []
        for ts, text, expr, gesture, next_expr, next_gesture, params_json in rows:
            try:
                params = json.loads(params_json) if params_json else {}
            except ValueError:
                params = {}
            items.append({"ts": ts, "text": text, "expression": expr, "gesture": gesture,
                          "params": params, "next_expression": next_expr, "next_gesture": next_gesture})
        return items

    def similar(self, snapshot, k=3, min_score=0.45):
        vector = embed(self.snapshot_text(snapshot))
        if vector is None:
            return []
        query = struct.unpack(f"<{DIMS}f", vector)
        rows = self.conn.execute(
            "SELECT id,text,expression,gesture,next_expression,next_gesture,embedding FROM trajectories ORDER BY id DESC LIMIT 300"
        ).fetchall()
        scored = []
        for rid, text, expr, gesture, next_expr, next_gesture, embedding in rows:
            if not embedding or len(embedding) != DIMS * 4:
                continue
            vec = struct.unpack(f"<{DIMS}f", embedding)
            score = sum(a * b for a, b in zip(query, vec))
            if score >= min_score:
                scored.append((score, rid, text, expr, gesture, next_expr, next_gesture))
        scored.sort(reverse=True)
        return [
            {"score": round(score, 2), "text": text, "expression": expr, "gesture": gesture,
             "next_expression": next_expr, "next_gesture": next_gesture}
            for score, _, text, expr, gesture, next_expr, next_gesture in scored[:k]
        ]

    def predict(self, snapshot, k=5):
        """用最近邻历史轨迹预测下一步表情/动作（关联记忆，不训练新网络）。"""
        hits = self.similar(snapshot, k=k, min_score=0.4)
        votes_expr = {}
        votes_gest = {}
        for hit in hits:
            expr = hit.get("next_expression")
            gest = hit.get("next_gesture")
            if expr:
                votes_expr[expr] = votes_expr.get(expr, 0) + 1
            if gest:
                votes_gest[gest] = votes_gest.get(gest, 0) + 1
        expression = max(votes_expr, key=votes_expr.get) if votes_expr else "neutral"
        gesture = max(votes_gest, key=votes_gest.get) if votes_gest else ""
        return {"expression": expression, "gesture": gesture, "based_on": len(hits), "samples": hits[:2]}

    def prune(self, max_rows=3000, max_age_days=45):
        """轨迹库只增不减会拖慢训练，按时间和条数清理。"""
        cutoff = time.time() - max_age_days * 86400
        with self.lock:
            self.conn.execute("DELETE FROM trajectories WHERE ts < ?", (cutoff,))
            count = self.conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
            if count > max_rows:
                self.conn.execute(
                    "DELETE FROM trajectories WHERE id IN (SELECT id FROM trajectories ORDER BY id ASC LIMIT ?)",
                    (count - max_rows,),
                )
            self.conn.commit()
            return count

    def stats(self):
        row = self.conn.execute("SELECT COUNT(*), COUNT(DISTINCT expression) FROM trajectories").fetchone()
        return {"rows": row[0] or 0, "expressions": row[1] or 0}
