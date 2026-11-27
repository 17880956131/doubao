# 长期记忆：SQLite + 本地 embedding 语义检索
import hashlib
import json
import math
import sqlite3
import struct
import threading
import time
import urllib.request

EMBED_URL = "http://127.0.0.1:8182/v1/embeddings"
EMBED_DIMS = 768


class MemoryStore:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.cache = {}
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, kind TEXT, text TEXT, importance REAL,
                embedding BLOB, last_used REAL, hits INTEGER)"""
        )
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "source" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'perception'")
            self.conn.commit()
        self.conn.commit()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories(ts)")
        self.conn.commit()

    @staticmethod
    def infer_source(kind):
        mapping = {
            "imagination": "imagined",
            "system": "perception", "image": "perception", "voice": "perception",
            "scene": "perception",
            "chat": "chat",
            "fact": "fact",
            "diary": "reflection",
            "joke": "generated",
        }
        return mapping.get(kind, "unknown")

    def _embed(self, text):
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        try:
            body = json.dumps({"input": [text[:500]]}, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=6) as response:
                vector = json.loads(response.read().decode("utf-8"))["data"][0]["embedding"]
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            blob = struct.pack(f"<{EMBED_DIMS}f", *[v / norm for v in vector])
        except Exception:  # noqa: BLE001 - 向量服务没起时降级为无记忆
            return None
        self.cache[key] = blob
        if len(self.cache) > 2000:
            self.cache.clear()
        return blob

    def add(self, kind, text, importance=0.35, source=None):
        text = str(text).strip()[:1000]
        if not text:
            return False
        blob = self._embed(text)
        source = source or self.infer_source(kind)
        with self.lock:
            self.conn.execute(
                "INSERT INTO memories (ts,kind,text,importance,embedding,last_used,hits,source) VALUES (?,?,?,?,?,?,0,?)",
                (time.time(), kind, text, float(importance), blob, time.time(), source),
            )
            self.conn.commit()
        return blob is not None

    def search(self, query, k=4, min_score=0.3, include_imagined=False, with_source=False):
        blob = self._embed(query)
        if blob is None:
            return []
        query_vec = struct.unpack(f"<{EMBED_DIMS}f", blob)
        rows = self.conn.execute(
            "SELECT id,kind,text,importance,embedding,source FROM memories WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT 400"
        ).fetchall()
        scored = []
        for rid, kind, text, importance, embedding, source in rows:
            if not embedding or len(embedding) != EMBED_DIMS * 4:
                continue
            if not include_imagined and (source or "unknown") == "imagined":
                continue
            vec = struct.unpack(f"<{EMBED_DIMS}f", embedding)
            score = sum(a * b for a, b in zip(query_vec, vec)) + (importance or 0) * 0.05
            scored.append((score, rid, kind, text, source))
        scored.sort(reverse=True)
        hits = [item for item in scored[:k] if item[0] >= min_score]
        with self.lock:
            for _, rid, _, _, _ in hits:
                self.conn.execute("UPDATE memories SET last_used=?, hits=hits+1 WHERE id=?", (time.time(), rid))
            self.conn.commit()
        if with_source:
            return [{"kind": kind, "text": text, "source": source or "unknown"}
                    for _, _, kind, text, source in hits]
        return [(kind, text) for _, _, kind, text, _ in hits]

    def source_stats(self):
        rows = self.conn.execute("SELECT COALESCE(source,'unknown'), COUNT(*) FROM memories GROUP BY source").fetchall()
        return {str(source): int(count) for source, count in rows}

    def recent(self, limit=8):
        rows = self.conn.execute("SELECT ts,kind,text,importance FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": ts, "kind": kind, "text": text, "importance": importance} for ts, kind, text, importance in rows]

    def day_memories(self, day_start):
        rows = self.conn.execute(
            "SELECT ts,kind,text FROM memories WHERE ts >= ? AND kind IN ('chat','diary','fact') ORDER BY ts",
            (day_start,),
        ).fetchall()
        return [{"ts": ts, "kind": kind, "text": text} for ts, kind, text in rows]

    def decay_and_prune(self, max_rows=800):
        """按年龄衰减重要度，删掉几乎没用的，控制规模。"""
        now = time.time()
        with self.lock:
            rows = self.conn.execute("SELECT id,ts,importance FROM memories").fetchall()
            for rid, ts, importance in rows:
                age_days = max(0.0, (now - (ts or now)) / 86400)
                decayed = (importance or 0.3) * (0.5 ** (age_days / 45))
                if decayed < 0.03:
                    self.conn.execute("DELETE FROM memories WHERE id=?", (rid,))
                else:
                    self.conn.execute("UPDATE memories SET importance=? WHERE id=?", (decayed, rid))
            count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count > max_rows:
                self.conn.execute(
                    "DELETE FROM memories WHERE id IN (SELECT id FROM memories ORDER BY importance ASC, id ASC LIMIT ?)",
                    (count - max_rows,),
                )
            self.conn.commit()

    def stats(self):
        row = self.conn.execute("SELECT COUNT(*), SUM(hits) FROM memories").fetchone()
        return {"count": row[0] or 0, "hits": row[1] or 0}
