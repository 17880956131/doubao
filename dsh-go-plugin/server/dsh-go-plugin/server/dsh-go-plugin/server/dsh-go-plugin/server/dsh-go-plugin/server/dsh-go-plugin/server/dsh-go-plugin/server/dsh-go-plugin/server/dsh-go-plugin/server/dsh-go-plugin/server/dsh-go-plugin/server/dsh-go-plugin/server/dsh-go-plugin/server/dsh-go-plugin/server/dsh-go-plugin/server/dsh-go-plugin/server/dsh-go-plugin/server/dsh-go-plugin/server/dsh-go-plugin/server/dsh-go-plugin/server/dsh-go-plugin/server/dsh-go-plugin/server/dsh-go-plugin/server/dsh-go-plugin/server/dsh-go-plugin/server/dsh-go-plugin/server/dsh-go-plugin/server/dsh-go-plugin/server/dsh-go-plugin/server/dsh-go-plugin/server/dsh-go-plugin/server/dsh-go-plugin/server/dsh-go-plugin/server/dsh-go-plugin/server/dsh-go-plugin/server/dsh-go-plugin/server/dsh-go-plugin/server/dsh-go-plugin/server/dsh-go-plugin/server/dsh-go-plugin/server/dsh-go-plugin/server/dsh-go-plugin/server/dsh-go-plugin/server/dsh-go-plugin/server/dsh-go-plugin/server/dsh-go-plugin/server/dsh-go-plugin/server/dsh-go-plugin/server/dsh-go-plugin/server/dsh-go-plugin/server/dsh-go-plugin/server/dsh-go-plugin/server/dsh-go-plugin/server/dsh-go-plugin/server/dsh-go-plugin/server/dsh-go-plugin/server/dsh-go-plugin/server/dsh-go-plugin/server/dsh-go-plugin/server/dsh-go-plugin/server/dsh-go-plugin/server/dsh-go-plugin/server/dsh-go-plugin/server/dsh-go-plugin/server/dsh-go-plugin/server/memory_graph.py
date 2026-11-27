# 记忆图谱：从平铺记忆里抽实体、连关系（谁/什么/何时/共现）
import json
import os
import re
import sqlite3
import threading
import time

import numpy as np

DB_PATH = "/root/workspace/digital-human/data/memory.db"
GRAPH_PATH = "/root/workspace/digital-human/data/memory_graph.db"

STOP_WORDS = {"我", "你", "她", "他", "我们", "你们", "的", "了", "是", "在", "有", "和", "就", "也", "都", "很", "太", "这", "那", "个", "一", "点", "说", "觉得", "现在", "今天", "明天", "昨天"}
ENTITY_PATTERNS = [
    r"(?:喜欢|爱|讨厌|害怕|怕|想要|想喝|想吃|喝|吃)([^，。；！？\s]{1,10})",
    r"(?:记住|记得)([^，。；！？\s]{1,15})",
]


def _segments(text):
    cleaned = re.sub(r"^(用户|助手|她|他|系统)[：:]\s*", "", str(text or ""))
    return [part.strip() for part in re.split(r"[，。；！？、\n,.;!?]", cleaned) if part.strip()]


def extract_entities(text):
    entities = []
    for segment in _segments(text):
        for pattern in ENTITY_PATTERNS:
            for match in re.findall(pattern, segment):
                cleaned = re.split(r"[能帮会要让把是有]", match)[0].strip("的了呢啊呀吧啦嘛哈哦哟")
                if 1 <= len(cleaned) <= 10 and cleaned not in STOP_WORDS:
                    entities.append(cleaned)
        cleaned = segment.strip("的了呢啊呀吧啦嘛哈哦哟")
        cleaned = re.split(r"[能帮会要让把是有]", cleaned)[0].strip("的了呢啊呀吧啦嘛哈哦哟")
        if 2 <= len(cleaned) <= 8 and not re.search(r"[喜欢爱怕喝吃想要记住]", cleaned) \
                and cleaned not in STOP_WORDS:
            entities.append(cleaned)
    seen = []
    for entity in entities:
        if entity not in seen:
            seen.append(entity)
    return seen[:6]


class MemoryGraph:
    def __init__(self, graph_path=GRAPH_PATH, memory_db=DB_PATH):
        self.path = graph_path
        self.memory_db = memory_db
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(graph_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, layer REAL, type TEXT,
                embedding BLOB, mentions INTEGER DEFAULT 0, last_ts REAL,
                UNIQUE(name, layer))"""
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS edges (
                src INTEGER, dst INTEGER, relation TEXT, layer REAL, weight REAL DEFAULT 1,
                last_ts REAL, PRIMARY KEY (src, dst, relation, layer))"""
        )
        self.conn.commit()

    def _embed(self, text):
        try:
            from trajectory_store import embed
            return embed(text)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _day(ts):
        return time.mktime(time.localtime(ts)[:3] + (0, 0, 0, 0, 0, -1))

    def _node(self, name, node_type="概念", layer=0.0):
        name = str(name or "").strip()
        canonical = name
        for exist_name in [row[0] for row in self.conn.execute("SELECT DISTINCT name FROM nodes").fetchall()]:
            if exist_name and exist_name != name:
                if exist_name in name and 2 <= len(exist_name) < len(canonical):
                    canonical = exist_name
                elif name in exist_name and 2 <= len(name) < len(exist_name):
                    canonical = name
        row = self.conn.execute("SELECT id, mentions FROM nodes WHERE name = ? AND layer = ?",
                                (canonical, layer)).fetchone()
        if row:
            self.conn.execute("UPDATE nodes SET mentions = mentions + 1, last_ts = ? WHERE id = ?",
                              (time.time(), row[0]))
            self.conn.commit()
            return row[0]
        embedding = self._embed(canonical)
        cursor = self.conn.execute(
            "INSERT INTO nodes (name, layer, type, embedding, mentions, last_ts) VALUES (?,?,?,?,?,?)",
            (canonical, layer, node_type, embedding, 1, time.time()))
        self.conn.commit()
        return cursor.lastrowid

    def _edge(self, src, dst, relation, weight=1.0, layer=0.0):
        if src == dst:
            return
        row = self.conn.execute(
            "SELECT weight FROM edges WHERE src=? AND dst=? AND relation=? AND layer=?",
            (src, dst, relation, layer)).fetchone()
        if row:
            self.conn.execute(
                "UPDATE edges SET weight = weight + ?, last_ts=? WHERE src=? AND dst=? AND relation=? AND layer=?",
                (weight, time.time(), src, dst, relation, layer))
        else:
            self.conn.execute("INSERT INTO edges (src,dst,relation,layer,weight,last_ts) VALUES (?,?,?,?,?,?)",
                              (src, dst, relation, layer, weight, time.time()))
        self.conn.commit()

    def build(self, limit=500):
        try:
            source = sqlite3.connect(self.memory_db, timeout=5)
            rows = source.execute("SELECT ts, kind, text, importance FROM memories ORDER BY id DESC LIMIT ?",
                                  (limit,)).fetchall()
            source.close()
        except sqlite3.Error:
            rows = []
        built = 0
        with self.lock:
            for ts, kind, text, importance in rows:
                layer = self._day(ts or time.time())
                entities = extract_entities(text)
                if not entities:
                    continue
                ids = [self._node(entity, layer=layer) for entity in entities]
                for i, left in enumerate(ids):
                    for right in ids[i + 1:]:
                        self._edge(left, right, "共现", 1.0, layer)
                built += 1
        return {"ok": True, "memories": len(rows), "built": built, **self.stats()}

    def layers(self):
        rows = self.conn.execute(
            "SELECT layer, COUNT(*), SUM(mentions) FROM nodes GROUP BY layer ORDER BY layer DESC").fetchall()
        items = []
        for layer, node_count, mentions in rows:
            edge_count = self.conn.execute("SELECT COUNT(*) FROM edges WHERE layer=?", (layer,)).fetchone()[0]
            top = self.conn.execute(
                "SELECT name, mentions FROM nodes WHERE layer=? ORDER BY mentions DESC LIMIT 3",
                (layer,)).fetchall()
            items.append({
                "layer": layer,
                "label": time.strftime("%m-%d", time.localtime(layer)),
                "nodes": node_count,
                "edges": edge_count,
                "mentions": mentions or 0,
                "top": [{"name": name, "mentions": count} for name, count in top],
            })
        return {"layers": items, "count": len(items)}

    def cross_layers(self, name):
        rows = self.conn.execute(
            "SELECT layer, mentions FROM nodes WHERE name=? ORDER BY layer", (name,)).fetchall()
        return {"name": name,
                "appears_in": [{"label": time.strftime("%m-%d", time.localtime(layer)),
                                "mentions": mentions} for layer, mentions in rows]}

    def stats(self):
        nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        top = self.conn.execute(
            "SELECT n.name, n.mentions FROM nodes n ORDER BY n.mentions DESC LIMIT 5").fetchall()
        return {"nodes": nodes, "edges": edges,
                "top_nodes": [{"name": name, "mentions": mentions} for name, mentions in top]}

    def search(self, query, limit=8, layer=None):
        entities = extract_entities(str(query or ""))
        nodes = []
        query_embedding = self._embed(str(query or ""))
        if layer is not None:
            rows = self.conn.execute(
                "SELECT id, name, type, embedding, mentions FROM nodes WHERE layer=?", (layer,)).fetchall()
        else:
            rows = self.conn.execute("SELECT id, name, type, embedding, mentions FROM nodes").fetchall()
        for node_id, name, node_type, embedding, mentions in rows:
            score = 0.0
            if name in str(query):
                score = 1.0
            elif query_embedding and embedding:
                a = np.frombuffer(query_embedding, dtype=np.float32)
                b = np.frombuffer(embedding, dtype=np.float32)
                score = float(a @ b)
            if score > 0.35:
                nodes.append({"id": node_id, "name": name, "type": node_type,
                              "mentions": mentions, "score": round(score, 3)})
        nodes.sort(key=lambda item: -item["score"])
        nodes = nodes[:limit]
        related = []
        for node in nodes[:3]:
            if layer is not None:
                edges = self.conn.execute(
                    "SELECT dst, relation, weight FROM edges WHERE src=? AND layer=? ORDER BY weight DESC LIMIT 5",
                    (node["id"], layer)).fetchall()
            else:
                edges = self.conn.execute(
                    "SELECT dst, relation, weight FROM edges WHERE src=? ORDER BY weight DESC LIMIT 5",
                    (node["id"],)).fetchall()
            for dst, relation, weight in edges:
                other = self.conn.execute("SELECT name FROM nodes WHERE id=?", (dst,)).fetchone()
                if other:
                    related.append({"from": node["name"], "to": other[0],
                                    "relation": relation, "weight": round(weight, 2)})
        return {"query": str(query or ""), "entities": entities,
                "nodes": nodes, "related": related[:12]}
