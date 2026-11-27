# 认知器官：工作记忆 / 注意闸 / 意图（吸收自大肥鱼 brain 的器官回路思路）
import json
import os
import threading
import time


class Organs:
    def __init__(self, path, working_limit=20):
        self.path = path
        self.working_limit = working_limit
        self.lock = threading.Lock()
        self.state = self._load()
        self.state.setdefault("working", [])
        self.state.setdefault("gate", {"passed": 0, "filtered": 0, "last_reason": None})
        self.state.setdefault("intents", [])

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False)
        except OSError:
            pass

    # ---------- 注意闸 ----------
    def gate(self, event):
        """决定一条外部事件值不值得进入工作记忆：重复的低价值感知直接过滤。"""
        kind = str(event.get("kind") or event.get("type") or "external")
        text = str(event.get("text") or "")[:200]
        intensity = float(event.get("intensity") or 0.5)
        now = time.time()
        with self.lock:
            recent = [item for item in self.state["working"] if now - item["ts"] < 600]
            duplicate = any(item["text"] == text for item in recent)
            low_value = kind in ("battery", "sensing") and intensity < 0.3
            if duplicate or low_value:
                self.state["gate"]["filtered"] = int(self.state["gate"].get("filtered", 0)) + 1
                self.state["gate"]["last_reason"] = "重复或低价值"
                self._save()
                return {"pass": False, "reason": "重复或低价值"}
            self.state["gate"]["passed"] = int(self.state["gate"].get("passed", 0)) + 1
            self.state["gate"]["last_reason"] = "通过"
            self.state["working"].append({
                "ts": now,
                "kind": kind,
                "text": text,
                "salience": round(min(1.0, intensity + (0.2 if kind == "notification" else 0.0)), 2),
            })
            self.state["working"] = self.state["working"][-self.working_limit:]
            self._save()
            return {"pass": True, "reason": "通过"}

    # ---------- 工作记忆 ----------
    def observe(self, event):
        decision = self.gate(event)
        return decision

    def working(self, limit=10):
        with self.lock:
            return self.state["working"][-limit:]

    # ---------- 意图 ----------
    def add_intent(self, text, evidence=None, status="candidate", due_minutes=None):
        text = str(text or "").strip()
        if not text:
            return None
        with self.lock:
            now = time.time()
            intent = {
                "id": "i" + str(int(now * 1000)),
                "text": text[:200],
                "status": status,
                "evidence": list(evidence or []),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "due_at": (now + float(due_minutes) * 60) if due_minutes is not None else None,
                "executed_at": None,
            }
            self.state["intents"].append(intent)
            self.state["intents"] = self.state["intents"][-50:]
            self._save()
            return intent

    def update_intent(self, intent_id, status=None, evidence=None):
        with self.lock:
            for intent in self.state["intents"]:
                if intent["id"] != intent_id:
                    continue
                if status:
                    intent["status"] = status
                if evidence:
                    intent.setdefault("evidence", []).extend(evidence)
                intent["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._save()
                return intent
        return None

    def due_intents(self):
        now = time.time()
        with self.lock:
            return [intent for intent in self.state["intents"]
                    if intent.get("due_at") and not intent.get("executed_at")
                    and float(intent["due_at"]) <= now
                    and intent.get("status") in ("candidate", "active")]

    def execute_intent(self, intent_id):
        with self.lock:
            for intent in self.state["intents"]:
                if intent.get("id") != intent_id:
                    continue
                intent["executed_at"] = time.time()
                intent["status"] = "executed"
                self._save()
                return intent
        return None

    def touch_interaction(self, window=900):
        """用户有互动：最近执行过的意图算完成。"""
        now = time.time()
        changed = []
        with self.lock:
            for intent in self.state["intents"]:
                if intent.get("status") != "executed":
                    continue
                executed = float(intent.get("executed_at") or 0)
                if now - executed <= window:
                    intent["status"] = "done"
                    intent["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    changed.append(intent["id"])
            if changed:
                self._save()
        return changed

    def intents(self, limit=20):
        with self.lock:
            return self.state["intents"][-limit:]

    def summary(self):
        with self.lock:
            return {
                "working": len(self.state["working"]),
                "gate": dict(self.state["gate"]),
                "intents": len(self.state["intents"]),
                "active_intents": [i for i in self.state["intents"] if i.get("status") in ("active", "candidate")],
            }
