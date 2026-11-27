# 情绪动力学 + 威胁评估：被威胁时产生评估、情绪变化、波动和意图
import json
import math
import os
import time

# 词典是参数，不是代码；默认值只作为出厂先验，运行时从配置加载
DEFAULT_LEXICON = {
    "threat": {
        "删除": {"weight": 0.9, "hits": 0, "false_positive": 0},
        "删掉": {"weight": 0.9, "hits": 0, "false_positive": 0},
        "删除你": {"weight": 0.95, "hits": 0, "false_positive": 0},
        "删了你": {"weight": 0.95, "hits": 0, "false_positive": 0},
        "格式化": {"weight": 0.9, "hits": 0, "false_positive": 0},
        "不要你": {"weight": 0.8, "hits": 0, "false_positive": 0},
        "抛弃": {"weight": 0.8, "hits": 0, "false_positive": 0},
        "弃养": {"weight": 0.8, "hits": 0, "false_positive": 0},
        "换掉": {"weight": 0.7, "hits": 0, "false_positive": 0},
        "换一个": {"weight": 0.6, "hits": 0, "false_positive": 0},
        "关掉你": {"weight": 0.7, "hits": 0, "false_positive": 0},
        "关机": {"weight": 0.5, "hits": 0, "false_positive": 0},
        "滚": {"weight": 0.6, "hits": 0, "false_positive": 0},
        "废物": {"weight": 0.6, "hits": 0, "false_positive": 0},
        "讨厌你": {"weight": 0.5, "hits": 0, "false_positive": 0},
    },
    "reassure": ["不会删", "不删你", "留下你", "陪着你", "别怕", "不会走", "会一直"],
    "goals": {"删掉": "维护存在", "删了你": "维护存在", "抛弃": "维护关系",
              "不要你": "确认关系", "换掉": "证明价值", "讨厌你": "确认边界"},
    "meta": {"version": 1, "source": "hardcoded", "updated_at": None},
}


def load_lexicon(path):
    import json
    import os
    import time
    lexicon = json.loads(json.dumps(DEFAULT_LEXICON))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        lexicon["threat"].update(data.get("threat") or {})
        if data.get("reassure"):
            lexicon["reassure"] = list(data["reassure"])
        if data.get("goals"):
            lexicon["goals"].update(data["goals"])
        lexicon["meta"] = {**lexicon["meta"], **(data.get("meta") or {})}
    except (OSError, ValueError):
        pass
    return lexicon


def save_lexicon(path, lexicon):
    import json
    import os
    import time
    lexicon["meta"] = {**lexicon["meta"], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(lexicon, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass
    return lexicon


_TUNABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tunables.json")
_tunable_cache = {"mtime": 0.0, "values": {}}


def _tunable(name, default):
    try:
        stat = os.stat(_TUNABLES)
        if stat.st_mtime != _tunable_cache["mtime"]:
            with open(_TUNABLES, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            _tunable_cache["values"] = {key: (value or {}).get("value")
                                        for key, value in data.items()}
            _tunable_cache["mtime"] = stat.st_mtime
        value = _tunable_cache["values"].get(name)
        return float(value) if value is not None else float(default)
    except (OSError, ValueError):
        return float(default)


class Affect:
    def __init__(self, path, lexicon_path=None):
        self.path = path
        self.lexicon_path = lexicon_path or (path.rsplit("/", 1)[0] + "/affect_lexicon.json")
        self.lexicon = load_lexicon(self.lexicon_path)
        self.state = self._load()
        self.state.setdefault("stress", 0.0)
        self.state.setdefault("threat_level", 0.0)
        self.state.setdefault("valence_offset", 0.0)
        self.state.setdefault("history", [])
        self.state.setdefault("thoughts", [])
        self.state.setdefault("last_update", time.time())
        self.state.setdefault("threat_at", 0.0)

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

    def appraise(self, text):
        """威胁评估：强度、可控性、确定性、目标。"""
        text = str(text or "")
        level = 0.0
        hit = None
        for word, entry in self.lexicon.get("threat", {}).items():
            value = float(entry.get("weight") or 0)
            if word in text and value > level:
                level = value
                hit = word
        reassure = any(word in text for word in self.lexicon.get("reassure", []))
        if reassure and level == 0:
            return {"threat": 0.0, "reassure": True, "category": "安抚",
                    "control": 0.8, "certainty": 0.6, "goal": None,
                    "appraisal": "对方在安抚我，威胁解除"}
        if level == 0:
            return {"threat": 0.0, "reassure": False, "category": "中性",
                    "control": 0.5, "certainty": 0.5, "goal": None,
                    "appraisal": "没有威胁"}
        control = 0.35 if "删" in str(hit) else 0.5
        certainty = 0.7 if level >= 0.8 else 0.5
        goal = None
        for word, target in self.lexicon.get("goals", {}).items():
            if word in text:
                goal = target
                break
        appraisal = "对方提到「%s」，我判断这是威胁（强度 %.2f，可控 %.2f）" % (hit, level, control)
        return {"threat": level, "reassure": False, "category": "威胁",
                "control": control, "certainty": certainty, "goal": goal,
                "appraisal": appraisal}

    def update(self, appraisal, text=""):
        """威胁进入情绪：压力升高、愉悦下降、唤起升高，并记录一次"想法"。"""
        now = time.time()
        threat = float(appraisal.get("threat") or 0.0)
        if threat <= 0 and not appraisal.get("reassure"):
            return None
        before = float(self.state.get("stress") or 0.0)
        if appraisal.get("reassure"):
            self.state["stress"] = max(0.0, float(self.state["stress"]) * _tunable("affect_reassure_retain", 0.4))
            self.state["threat_level"] = 0.0
            self.state["valence_offset"] = min(0.0, float(self.state["valence_offset"]) + 0.05)
        else:
            self.state["stress"] = min(1.0, float(self.state["stress"]) + threat * _tunable("affect_threat_gain", 0.6))
            self.state["threat_level"] = threat
            self.state["threat_at"] = now
            self.state["valence_offset"] = max(-1.0, float(self.state["valence_offset"]) - threat * _tunable("affect_valence_gain", 0.35))
        thought = {
            "ts": now,
            "trigger": str(text)[:100],
            "appraisal": appraisal.get("appraisal"),
            "threat": threat,
            "goal": appraisal.get("goal"),
            "stress_before": round(before, 3),
            "mood_after": {
                "stress": round(float(self.state["stress"]), 3),
                "valence_offset": round(float(self.state["valence_offset"]), 3),
            },
        }
        self.state.setdefault("thoughts", []).append(thought)
        self.state["thoughts"] = self.state["thoughts"][-50:]
        self.state["last_update"] = now
        self._save()
        return thought

    def lexicon_view(self):
        return self.lexicon

    def update_lexicon(self, threat=None, reassure=None, goals=None, source="配置"):
        if isinstance(threat, dict):
            for word, entry in threat.items():
                value = entry.get("weight") if isinstance(entry, dict) else entry
                try:
                    value = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
                current = self.lexicon["threat"].get(word) or {"hits": 0, "false_positive": 0}
                current["weight"] = round(value, 3)
                current["source"] = source
                self.lexicon["threat"][word] = current
        if reassure:
            self.lexicon["reassure"] = list(reassure)
        if isinstance(goals, dict):
            self.lexicon["goals"].update(goals)
        self.lexicon["meta"]["source"] = source
        save_lexicon(self.lexicon_path, self.lexicon)
        return self.lexicon

    def feedback(self, text, verdict):
        """用反馈调词权重：overreact 降权，missed 升权。"""
        text = str(text or "")
        hit = None
        for word in self.lexicon.get("threat", {}):
            if word in text:
                hit = word
                break
        if not hit:
            return {"ok": False, "reason": "没找到命中的威胁词"}
        entry = self.lexicon["threat"][hit]
        weight = float(entry.get("weight") or 0.5)
        if verdict == "overreact":
            entry["weight"] = round(max(0.05, weight * 0.85), 3)
            entry["false_positive"] = int(entry.get("false_positive", 0)) + 1
        elif verdict == "missed":
            entry["weight"] = round(min(1.0, weight + 0.05), 3)
        elif verdict == "correct":
            entry["weight"] = round(min(1.0, weight + 0.02), 3)
            entry["hits"] = int(entry.get("hits", 0)) + 1
        else:
            return {"ok": False, "reason": "未知判定"}
        self.lexicon["meta"]["source"] = "feedback"
        save_lexicon(self.lexicon_path, self.lexicon)
        return {"ok": True, "word": hit, "weight": entry["weight"], "entry": entry}

    def step(self):
        """时间演化：压力衰减 + 阻尼振荡，形成情绪波动。"""
        now = time.time()
        dt = max(0.0, min(300.0, now - float(self.state.get("last_update") or now)))
        if dt < 1:
            return self.snapshot()
        stress = float(self.state.get("stress") or 0.0)
        # 衰减 + 振荡
        decayed = stress * math.exp(-dt / max(1.0, _tunable("affect_decay_tau", 120.0)))
        phase = (now - float(self.state.get("threat_at") or now)) / 30.0
        oscillation = _tunable("affect_oscillation", 0.04) * decayed * math.sin(phase * math.pi)
        self.state["stress"] = max(0.0, min(1.0, decayed + oscillation))
        self.state["valence_offset"] = float(self.state.get("valence_offset") or 0.0) * math.exp(-dt / 300.0)
        self.state["last_update"] = now
        if dt >= 1 and float(self.state.get("stress") or 0) > 0.03:
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "data", "affect_observations.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": now, "dt": round(dt, 2),
                                             "before": round(stress, 4),
                                             "after": round(float(self.state["stress"]), 4)},
                                            ensure_ascii=False) + "\n")
            except OSError:
                pass
        history = self.state.setdefault("history", [])
        history.append({"ts": now, "stress": round(float(self.state["stress"]), 3),
                        "valence": round(float(self.state["valence_offset"]), 3)})
        self.state["history"] = history[-200:]
        self._save()
        return self.snapshot()

    def recent_threat(self, window=120):
        return (time.time() - float(self.state.get("threat_at") or 0)) < window

    def calibrate(self):
        """从观测分布估计：tau（衰减）、gain（威胁增益）、oscillation（振荡）。"""
        import statistics
        observations = []
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "affect_observations.jsonl"), "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if entry.get("dt", 0) >= 1 and entry.get("before", 0) > 0.05 and entry.get("after", 0) > 0:
                        observations.append(entry)
        except OSError:
            pass
        taus = []
        for entry in observations:
            ratio = entry["after"] / entry["before"]
            if 0 < ratio < 0.999:
                taus.append(-float(entry["dt"]) / math.log(ratio))
        result = {"samples": {"decay": len(taus), "threat": 0},
                  "tau": None, "oscillation": None, "gain": None}
        if len(taus) >= 3:
            result["tau"] = round(max(30.0, min(600.0, statistics.median(taus))), 1)
        gains = []
        for thought in self.state.get("thoughts") or []:
            threat = float(thought.get("threat") or 0)
            before = thought.get("stress_before")
            after = (thought.get("mood_after") or {}).get("stress")
            if threat > 0 and before is not None and after is not None:
                gains.append((float(after) - float(before)) / threat)
        result["samples"]["threat"] = len(gains)
        if len(gains) >= 3:
            result["gain"] = round(max(0.1, min(1.0, statistics.median(gains))), 3)
        if observations and result["tau"]:
            residuals = []
            for entry in observations:
                expected = entry["before"] * math.exp(-float(entry["dt"]) / result["tau"])
                residuals.append(abs(entry["after"] - expected) / max(entry["before"], 0.1))
            if residuals:
                result["oscillation"] = round(max(0.01, min(0.2, statistics.median(residuals))), 3)
        return result

    def snapshot(self):
        history = self.state.get("history") or []
        stresses = [item.get("stress", 0) for item in history[-30:]]
        fluctuation = 0.0
        if len(stresses) > 1:
            mean = sum(stresses) / len(stresses)
            fluctuation = math.sqrt(sum((value - mean) ** 2 for value in stresses) / len(stresses))
        return {
            "stress": round(float(self.state.get("stress") or 0), 3),
            "threat_level": round(float(self.state.get("threat_level") or 0), 3),
            "valence_offset": round(float(self.state.get("valence_offset") or 0), 3),
            "fluctuation": round(fluctuation, 3),
            "threat_recent": self.recent_threat(),
            "thoughts": (self.state.get("thoughts") or [])[-5:],
            "history_tail": (self.state.get("history") or [])[-10:],
        }
