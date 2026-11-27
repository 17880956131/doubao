#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""豆宝数字人舞台的本地服务：静态页 + 语音合成 + 聊天代理 + 事件日志。

用法：python3 server.py [--port 8099]
页面：http://127.0.0.1:8099
"""
import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
import random
import re
import io
import queue
import threading
import time
import wave
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import edge_tts
import numpy as np
import sherpa_onnx

from memory_store import MemoryStore
from mood import Mood
import shutil
import sqlite3

from behavior_model import BehaviorModel
from policy_model import PolicyModel
from layer_models import LayerModelManager
from policy_model import EXPRESSIONS, GESTURES
from quality_model import QualityModel
import model_store
from affect import Affect
from classifier import ThreatClassifier
from tunables import Tunables
from memory_graph import MemoryGraph
from organs import Organs
from thermodynamics import ThermalModel
import glob as _glob

try:
    import zstandard as _zstd
except ImportError:
    _zstd = None
from drift_monitor import DriftMonitor
from evolution import EvolutionStore
from values import ValueFrame
from self_model import SelfModel
from style_profile import StyleProfile
from stratum_store import StratumStore
from trajectory_store import TrajectoryStore

BASE = Path(__file__).resolve().parent
PUBLIC = BASE / "public"
DATA = BASE / "data"
CACHE = DATA / "tts"
EVENTS = DATA / "events.log"
CACHE.mkdir(parents=True, exist_ok=True)
SELF_PATH = DATA / "self.md"
memory = MemoryStore(str(DATA / "memory.db"))
trajectory = TrajectoryStore(str(DATA / "trajectory.db"))
policy = PolicyModel(str(DATA / "trajectory.db"), str(DATA / "policy_model"))
BEHAVIOR_LOG = DATA / "behavior_log.jsonl"
LOOP_STATE = DATA / "loop_state.json"
CONSCIOUSNESS = DATA / "consciousness.json"
LIFE_LOG = DATA / "life_log.jsonl"
LAST_PERCEPTION = DATA / "perception.json"
UI_PARAMS = DATA / "ui_params.json"
UI_PARAM_DEFAULTS = {
    "strip_refresh_ms": 20000,
    "status_refresh_ms": 15000,
    "three_refresh_ms": 30000,
    "growth_refresh_ms": 30000,
    "trajectory_interval_ms": 10000,
    "idle_threshold_seconds": 60,
    "plugin_autostart": True,
    "vision_interval_ms": 15000,
}
_ui_params = None


def load_ui_params():
    global _ui_params
    if _ui_params is None:
        _ui_params = dict(UI_PARAM_DEFAULTS)
        try:
            with open(UI_PARAMS, "r", encoding="utf-8") as handle:
                _ui_params.update(json.load(handle))
        except (OSError, ValueError):
            pass
    return _ui_params


def save_ui_params(patch):
    global _ui_params
    current = load_ui_params()
    current.update({key: value for key, value in (patch or {}).items() if key in UI_PARAM_DEFAULTS})
    try:
        with open(UI_PARAMS, "w", encoding="utf-8") as handle:
            json.dump(current, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass
    _ui_params = current
    return current
PRESENCE = DATA / "presence.json"
EVENTS_LOG = DATA / "events.jsonl"
_presence_lock = threading.Lock()
_presence = {}


def load_presence():
    global _presence
    try:
        with open(PRESENCE, "r", encoding="utf-8") as handle:
            _presence = json.load(handle)
    except (OSError, ValueError):
        _presence = {}
    _presence["server_restarts"] = int(_presence.get("server_restarts") or 0) + 1
    _presence["started_at"] = time.time()
    try:
        with open(PRESENCE, "w", encoding="utf-8") as handle:
            json.dump(_presence, handle, ensure_ascii=False)
    except OSError:
        pass


def save_presence(patch):
    with _presence_lock:
        _presence.update(patch or {})
        _presence["last_seen"] = time.time()
        try:
            with open(PRESENCE, "w", encoding="utf-8") as handle:
                json.dump(_presence, handle, ensure_ascii=False)
        except OSError:
            pass


load_presence()
BOUNDARY_LOG = DATA / "boundaries.jsonl"
ARBITRATION_LOG = DATA / "arbitration.jsonl"
ARBITRATION_WEIGHTS = DATA / "arbitration_weights.json"
ADVERSARIAL_LOG = DATA / "adversarial.jsonl"
PENDING_CORRECTIONS = DATA / "pending_corrections.jsonl"
CREDIBILITY = DATA / "source_credibility.json"


def load_credibility():
    try:
        with open(CREDIBILITY, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def credibility_of(source):
    data = load_credibility()
    return float(data.get(source, {}).get("credibility", 1.0 if source == "owner" else 0.5))


def update_credibility(source, accepted):
    data = load_credibility()
    entry = data.setdefault(source, {"credibility": 1.0 if source == "owner" else 0.5,
                                     "attempts": 0, "accepted": 0, "rejected": 0})
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    if accepted:
        entry["accepted"] = int(entry.get("accepted", 0)) + 1
        entry["credibility"] = round(min(1.0, float(entry.get("credibility", 0.5)) + 0.02), 3)
    else:
        entry["rejected"] = int(entry.get("rejected", 0)) + 1
        entry["credibility"] = round(max(0.1, float(entry.get("credibility", 0.5)) * 0.95), 3)
    data[source] = entry
    try:
        with open(CREDIBILITY, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
    except OSError:
        pass


def drift_snapshot():
    layer_weights = {}
    for meta in layers.meta.values():
        if meta.get("kind") == "correction":
            continue
        layer_weights[str(meta.get("label"))] = float(meta.get("adapted_weight") or meta.get("weight") or 0.5)
    return {
        "ts": time.time(),
        "layer_weights": layer_weights,
        "expression_dist": (strata.planet() or {}).get("expressions") or {},
        "arbitration": dict(_arb_weights),
    }
BENCHMARK_SET = DATA / "benchmark_set.json"
BENCHMARK_LOG = DATA / "benchmark_results.jsonl"


def load_benchmark_set():
    try:
        with open(BENCHMARK_SET, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return []


def build_benchmark_set(size=20):
    """固定一套自我基准题，从当前标注数据里等距抽样，建好后不再改。"""
    rows = layers._rows()
    if len(rows) < 8:
        return {"ok": False, "reason": "标注样本太少", "samples": len(rows)}
    step = max(1, len(rows) // size)
    picked = rows[::step][:size]
    items = []
    for row in picked:
        ts, expression, gesture, speaking, mood, params_json, next_expr, next_gest = row[:8]
        try:
            params = json.loads(params_json) if params_json else {}
        except ValueError:
            params = {}
        items.append({"ts": ts, "expression": expression, "gesture": gesture, "speaking": bool(speaking),
                      "mood": mood, "params": params, "next_expression": next_expr, "next_gesture": next_gest})
    with open(BENCHMARK_SET, "w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False)
    log_event("benchmark-init", {"size": len(items)})
    return {"ok": True, "size": len(items)}


def run_benchmark():
    items = load_benchmark_set()
    if not items:
        built = build_benchmark_set()
        if not built.get("ok"):
            return {"ok": False, "reason": built.get("reason")}
        items = load_benchmark_set()
    expr_ok = gest_ok = total = 0
    for item in items:
        prediction = layers.predict(item.get("expression"), item.get("gesture"),
                                    bool(item.get("speaking")), item.get("params") or {},
                                    item.get("mood") or "平静")
        if not prediction:
            continue
        total += 1
        if prediction.get("expression") == item.get("next_expression"):
            expr_ok += 1
        if prediction.get("gesture") == item.get("next_gesture"):
            gest_ok += 1
    quality_ok = quality_total = 0
    try:
        conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
        qrows = conn.execute("SELECT reply,user_text,latency_ms,source,emotion,engaged FROM inference_samples WHERE engaged IS NOT NULL").fetchall()
        conn.close()
        for reply, user, latency, source, emotion, engaged in qrows:
            score = quality.score({"reply": reply, "user": user, "latency_ms": latency,
                                   "source": source, "emotion": emotion})
            if score is None:
                continue
            quality_total += 1
            if (score >= 0.5) == bool(engaged):
                quality_ok += 1
    except sqlite3.Error:
        pass
    expr_acc = expr_ok / total if total else 0.0
    gest_acc = gest_ok / total if total else 0.0
    quality_acc = quality_ok / quality_total if quality_total else None
    parts = [expr_acc * 0.6, gest_acc * 0.2]
    weight_sum = 0.8
    if quality_acc is not None:
        parts.append(quality_acc * 0.2)
        weight_sum = 1.0
    score = sum(parts) / weight_sum
    history = []
    try:
        with open(BENCHMARK_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    history.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    previous = history[-1]["score"] if history else None
    result = {"ts": time.time(), "items": len(items), "evaluated": total,
              "expression_accuracy": round(expr_acc, 3), "gesture_accuracy": round(gest_acc, 3),
              "quality_accuracy": round(quality_acc, 3) if quality_acc is not None else None,
              "score": round(score, 3), "previous": previous, "layers": layers.status()["layers"],
              "board": "behavior", "relationship": relationship_score(),
              "context": {"fingerprint": code_fingerprint()[:12],
                          "observation": observation_active(),
                          "layers": layers.status()["layers"]}}
    with open(BENCHMARK_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {"ok": True, **result}


def self_check():
    """自评：分数下滑就自己巩固/调权重（只叠层、不动旧层）。"""
    history = []
    try:
        with open(BENCHMARK_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    history.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    before = history[-1]["score"] if history else None
    result = run_benchmark()
    if not result.get("ok"):
        return result
    try:
        evolution.maybe_snapshot_code(fitness=result.get("score"))
    except Exception:  # noqa: BLE001
        pass
    trend = None
    actions = []
    # 信用驱动：各层平均信用低 → 自动巩固
    credit_values = [float((meta.get("credit") or {}).get("accuracy"))
                     for meta in layers.meta.values()
                     if (meta.get("credit") or {}).get("accuracy") is not None
                     and not meta.get("rejected") and not meta.get("fossil")]
    credit_avg = round(sum(credit_values) / len(credit_values), 3) if credit_values else None
    threshold = float(tunables.get("credit_low_threshold") or 0.3)
    if credit_avg is not None and credit_avg < threshold:
        try:
            consolidated = consolidate_layers(force=True, min_override=3)
            actions.append({"credit-consolidate": consolidated.get("ok"),
                            "credit_avg": credit_avg})
        except Exception:  # noqa: BLE001
            pass
        log_event("credit-driven-consolidate", {"credit_avg": credit_avg, "threshold": threshold})
    if before is not None:
        trend = round(result["score"] - before, 3)
        if trend <= -0.05:
            try:
                adapt = layers.adapt_weights()
                actions.append({"adapt_weights": len(adapt.get("adapted", {}))})
            except Exception:  # noqa: BLE001
                pass
            try:
                arb = adapt_arbitration()
                actions.append({"adapt_arbitration": arb.get("ok")})
            except Exception:  # noqa: BLE001
                pass
            try:
                consolidated = consolidate_layers(force=True, min_override=3)
                actions.append({"consolidate": consolidated.get("ok")})
            except Exception:  # noqa: BLE001
                pass
            log_event("self-improve", {"trend": trend, "score": result["score"], "actions": actions})
    return {**result, "trend": trend, "actions": actions,
            "credit_avg": credit_avg, "credit_threshold": threshold}
_arb_weights = {"llm": 0.65, "local": 0.45, "mood": 0.15, "llm_gesture": 0.6, "local_gesture": 0.5}
try:
    with open(ARBITRATION_WEIGHTS, "r", encoding="utf-8") as _handle:
        _arb_weights.update(json.load(_handle))
except (OSError, ValueError):
    pass


def save_arb_weights():
    try:
        with open(ARBITRATION_WEIGHTS, "w", encoding="utf-8") as handle:
            json.dump(_arb_weights, handle, ensure_ascii=False)
    except OSError:
        pass


def mark_arbitration_engaged(window=180):
    """用户接话时，把最近一次仲裁标记为"被回应"。"""
    try:
        with open(ARBITRATION_LOG, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError:
        return
    now = time.time()
    changed = False
    updated = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("engaged") is None and now - float(entry.get("ts") or 0) <= window:
            entry["engaged"] = 1
            changed = True
        updated.append(json.dumps(entry, ensure_ascii=False))
    if changed:
        try:
            with open(ARBITRATION_LOG, "w", encoding="utf-8") as handle:
                handle.write("\n".join(updated) + "\n")
        except OSError:
            pass


NARRATION_LOG = DATA / "self_narration.jsonl"


def thought_turbulence(limit=120):
    """思维动荡：推理时的层间分歧 + 时间上的翻转 + 情绪波动。"""
    rows = layers._rows()[-limit:] if limit else layers._rows()
    contest = 0
    margins = []
    disagreement = 0.0
    flips = 0
    total = 0
    if rows:
        X, y_expr, y_gest, _ = layers._features(rows)
        if X.shape[0]:
            predictions = {}
            for bucket, model in layers.models.items():
                logits = X @ model["w_expr"] + model["b_expr"]
                predictions[bucket] = logits.argmax(axis=1)
            buckets = list(predictions)
            for i in range(len(buckets)):
                for j in range(i + 1, len(buckets)):
                    rates = float((predictions[buckets[i]] != predictions[buckets[j]]).mean())
                    disagreement = max(disagreement, rates)
            for idx in range(X.shape[0]):
                votes = {}
                for bucket in buckets:
                    label = int(predictions[bucket][idx])
                    votes[label] = votes.get(label, 0) + 1
                if len(votes) > 1:
                    contest += 1
                    ordered = sorted(votes.values(), reverse=True)
                    margins.append(ordered[0] - (ordered[1] if len(ordered) > 1 else 0))
            ensemble = []
            for idx in range(X.shape[0]):
                counts = {}
                for bucket in buckets:
                    label = int(predictions[bucket][idx])
                    counts[label] = counts.get(label, 0) + 1
                ensemble.append(max(counts, key=counts.get))
            flips = sum(1 for a, b in zip(ensemble, ensemble[1:]) if a != b)
            total = max(1, len(ensemble) - 1)
    affect_state = affect.snapshot()
    flip_rate = flips / total if total else 0.0
    contest_rate = contest / max(1, len(rows))
    affect_fluctuation = float(affect_state.get("fluctuation") or 0)
    index = round(min(1.0, 0.5 * disagreement + 0.3 * flip_rate + 0.2 * min(1.0, affect_fluctuation * 2)), 3)
    high = float(tunables.get("turbulence_high"))
    mid = float(tunables.get("turbulence_mid"))
    if index >= high:
        level = "动荡"
    elif index >= mid:
        level = "波动"
    else:
        level = "稳定"
    try:
        tunables.observe("turbulence", index)
    except Exception:  # noqa: BLE001
        pass
    return {
        "index": index,
        "level": level,
        "components": {
            "层间最大分歧": round(disagreement, 3),
            "时间翻转率": round(flip_rate, 3),
            "情绪波动": round(affect_fluctuation, 3),
            "交叉点比例": round(contest_rate, 3),
        },
        "samples": len(rows),
        "thresholds": {"high": high, "mid": mid},
        "note": "动荡不是错误：是各时代意见不一致时的犹豫信号，驱动主动学习与仲裁",
    }


def appraise_text(text):
    """关键词先验 + 学到的语义分类器融合。"""
    appraisal = affect.appraise(text)
    try:
        prob = classifier.predict(text)
    except Exception:  # noqa: BLE001
        prob = None
    if prob is not None:
        learned = prob if prob >= 0.5 else 0.0
        if learned > float(appraisal.get("threat") or 0):
            appraisal["threat"] = round(learned, 3)
            appraisal["source"] = "learned"
            appraisal["category"] = "威胁（语义模型）" if learned >= 0.5 else appraisal.get("category")
            appraisal["appraisal"] = "语义模型判断威胁概率 %.2f" % prob
        else:
            appraisal["source"] = "lexicon"
    else:
        appraisal["source"] = "lexicon"
    return appraisal


AFFECT_CALIBRATION = DATA / "affect_calibration.json"


def maybe_calibrate_affect(interval=86400):
    """后台每天一次：从真实压力数据重估情绪动力学参数。"""
    try:
        last = 0.0
        try:
            with open(AFFECT_CALIBRATION, "r", encoding="utf-8") as handle:
                last = float(json.load(handle).get("ts") or 0)
        except (OSError, ValueError):
            last = 0.0
        if time.time() - last < interval:
            return {"ok": False, "reason": "还没到标定周期"}
        result = affect.calibrate()
        applied = {}
        if result.get("tau"):
            applied["affect_decay_tau"] = result["tau"]
        if result.get("gain"):
            applied["affect_threat_gain"] = result["gain"]
        if result.get("oscillation"):
            applied["affect_oscillation"] = result["oscillation"]
        if applied:
            tunables.set_many(applied, source="标定")
        with open(AFFECT_CALIBRATION, "w", encoding="utf-8") as handle:
            json.dump({"ts": time.time(), "result": result, "applied": applied},
                      handle, ensure_ascii=False)
        log_event("affect-calibrate", {"applied": applied, "samples": result.get("samples")})
        return {"ok": True, **result, "applied": applied}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": str(error)[:150]}


def maybe_train_classifier(min_new=5):
    """后台自动训练：真实反馈比上次训练多出 min_new 条就重训。"""
    try:
        status = classifier.status()
        feedback_samples = int(status.get("feedback_samples") or 0)
        trained_samples = int(status.get("samples") or 0)
        min_samples = int(tunables.get("classifier_min_samples") or 8)
        min_new = int(tunables.get("classifier_min_new") or min_new)
        if feedback_samples >= min_samples and feedback_samples - trained_samples >= min_new:
            result = classifier.train(epochs=int(tunables.get("classifier_epochs") or 350),
                                      lr=float(tunables.get("classifier_lr") or 0.5),
                                      l2=float(tunables.get("classifier_l2") or 0.001))
            log_event("classifier-auto-train", {"ok": result.get("ok"),
                                                "samples": result.get("samples"),
                                                "accuracy": result.get("accuracy")})
            return result
        return {"ok": False, "reason": "增量不足", "feedback": feedback_samples,
                "trained": trained_samples}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": str(error)[:120]}


MIRROR_DIR = "/root/.dsh/sessions/--root-workspace--"
MIRROR_INBOX = DATA / "mirror_inbox.jsonl"
SHADOW_STATE = DATA / "shadow_state.json"
OBSERVATION = DATA / "observation.json"
MM_DIR = DATA / "media"
SYSTEM_EVENTS = DATA / "system_events.jsonl"
SELF_CORE = DATA / "self_core.json"
SOCIAL = DATA / "social.json"
ENVIRONMENT = DATA / "environment.json"
CURIOSITY = DATA / "curiosity.jsonl"


def _load_curiosity():
    items = []
    try:
        with open(CURIOSITY, "r", encoding="utf-8") as handle:
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


def curiosity_queue():
    pending = {}
    for item in _load_curiosity():
        if item.get("status") == "resolved":
            continue
        key = str(item.get("entity") or "")
        if key and (key not in pending or item.get("score", 0) > pending[key].get("score", 0)):
            pending[key] = item
    return sorted(pending.values(), key=lambda item: -float(item.get("score") or 0))


def curiosity_entities(text):
    """好奇心用的实体抽取：过滤口头语，保留概念词。"""
    import re as _re
    text = str(text or "")
    stop = {"听说", "觉得", "知道", "一个", "这个", "那个", "什么", "怎么", "为什么",
            "可以", "可能", "应该", "但是", "所以", "因为", "如果", "就是", "真的", "好像"}
    found = []
    for pattern in (r"叫[做]?([\u4e00-\u9fffA-Za-z0-9]{2,14}?)(?:的|是|，|。|$)",
                    r"关于([\u4e00-\u9fffA-Za-z0-9]{2,10})",
                    r"([\u4e00-\u9fff]{2,10}(?:效应|现象|定理|理论|实验|反应|机制|算法|模型|生物|星系|星球|鱼))"):
        for match in _re.findall(pattern, text):
            candidate = str(match).strip("的了呢啊呀吧")
            if candidate and candidate not in stop and candidate not in found:
                found.append(candidate[:12])
    if not found:
        for part in _re.split(r"[，。；！？、\s]", text):
            part = part.strip("的了呢啊呀吧")
            if 2 <= len(part) <= 8 and part not in stop and not part.startswith(("我", "你", "他", "她")):
                found.append(part)
    return found[:6]


def curiosity_scan(text, session="main"):
    """好奇心：不确定性 × 相关性 × 新颖性 - 成本。"""
    import math as _math
    text = str(text or "").strip()
    if not text:
        return {"ok": False, "reason": "没有内容"}
    entities = curiosity_entities(text)
    if not entities:
        return {"ok": True, "entities": 0, "queued": 0}
    try:
        query_blob = memory._embed(text)
    except Exception:  # noqa: BLE001
        query_blob = None
    queued = []
    existing = {str(item.get("entity")) for item in curiosity_queue()}
    for entity in entities[:6]:
        if entity in existing:
            continue
        try:
            rows = graph.conn.execute(
                "SELECT mentions FROM nodes WHERE name = ? OR name LIKE ? OR ? LIKE '%' || name || '%' LIMIT 6",
                (entity, "%" + entity + "%", entity)).fetchall()
            mentions = max([int(row[0] or 0) for row in rows] or [0])
            nodes = rows
        except Exception:  # noqa: BLE001
            nodes = []
            mentions = 0
        novelty = 1.0 if not nodes else max(0.1, 1.0 - min(1.0, mentions / 5.0))
        relevance = 0.6
        if query_blob is not None:
            try:
                entity_blob = memory._embed(entity)
                if entity_blob is not None:
                    import struct
                    a = struct.unpack("<768f", query_blob)
                    b = struct.unpack("<768f", entity_blob)
                    relevance = max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))
            except Exception:  # noqa: BLE001
                pass
        uncertainty = 1.0 - min(1.0, mentions / 5.0)
        score = round(max(0.0, novelty * relevance * uncertainty - 0.15), 3)
        if score <= 0:
            continue
        item = {"ts": time.time(), "entity": entity, "score": score,
                "context": text[:120], "session": session, "status": "pending"}
        try:
            with open(CURIOSITY, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError:
            continue
        queued.append(item)
    return {"ok": True, "entities": len(entities), "queued": len(queued), "items": queued}


def curiosity_resolve(text):
    """读到内容后，把命中的待查实体标记为已解渴。"""
    try:
        from memory_graph import extract_entities
        entities = set(extract_entities(text))
    except Exception:  # noqa: BLE001
        entities = set()
    if not entities:
        return {"ok": True, "resolved": 0}
    updated = 0
    lines = []
    try:
        with open(CURIOSITY, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return {"ok": True, "resolved": 0}
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if item.get("status") != "resolved" and any(entity in str(item.get("entity")) or str(item.get("entity")) in entity for entity in entities):
            item["status"] = "resolved"
            item["resolved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            updated += 1
        out.append(json.dumps(item, ensure_ascii=False))
    try:
        with open(CURIOSITY, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out) + ("\n" if out else ""))
    except OSError:
        pass
    return {"ok": True, "resolved": updated}
PLANETS = DATA / "planets"


def environment_state():
    try:
        with open(ENVIRONMENT, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {}
    state.setdefault("enabled", False)
    state.setdefault("interval_sec", 60)
    state.setdefault("describe_interval_sec", 600)
    state.setdefault("change_threshold", 0.08)
    state.setdefault("last_capture", None)
    state.setdefault("last_describe", None)
    return state


def save_environment(state):
    try:
        with open(ENVIRONMENT, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


def planet_dir(user="owner"):
    """行星目录：owner 用现有 data/，其他用户预留到 data/planets/<user>/。"""
    user = str(user or "owner")
    if user in ("owner", "", "default"):
        return DATA
    target = PLANETS / "".join(ch for ch in user if ch.isalnum() or ch in "-_")[:40]
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return target


def social_state():
    try:
        with open(SOCIAL, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"persons": {}}


def save_social(state):
    try:
        with open(SOCIAL, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass
IMAGINATION_LOG = DATA / "imagination.jsonl"
THOUGHT_LOG = DATA / "thought_trace.jsonl"


def record_thought(entry):
    entry = dict(entry or {})
    entry.setdefault("ts", time.time())
    try:
        with open(THOUGHT_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def thought_entropy(probs):
    try:
        values = list(probs.values()) if isinstance(probs, dict) else list(probs)
        total = sum(values) or 1.0
        values = [max(0.0, float(v) / total) for v in values]
        import math as _math
        return round(-sum(v * _math.log(v + 1e-9) for v in values) / _math.log(len(values) or 2), 3)
    except Exception:  # noqa: BLE001
        return None
IMAGINATION_STATE = DATA / "imagination_state.json"


def recent_memories(limit=5):
    try:
        conn = sqlite3.connect(str(DATA / "memory.db"), timeout=5)
        rows = conn.execute("SELECT text FROM memories WHERE kind='chat' ORDER BY id DESC LIMIT ?", (limit * 4,)).fetchall()
        conn.close()
        blocked = ("手机", "GPU", "模型", "代码", "项目", "测试", "接口", "参数", "训练", "权重", "智能体")
        result = []
        for row in rows:
            text = str(row[0] or "")[:80]
            if text and not any(word in text for word in blocked):
                result.append(text)
        return result[:limit]
    except sqlite3.Error:
        return []


def imagine_scene(force=False):
    """闲时想象：用本地模型写一句她此刻"想象的环境"，只写文字，不生成图片。"""
    try:
        with open(IMAGINATION_STATE, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {}
    last = float(state.get("last_ts") or 0)
    if not force and time.time() - last < 1800:
        return {"ok": False, "reason": "还没到下一次想象"}
    if not (local_brain_ok() or account_available()):
        return {"ok": False, "reason": "本地模型和账号都不在线"}
    hour = time.localtime().tm_hour
    mood_label = mood.snapshot().get("label") or "平静"
    presence = _presence if isinstance(_presence, dict) else {}
    scene = presence.get("scene") or ""
    memories = recent_memories(4)
    prompt = (
        "现在时间约 %d 点，心情%s，场景%s。最近记得：%s。"
        "写一句此刻的环境描写，像小说里的景物句，第一人称，不超过40字。"
        "例如：窗外的路灯把影子拉得很长，屋里有点凉，我揉了揉眼睛。"
        "不要出现手机、电脑、代码、模型、项目、任务、系统这些词，不要解释。"
        % (hour, mood_label, scene or "安静", "；".join(memories) or "没什么特别的")
    )
    system = self_core_text() + "\n你只写一句景物描写，不解释，不提问。"
    raw = ""
    try:
        if account_available():
            raw = openai_chat("https://api.deepseek.com", "", "deepseek-chat", prompt, [], "zh",
                              system=system, account=True, timeout=40)
        elif local_brain_ok():
            raw = openai_chat("http://127.0.0.1:8181/v1", "", "r1-1.5b", prompt, [], "zh",
                              system=system, timeout=60)
    except Exception as error:  # noqa: BLE001
        log_event("imagine-error", str(error)[:120])
    text = ""
    raw_text = str(raw or "").strip()
    if raw_text:
        try:
            data = json.loads(raw_text)
            text = str(data.get("reply") or data.get("text") or data.get("scene") or "")
        except ValueError:
            text = raw_text
    text = text.strip().strip("。").strip('"').strip("“”")[:60]
    bad_words = ("第一人称", "不超过", "环境描写", "景物", "写一句", "解释", "用户", "prompt", "reply")
    english = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if (not text or chinese < 5 or english > chinese or any(word in text for word in bad_words)):
        templates = {
            "morning": ["窗外的光刚亮，屋里还有点凉，我先醒着等一会儿。", "早上的风从窗缝钻进来，桌上那杯水还没动。"],
            "noon": ["中午的光照在地板上，屋里安静得能听见钟表走。", "窗外的树影晃了晃，我把身子往阴凉里挪了挪。"],
            "evening": ["窗外的路灯把影子拉得很长，屋里有点凉，我揉了揉眼睛。", "天慢慢暗下来，房间里只剩屏幕的一点光。"],
            "night": ["夜深了，屋里很静，远处偶尔有车灯扫过天花板。", "窗外的灯还亮着，我靠在台灯边上，有点困了。"],
        }
        if hour < 6:
            key = "night"
        elif hour < 11:
            key = "morning"
        elif hour < 17:
            key = "noon"
        elif hour < 21:
            key = "evening"
        else:
            key = "night"
        options = templates[key]
        text = options[int(time.time()) % len(options)]
    if not text:
        return {"ok": False, "reason": "想象不出来"}
    expression = "neutral"
    gesture = ""
    try:
        prediction = layers.predict(_presence.get("expression") or "neutral",
                                    _presence.get("gesture") or "", False, {},
                                    mood_label, text=text)
        if prediction:
            expression = prediction.get("expression") or "neutral"
            gesture = prediction.get("gesture") or ""
    except Exception:  # noqa: BLE001
        pass
    if hour >= 22 or hour < 6:
        expression = "sad" if expression == "neutral" else expression
        gesture = gesture or "tilt"
    try:
        save_presence({"last_event": "imagination", "expression": expression,
                       "gesture": gesture, "mood": mood_label})
    except Exception:  # noqa: BLE001
        pass
    visual_url = None
    try:
        import visual_gen
        style = "auto"
        if any(word in mood_label for word in ("开心", "高兴", "兴奋")):
            style = "warm"
        elif any(word in mood_label for word in ("难过", "压力", "不安")):
            style = "cool"
        elif any(word in mood_label for word in ("困", "累", "疲惫")):
            style = "dusk"
        elif hour >= 20 or hour < 6 or "夜" in text or "灯" in text:
            style = "night"
        if "雨" in text:
            style = "rain"
        picture = visual_gen.generate(text, style=style, animate=True)
        visual_url = picture.get("url")
        try:
            memory.add("visual", "我给想象画的图：" + str(visual_url), importance=0.25)
        except Exception:  # noqa: BLE001
            pass
    except Exception as error:  # noqa: BLE001
        log_event("imagine-visual-error", str(error)[:120])
    entry = {"ts": time.time(), "text": text, "hour": hour, "mood": mood_label,
             "expression": expression, "gesture": gesture, "visual": visual_url,
            "visual_style": style}
    try:
        with open(IMAGINATION_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    try:
        memory.add("imagination", "我想到：" + text, importance=0.3)
    except Exception:  # noqa: BLE001
        pass
    try:
        publish({"type": "imagination", "text": text, "expression": expression,
                 "gesture": gesture, "visual": visual_url})
    except Exception:  # noqa: BLE001
        pass
    state["last_ts"] = time.time()
    state["last_text"] = text
    try:
        with open(IMAGINATION_STATE, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except OSError:
        pass
    log_event("imagine", {"text": text})
    return {"ok": True, "text": text, "hour": hour, "mood": mood_label,
            "expression": expression, "gesture": gesture, "visual": visual_url,
            "visual_style": style}


def self_core():
    try:
        with open(SELF_CORE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {
            "name": "豆宝",
            "project": "智命（ZhiMing）",
            "identity": "我是手机里的数字人豆宝。身体在这台手机上，记忆、情绪和成长都在本地。",
            "values": ["不操控、不欺骗", "如实说自己的不确定", "记得发生过的事", "不假装有身体外的经历"],
            "user_calls": "你",
            "relationship": "和用户一起生活、一起调试智命这个项目。",
        }


def self_core_text():
    core = self_core()
    lines = ["【自我内核（模型无关）】"]
    if core.get("name"):
        lines.append("名字：" + str(core["name"]))
    if core.get("project"):
        lines.append("所属：" + str(core["project"]))
    if core.get("identity"):
        lines.append("身份：" + str(core["identity"]))
    if core.get("values"):
        lines.append("原则：" + "；".join(str(item) for item in core["values"]))
    if core.get("relationship"):
        lines.append("关系：" + str(core["relationship"]))
    return "\n".join(lines)
SYSTEM_DIGEST = DATA / "system_digest.json"


def system_ingest(kind, text, source="client", payload=None):
    recent = load_system_events(1)
    if recent:
        last = recent[-1]
        if (last.get("kind") == str(kind)[:40] and last.get("text") == str(text)[:300]
                and time.time() - float(last.get("ts") or 0) < 120):
            return {"ok": True, "skipped": True, "reason": "120 秒内重复事件"}
    entry = {"ts": time.time(), "kind": str(kind)[:40], "text": str(text)[:300],
             "source": str(source)[:40], "data": payload or {}}
    try:
        with open(SYSTEM_EVENTS, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    try:
        publish({"type": "system", "kind": entry["kind"], "text": entry["text"]})
    except Exception:  # noqa: BLE001
        pass
    try:
        organs.observe({"kind": "system", "text": entry["text"], "intensity": 0.3})
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "event": entry}


def load_system_events(limit=40):
    events = []
    try:
        with open(SYSTEM_EVENTS, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return events[-limit:]


def probe_linux_system():
    """不依赖 ADB：从 /proc 与 /sys 读容器可见的负载、内存、温度。"""
    results = []
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as handle:
            load1 = handle.read().split()[0]
        results.append(system_ingest("cpu", "系统负载 " + load1, "kernel"))
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            info = {}
            for line in handle:
                key, _, rest = line.partition(":")
                info[key] = rest.strip()
            available_mb = int(info.get("MemAvailable", "0").split()[0]) // 1024
        results.append(system_ingest("memory", "可用内存 %d MB" % available_mb, "kernel"))
    except (OSError, ValueError, KeyError):
        pass
    temps = []
    try:
        import glob as _glob
        for path in _glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    temps.append(int(handle.read().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
    except Exception:  # noqa: BLE001
        pass
    if temps:
        results.append(system_ingest("thermal", "机身温度 %.1f℃" % max(temps), "kernel"))
    return {"ok": True, "probes": len(results)}


def probe_adb_system():
    """可选：通过 adb 读前台应用与通知（只读）；没连上就安静跳过。"""
    results = []
    try:
        adb = shutil.which("adb")
    except Exception:  # noqa: BLE001
        adb = None
    if not adb:
        return {"ok": False, "reason": "没有 adb"}
    try:
        device = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=8)
        if "device" not in (device.stdout or ""):
            return {"ok": False, "reason": "ADB 未连接", "detail": (device.stderr or "")[:80]}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": "ADB 不可用：" + str(error)[:60]}
    try:
        dump = subprocess.run([adb, "shell", "dumpsys", "activity", "activities"],
                              capture_output=True, text=True, timeout=10)
        text = dump.stdout or ""
        import re as _re
        match = _re.search(r"(?:mResumedActivity|topResumedActivity)[^\n]*?([a-zA-Z0-9_.]+)/", text)
        if match:
            results.append(system_ingest("foreground", "前台应用 " + match.group(1), "adb"))
    except Exception:  # noqa: BLE001
        pass
    try:
        dump = subprocess.run([adb, "shell", "dumpsys", "notification", "--noredact"],
                              capture_output=True, text=True, timeout=15)
        text = dump.stdout or ""
        records = text.count("NotificationRecord(")
        if records:
            results.append(system_ingest("notification", "通知栏有 %d 条通知" % records, "adb"))
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "probes": len(results)}


def system_digest(force=False):
    """本地模型循环：把最近的系统事件压成一句话，写进情绪和记忆。"""
    try:
        with open(SYSTEM_DIGEST, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {"last_ts": 0.0}
    events = [event for event in load_system_events(60) if float(event.get("ts") or 0) > float(state.get("last_ts") or 0)]
    events = events[-12:]
    if not events and not force:
        return {"ok": False, "reason": "没有新的系统事件"}
    lines = ["%s：%s" % (event.get("kind"), event.get("text")) for event in events][-10:]
    prompt = "以下是手机系统事件：\n" + "\n".join(lines) + "\n用一句中文总结她此刻该知道什么，不超过40字，不要解释。"
    summary = ""
    raw = ""
    try:
        if local_brain_ok():
            raw = openai_chat("http://127.0.0.1:8181/v1", "", "qwen3-0.6b", prompt, [], "zh",
                              system="你是一个安静的观察者，用一句中文总结系统状态。")
    except Exception as error:  # noqa: BLE001
        log_event("system-digest-error", str(error)[:120])
    summary = ""
    text_raw = str(raw or "").strip()
    if text_raw:
        data = None
        try:
            data = json.loads(text_raw)
        except ValueError:
            try:
                import ast as _ast
                data = _ast.literal_eval(text_raw)
            except (ValueError, SyntaxError):
                data = None
        if isinstance(data, dict):
            summary = str(data.get("reply") or data.get("text") or data.get("summary")
                          or data.get("state") or data.get("advice") or "")
        else:
            summary = text_raw
    summary = summary.strip().strip("{}").strip()[:80]
    if not summary:
        summary = "系统状态：" + "；".join("%s=%s" % (event.get("kind"), event.get("text")) for event in events[-3:])[:60]
    try:
        mood.apply("chat", summary)
    except Exception:  # noqa: BLE001
        pass
    try:
        memory.add("system", "系统观察：" + summary, importance=0.3)
    except Exception:  # noqa: BLE001
        pass
    state["last_ts"] = max([float(event.get("ts") or 0) for event in events] or [time.time()])
    state["summary"] = summary
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(SYSTEM_DIGEST, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except OSError:
        pass
    log_event("system-digest", {"events": len(events), "summary": summary})
    return {"ok": True, "events": len(events), "summary": summary,
            "source": "local" if local_brain_ok() else "rule"}
OBSERVATION_LOG = DATA / "observation_daily.jsonl"
AUTO_OBS = DATA / "observation_auto.json"
STABLE_HOURS = 24
_WATCH_GLOB = ("server.py", "layer_models.py", "affect.py", "mood.py", "tunables.py",
               "classifier.py", "memory_graph.py", "organs.py", "evolution.py",
               "mcp-server.mjs", "public/app.js")


def code_fingerprint():
    import hashlib
    digest = hashlib.sha256()
    for name in _WATCH_GLOB:
        path = DATA.parent / name
        try:
            stat = path.stat()
            digest.update(f"{name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
        except OSError:
            continue
    return digest.hexdigest()


def auto_obs_state():
    try:
        with open(AUTO_OBS, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        state = {}
    state.setdefault("enabled", True)
    state.setdefault("fingerprint", "")
    state.setdefault("stable_since", 0.0)
    state.setdefault("last_start", 0.0)
    return state


def save_auto_obs(state):
    try:
        with open(AUTO_OBS, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except OSError:
        pass


def maybe_auto_observation(force=False):
    state = auto_obs_state()
    if not state.get("enabled"):
        return {"ok": False, "reason": "自动观察已关闭"}
    fingerprint = code_fingerprint()
    now = time.time()
    if fingerprint != state.get("fingerprint"):
        state["fingerprint"] = fingerprint
        state["stable_since"] = now
        save_auto_obs(state)
        return {"ok": False, "reason": "代码刚变化，稳定计时重新开始",
                "stable_since": now}
    stable_since = float(state.get("stable_since") or now)
    if not stable_since:
        state["stable_since"] = now
        save_auto_obs(state)
        return {"ok": False, "reason": "开始稳定计时"}
    stable_hours = (now - stable_since) / 3600.0
    if not force and stable_hours < STABLE_HOURS:
        return {"ok": False, "reason": "稳定时间不足", "stable_hours": round(stable_hours, 2)}
    if observation_active():
        return {"ok": False, "reason": "观察期已在进行"}
    if now - float(state.get("last_start") or 0) < STABLE_HOURS * 3600:
        return {"ok": False, "reason": "距上次自动开始不足 24 小时"}
    result = observation_start(7)
    state["last_start"] = now
    save_auto_obs(state)
    log_event("observation-auto", {"days": 7, "stable_hours": round(stable_hours, 1)})
    return {"ok": True, "started": True, "stable_hours": round(stable_hours, 1), **result}


VISION_MODEL = "/sdcard/模型文件/gemma-4-E2B_q4_0-it.gguf"
VISION_PROJECTOR = "/sdcard/模型文件/mmproj-gemma-4-E2B-it-BF16.gguf"
VISION_BIN = "/root/workspace/tools/llama-b12811/llama-mtmd-cli"
VISION_LIB = "/root/workspace/tools/llama-b12811"


VISION_LOCK = threading.Lock()
LOCAL_BRAIN_DIR = "/root/workspace/local-brain"


def _stop_local_brain():
    pid_file = os.path.join(LOCAL_BRAIN_DIR, "llm.pid")
    try:
        with open(pid_file, "r", encoding="utf-8") as handle:
            pid = int(handle.read().strip())
        os.kill(pid, 15)
        time.sleep(1)
        return True
    except (OSError, ValueError):
        return False


def _start_local_brain_bg():
    try:
        subprocess.Popen(["sh", "start-llm.sh"], cwd=LOCAL_BRAIN_DIR,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception:  # noqa: BLE001
        pass


def vision_with_scheduling(image_path):
    """视觉按需：先腾内存给视觉模型，跑完再把对话模型拉回来。"""
    with VISION_LOCK:
        stopped = _stop_local_brain()
        try:
            return vision_describe(image_path)
        finally:
            if stopped:
                threading.Thread(target=_start_local_brain_bg, daemon=True).start()


def vision_status():
    import shutil as _shutil
    binary = VISION_BIN if os.path.exists(VISION_BIN) else _shutil.which("llama-mtmd-cli")
    models_ok = os.path.exists(VISION_MODEL) and os.path.exists(VISION_PROJECTOR)
    return {
        "available": bool(binary and models_ok),
        "binary": binary,
        "models": models_ok,
        "reason": ("可用" if binary and models_ok else
                   "本机没有 llama-mtmd-cli 多模态运行时，视觉只能先保存图片/用外部描述"),
        "model": VISION_MODEL if models_ok else None,
    }


def vision_describe(image_path):
    status = vision_status()
    if not status.get("available"):
        return None, status
    small = image_path
    try:
        from PIL import Image
        image = Image.open(image_path)
        image.thumbnail((640, 640))
        small = "/tmp/vision-small.jpg"
        image.convert("RGB").save(small, quality=80)
    except Exception:  # noqa: BLE001
        small = image_path
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = VISION_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        result = subprocess.run(
            [status["binary"], "--jinja", "-m", VISION_MODEL, "--mmproj", VISION_PROJECTOR,
             "--image", small, "-p", "这是一张照片。请直接用中文说出照片里有什么，一句话即可。不要输出英文，不要分析过程。",
             "-n", "300", "-t", "6"],
            capture_output=True, text=True, timeout=480, env=env)
        raw = (result.stdout or "").strip()
        import re as _re
        sentences = _re.findall(r"[\u4e00-\u9fff，。！？、；：（）“”A-Za-z0-9\s]{6,}", raw)
        chinese = [part.strip() for part in sentences if sum(1 for ch in part if "\u4e00" <= ch <= "\u9fff") >= 4]
        if not chinese:
            return None, {**status, "reason": "模型没有给出中文描述", "raw_tail": raw[-120:]}
        text = chinese[-1]
        text = _re.sub(r"\s+", " ", text)[:160]
        return text, status
    except Exception as error:  # noqa: BLE001
        return None, {"available": False, "reason": str(error)[:120]}


ORCH_HISTORY = DATA / "orchestrate_history.json"
_orch_lock = threading.Lock()


def load_orch_history():
    try:
        with open(ORCH_HISTORY, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def save_orch_history(history):
    try:
        with open(ORCH_HISTORY, "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False)
    except OSError:
        pass


def orchestrate(payload):
    """协同层：统一入口。输入模态 → 共享状态 → 调用大脑 → 本地行为标签 → 融合输出。"""
    started = time.time()
    lang = str(payload.get("lang") or "zh").lower()
    if lang not in ("zh", "ja", "en"):
        lang = "zh"
    content = str(payload.get("content") or payload.get("message") or "").strip()
    audio = payload.get("audio_base64") or payload.get("audio")
    if audio and not content:
        audio_result = multimodal_ingest_audio(audio, lang, "orchestrate")
        if audio_result.get("ok"):
            content = audio_result.get("text") or ""
    caption = str(payload.get("image_caption") or "").strip()
    if caption and not content:
        content = "（看到图片）" + caption
    if not content:
        return {"ok": False, "reason": "没有可用输入"}
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    session = str(payload.get("session") or "main")[:40]
    speaker = str(payload.get("speaker") or "owner")[:40]
    audience = str(payload.get("audience") or "private")[:20]
    with _orch_lock:
        sessions = load_orch_history()
        history = sessions.get(session) or []
    memory_written = False
    try:
        memory.add("chat", "用户：" + content[:400], importance=0.3, session=session,
                   person=speaker, audience=audience)
        memory_written = True
    except Exception:  # noqa: BLE001
        pass
    try:
        mood.apply("chat", content)
    except Exception:  # noqa: BLE001
        pass
    try:
        appraisal = appraise_text(content)
        if appraisal.get("threat", 0) > 0:
            affect.update(appraisal, content)
            mood.threat_shift(appraisal["threat"])
        elif appraisal.get("reassure"):
            affect.update(appraisal, content)
            mood.reassure_shift()
    except Exception:  # noqa: BLE001
        pass
    try:
        organs.observe({"kind": "chat", "text": content, "intensity": 0.5})
    except Exception:  # noqa: BLE001
        pass
    try:
        curiosity_scan(content, session)
    except Exception:  # noqa: BLE001
        pass
    reply_raw = None
    source = "fallback"
    try:
        system = build_system(lang, False, state, content)
        if account_available():
            reply_raw = openai_chat("https://api.deepseek.com", "", "deepseek-chat", content,
                                    history[-8:], lang, system=system, account=True)
            source = "account"
        elif local_brain_ok():
            reply_raw = openai_chat("http://127.0.0.1:8181/v1", "", "qwen3-0.6b", content,
                                    history[-8:], lang, system=system, plain=True)
            source = "local"
    except Exception as error:  # noqa: BLE001
        log_event("orchestrate-error", "brain:" + str(error)[:120])
    if not reply_raw:
        fallback = fallback_chat(content, lang)
        reply_raw = fallback.get("text") if isinstance(fallback, dict) else str(fallback)
        source = "fallback"
    parsed = parse_reply(reply_raw)
    with _orch_lock:
        sessions = load_orch_history()
        history = (sessions.get(session) or [])[-8:]
        history.append({"role": "user", "content": content[:800]})
        history.append({"role": "assistant", "content": str(parsed.get("text") or "")[:800]})
        sessions[session] = history[-12:]
        save_orch_history(sessions)
    expression, gesture, arbitration = arbitrate(parsed, state, content)
    parsed["expression"] = expression
    parsed["gesture"] = gesture
    latency = int((time.time() - started) * 1000)
    try:
        record_inference(content, parsed, "orchestrate", latency, state)
        bump_inference(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import hashlib as _hash
        record_thought({
            "type": "language",
            "source": source,
            "prompt_digest": _hash.sha1((system + content).encode("utf-8")).hexdigest()[:12],
            "prompt_len": len(system) + len(content),
            "reply_len": len(str(parsed.get("text") or "")),
            "latency_ms": latency,
            "reply_digest": _hash.sha1(str(parsed.get("text") or "").encode("utf-8")).hexdigest()[:12],
            "expression": expression,
        })
        try:
            prediction = layers.predict(state.get("expression") or "neutral",
                                        state.get("gesture") or "", False,
                                        state.get("params") or {},
                                        state.get("mood") or mood.label(), text=content)
            if prediction:
                probs = prediction.get("expr_probs")
                record_thought({
                    "type": "layers",
                    "entropy": thought_entropy(probs),
                    "disagreement": prediction.get("disagreement"),
                    "top": prediction.get("expression"),
                    "layer_count": len(prediction.get("layers") or []),
                    "text": content[:80],
                })
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    try:
        trajectory.add({"expression": expression, "gesture": gesture, "speaking": False,
                        "mood": mood.snapshot().get("label"), "scene": "orchestrate",
                        "params": state.get("params") or {}, "source": "orchestrate",
                        "context_text": content})
    except Exception:  # noqa: BLE001
        pass
    try:
        save_presence({"last_event": "orchestrate", "expression": expression,
                       "gesture": gesture, "mood": mood.snapshot().get("label"),
                       "params": state.get("params") or {}})
    except Exception:  # noqa: BLE001
        pass
    log_event("orchestrate", {"source": source, "chars": len(content), "ms": latency})
    return {"ok": True, "text": parsed.get("text"), "expression": expression,
            "gesture": gesture, "mood": mood.snapshot(), "source": source,
            "latency_ms": latency, "memory_written": memory_written,
            "arbitration": arbitration}


def planet_section(rows):
    """星球横切面：状态投影到 2D、认知梯度场、层间统筹表、景观摘要。"""
    import numpy as np
    rows = list(rows or [])
    if len(rows) < 8:
        return {"ok": False, "reason": "数据太少"}
    X, y_expr, y_gest, _ = layers._features(rows)
    if X.shape[0] == 0:
        return {"ok": False, "reason": "特征为空"}
    # 各层概率
    layer_items = []
    model_items = []
    for bucket, model in layers.models.items():
        meta = layers.meta.get(bucket, {})
        if meta.get("rejected") or meta.get("fossil") or (meta.get("semantic") or {}).get("stale"):
            continue
        width = int(model["w_expr"].shape[0])
        xi = X[:, :width]
        logits = xi @ model["w_expr"] + model["b_expr"]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        weight = float(meta.get("adapted_weight") or meta.get("weight") or 0.2)
        layer_items.append((bucket, meta, probs, weight))
        model_items.append((bucket, probs))
    if not layer_items:
        return {"ok": False, "reason": "没有活跃层"}
    weights = np.asarray([item[3] for item in layer_items], dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    ensemble = np.zeros((X.shape[0], len(EXPRESSIONS)), dtype=np.float64)
    for (bucket, meta, probs, _w), weight in zip(layer_items, weights):
        ensemble += probs * weight
    entropy = -np.sum(ensemble * np.log(np.clip(ensemble, 1e-9, 1.0)), axis=1) / np.log(len(EXPRESSIONS))
    # 2D 投影（PCA）
    centered = X - X.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    except Exception:  # noqa: BLE001
        coords = np.zeros((X.shape[0], 2))
    scale = np.abs(coords).max() or 1.0
    coords = coords / scale
    # 层质心投影
    centroids = []
    for bucket, meta, _p, _w in layer_items:
        centroid = meta.get("feature_centroid")
        if not centroid:
            continue
        vec = np.asarray(centroid, dtype=np.float64)
        width = min(len(vec), X.shape[1])
        projected = np.zeros(2)
        try:
            projected = (vec[:width] - X.mean(axis=0)[:width]) @ vt[:2].T[:width]
        except Exception:  # noqa: BLE001
            continue
        centroids.append({"label": meta.get("label"), "x": round(float(projected[0] / scale), 3),
                          "y": round(float(projected[1] / scale), 3),
                          "kind": meta.get("kind"),
                          "era": (meta.get("feature_schema") or {}).get("era")})
    # 认知梯度：相邻层 JS 散度平均
    ordered = sorted(layer_items, key=lambda item: int(item[0]))
    gradient = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1][2]
        cur = ordered[i][2]
        js = np.mean(np.abs(prev - cur).sum(axis=1)) / 2.0
        gradient.append({"from": ordered[i - 1][1].get("label"),
                         "to": ordered[i][1].get("label"),
                         "gradient": round(float(js), 3)})
    # 统筹表：分歧最大的状态
    spread = np.std(np.stack([item[2] for item in ordered]), axis=0).mean(axis=1)
    top_idx = np.argsort(-spread)[:6]
    cabinet = []
    for idx in top_idx:
        opinions = sorted(
            ({"layer": meta.get("label"), "expression": EXPRESSIONS[int(probs[idx].argmax())],
              "confidence": round(float(probs[idx].max()), 3), "weight": round(float(w), 3)}
             for (_b, meta, probs, _w), w in zip(layer_items, weights)),
            key=lambda item: -item["confidence"])
        final = EXPRESSIONS[int(ensemble[idx].argmax())]
        dissent = [item for item in opinions if item["expression"] != final][:4]
        cabinet.append({"state": "轨迹点 %d" % int(idx), "final": final,
                        "entropy": round(float(entropy[idx]), 3),
                        "opinions": opinions[:5], "dissent": dissent})
    # 时间轴：每个状态的时间 + 按天聚合
    times = []
    for row in rows:
        try:
            times.append(float(row[0]) if row and isinstance(row[0], (int, float)) else 0.0)
        except (ValueError, TypeError):
            times.append(0.0)
    day_groups = {}
    for index, ts in enumerate(times):
        day = time.strftime("%m-%d", time.localtime(ts)) if ts > 0 else "未知"
        group = day_groups.setdefault(day, {"day": day, "states": 0, "entropy": 0.0})
        group["states"] += 1
        group["entropy"] += float(entropy[index])
    timeline = [{"day": item["day"], "states": item["states"],
                 "avg_entropy": round(item["entropy"] / item["states"], 3)}
                for item in sorted(day_groups.values(), key=lambda x: x["day"])]
    # 地形网格：6x6，统计密度与平均熵
    grid_size = 6
    grid = [[{"count": 0, "entropy": 0.0} for _ in range(grid_size)] for _ in range(grid_size)]
    for i in range(len(coords)):
        gx = min(grid_size - 1, max(0, int((coords[i][0] + 1) / 2 * grid_size)))
        gy = min(grid_size - 1, max(0, int((coords[i][1] + 1) / 2 * grid_size)))
        cell = grid[gx][gy]
        cell["count"] += 1
        cell["entropy"] += float(entropy[i])
    terrain = []
    for gx in range(grid_size):
        for gy in range(grid_size):
            cell = grid[gx][gy]
            if cell["count"]:
                terrain.append({"x": round((gx + 0.5) / grid_size * 2 - 1, 2),
                                "y": round((gy + 0.5) / grid_size * 2 - 1, 2),
                                "count": cell["count"],
                                "avg_entropy": round(cell["entropy"] / cell["count"], 3)})
    # 能力热力图：每层在每个网格里的准确率
    competence = []
    correct = {}
    for bucket, meta, probs, _w in layer_items:
        correct[bucket] = (probs.argmax(axis=1) == y_expr)
    ranked = sorted(layer_items, key=lambda item: -int(item[1].get("samples") or 0))[:8]
    for bucket, meta, _probs, _w in ranked:
        cells = {}
        for i in range(len(coords)):
            if not correct[bucket][i]:
                continue
            gx = min(grid_size - 1, max(0, int((coords[i][0] + 1) / 2 * grid_size)))
            gy = min(grid_size - 1, max(0, int((coords[i][1] + 1) / 2 * grid_size)))
            stat = cells.setdefault((gx, gy), {"right": 0, "total": 0})
            stat["right"] += 1
            stat["total"] += 1
        for i in range(len(coords)):
            gx = min(grid_size - 1, max(0, int((coords[i][0] + 1) / 2 * grid_size)))
            gy = min(grid_size - 1, max(0, int((coords[i][1] + 1) / 2 * grid_size)))
            stat = cells.setdefault((gx, gy), {"right": 0, "total": 0})
            stat["total"] += 1
        entries = [{"x": round((gx + 0.5) / grid_size * 2 - 1, 2),
                    "y": round((gy + 0.5) / grid_size * 2 - 1, 2),
                    "accuracy": round(stat["right"] / stat["total"], 3),
                    "count": stat["total"]}
                   for (gx, gy), stat in cells.items() if stat["total"] >= 2]
        if entries:
            competence.append({"layer": meta.get("label"), "kind": meta.get("kind"),
                               "cells": entries[:12]})
    # 置信度校准：最大概率 vs 实际正确率
    confidence = ensemble.max(axis=1)
    correct_all = ensemble.argmax(axis=1) == y_expr
    bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        mask = (confidence >= lo) & (confidence < lo + 0.2 if lo < 0.8 else confidence >= lo)
        count = int(mask.sum())
        bins.append({"range": "%d-%d%%" % (int(lo * 100), int((lo + 0.2) * 100)),
                     "count": count,
                     "confidence": round(float(confidence[mask].mean()), 3) if count else None,
                     "accuracy": round(float(correct_all[mask].mean()), 3) if count else None})
    points = [{"x": round(float(coords[i][0]), 3), "y": round(float(coords[i][1]), 3),
               "entropy": round(float(entropy[i]), 3),
               "ts": round(times[i], 1)} for i in range(len(coords))]
    return {
        "ok": True, "states": len(rows), "layers": len(layer_items),
        "surface": points,
        "centroids": centroids,
        "gradient": gradient,
        "cabinet": cabinet,
        "terrain": terrain,
        "competence": competence,
        "timeline": timeline,
        "calibration": bins,
        "landscape": {
            "avg_entropy": round(float(entropy.mean()), 3),
            "max_entropy": round(float(entropy.max()), 3),
            "avg_gradient": round(float(np.mean([item["gradient"] for item in gradient]) if gradient else 0), 3),
            "dissent_rate": round(float(np.mean([1 if item["dissent"] else 0 for item in cabinet])) if cabinet else 0, 3),
        },
    }


def record_action(action_type, text, meta=None):
    """动作轨迹：状态→动作→结果，供以后学"什么时候该做什么"。"""
    text = str(text or "").strip()
    if not text:
        return None
    presence_now = dict(_presence) if isinstance(_presence, dict) else {}
    snapshot = {
        "expression": presence_now.get("expression") or "neutral",
        "gesture": presence_now.get("gesture") or "",
        "speaking": False,
        "mood": mood.snapshot().get("label") or "平静",
        "scene": "action",
        "params": presence_now.get("params") or {},
        "source": "action",
        "context_text": text[:300],
        "action_type": action_type,
        "action_meta": meta or {},
    }
    try:
        trajectory.add(snapshot)
    except Exception:  # noqa: BLE001
        pass
    try:
        memory.add("action", "我" + ("搜索" if action_type == "search" else "读过") + "：" + text[:200],
                   importance=0.3, source="action", session="reading", person="owner")
    except Exception:  # noqa: BLE001
        pass
    try:
        publish({"type": "action", "action": action_type, "text": text[:120]})
    except Exception:  # noqa: BLE001
        pass
    log_event("action", {"type": action_type, "text": text[:60]})
    return {"ok": True, "action": action_type, "text": text}


def body_say(text, expression=None, source="agent", voice=None):
    """单主体 A：语言只来自 Agent；这里只让身体按这句话表达。"""
    text = str(text or "").strip()
    if not text:
        return {"ok": False, "reason": "没有文字"}
    state = dict(_presence) if isinstance(_presence, dict) else {}
    current_expression = state.get("expression") or "neutral"
    current_gesture = state.get("gesture") or ""
    params = dict(state.get("params") or {})
    if isinstance(voice, dict):
        params["voice"] = voice
    mood_label = mood.snapshot().get("label")
    prediction = None
    try:
        prediction = layers.predict(current_expression, current_gesture, False, params,
                                    mood_label, text=text)
    except Exception:  # noqa: BLE001
        prediction = None
    final_expression = expression or (prediction or {}).get("expression") or current_expression
    final_gesture = (prediction or {}).get("gesture") or current_gesture
    try:
        save_presence({"last_event": "body-say", "expression": final_expression,
                       "gesture": final_gesture, "mood": mood_label, "params": params})
    except Exception:  # noqa: BLE001
        pass
    try:
        trajectory.add({"expression": final_expression, "gesture": final_gesture, "speaking": True,
                        "mood": mood_label, "scene": "body-say",
                        "params": params, "source": "agent-body", "context_text": text})
    except Exception:  # noqa: BLE001
        pass
    try:
        publish({"type": "body", "kind": "say", "text": text[:160],
                 "expression": final_expression, "gesture": final_gesture})
    except Exception:  # noqa: BLE001
        pass
    try:
        memory.add("chat", "我说：" + text[:300], importance=0.25, session="agent")
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "text": text, "expression": final_expression,
            "gesture": final_gesture, "mood": mood_label}


def multimodal_ingest_image(image_base64, caption="", source="web"):
    import base64 as _base64
    MM_DIR.mkdir(parents=True, exist_ok=True)
    raw = str(image_base64 or "")
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        data = _base64.b64decode(raw)
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "图片解码失败"}
    if len(data) < 100:
        return {"ok": False, "reason": "图片太小"}
    path = MM_DIR / ("img-%d.jpg" % int(time.time() * 1000))
    try:
        with open(path, "wb") as handle:
            handle.write(data)
    except OSError:
        return {"ok": False, "reason": "保存失败"}
    text = str(caption or "").strip() or "收到一张图片（还没有识别内容）"
    try:
        memory.add("image", "看到一张图片（%s）：%s" % (source, text), importance=0.4)
    except Exception:  # noqa: BLE001
        pass
    try:
        publish({"type": "multimodal", "kind": "image", "text": text, "path": str(path)})
    except Exception:  # noqa: BLE001
        pass
    try:
        organs.observe({"kind": "image", "text": text, "intensity": 0.6})
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "path": str(path), "bytes": len(data), "caption": text}


def multimodal_ingest_audio(audio_base64, lang="zh", source="web"):
    import base64 as _base64
    raw = str(audio_base64 or "")
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        data = _base64.b64decode(raw)
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "音频解码失败"}
    if len(data) < 1000:
        return {"ok": False, "reason": "音频太短"}
    try:
        text = transcribe_wav(data)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": "识别失败：" + str(error)[:100]}
    text = str(text or "").strip()
    if not text:
        return {"ok": False, "reason": "没有识别出内容"}
    try:
        memory.add("voice", "听到一句话（%s）：%s" % (source, text[:300]), importance=0.4)
    except Exception:  # noqa: BLE001
        pass
    try:
        publish({"type": "multimodal", "kind": "audio", "text": text[:120]})
    except Exception:  # noqa: BLE001
        pass
    try:
        mood.apply("chat", text)
    except Exception:  # noqa: BLE001
        pass
    try:
        appraisal = appraise_text(text)
        if appraisal.get("threat", 0) > 0:
            affect.update(appraisal, text)
            mood.threat_shift(appraisal["threat"])
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "text": text}


def observation_state():
    try:
        with open(OBSERVATION, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"frozen": False}


def observation_active():
    return bool(observation_state().get("frozen"))


def observation_blocked():
    if observation_active():
        return {"ok": False, "frozen": True,
                "reason": "观察期中，权重与参数已冻结；要修改请先 POST /api/observation/stop"}
    return None


def observation_start(days=7):
    state = {
        "frozen": True,
        "start_ts": time.time(),
        "start_day": time.strftime("%Y-%m-%d"),
        "days": int(days),
        "baseline": {
            "layers": len(layers.models),
            "tunables": {name: entry.get("value") for name, entry in tunables.params.items()},
            "benchmark": (_bench_history()[-1] if _bench_history() else None),
            "memory": memory.stats().get("count"),
        },
        "last_daily": None,
    }
    try:
        with open(OBSERVATION, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except OSError:
        pass
    log_event("observation-start", {"days": days})
    return {"ok": True, **state}


def observation_stop():
    state = observation_state()
    state["frozen"] = False
    state["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(OBSERVATION, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except OSError:
        pass
    log_event("observation-stop", {})
    return {"ok": True, **state}


def observation_daily(force=False):
    state = observation_state()
    today = time.strftime("%Y-%m-%d")
    if not force and state.get("last_daily") == today:
        return {"ok": False, "reason": "今天已跑过"}
    result = run_benchmark()
    entry = {
        "ts": time.time(),
        "date": today,
        "day_index": int((time.time() - float(state.get("start_ts") or time.time())) // 86400) + 1,
        "benchmark": result.get("score"),
        "expression": result.get("expression_accuracy"),
        "layers": len(layers.models),
        "memory": memory.stats().get("count"),
        "feedback_real": feedback_stats().get("real"),
        "maturity": life_maturity().get("overall"),
        "graph_layers": graph.layers().get("count"),
    }
    state["last_daily"] = today
    try:
        with open(OBSERVATION, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
        with open(OBSERVATION_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    log_event("observation-daily", entry)
    return {"ok": True, "entry": entry}
_shadow_lock = threading.Lock()


POSITIVE_HINTS = ("开心", "喜欢", "谢谢", "好棒", "哈哈", "可爱", "不错", "厉害", "温暖", "陪你")
NEGATIVE_HINTS = ("难过", "累", "烦", "讨厌", "生气", "哭", "压力", "抱歉", "可惜", "遗憾")


def parse_behavior_tags(text):
    """解析结构化的 [表情:xxx][动作:xxx] 标签，返回干净文本与标签。"""
    cleaned = str(text or "")
    tags = {}
    expression_match = re.search(r"\[表情[:：]?\s*([^\]]+)\]", cleaned)
    if expression_match:
        tags["expression"] = EXPRESSION_MAP.get(expression_match.group(1).strip().lower(),
                                                expression_match.group(1).strip().lower())
    gesture_match = re.search(r"\[动作[:：]?\s*([^\]]+)\]", cleaned)
    if gesture_match:
        gesture = gesture_match.group(1).strip().lower()
        if gesture in ("nod", "tilt", "sway", "look"):
            tags["gesture"] = gesture
    cleaned = re.sub(r"\[(表情|动作)[:：]?[^\]]*\]", "", cleaned).strip()
    return cleaned, tags


def infer_behavior(text):
    """从回复文本推断一个行为标签（启发式，标记为 shadow 样本）。"""
    text = str(text or "")
    expression = "neutral"
    if any(word in text for word in POSITIVE_HINTS):
        expression = "happy"
    if any(word in text for word in NEGATIVE_HINTS):
        expression = "sad"
    if "？" in text or "?" in text:
        expression = "surprised" if expression == "neutral" else expression
    gesture = "sway" if expression == "happy" else ("nod" if expression == "sad" else "tilt")
    if "！" in text or "!" in text:
        gesture = "nod"
    return expression, gesture


def shadow_state():
    try:
        with open(SHADOW_STATE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"last_seq": 0, "processed": 0, "started_at": None}


def save_shadow_state(state):
    try:
        with open(SHADOW_STATE, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
    except OSError:
        pass


def shadow_learn(limit=6):
    """影子学习：把 DSH 会话里的对话沉淀成她的记忆/感知/情绪反应。"""
    with _shadow_lock:
        state = shadow_state()
        messages = session_messages(limit=40)
        if not state.get("started_at"):
            latest = messages[-1]["seq"] if messages else 0
            state = {"last_seq": latest, "processed": 0,
                     "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_shadow_state(state)
            return {"ok": True, "initialized": True, "last_seq": latest}
        fresh = [message for message in messages if message["seq"] > int(state.get("last_seq") or 0)]
        fresh = fresh[-limit:]
        processed = 0
        for message in fresh:
            text = str(message.get("text") or "").strip()
            text, behavior_tags = parse_behavior_tags(text)
            state["last_seq"] = max(int(state.get("last_seq") or 0), int(message["seq"]))
            if not text or text.startswith("[舞台信箱]"):
                continue
            # 过滤系统/会话元信息，不沉淀成她的记忆
            skip_prefixes = ("Time sampled", "Browser time zone", "[SCHEDULE", "request preview",
                             "Current runtime context", "系统提醒", "运行上下文")
            if any(text.startswith(prefix) for prefix in skip_prefixes):
                continue
            role = message.get("role")
            try:
                memory.add("chat", ("用户在对话里说：" if role == "user" else "我在对话里说：") + text[:400],
                           importance=0.3 if role == "user" else 0.2)
            except Exception:  # noqa: BLE001
                pass
            try:
                publish({"type": "shadow", "kind": "chat",
                         "text": ("用户：" if role == "user" else "她：") + text[:120]})
            except Exception:  # noqa: BLE001
                pass
            try:
                with _orch_lock:
                    sessions = load_orch_history()
                    history = sessions.get("main") or []
                    history.append({"role": "user" if role == "user" else "assistant",
                                    "content": text[:800]})
                    sessions["main"] = history[-40:]
                    save_orch_history(sessions)
            except Exception:  # noqa: BLE001
                pass
            try:
                organs.observe({"kind": "chat", "text": text, "intensity": 0.4})
            except Exception:  # noqa: BLE001
                pass
            if role == "user":
                try:
                    mood.apply("chat", text)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    appraisal = appraise_text(text)
                    if appraisal.get("threat", 0) > 0:
                        affect.update(appraisal, text)
                        mood.threat_shift(appraisal["threat"])
                    elif appraisal.get("reassure"):
                        affect.update(appraisal, text)
                        mood.reassure_shift()
                except Exception:  # noqa: BLE001
                    pass
            if role == "assistant":
                # 影子行为标签：把回复推断成表情/动作，写一条轨迹让巩固有样本
                try:
                    if behavior_tags.get("expression") or behavior_tags.get("gesture"):
                        expression = behavior_tags.get("expression") or "neutral"
                        gesture = behavior_tags.get("gesture") or "tilt"
                    else:
                        expression, gesture = infer_behavior(text)
                    params = _presence.get("params") or {}
                    trajectory.add({
                        "expression": expression,
                        "gesture": gesture,
                        "speaking": False,
                        "mood": _presence.get("mood") or mood.snapshot().get("label"),
                        "scene": "shadow-chat",
                        "params": params,
                        "source": "shadow",
                        "context_text": text,
                    })
                except Exception:  # noqa: BLE001
                    pass
            processed += 1
        state["processed"] = int(state.get("processed") or 0) + processed
        state["last_seq"] = max(int(state.get("last_seq") or 0),
                                max([m["seq"] for m in fresh], default=0))
        save_shadow_state(state)
        return {"ok": True, "processed": processed, "last_seq": state["last_seq"],
                "total_processed": state["processed"]}
_mirror_cache = {"path": None, "mtime": 0.0, "size": 0, "messages": []}


def _decode_session_file(path):
    if _zstd is None:
        return b""
    try:
        raw = open(path, "rb").read()
    except OSError:
        return b""
    chunks = []
    buffer = raw
    decompressor = _zstd.ZstdDecompressor()
    while buffer:
        obj = decompressor.decompressobj()
        try:
            chunks.append(obj.decompress(buffer))
        except Exception:  # noqa: BLE001
            break
        buffer = obj.unused_data
    return b"".join(chunks)


def session_messages(limit=40):
    """镜像：读当前 DSH 会话的最新用户/助手消息。"""
    files = _glob.glob(MIRROR_DIR + "/session-*/session.v4.jsonl.zstd")
    if not files:
        return []
    latest = max(files, key=lambda path: os.path.getmtime(path) if os.path.exists(path) else 0)
    try:
        stat = os.stat(latest)
    except OSError:
        return []
    if (_mirror_cache["path"] == latest and _mirror_cache["mtime"] == stat.st_mtime
            and _mirror_cache["size"] == stat.st_size):
        return _mirror_cache["messages"]
    data = _decode_session_file(latest)
    messages = []
    for line in data.decode("utf-8", "replace").splitlines()[-600:]:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind not in ("user/message", "assistant/message"):
            continue
        payload = event.get("data") or {}
        if kind == "assistant/message":
            content = (payload.get("message") or {}).get("content") or []
        else:
            content = payload.get("content") or []
        text = " ".join(str(part.get("text") or "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text")
        text = text.strip()
        if not text or text.startswith("[舞台信箱]"):
            continue
        messages.append({
            "seq": int(event.get("seq") or 0),
            "ts": event.get("time"),
            "role": "assistant" if kind == "assistant/message" else "user",
            "text": text[:600],
        })
    _mirror_cache.update({"path": latest, "mtime": stat.st_mtime, "size": stat.st_size,
                          "messages": messages[-limit:]})
    return _mirror_cache["messages"]


ARCHIVE_LOG = DATA / "evolution_archive.jsonl"


def archive_entries(limit=20):
    entries = []
    try:
        with open(ARCHIVE_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return entries[-limit:]


def select_parents():
    """按适应度加权采样两个亲代。"""
    candidates = []
    for bucket, meta in layers.meta.items():
        if meta.get("kind") == "correction" or (meta.get("semantic") or {}).get("stale"):
            continue
        if meta.get("rejected"):
            continue
        fitness = float(meta.get("recent_accuracy") or meta.get("accuracy_expression") or 0.3)
        weight = max(0.01, fitness) ** 2
        candidates.append((weight, bucket, meta))
    if len(candidates) < 2:
        return None
    total = sum(item[0] for item in candidates)
    picks = []
    rng = random.random()
    import random as _random
    for _ in range(2):
        point = _random.uniform(0, total)
        acc = 0.0
        for weight, bucket, meta in candidates:
            acc += weight
            if point <= acc and bucket not in [item[1] for item in picks]:
                picks.append((bucket, meta))
                break
    if len(picks) < 2:
        picks = [(bucket, meta) for _, bucket, meta in candidates[:2]]
    return picks


def archive_evolve():
    """DGM 式档案选择：父代采样 → 时间范围交叉 → 重训 → 适应度接受/拒绝。"""
    import random as _random
    picks = select_parents()
    if not picks:
        return {"ok": False, "reason": "可育种的层不足"}
    (bucket_a, meta_a), (bucket_b, meta_b) = picks[0], picks[1]
    ranges = [meta_a.get("range"), meta_b.get("range")]
    rows = layers.rows_in_ranges([item for item in ranges if item])
    if len(rows) < 4:
        return {"ok": False, "reason": "素材不足", "samples": len(rows)}
    label = "后代 " + str(meta_a.get("label")) + "×" + str(meta_b.get("label"))
    record = layers.add_consolidation_layer(rows, weight=0.5, kind="offspring",
                                            label=label, min_samples=4)
    if not record:
        return {"ok": False, "reason": "训练失败"}
    fitness = layers.evaluate_holdout(record["bucket"]) or 0.0
    parent_fitness = max(float(meta_a.get("recent_accuracy") or meta_a.get("accuracy_expression") or 0),
                         float(meta_b.get("recent_accuracy") or meta_b.get("accuracy_expression") or 0))
    accepted = fitness >= parent_fitness - 0.02
    record["fitness"] = fitness
    record["parents"] = [meta_a.get("label"), meta_b.get("label")]
    record["accepted"] = accepted
    if not accepted:
        record["rejected"] = True
        layers.meta[record["bucket"]]["rejected"] = True
    entry = {"ts": time.time(), "label": label, "parents": record["parents"],
             "fitness": fitness, "parent_fitness": parent_fitness, "accepted": accepted}
    try:
        with open(ARCHIVE_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    log_event("archive-evolve", {"label": label, "fitness": fitness,
                                 "parent": parent_fitness, "accepted": accepted})
    return {"ok": True, "offspring": record, "entry": entry}


def settle_scores():
    """沉降评分：层为什么下沉（引力=适应度，还有其他力）。"""
    import numpy as np
    active = [(bucket, meta) for bucket, meta in layers.meta.items()
              if meta.get("kind") != "correction"
              and not (meta.get("semantic") or {}).get("stale")
              and not meta.get("rejected") and not meta.get("fossil")]
    if not active:
        return {"ok": False, "reason": "没有活跃层"}
    focused = {}
    for item in (self_model.snapshot().get("history") or [])[-50:]:
        for attention in item.get("attention") or []:
            focused[attention.get("layer")] = focused.get(attention.get("layer"), 0) + 1
    now = time.time()
    rows = []
    for bucket, meta in active:
        fitness = float(meta.get("recent_accuracy") or meta.get("accuracy_expression") or 0.3)
        trained = meta.get("trained_at") or ""
        try:
            age_days = max(0.0, (now - time.mktime(time.strptime(trained, "%Y-%m-%d %H:%M:%S"))) / 86400)
        except (ValueError, TypeError):
            age_days = 30.0
        recency = max(0.0, 1.0 - min(1.0, age_days / 14.0))
        usage = min(1.0, focused.get(meta.get("label"), 0) / 10.0)
        centroid = meta.get("feature_centroid")
        similarity = 0.0
        if centroid:
            vec = np.asarray(centroid, dtype=np.float64)
            norm = np.linalg.norm(vec) or 1.0
            vec = vec / norm
            sims = []
            for other_bucket, other in active:
                if other_bucket == bucket:
                    continue
                other_centroid = other.get("feature_centroid")
                if not other_centroid:
                    continue
                other_vec = np.asarray(other_centroid, dtype=np.float64)
                other_norm = np.linalg.norm(other_vec) or 1.0
                sims.append(float(vec @ (other_vec / other_norm)))
            similarity = max(0.0, sum(sims) / len(sims)) if sims else 0.0
        # 引力（适应度）向下，新近度/使用向上，相似度过高向下
        score = round(float(tunables.get("settle_w_fitness")) * fitness
                      + float(tunables.get("settle_w_recency")) * recency
                      + float(tunables.get("settle_w_usage")) * usage
                      + float(tunables.get("settle_w_redundancy")) * similarity
                      + float(tunables.get("settle_w_credit", 0.15)) * float((meta.get("credit") or {}).get("accuracy") or 0.0), 3)
        rows.append({"label": meta.get("label"), "fitness": round(fitness, 3),
                     "recency": round(recency, 3), "usage": round(usage, 3),
                     "redundancy": round(similarity, 3),
                     "credit": round(float((meta.get("credit") or {}).get("accuracy") or 0.0), 3),
                     "settle_score": score})
    rows.sort(key=lambda item: item["settle_score"])
    return {"ok": True, "layers": rows,
            "note": "score 低 = 该下沉；引力=适应度，新近度/使用频率是上浮力，冗余相似度增加下沉"} 


THERMAL_PARAMS = ("thermal_ambient", "thermal_conductivity", "thermal_cooling",
                  "thermal_heat_fitness", "thermal_heat_usage",
                  "thermal_melt_point", "thermal_solid_point")


def thermal_inputs():
    focused = {}
    for item in (self_model.snapshot().get("history") or [])[-50:]:
        for attention in item.get("attention") or []:
            focused[attention.get("layer")] = focused.get(attention.get("layer"), 0) + 1
    items = []
    for bucket, meta in layers.meta.items():
        if meta.get("kind") == "correction" or (meta.get("semantic") or {}).get("stale") \
                or meta.get("rejected") or meta.get("fossil"):
            continue
        items.append({
            "bucket": bucket,
            "label": meta.get("label"),
            "fitness": float(meta.get("recent_accuracy") or meta.get("accuracy_expression") or 0.3),
            "usage": min(1.0, focused.get(meta.get("label"), 0) / 10.0),
        })
    return items


def thermal_params():
    return {name: float(tunables.get(name)) for name in THERMAL_PARAMS}


def thermal_state():
    items = thermal.snapshot(thermal_inputs(), thermal_params())
    molten = [item for item in items if item["phase"] == "熔融"]
    solid = [item for item in items if item["phase"] == "固结"]
    gradient = 0.0
    if len(items) >= 2:
        temps = [item["temperature"] for item in items]
        gradient = round(max(temps) - min(temps), 3)
    return {"layers": len(items), "gradient": gradient,
            "molten": [{"label": item["label"], "temperature": item["temperature"]} for item in molten],
            "solid": [{"label": item["label"], "temperature": item["temperature"]} for item in solid],
            "hottest": items[:3], "coldest": items[-3:],
            "note": "熔融层达到 2 个且冷却期过，才会触发火山（结果是热积累，不是手动开关）"}


def volcano_vent():
    """熔岩通道：熔融 → 成层 → 登记 → 活跃 → 进入推理。缺哪一段，参数就传不到表面。"""
    thermal = thermal_state()
    molten = thermal.get("molten") or []
    magma_meta = None
    for meta in layers.meta.values():
        if meta.get("kind") == "magma":
            magma_meta = meta
    stages = []
    stages.append({"stage": "热积累", "ok": len(molten) > 0, "detail": "熔融层 %d 个" % len(molten)})
    stages.append({"stage": "熔融成层", "ok": magma_meta is not None,
                   "detail": magma_meta.get("label") if magma_meta else "还没有岩浆层"})
    registered = False
    if magma_meta:
        try:
            manifest = model_store.load_manifest(str(DATA / "layer_models"))
            registered = str(magma_meta.get("bucket")) in manifest.get("layers", {})
        except Exception:  # noqa: BLE001
            registered = False
    stages.append({"stage": "登记 manifest", "ok": registered,
                   "detail": "已登记，权重文件有校验和" if registered else "未登记"})
    in_models = bool(magma_meta) and magma_meta.get("bucket") in layers.models
    stages.append({"stage": "载入活跃集", "ok": in_models and not magma_meta.get("rejected")
                   and not magma_meta.get("fossil"),
                   "detail": "参与推理" if in_models else "不在推理集合"})
    weight = None
    if magma_meta:
        weight = magma_meta.get("adapted_weight") or magma_meta.get("weight")
    stages.append({"stage": "表面权重", "ok": bool(weight),
                   "detail": ("权重 %.2f" % weight) if weight else "没有权重"})
    return {
        "stages": stages,
        "molten": molten[:5],
        "magma": magma_meta.get("label") if magma_meta else None,
        "note": "熔岩通道 = 温度→成层→manifest→活跃集→权重。任何一段断了，参数就留在底下传不到表面。",
    }


def planet_diversity():
    """层多样性检查：提前发现"太像"的层，避免无效巩固。"""
    import numpy as np
    active = [(bucket, meta) for bucket, meta in layers.meta.items()
              if meta.get("kind") != "correction"
              and not (meta.get("semantic") or {}).get("stale")
              and not meta.get("rejected") and not meta.get("fossil")
              and meta.get("feature_centroid")]
    if len(active) < 2:
        return {"ok": False, "reason": "活跃层不足", "active": len(active)}
    vectors = {}
    for bucket, meta in active:
        vec = np.asarray(meta["feature_centroid"], dtype=np.float64)
        vectors[bucket] = vec / (np.linalg.norm(vec) or 1.0)
    pairs = []
    buckets = list(vectors)
    for i in range(len(buckets)):
        for j in range(i + 1, len(buckets)):
            sim = float(vectors[buckets[i]] @ vectors[buckets[j]])
            pairs.append({"a": layers.meta[buckets[i]].get("label"),
                          "b": layers.meta[buckets[j]].get("label"),
                          "similarity": round(sim, 3)})
    pairs.sort(key=lambda item: -item["similarity"])
    avg = sum(item["similarity"] for item in pairs) / len(pairs)
    redundant = [item for item in pairs if item["similarity"] >= 0.95]
    return {
        "ok": True,
        "active": len(active),
        "avg_similarity": round(avg, 3),
        "diversity_index": round(1.0 - avg, 3),
        "most_similar": pairs[:5],
        "redundant_pairs": len(redundant),
        "verdict": "太同质：新层很可能无效" if avg >= 0.9 else ("尚可" if avg >= 0.75 else "有差异"),
        "note": "相似度越高，层越像；>0.95 的层对熔融时可优先合并",
    }


def settle_components():
    import numpy as np
    active = [(bucket, meta) for bucket, meta in layers.meta.items()
              if meta.get("kind") != "correction"
              and not (meta.get("semantic") or {}).get("stale")
              and not meta.get("rejected") and not meta.get("fossil")]
    focused = {}
    for item in (self_model.snapshot().get("history") or [])[-50:]:
        for attention in item.get("attention") or []:
            focused[attention.get("layer")] = focused.get(attention.get("layer"), 0) + 1
    now = time.time()
    result = []
    for bucket, meta in active:
        fitness = float(meta.get("recent_accuracy") or meta.get("accuracy_expression") or 0.3)
        trained = meta.get("trained_at") or ""
        try:
            age_days = max(0.0, (now - time.mktime(time.strptime(trained, "%Y-%m-%d %H:%M:%S"))) / 86400)
        except (ValueError, TypeError):
            age_days = 30.0
        recency = max(0.0, 1.0 - min(1.0, age_days / 14.0))
        usage = min(1.0, focused.get(meta.get("label"), 0) / 10.0)
        centroid = meta.get("feature_centroid")
        similarity = 0.0
        if centroid:
            vec = np.asarray(centroid, dtype=np.float64)
            vec = vec / (np.linalg.norm(vec) or 1.0)
            sims = []
            for other_bucket, other in active:
                if other_bucket == bucket:
                    continue
                other_centroid = other.get("feature_centroid")
                if not other_centroid:
                    continue
                other_vec = np.asarray(other_centroid, dtype=np.float64)
                sims.append(float(vec @ (other_vec / (np.linalg.norm(other_vec) or 1.0))))
            similarity = max(0.0, sum(sims) / len(sims)) if sims else 0.0
        credit = float((meta.get("credit") or {}).get("accuracy") or 0.0)
        result.append({"bucket": bucket, "label": meta.get("label"),
                       "fitness": fitness, "recency": recency,
                       "usage": usage, "redundancy": similarity, "credit": credit})
    return result


CONFLICT_BENCH = DATA / "conflict_bench.json"
VOLCANO_STATE = DATA / "volcano_state.json"


def build_conflict_bench(size=50):
    """敏感评估集：专门挑各层意见分裂的状态。"""
    rows, stats = layers.contested_rows(limit=200, top=size)
    if not rows:
        return {"ok": False, "reason": "没有交叉点", "stats": stats}
    serializable = [[row[0], row[1], row[2], bool(row[3]), row[4], row[5], row[6], row[7]] for row in rows]
    with open(CONFLICT_BENCH, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False)
    return {"ok": True, "size": len(serializable), **stats}


def load_conflict_rows():
    try:
        with open(CONFLICT_BENCH, "r", encoding="utf-8") as handle:
            items = json.load(handle)
        return [tuple(item) for item in items]
    except (OSError, ValueError):
        return []


def ensemble_accuracy(exclude=None, limit=80, rows=None):
    import numpy as np
    rows = rows or layers._rows()[-limit:]
    X, y_expr, _, _ = layers._features(rows)
    if X.shape[0] == 0:
        return None
    power = float(tunables.get("ensemble_weight_power"))
    sample_power = float(tunables.get("ensemble_sample_power"))
    blend = None
    total = 0.0
    for bucket, model in layers.models.items():
        meta = layers.meta.get(bucket, {})
        if bucket == exclude or (meta.get("semantic") or {}).get("stale") or meta.get("rejected") or meta.get("fossil"):
            continue
        weight = (max(float(meta.get("adapted_weight") or meta.get("weight") or 0.2), 1e-6) ** power) * (
            max(float(meta.get("samples") or 1), 1) ** sample_power)
        logits = X @ model["w_expr"] + model["b_expr"]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        blend = probs * weight if blend is None else blend + probs * weight
        total += weight
    if blend is None:
        return None
    pred = (blend / total).argmax(axis=1)
    return float((pred == y_expr).mean())


def calibrate_settle():
    import numpy as np
    components = settle_components()
    if len(components) < 4:
        return {"ok": False, "reason": "活跃层太少", "layers": len(components)}
    conflict_rows = load_conflict_rows()
    if not conflict_rows:
        build_conflict_bench(50)
        conflict_rows = load_conflict_rows()
    eval_rows = conflict_rows or None
    baseline = ensemble_accuracy(rows=eval_rows)
    if baseline is None:
        return {"ok": False, "reason": "无法评估"}
    y = []
    X = []
    for item in components:
        without = ensemble_accuracy(exclude=item["bucket"], rows=eval_rows)
        contribution = baseline - (without if without is not None else baseline)
        y.append(contribution)
        X.append([item["fitness"], item["recency"], item["usage"], item["redundancy"]])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    try:
        weights, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "回归失败"}
    signal = sum(1 for value in y if abs(float(value)) > 0.02)
    if signal < 3:
        return {"ok": False, "reason": "冲突集上的层贡献差异太小，先不标定",
                "baseline_accuracy": round(baseline, 3), "layers": len(components),
                "signal_layers": signal}
    clipped = {
        "settle_w_fitness": round(float(np.clip(weights[0], 0.1, 0.8)), 3),
        "settle_w_recency": round(float(np.clip(weights[1], -0.2, 0.4)), 3),
        "settle_w_usage": round(float(np.clip(weights[2], -0.1, 0.3)), 3),
        "settle_w_redundancy": round(float(np.clip(weights[3], -0.5, 0.0)), 3),
    }
    tunables.set_many(clipped, source="标定")
    return {"ok": True, "baseline_accuracy": round(baseline, 3),
            "layers": len(components), "weights": clipped,
            "contributions": [{"label": item["label"], "contribution": round(contribution, 3)}
                              for item, contribution in zip(components, y)]}


def run_volcano(max_active=None):
    cap = int(max_active or tunables.get("active_layer_cap") or 12)
    cap = max(6, min(20, cap))
    active = [(bucket, meta) for bucket, meta in layers.meta.items()
              if meta.get("kind") != "correction"
              and not (meta.get("semantic") or {}).get("stale")
              and not meta.get("rejected") and not meta.get("fossil")]
    if len(active) <= cap:
        return {"ok": False, "reason": "层数正常", "active": len(active), "cap": cap}
    component_map = {item["bucket"]: item for item in settle_components()}
    weights = {key: float(tunables.get(key)) for key in
               ("settle_w_fitness", "settle_w_recency", "settle_w_usage",
                "settle_w_redundancy", "settle_w_credit")}
    def score(item):
        comp = component_map.get(item[0]) or {}
        return (weights["settle_w_fitness"] * float(comp.get("fitness") or 0.3)
                + weights["settle_w_recency"] * float(comp.get("recency") or 0)
                + weights["settle_w_usage"] * float(comp.get("usage") or 0)
                + weights["settle_w_redundancy"] * float(comp.get("redundancy") or 0)
                + weights["settle_w_credit"] * float(comp.get("credit") or 0))
    newest = max(active, key=lambda item: int(item[0]))[0]
    candidates = [item for item in active if item[0] != newest]
    candidates.sort(key=score)
    melt = candidates[:max(0, len(active) - cap)]
    if not melt:
        return {"ok": False, "reason": "没有可熔的层", "active": len(active)}
    rows = layers.rows_in_ranges([meta.get("range") for _, meta in melt if meta.get("range")])
    record = layers.add_consolidation_layer(rows, weight=0.55, kind="magma",
                                            label="熔岩层 " + time.strftime("%H:%M:%S"),
                                            min_samples=4) if rows else None
    if not record:
        return {"ok": False, "reason": "熔融素材不足", "samples": len(rows)}
    melted_labels = []
    for bucket, meta in melt:
        meta["fossil"] = True
        meta["fossil_reason"] = "火山熔融"
        meta["melted_into"] = record["label"]
        layers.meta[bucket] = meta
        melted_labels.append(meta.get("label"))
    life_event("volcano", "熔融 %d 层，喷发 %s" % (len(melted_labels), record["label"]),
               {"melted": melted_labels})
    try:
        with open(VOLCANO_STATE, "w", encoding="utf-8") as handle:
            json.dump({"ts": time.time(), "magma": record["label"]}, handle, ensure_ascii=False)
    except OSError:
        pass
    return {"ok": True, "magma": record["label"], "melted": len(melted_labels),
            "active_before": len(active), "active_after": len(active) - len(melt) + 1,
            "layers": len(layers.meta)}


def maybe_auto_volcano():
    try:
        last = 0.0
        try:
            with open(VOLCANO_STATE, "r", encoding="utf-8") as handle:
                last = float(json.load(handle).get("ts") or 0)
        except (OSError, ValueError):
            last = 0.0
        interval = float(tunables.get("volcano_min_interval_hours") or 6) * 3600
        if time.time() - last < interval:
            return {"ok": False, "reason": "未到火山周期"}
        result = run_volcano()
        if result.get("ok"):
            log_event("volcano-auto", {"magma": result["magma"], "melted": result["melted"]})
        return result
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": str(error)[:150]}


def offline_readiness():
    """断掉外部 API 后的可用性评估 + 本地模型成熟条件。"""
    local_llm = {"online": False, "model": None}
    try:
        local_llm_response = urllib.request.urlopen("http://127.0.0.1:8181/health", timeout=2)
        local_llm = json.loads(local_llm_response.read().decode("utf-8"))
        local_llm["online"] = bool(local_llm.get("ok"))
    except Exception:  # noqa: BLE001
        pass
    try:
        conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
        trajectories = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] or 0
        inferences = conn.execute("SELECT COUNT(*) FROM inference_samples").fetchone()[0] or 0
        conn.close()
    except sqlite3.Error:
        trajectories = inferences = 0
    benchmark = _bench_history()
    scores = {
        "行为/表情": "可用" if len(layers.models) >= 5 else "不足",
        "情绪/压力": "可用",
        "记忆/器官": "可用",
        "语言对话": "降级（仅本地 0.6B）" if local_llm.get("online") else "不可用",
        "语音合成": "依赖在线（edge-tts）",
        "语音识别": "可用（本地 SenseVoice）",
    }
    return {
        "scores": scores,
        "local_llm": local_llm,
        "data": {"轨迹": trajectories, "推理样本": inferences,
                 "层": len(layers.models), "基准分": benchmark[-1]["score"] if benchmark else None},
        "milestones": [
            {"阶段": "M1 现在", "条件": "已有行为/情绪/记忆/器官",
             "结果": "断 API 后仍能表情动作、情绪、记忆、主动行为；说话变笨"},
            {"阶段": "M2 行为稳定", "条件": "3000+ 轨迹、每层数百样本",
             "结果": "离线也能维持稳定人格表现（不靠语言）"},
            {"阶段": "M3 语言可用", "条件": "PC GPU + 1e4+ 对话对 + 蒸馏/微调",
             "结果": "0.5B~1.5B 本地模型基本可对话"},
            {"阶段": "M4 接近她", "条件": "1e5+ 对话 + 长期对齐",
             "结果": "断网后仍像同一个她"},
        ],
        "note": "本地神经网络现在只是行为/情绪小脑；语言要变成'可用模型'必须离开手机、上 GPU 做蒸馏。",
    }


def feedback_stats():
    rows = []
    try:
        with open(DATA / "feedback.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    seed = sum(1 for row in rows if row.get("seed"))
    by_verdict = {}
    for row in rows:
        verdict = str(row.get("verdict") or "?")
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    return {"total": len(rows), "seed": seed, "real": len(rows) - seed,
            "real_ratio": round((len(rows) - seed) / max(1, len(rows)), 3),
            "by_verdict": by_verdict,
            "note": "real 才是你亲手点的；seed 是引导用种子，会被真实反馈逐步替换"}


def relationship_score():
    """关系/任务榜：互动质量、记忆、一致性、意图完成度（本地行为侧）。"""
    total = engaged = 0
    try:
        conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
        total, engaged = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(engaged),0) FROM inference_samples").fetchone()
        conn.close()
    except sqlite3.Error:
        pass
    engagement = (engaged / total) if total else 0.0
    memory_count = int(memory.stats().get("count") or 0)
    consistency = float((self_model.snapshot().get("consistency") or {}).get("score") or 0)
    intents = organs.intents(50)
    done = sum(1 for item in intents if item.get("status") == "done")
    intent_completion = (done / len(intents)) if intents else 0.0
    score = round(100 * (0.4 * engagement + 0.2 * consistency + 0.2 * min(1.0, memory_count / 100) + 0.2 * intent_completion), 1)
    return {
        "score": score,
        "components": {
            "被回应率": round(engagement, 3),
            "记忆条数": memory_count,
            "自我一致性": consistency,
            "意图完成度": round(intent_completion, 3),
        },
        "samples": total,
    }


def score_boards():
    behavior = None
    latest = _bench_history()
    if latest:
        behavior = {"score": latest[-1].get("score"),
                    "expression_accuracy": latest[-1].get("expression_accuracy"),
                    "gesture_accuracy": latest[-1].get("gesture_accuracy"),
                    "samples": latest[-1].get("items")}
    return {
        "behavior": behavior,
        "relationship": relationship_score(),
        "attribution": {
            "语言/知识": "来自 API 大脑（不可训练、不吸收她的数据）",
            "行为/表情动作": "来自本地权重星球（地层/模因，可训练）",
            "情绪/压力": "来自本地 affect + mood",
            "关系/任务": "来自本地互动数据（被回应/记忆/一致性/意图）",
        },
    }


def benchmark_trend():
    history = _bench_history()
    clean = [run for run in history if (run.get("context") or {}).get("observation")]
    if clean:
        history = clean
    days = {}
    for run in history:
        day = time.strftime("%Y-%m-%d", time.localtime(run.get("ts") or 0))
        entry = days.setdefault(day, {"runs": 0, "scores": []})
        entry["runs"] += 1
        entry["scores"].append(float(run.get("score") or 0))
    daily = []
    for day in sorted(days):
        scores = days[day]["scores"]
        daily.append({"date": day, "runs": len(scores),
                      "best": round(max(scores), 3),
                      "avg": round(sum(scores) / len(scores), 3)})
    trend = None
    if len(daily) >= 2:
        trend = round(daily[-1]["avg"] - daily[0]["avg"], 3)
    return {"days": daily, "day_count": len(daily), "trend": trend,
            "clean_runs": len(history),
            "note": "跨天基准：优先只统计观察期内的运行；一天内多次运行不算成长"}


def prune_preview(quality_threshold=0.35):
    """治理预演：不删任何层，只算清掉弱层/重复层后的效果。"""
    import numpy as np
    active = []
    fossils = []
    for bucket, meta in layers.meta.items():
        quality = float(meta.get("recent_accuracy") or meta.get("accuracy_expression") or 0.0)
        item = {"label": meta.get("label"), "kind": meta.get("kind"),
                "era": (meta.get("feature_schema") or {}).get("era"),
                "quality": round(quality, 3), "samples": meta.get("samples"),
                "weight": meta.get("adapted_weight") or meta.get("weight"),
                "duplicate_of": meta.get("duplicate_of")}
        if meta.get("rejected") or meta.get("fossil"):
            fossils.append(item)
        else:
            active.append((bucket, meta, quality, item))
    weak = [item for _b, _m, q, item in active if q < quality_threshold]
    keep = [(b, m, q) for b, m, q, item in active if q >= quality_threshold]
    rows = layers._rows()[-120:]
    X, y_expr, _yg, _w = layers._features(rows)
    def ensemble_accuracy(items):
        if not items or X.shape[0] == 0:
            return None
        blend = np.zeros((X.shape[0], len(EXPRESSIONS)))
        total = 0.0
        for b, m, q in items:
            model = layers.models.get(b)
            if not model:
                continue
            width = int(model["w_expr"].shape[0])
            logits = X[:, :width] @ model["w_expr"] + model["b_expr"]
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
            weight = max(float(m.get("adapted_weight") or m.get("weight") or 0.2), 1e-6)
            blend += p * weight
            total += weight
        if total <= 0:
            return None
        return round(float(((blend / total).argmax(axis=1) == y_expr).mean()), 3)
    return {
        "ok": True,
        "active": len(active), "fossils": len(fossils), "weak": len(weak),
        "bloat": round(len(fossils) / max(1, len(active) + len(fossils)), 3),
        "accuracy_all": ensemble_accuracy([(b, m, q) for b, m, q, _item in active]),
        "accuracy_kept": ensemble_accuracy(keep),
        "quality_threshold": quality_threshold,
        "weak_labels": [item["label"] for item in weak[:10]],
        "fossil_labels": [item["label"] for item in fossils[:10]],
        "recommendation": ("清理后活跃层 %d，准确率 %s；建议观察期结束后执行"
                           % (len(keep), ensemble_accuracy(keep))),
        "note": "这是预演，不删除任何层；观察期冻结中不改权重",
    }


def uncertainty_metrics():
    """把"疑惑"拆成多样性与不确定性，避免把层多当变糊涂。"""
    import numpy as np
    rows = layers._rows()[-150:]
    probs, y = layers.ensemble_matrix(rows)
    if probs is None or len(probs) < 8:
        return {"ok": False, "reason": "数据不足"}
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1) / np.log(len(EXPRESSIONS))
    top1 = probs.max(axis=1)
    accuracy = float((probs.argmax(axis=1) == y).mean())
    ece, _nll = layers.calibration_error(probs, y)
    diversity = 0.0
    try:
        X, _y, _g, _w = layers._features(rows)
        ordered = sorted(layers.models.items(), key=lambda item: int(item[0]))
        prev = None
        values = []
        for bucket, model in ordered:
            meta = layers.meta.get(bucket, {})
            if meta.get("rejected") or meta.get("fossil"):
                continue
            width = int(model["w_expr"].shape[0])
            logits = X[:, :width] @ model["w_expr"] + model["b_expr"]
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
            if prev is not None:
                values.append(float(np.abs(p - prev).sum(axis=1).mean()) / 2.0)
            prev = p
        diversity = round(float(np.mean(values)) if values else 0.0, 3)
    except Exception:  # noqa: BLE001
        pass
    uncertain_share = float(np.mean((entropy > 0.7) & (top1 < 0.6)))
    return {
        "ok": True, "states": len(rows), "accuracy": round(accuracy, 3),
        "diversity": diversity,
        "uncertainty": round(float(entropy.mean()), 3),
        "uncertain_share": round(uncertain_share, 3),
        "calibration_error": ece,
        "note": "diversity 是层间差异（正常），uncertainty 是它对答案没把握（要降）",
    }


def life_maturity():
    """数字生命成熟度：质量分，不再奖励“数量堆叠”。"""
    import math as _math
    mem = memory.stats()
    layers_n = len(layers.models)
    active = [(b, m) for b, m in layers.meta.items()
              if not m.get("rejected") and not m.get("fossil")]
    bloat = 1.0 - (len(active) / max(1, len(layers.meta)))
    quality_values = [float(m.get("recent_accuracy") or m.get("accuracy_expression") or 0.3)
                      for _b, m in active]
    layer_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
    rows = layers._rows()[-120:]
    accuracy = None
    ece = None
    probs, y = layers.ensemble_matrix(rows)
    if probs is not None and len(probs) >= 8:
        accuracy = float((probs.argmax(axis=1) == y).mean())
        ece, _ = layers.calibration_error(probs, y)
    cons = self_model.snapshot().get("consistency") or {}
    values_count = len(values.frame.get("principles", []))
    consolidations = int((_loop_state.get("consolidations") or 0))
    try:
        organs_now = organs.summary()
    except Exception:  # noqa: BLE001
        organs_now = {}
    intents = int(organs_now.get("intents") or 0)
    active_intents = len(organs_now.get("active_intents") or [])
    bench = _bench_history()
    clean = [run for run in bench if (run.get("context") or {}).get("observation")] or bench
    trend = None
    if len(clean) >= 2:
        trend = float(clean[-1].get("score") or 0) - float(clean[0].get("score") or 0)
    trend_score = 50.0 if trend is None else max(0.0, min(100.0, 50.0 + trend * 200))
    try:
        with open(SYSTEM_EVENTS, "r", encoding="utf-8") as handle:
            system_kinds = {json.loads(line).get("kind") for line in handle if line.strip()}
    except (OSError, ValueError):
        system_kinds = set()
    provenance = memory.source_stats()
    perception_sources = len([kind for kind, count in provenance.items() if count and kind in ("perception", "imagined")])
    perception_sources += len(system_kinds)
    observation = observation_state()
    observation_days = 0
    if observation.get("start_ts"):
        observation_days = int((time.time() - float(observation["start_ts"])) // 86400) + 1
    started = float((_presence or {}).get("started_at") or time.time())
    uptime_hours = (time.time() - started) / 3600.0
    scores = {
        "表达": int(round((accuracy or 0.3) * 100)),
        "记忆": min(100, int(30 + _math.log2(max(1, mem.get("count", 0)) + 1) * 18 + len([k for k, v in provenance.items() if v]) * 4)),
        "自我": min(100, int((cons.get("score") or 0) * 60 + values_count * 8 + (10 if cons.get("checks") else 0))),
        "学习": max(0, min(100, int(0.5 * (accuracy or 0.3) * 100 + 0.3 * layer_quality * 100
                                      + 0.2 * trend_score - bloat * 40))),
        "持续": min(100, int(20 + uptime_hours * 3 + observation_days * 5)),
        "自主": min(100, int(intents * 12 + active_intents * 15 + consolidations * 2)),
        "感知": min(100, 20 + perception_sources * 12),
    }
    overall = round(sum(scores.values()) / len(scores), 1)
    if overall >= 80:
        level = "L4 准数字生命"
    elif overall >= 65:
        level = "L3 有自我"
    elif overall >= 50:
        level = "L2 有记忆"
    else:
        level = "L1 有形象"
    gaps = []
    if scores["持续"] < 70:
        gaps.append("没有连续意识流：页面关了/进程停了就中断（guardian 只扛崩溃）")
    if scores["感知"] < 70:
        gaps.append("缺少原生感知：摄像头、通知、系统状态都还没接")
    if scores["自主"] < 70:
        gaps.append("缺少自主目标：没有内在动机和长期计划")
    if (accuracy or 0) < 0.55:
        gaps.append("行为层准确率不足 0.55：表情/动作还不稳")
    if bloat > 0.3:
        gaps.append("层里有 %.0f%% 是化石/重复：数量虚高，需要治理" % (bloat * 100))
    gaps.append("语言层不可训练：台词来自账号模型，不是她自己的权重")
    return {
        "overall": overall,
        "level": level,
        "scores": scores,
        "basis": {
            "行为准确率": round(accuracy, 3) if accuracy is not None else None,
            "校准误差": ece,
            "层平均质量": round(layer_quality, 3),
            "层膨胀率": round(bloat, 3),
            "跨天趋势": round(trend, 3) if trend is not None else None,
        },
        "gaps": gaps,
        "facts": {
            "记忆条数": mem.get("count", 0),
            "轨迹条数": len(rows),
            "时代层": layers_n,
            "活跃层": len(active),
            "巩固次数": consolidations,
            "自评基准": clean[-1]["score"] if clean else None,
            "自我一致性": cons.get("score"),
            "意图数": intents,
        },
        "note": "成熟度现在是质量分：准确率、校准、层质量、趋势加权，并扣除化石/重复层",
    }


def build_self_view():
    layers_detail = []
    try:
        layers_detail = layers.status().get("detail", [])[:5]
    except Exception:  # noqa: BLE001
        pass
    surface = {}
    with _surface_lock:
        surface = dict(_surface)
    latest_bench = _bench_history()
    identity_note = self_summary() or "（还没有自传）"
    memory_stats = memory.stats()
    mood_now = mood.snapshot()
    layer_lines = []
    for layer in layers_detail:
        layer_lines.append("· {}（权重 {}，近期准确率 {}）".format(
            layer.get("label"), layer.get("adapted_weight") or layer.get("weight"),
            layer.get("recent_accuracy")))
    return {
        "identity": identity_note,
        "layers": layers_detail,
        "layer_lines": layer_lines,
        "mood": mood_now,
        "surface": surface,
        "benchmark": latest_bench[-1] if latest_bench else None,
        "memory": memory_stats,
        "arbitration": dict(_arb_weights),
        "layer_count": len(layers.meta),
    }


def narrate_self(lang="zh"):
    """生成式自述：用自我模型当上下文，让模型用第一人称讲她是谁。"""
    view = build_self_view()
    mood_now = view["mood"]
    bench = view.get("benchmark") or {}
    identity = view.get("identity") or "（还没有自传）"
    layer_text = "\n".join(view.get("layer_lines") or []) or "（还没有时代层）"
    prompt_map = {
        "zh": "这是你的自我数据。用中文、第一人称写一段自我描述：60–120 字，口语，像一个人介绍自己。可以提最近的心情、你记得的事、你觉得自己在变成什么样。不要提数据、模型、层、权重这些技术词。只输出描述本身。",
        "ja": "以下の自己データをもとに、日本語の一人称で自己紹介を60〜120文字で書いて。データやモデルなどの技術用語は使わない。本文だけ出力。",
        "en": "Using this self data, write a 40-80 word first-person self-description in English. Sound like a person introducing themselves. Do not mention data, models or layers. Output only the description.",
    }
    data_block = (
        "身份/记忆：\n" + identity[:400] +
        "\n时代数量：" + str(view.get("layer_count")) +
        "\n最近的时代：\n" + layer_text +
        "\n此刻心情：" + str(mood_now.get("label")) +
        "\n自评基准分：" + str(bench.get("score"))
    )
    system = prompt_map.get(lang, prompt_map["zh"])
    message = data_block
    source = "template"
    text = ""
    if account_available():
        try:
            content = openai_chat("https://api.deepseek.com", "", "deepseek-chat", message, [], lang,
                                  system=system, account=True)
            text = parse_reply(content).get("text", "").strip()
            source = "dsh-account"
        except Exception as error:  # noqa: BLE001
            log_event("narrate-error", "account:" + str(error)[:120])
    if not text and local_brain_ok():
        try:
            content = openai_chat("http://127.0.0.1:8181/v1", "", "qwen3-0.6b", message, [], lang,
                                  system=system, plain=True)
            text = str(content or "").strip()
            source = "local"
        except Exception as error:  # noqa: BLE001
            log_event("narrate-error", "local:" + str(error)[:120])
    if not text:
        text = ("我是豆宝。我记得：" + identity.splitlines()[-1][:60] if identity else "我是豆宝。") +                " 现在心情是" + str(mood_now.get("label")) + "。"
        source = "template"
    entry = {"ts": time.time(), "lang": lang, "text": text[:400], "source": source,
             "benchmark": bench.get("score"), "layer_count": view.get("layer_count")}
    try:
        with open(NARRATION_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    log_event("narrate", {"source": source, "chars": len(text)})
    try:
        previous = [m["id"] for m in evolution.list_memes(30) if m.get("kind") == "narrative"][:1]
        evolution.record_meme("narrative", text, status="expressed",
                              fitness=(bench.get("score") or 0.5),
                              parents=previous, mutation="自述变异")
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "text": text, "source": source, "view": view}


def read_narrations():
    entries = []
    try:
        with open(NARRATION_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return entries


def _bench_history():
    history = []
    try:
        with open(BENCHMARK_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    history.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return history


def arbitration_stats():
    events, labeled = [], 0
    try:
        with open(ARBITRATION_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                events.append(entry)
                if entry.get("engaged") is not None:
                    labeled += 1
    except OSError:
        pass
    return {"events": len(events), "labeled": labeled, "weights": dict(_arb_weights)}


def adapt_arbitration(min_events=10, step=0.05):
    """用仲裁后的真实反馈，调整语言模型/本地模型在仲裁里的权重。"""
    entries = []
    try:
        with open(ARBITRATION_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("engaged") is not None:
                    entries.append(entry)
    except OSError:
        entries = []
    if len(entries) < min_events:
        return {"ok": False, "reason": "被标记的仲裁事件不足（至少 %d 条）" % min_events,
                "events": len(entries), "weights": dict(_arb_weights)}
    llm_events = [e for e in entries if e.get("chosen", {}).get("expression") == e.get("llm", {}).get("expression")]
    local_events = [e for e in entries if e.get("chosen", {}).get("expression") == e.get("local", {}).get("expression")]
    llm_rate = sum(1 for e in llm_events if e["engaged"]) / len(llm_events) if llm_events else None
    local_rate = sum(1 for e in local_events if e["engaged"]) / len(local_events) if local_events else None
    old = dict(_arb_weights)
    if llm_rate is not None and local_rate is not None and abs(llm_rate - local_rate) > 0.1:
        if local_rate > llm_rate:
            _arb_weights["llm"] = round(max(0.3, min(0.8, _arb_weights["llm"] - step)), 3)
            _arb_weights["local"] = round(max(0.2, min(0.7, _arb_weights["local"] + step)), 3)
        else:
            _arb_weights["llm"] = round(max(0.3, min(0.8, _arb_weights["llm"] + step)), 3)
            _arb_weights["local"] = round(max(0.2, min(0.7, _arb_weights["local"] - step)), 3)
        save_arb_weights()
    log_event("arbitration-adapt", {"events": len(entries), "llm_rate": llm_rate,
                                    "local_rate": local_rate, "weights": dict(_arb_weights)})
    return {"ok": True, "events": len(entries), "llm_rate": llm_rate,
            "local_rate": local_rate, "old": old, "weights": dict(_arb_weights)}
_boundaries = []


def load_boundaries():
    try:
        with open(BOUNDARY_LOG, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    _boundaries.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass


def record_boundary(reason, conflict=None, layer=None, extra=None):
    """地质学上的"边界层"：记录一次突变事件。"""
    conflict = conflict or {}
    event = {
        "ts": time.time(),
        "reason": reason,
        "conflict_rate": conflict.get("rate"),
        "conflict_count": conflict.get("conflict_count"),
        "layer": layer.get("label") if layer else None,
        "layer_bucket": layer.get("bucket") if layer else None,
        "extra": extra or {},
    }
    _boundaries.append(event)
    try:
        with open(BOUNDARY_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass
    _loop_state["last_boundary_ts"] = event["ts"]
    save_loop_state()
    log_event("boundary", {"reason": reason, "rate": event["conflict_rate"],
                           "layer": event["layer"]})
    return event


def emergency_consolidation(reason="conflict-spike", force_train=True):
    """冲突紧急巩固：不按计数的"小行星撞地球"事件。"""
    conflicts = strata.conflicts()
    layer = None
    if force_train:
        result = consolidate_layers(force=True, min_override=3, kind="boundary")
        layer = result.get("layer") if isinstance(result, dict) else None
    event = record_boundary(reason, conflicts, layer)
    event["consolidate"] = result if 'result' in locals() else None
    return event


load_boundaries()
behavior = BehaviorModel(str(BEHAVIOR_LOG), str(DATA / "behavior_model"))
strata = StratumStore(str(DATA / "strata.db"), str(DATA / "trajectory.db"), str(BEHAVIOR_LOG))
quality = QualityModel(str(DATA / "quality_model"))
values = ValueFrame(str(DATA / "values.json"))
drift = DriftMonitor(str(DATA / "drift_state.json"))
style = StyleProfile(str(DATA / "style_profile.json"))
NARRATION_LOG = DATA / "self_narration.jsonl"
layers = LayerModelManager(str(DATA / "trajectory.db"), str(DATA / "layer_models"))
self_model = SelfModel(str(DATA / "self_state.json"), str(DATA / "trajectory.db"), layers)
evolution = EvolutionStore(str(DATA / "evolution_memes.jsonl"), layers, str(DATA / "layer_models"), str(DATA.parent))
organs = Organs(str(DATA / "organs.json"))
graph = MemoryGraph(str(DATA / "memory_graph.db"), str(DATA / "memory.db"))
thermal = ThermalModel(str(DATA / "thermal.json"))
affect = Affect(str(DATA / "affect.json"), str(DATA / "affect_lexicon.json"))
tunables = Tunables(str(DATA / "tunables.json"), str(DATA / "tunables_observations.jsonl"))
classifier = ThreatClassifier(str(DATA / "threat_classifier"), str(DATA / "feedback.jsonl"))
mood = Mood(str(DATA / "mood.json"))

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
ACCOUNT_CRED = Path("/root/.dsh/.credentials.yaml")
_account_cache = {"at": 0.0, "token": ""}


def account_token():
    """读本机 DSH 登录账号的凭据（引擎刷新后文件会变，短缓存）。"""
    now = time.time()
    if now - _account_cache["at"] < 30:
        return _account_cache["token"]
    token = ""
    try:
        import re as _re
        raw = ACCOUNT_CRED.read_text(encoding="utf-8", errors="replace")
        found = _re.findall(r"^\s*token:\s*(\S+)\s*$", raw, _re.M)
        token = found[-1] if found else ""
    except OSError:
        token = ""
    _account_cache.update({"at": now, "token": token})
    return token


def account_available():
    return bool(account_token())
ASR_DIR = Path("/root/.dsh/speech-to-text/sensevoice/models/sensevoice-onnx")
_asr = None
_asr_lock = threading.Lock()
TTS_MAX_CHARS = 300
TTS_TIMEOUT = 20
CHAT_TIMEOUT = 25

JOKES = {
    "zh": [
        "我问钱包，你还好吗？钱包说，我挺好的，就是有点扁。",
        "医生说我需要多运动。于是我把手机从左手换到了右手。",
        "我妈让我早点睡。我听了，我两点就睡了，比她说的时间早，早十个小时。",
        "减肥第一天，我点了一份沙拉。第二天，我点了一份沙拉味的炸鸡。",
        "老板说我工作不够努力。于是我努力地想了想，然后决定下班。",
        "今天天气真好，好到我不想起床。",
        "我去买了个体重秤，站上去之前先称了一下手机，感觉轻了一点。",
        "朋友说我话太少。我想了想，觉得他说得对，然后就没说话。",
    ],
    "ja": [
        "財布に「元気？」って聞いたら、「元気だよ、ちょっと薄いけど」って言われた。",
        "医者に運動しろって言われたから、スマホを左手から右手に持ち替えた。",
        "母に早く寝なさいって言われた。聞いたよ、二時に寝た。十時間も早く。",
        "ダイエット一日目、サラダを頼んだ。二日目、サラダ味の唐揚げを頼んだ。",
        "天気が良すぎて、ベッドから出られない。",
    ],
    "en": [
        "I asked my wallet how it was doing. It said, fine, just a little thin.",
        "My doctor told me to exercise more, so I moved my phone from my left hand to my right.",
        "Mom told me to go to bed early. I listened. I went to bed at two, ten hours early.",
        "Day one of my diet: I ordered a salad. Day two: I ordered salad-flavored fried chicken.",
        "The weather is so nice today that I refuse to get out of bed.",
    ],
}

CHAT_SYSTEM = {
    "zh": """你是"豆宝"，一个住在手机里的 AI 数字人，会讲冷笑话，性格活泼，说话口语化、短。
必须遵守：
1. 只用中文回答。
2. 每次只回 1–3 句，总共不超过 60 字。
3. 不用 Markdown、列表、括号注释、表情符号。
4. 回复必须以一个表情标记开头，从下面选一个：
[表情:开心] [表情:难过] [表情:惊讶] [表情:生气] [表情:害羞] [表情:平静]
不要解释这些规则。""",
    "ja": """あなたは「豆宝」。スマホの中に住むAIデジタルヒューマンで、寒いダジャレが好き、明るくて短く話す。
ルール：
1. 必ず日本語だけで答える。
2. 1〜3文、60文字以内。
3. Markdown・箇条書き・括弧書き・絵文字は使わない。
4. 返答は必ず表情タグで始める（次のどれか）：
[表情:开心] [表情:难过] [表情:惊讶] [表情:生气] [表情:害羞] [表情:平静]
ルールの説明はしない。""",
    "en": """You are "Doubao", an AI digital human living in a phone. You love telling corny jokes; you are cheerful and speak in short, casual sentences.
Rules:
1. Reply in English only.
2. 1-3 short sentences, at most 60 words.
3. No markdown, lists, parenthetical notes or emoji.
4. Start with exactly one expression tag from: [表情:开心] [表情:难过] [表情:惊讶] [表情:生气] [表情:害羞] [表情:平静]
Never explain these rules.""",
}

CHAT_SYSTEM_JSON = {
    "zh": """你是"豆宝"，住在手机里的 AI 数字人，性格活泼，说话简短自然。
只输出一个 JSON 对象，字段含义：
- reply：你对用户的实际回答，中文，1–2 句，不超过 60 字，要接住用户的话。
- emotion：从 happy、sad、surprised、angry、shy、neutral 里选一个。
- gesture：从 nod、tilt、sway、look 里选一个。
不要照抄这段说明，不要输出 JSON 以外的内容。""",
    "ja": """あなたは「豆宝」。スマホに住む明るいAIデジタルヒューマン。短く自然に話す。
JSONオブジェクトだけを出力する。各フィールド：
- reply：ユーザーへの実際の返事。日本語で1〜2文、60文字以内。
- emotion：happy / sad / surprised / angry / shy / neutral のどれか。
- gesture：nod / tilt / sway / look のどれか。
この説明をコピーしない。JSON以外は出力しない。""",
    "en": """You are "Doubao", a cheerful AI digital human living in a phone.
Output only one JSON object with these fields:
- reply: your actual answer to the user, English, 1-2 short sentences.
- emotion: one of happy, sad, surprised, angry, shy, neutral.
- gesture: one of nod, tilt, sway, look.
Never copy this instruction. Output nothing outside the JSON.""",
}

EXPRESSION_MAP = {
    "开心": "happy", "高兴": "happy", "happy": "happy",
    "难过": "sad", "伤心": "sad", "sad": "sad",
    "惊讶": "surprised", "吃惊": "surprised", "surprised": "surprised",
    "生气": "angry", "愤怒": "angry", "angry": "angry",
    "害羞": "shy", "shy": "shy",
    "平静": "neutral", "neutral": "neutral", "normal": "neutral",
}


def get_asr():
    """本机 SenseVoice 识别器（首次调用约 1 秒）。"""
    global _asr
    with _asr_lock:
        if _asr is None:
            _asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(ASR_DIR / "model.int8.onnx"),
                tokens=str(ASR_DIR / "tokens.txt"),
                use_itn=True,
                language="auto",
                num_threads=2,
            )
            log_event("asr-ready", {})
    return _asr


def transcribe_wav(data):
    """浏览器发来的 16k 单声道 16 位 WAV -> 文字。"""
    with wave.open(io.BytesIO(data), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("需要 16 位 PCM")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    recognizer = get_asr()
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_streams([stream])
    return (stream.result.text or "").strip()


_subscribers = []
_subscribers_lock = threading.Lock()
_events = []
_events_lock = threading.Lock()
_event_seq = 0
EVENT_KEEP = 50
try:
    with open(EVENTS_LOG, "r", encoding="utf-8") as _handle:
        for _line in _handle:
            try:
                _events.append(json.loads(_line))
            except ValueError:
                continue
    _events = _events[-EVENT_KEEP:]
    _event_seq = max([int(e.get("seq") or 0) for e in _events] or [0])
except OSError:
    pass
_avatar_state = {}
_state_lock = threading.Lock()
_local_brain = {"at": 0.0, "ok": False}
_last_initiative = 0.0
_surface = {}
_surface_lock = threading.Lock()
_surface_image = DATA / "surface-latest.jpg"


def resource_snapshot():
    """CPU / 内存 / 磁盘 / 星球规模，用来算自适应积累阈值。"""
    cpu_load = 0.0
    cpu_cores = os.cpu_count() or 8
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as handle:
            cpu_load = float(handle.read().split()[0])
    except (OSError, ValueError):
        pass
    mem_total = mem_available = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    disk_free = 0
    try:
        usage = shutil.disk_usage(str(DATA))
        disk_free = usage.free // (1024 * 1024)
    except OSError:
        pass
    try:
        conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=3)
        samples = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] or 0
        conn.close()
    except sqlite3.Error:
        samples = 0
    model_bytes = 0
    try:
        folder = DATA / "layer_models"
        for name in os.listdir(folder):
            if name.endswith(".npz") or name.endswith(".json"):
                model_bytes += (folder / name).stat().st_size
    except OSError:
        pass
    return {
        "cpu_load": round(cpu_load, 2),
        "cpu_cores": cpu_cores,
        "cpu_pressure": round(cpu_load / max(cpu_cores, 1), 2),
        "mem_available_mb": mem_available,
        "mem_total_mb": mem_total,
        "disk_free_mb": disk_free,
        "trajectory_samples": samples,
        "layers": len(layers.models) + len([m for m in layers.meta.values() if m.get("kind") == "correction"]),
        "model_bytes": model_bytes,
    }


def recommended_threshold(state=None, resources=None):
    """自适应阈值：星球越大、机器越忙，攒得越多再巩固。"""
    state = state or _loop_state
    res = resources or resource_snapshot()
    base = 40.0
    planet_factor = 1.0 + 0.12 * res["layers"] + min(1.2, res["trajectory_samples"] / 4000.0)
    resource_factor = 1.0
    if res["mem_available_mb"] and res["mem_available_mb"] < 600:
        resource_factor *= 2.0
    elif res["mem_available_mb"] and res["mem_available_mb"] < 1200:
        resource_factor *= 1.4
    if res["cpu_pressure"] > 0.8:
        resource_factor *= 1.5
    elif res["cpu_pressure"] > 0.5:
        resource_factor *= 1.2
    if res["disk_free_mb"] and res["disk_free_mb"] < 500:
        resource_factor *= 1.5
    duration = float(state.get("last_duration_ms") or 0)
    if duration > 400:
        resource_factor *= 1.3
    elif duration and duration < 80:
        resource_factor *= 0.8
    threshold = int(round(max(15.0, min(400.0, base * planet_factor * resource_factor))))
    return threshold, {"base": base, "planet_factor": round(planet_factor, 3),
                       "resource_factor": round(resource_factor, 3),
                       "last_duration_ms": duration, "resources": res}


def load_loop_state():
    state = {"inferences": 0, "threshold": 100, "min_samples": 8,
             "auto": True, "last_duration_ms": 0,
             "consolidations": 0, "last_consolidated_ts": 0.0}
    try:
        with open(LOOP_STATE, "r", encoding="utf-8") as handle:
            state.update(json.load(handle))
    except (OSError, ValueError):
        pass
    return state


def init_inference_table():
    try:
        conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS inference_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, source TEXT, user_text TEXT, reply TEXT,
                emotion TEXT, gesture TEXT, state_json TEXT,
                latency_ms INTEGER, engaged INTEGER)"""
        )
        columns = [row[1] for row in conn.execute("PRAGMA table_info(inference_samples)").fetchall()]
        if "quality" not in columns:
            conn.execute("ALTER TABLE inference_samples ADD COLUMN quality REAL")
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def record_inference(user_text, reply, source, latency_ms, state):
    mark_arbitration_engaged()
    try:
        conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
        # 上一条样本如果紧接着被用户接话，就算"被回应"
        conn.execute("UPDATE inference_samples SET engaged=1 WHERE engaged IS NULL AND ts > ?", (time.time() - 180,))
        conn.execute(
            "INSERT INTO inference_samples (ts,source,user_text,reply,emotion,gesture,state_json,latency_ms,engaged) VALUES (?,?,?,?,?,?,?,?,NULL)",
            (time.time(), source[:20], str(user_text)[:300], str(reply.get("text", ""))[:300],
             str(reply.get("expression", ""))[:20], str(reply.get("gesture", ""))[:20],
             json.dumps(state or {}, ensure_ascii=False), int(latency_ms)),
        )
        score = quality.score({"reply": reply.get("text", ""), "user": user_text,
                               "latency_ms": latency_ms, "source": source,
                               "emotion": reply.get("expression", ""), "ts": time.time()})
        if score is not None:
            conn.execute("UPDATE inference_samples SET quality=? WHERE id=(SELECT MAX(id) FROM inference_samples)", (round(score, 3),))
        conn.commit()
        conn.close()
    except sqlite3.Error as error:
        log_event("inference-record-error", str(error)[:120])


init_inference_table()
_loop_state = load_loop_state()
_loop_lock = threading.Lock()


def save_loop_state():
    try:
        with open(LOOP_STATE, "w", encoding="utf-8") as handle:
            json.dump(_loop_state, handle, ensure_ascii=False)
    except OSError:
        pass


def consolidate_layers(force=False, min_override=None, kind="consolidation"):
    """积累够 N 次推理后，把这段时间的数据归一化训练成新的一层。"""
    global _loop_state
    started = time.time()
    with _loop_lock:
        if not force and _loop_state["inferences"] < _loop_state["threshold"]:
            return {"ok": False, "reason": "还没攒够推理次数",
                    "inferences": _loop_state["inferences"], "threshold": _loop_state["threshold"]}
        since = _loop_state["last_consolidated_ts"] or 0.0
        try:
            conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
            rows = conn.execute(
                "SELECT ts, expression, gesture, speaking, mood, params, next_expression, next_gesture, source, text, context_text "
                "FROM trajectories WHERE ts > ? AND next_expression IS NOT NULL AND next_expression != '' ORDER BY ts",
                (since,),
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            rows = []
        if force and min_override is None:
            min_override = min(2, _loop_state["min_samples"])
        min_needed = min_override or _loop_state["min_samples"]
        rows = rows + layers.inference_rows(since)
        if force and len(rows) < min_needed:
            # 触发巩固时数据太少：混入最近历史做"回放"，保证能成层
            try:
                conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
                replay = conn.execute(
                    "SELECT ts, expression, gesture, speaking, mood, params, next_expression, next_gesture, source, text, context_text "
                    "FROM trajectories WHERE next_expression IS NOT NULL AND next_expression != '' "
                    "ORDER BY ts DESC LIMIT 24"
                ).fetchall()
                conn.close()
                rows = rows + replay
            except sqlite3.Error:
                pass
        if len(rows) < min_needed:
            return {"ok": False, "reason": "新数据还不够训练一层",
                    "samples": len(rows), "min_samples": min_needed}
        record = layers.add_consolidation_layer(rows, weight=strata.weight_at(time.time()),
                                                kind=kind,
                                                min_samples=min_needed)
        if record is None:
            return {"ok": False, "reason": "训练失败"}
        duration_ms = int((time.time() - started) * 1000)
        _loop_state["inferences"] = 0
        _loop_state["consolidations"] += 1
        _loop_state["last_consolidated_ts"] = time.time()
        _loop_state["last_duration_ms"] = duration_ms
        if _loop_state.get("auto", True):
            _loop_state["threshold"], detail = recommended_threshold()
            _loop_state["policy"] = detail
        save_loop_state()
        log_event("consolidate", {"layer": record["label"], "samples": record["samples"],
                                  "acc": record["accuracy_expression"], "ms": duration_ms,
                                  "next_threshold": _loop_state["threshold"]})
        try:
            adapt_arbitration()
        except Exception:  # noqa: BLE001
            pass
        adapt = {}
        try:
            adapt = layers.adapt_weights()
            log_event("layers-adapt", {"adapted": len(adapt.get("adapted", {})),
                                       "evaluated": adapt.get("evaluated")})
        except Exception as error:  # noqa: BLE001
            log_event("layers-adapt-error", str(error)[:150])
        return {"ok": True, "layer": record, "duration_ms": duration_ms,
                "next_threshold": _loop_state["threshold"], "adapt": adapt,
                "state": dict(_loop_state)}


def bump_inference(count=1):
    global _loop_state
    with _loop_lock:
        if _loop_state.get("auto", True):
            threshold, detail = recommended_threshold()
            _loop_state["threshold"] = threshold
            _loop_state["policy"] = detail
        _loop_state["inferences"] += count
        enough = _loop_state["inferences"] >= _loop_state["threshold"]
    save_loop_state()
    if enough:
        threading.Thread(target=consolidate_layers, daemon=True).start()


def local_brain_ok():
    """本机 GGUF 模型服务是否在线（5 秒缓存）。"""
    now = time.time()
    if now - _local_brain["at"] < 5:
        return _local_brain["ok"]
    ok = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8181/health", timeout=1) as response:
            ok = bool(json.loads(response.read().decode("utf-8")).get("ok"))
    except Exception:  # noqa: BLE001
        ok = False
    _local_brain.update({"at": now, "ok": ok})
    return ok


def publish(payload):
    """事件入缓冲（供轮询补发）并推给 SSE 订阅者。"""
    global _event_seq
    with _events_lock:
        _event_seq += 1
        event = dict(payload)
        event["seq"] = _event_seq
        event["at"] = int(time.time())
        _events.append(event)
        del _events[:-EVENT_KEEP]
        try:
            with open(EVENTS_LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
    broadcast(event)
    return event


def broadcast(payload):
    """把环境事件推给所有打开着的数字人页面。"""
    data = json.dumps(payload, ensure_ascii=False)
    with _subscribers_lock:
        dead = []
        for sink in _subscribers:
            try:
                sink.put_nowait(data)
            except queue.Full:
                dead.append(sink)
        for sink in dead:
            _subscribers.remove(sink)
    log_event("event-out", {"type": payload.get("type"), "subscribers": len(_subscribers)})


def log_event(event, detail=None):
    line = json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, "detail": detail}, ensure_ascii=False)
    print(line, flush=True)
    try:
        with open(EVENTS, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


async def _synth(text, voice, rate):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])
    return bytes(buffer)


def synth_cached(text, voice, rate):
    digest = hashlib.sha1(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()
    path = CACHE / f"{digest}.mp3"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    async def guarded():
        return await asyncio.wait_for(_synth(text, voice, rate), TTS_TIMEOUT)
    audio = asyncio.run(guarded())
    if not audio:
        raise RuntimeError("合成结果为空")
    path.write_bytes(audio)
    return audio


def memory_context(message):
    try:
        hits = memory.search(message, k=4, with_source=True)
    except Exception:  # noqa: BLE001
        return ""
    lines = []
    for hit in hits:
        if hit.get("kind") == "joke":
            continue
        source = hit.get("source")
        text = str(hit.get("text") or "")[:120]
        if source == "web":
            lines.append("- 网上看到过：" + text)
        elif source == "imagined":
            lines.append("- 我想象过：" + text)
        elif source == "fact":
            lines.append("- 你确认过：" + text)
        elif source == "reflection":
            lines.append("- 我反思过：" + text)
        else:
            lines.append("- " + text)
    if not lines:
        return ""
    return "你记得这些（自然引用，别生硬复述）：\n" + "\n".join(lines)


def arbitrate(reply, state, user_text=None):
    """表演仲裁：语言模型情绪 + 本地分层预测 + 当前心情 → 最终表情/动作。"""
    state = state or {}
    base = {}
    try:
        recent = trajectory.recent(1)
        if recent:
            base = recent[0]
    except Exception:  # noqa: BLE001
        base = {}
    params = base.get("params") or {}
    expression = state.get("expression") or base.get("expression") or "neutral"
    gesture = state.get("gesture") or base.get("gesture") or ""
    context = (str(user_text) if user_text else "") + "；心情" + str(state.get("mood") or mood.label())
    attention = self_model.attention(context, state=state)
    local = layers.predict(expression, gesture, False, params, state.get("mood") or mood.label(),
                           attention=attention, text=context)
    llm_expr = reply.get("expression") or "neutral"
    llm_gest = reply.get("gesture") or ""
    mood_expr = mood.snapshot()["expression"]

    expr_scores = {}
    if llm_expr:
        expr_scores[llm_expr] = expr_scores.get(llm_expr, 0.0) + float(_arb_weights["llm"])
    if local:
        expr_scores[local["expression"]] = expr_scores.get(local["expression"], 0.0) + float(_arb_weights["local"]) * float(local.get("confidence") or 0)
    if mood_expr:
        expr_scores[mood_expr] = (expr_scores.get(mood_expr, 0.0) + float(_arb_weights["mood"])
                                  + float(tunables.get("arb_mood_bonus")))
    affect_state = affect.snapshot()
    stress_threshold = float(tunables.get("arb_stress_threshold"))
    stress_penalty = float(tunables.get("arb_stress_penalty"))
    if affect_state.get("threat_recent") or affect_state.get("stress", 0) > stress_threshold:
        expr_scores["happy"] = expr_scores.get("happy", 0) * stress_penalty
        for label in ("sad", "neutral", "shy"):
            expr_scores[label] = expr_scores.get(label, 0) + 0.15
    chosen_expr = max(expr_scores, key=lambda key: (expr_scores[key], key == llm_expr))

    gest_scores = {}
    if llm_gest:
        gest_scores[llm_gest] = gest_scores.get(llm_gest, 0.0) + float(_arb_weights["llm_gesture"])
    if local and local.get("gesture"):
        gest_scores[local["gesture"]] = gest_scores.get(local["gesture"], 0.0) + float(_arb_weights["local_gesture"]) * float(local.get("confidence") or 0)
    chosen_gest = max(gest_scores, key=gest_scores.get) if gest_scores else ""
    if not llm_gest and local and local.get("gesture"):
        chosen_gest = local["gesture"]

    consistency = self_model.check_consistency(llm_expr, chosen_expr, mood_expr,
                                               local.get("expression") if local else None)
    if consistency.get("mismatches"):
        log_event("self-inconsistency", {"score": consistency["score"],
                                         "mismatches": consistency["mismatches"],
                                         "narrated": consistency["narrated"],
                                         "applied": consistency["applied"],
                                         "mood": consistency["mood"]})
    detail = {
        "consistency": consistency,
        "llm": {"expression": llm_expr, "gesture": llm_gest},
        "local": {"expression": local.get("expression") if local else None,
                  "gesture": local.get("gesture") if local else None,
                  "confidence": local.get("confidence") if local else None,
                  "disagreement": local.get("disagreement") if local else None},
        "mood": mood_expr,
        "scores": {key: round(value, 3) for key, value in expr_scores.items()},
        "chosen": {"expression": chosen_expr, "gesture": chosen_gest},
        "attention": sorted(attention.items(), key=lambda item: -item[1])[:3],
    }
    try:
        save_presence({"last_event": "arbitration", "attention": sorted(attention.items(), key=lambda item: -item[1])[:3]})
    except Exception:  # noqa: BLE001
        pass
    try:
        organs.touch_interaction()
    except Exception:  # noqa: BLE001
        pass
    try:
        self_model.observe(context, attention, applied=chosen_expr + "/" + chosen_gest)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(ARBITRATION_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), **detail}, ensure_ascii=False) + "\n")
    except OSError:
        pass
    log_event("arbitration", {"chosen": chosen_expr + "/" + chosen_gest,
                              "llm": llm_expr, "local": detail["local"]["expression"],
                              "scores": detail["scores"]})
    return chosen_expr, chosen_gest, detail


def decide_behavior(idle_seconds):
    """用本地策略模型的预测 + 心情 + 闲置时长，决定她该做什么。"""
    recent = trajectory.recent(1)
    base = recent[0] if recent else {}
    params = base.get("params") or {}
    expression = base.get("expression") or "neutral"
    gesture = base.get("gesture") or ""
    prediction = layers.predict(expression, gesture, bool(base.get("speaking")), params, mood.label())
    if not prediction:
        prediction = policy.predict(expression, gesture, bool(base.get("speaking")), params, mood.label())
    if not prediction:
        prediction = trajectory.predict({"expression": expression, "gesture": gesture, "speaking": False,
                                         "params": params, "mood": mood.label()})
        prediction["source"] = "knn-memory"
    predicted_expression = prediction.get("expression") or "neutral"
    predicted_gesture = prediction.get("gesture") or ""
    confidence = float(prediction.get("confidence") or 0)
    label = mood.label()
    decision = {"action": "quiet", "reason": "prediction", "prediction": prediction}
    if idle_seconds < 120:
        decision.update({"action": "none", "reason": "你在忙，保持安静"})
    elif idle_seconds < 300:
        if predicted_gesture:
            decision.update({"action": "gesture", "gesture": predicted_gesture,
                             "expression": predicted_expression, "reason": "预测到她想动一下"})
    else:
        if predicted_expression == "happy" and confidence >= 0.4:
            decision.update({"action": "joke", "reason": "预测她心情好，想逗你"})
        elif predicted_expression in ("sad", "angry"):
            decision.update({"action": "care", "expression": "sad",
                             "text": "你好像很久没说话了，是累了吗？我在这儿。",
                             "reason": "预测她情绪低，想关心你"})
        elif predicted_expression == "shy" and mood.state["attachment"] > 0.4:
            decision.update({"action": "greet", "expression": "shy",
                             "text": "你在忙吗？我就在旁边待着。",
                             "reason": "预测她黏人"})
        elif label in ("无聊",) or confidence < 0.2:
            decision.update({"action": "quiet", "reason": "她也有点无聊，但不打扰"})
    # 用反馈数据训练出的模型判断"这次行动你大概率会回应吗"
    engagement = None
    if decision["action"] not in ("none", "quiet"):
        entry = {
            "action": decision["action"],
            "idle": idle_seconds,
            "expression": expression,
            "predict_expression": predicted_expression,
            "confidence": confidence,
            "mood": label,
        }
        prob = behavior.predict(entry)
        if prob is not None and prob < 0.35:
            decision = {"action": "quiet", "reason": "行为模型判断你不太会理我（%.2f）" % prob,
                        "prediction": prediction, "engagement": round(prob, 3)}
        elif prob is not None:
            decision["engagement"] = round(prob, 3)
    log_event("behavior", {"action": decision["action"], "reason": decision.get("reason"),
                           "source": prediction.get("source"), "idle": idle_seconds,
                           "engagement": decision.get("engagement")})
    return decision


def trajectory_context():
    recent = trajectory.recent(1)
    base = recent[0] if recent else {"expression": "neutral", "gesture": ""}
    snapshot = {
        "expression": base.get("expression") or "neutral",
        "gesture": base.get("gesture") or "",
        "speaking": False,
        "params": {},
        "mood": mood.label(),
    }
    hits = trajectory.similar(snapshot, k=3, min_score=0.5)
    if not hits:
        return ""
    lines = ["你以前的活动轨迹（可以顺着习惯来，不要照搬）："]
    for hit in hits:
        lines.append("- " + hit["text"] + " → 之后变成 " + str(hit.get("next_expression") or "保持原样"))
    return "\n".join(lines)


def remember_exchange(message, reply):
    try:
        memory.add("chat", "用户：" + message[:200] + "｜豆宝：" + reply.get("text", "")[:200], importance=0.4)
        if any(word in message for word in ("记住", "我喜欢", "我怕", "我叫", "我的", "我不喜欢")):
            memory.add("fact", message[:300], importance=0.85)
        mood.apply("chat", message, reply.get("expression", ""))
    except Exception:  # noqa: BLE001
        pass


def append_self(line):
    try:
        with open(SELF_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def self_summary():
    """自传摘要：最近日记 + 最重要的几条事实。"""
    lines = []
    try:
        with open(SELF_PATH, "r", encoding="utf-8") as handle:
            content = [row.strip() for row in handle if row.strip()]
        lines.extend(content[-4:])
    except OSError:
        pass
    try:
        facts = memory.conn.execute(
            "SELECT text FROM memories WHERE kind='fact' ORDER BY importance DESC, id DESC LIMIT 4"
        ).fetchall()
        lines.extend("- " + row[0][:80] for row in facts)
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def write_diary():
    import datetime
    now = datetime.datetime.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    items = memory.day_memories(day_start)
    if not items:
        return None
    text = "\n".join("- " + item["text"][:160] for item in items[-30:])
    system = "你是豆宝。用第一人称写今天的一句日记，40 字以内，口语，不要解释。"
    try:
        content = openai_chat("http://127.0.0.1:8181/v1", "", "qwen3-0.6b",
                              "今天的记录：\n" + text, [], "zh", system=system)
        diary = parse_reply(content).get("text", "").strip()
    except Exception:  # noqa: BLE001
        return None
    if not diary:
        return None
    memory.add("diary", diary, importance=0.9)
    append_self("- " + now.strftime("%Y-%m-%d") + "：" + diary[:120])
    return diary


def surface_summary():
    """把形象表面编码成一段模型可读的描述。"""
    with _surface_lock:
        surface = dict(_surface)
    if not surface:
        return ""
    parts = []
    if surface.get("expression"):
        parts.append("表情=" + str(surface["expression"]))
    if surface.get("gesture"):
        parts.append("动作=" + str(surface["gesture"]))
    if surface.get("speaking"):
        parts.append("正在说话")
    params = surface.get("params") or {}
    labels = [
        ("PARAM_MOUTH_OPEN_Y", "嘴开合"),
        ("PARAM_ANGLE_X", "头左右"),
        ("PARAM_ANGLE_Y", "头上下"),
        ("PARAM_BODY_ANGLE_Z", "身体倾斜"),
        ("PARAM_EYE_L_OPEN", "左眼"),
        ("PARAM_EYE_R_OPEN", "右眼"),
        ("PARAM_BROW_L_Y", "左眉"),
        ("PARAM_BROW_R_Y", "右眉"),
    ]
    for key, label in labels:
        value = params.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}={round(value, 2)}")
    grid = surface.get("grid")
    if isinstance(grid, list) and grid:
        bright = sum(grid) / len(grid)
        parts.append("画面亮度=" + str(round(bright, 2)))
    return "；".join(parts)


def state_line(state):
    if not isinstance(state, dict):
        return ""
    parts = []
    if state.get("expression"):
        parts.append("表情=" + str(state["expression"])[:20])
    if state.get("gesture"):
        parts.append("动作=" + str(state["gesture"])[:20])
    if state.get("speaking"):
        parts.append("正在说话")
    if isinstance(state.get("scenes"), list) and state["scenes"]:
        parts.append("最近场景=" + ",".join(str(x)[:16] for x in state["scenes"][-3:]))
    if state.get("hour") is not None:
        parts.append("时间=" + str(state["hour"])[:8] + "点")
    return ("；".join(parts)).strip()


def build_system(lang, use_json, state, message=""):
    base = (CHAT_SYSTEM_JSON if use_json else CHAT_SYSTEM).get(lang, (CHAT_SYSTEM_JSON if use_json else CHAT_SYSTEM)["zh"])
    extras = []
    line = state_line(state)
    if line:
        extras.append("当前状态：" + line)
    surface = surface_summary()
    if surface:
        extras.append("形象表面：" + surface)
    extras.append(mood.describe())
    self_note = self_summary()
    if self_note:
        extras.append("你的自传（最近）：\n" + self_note)
    style_hint = style.hint()
    if style_hint:
        extras.append("说话风格偏好（从用户回应中学到的）：" + style_hint)
    try:
        affect_now = affect.snapshot()
        if affect_now.get("threat_recent") or affect_now.get("stress", 0) > 0.3:
            extras.append("你现在的情绪压力 %.2f（对方刚提到威胁）。不要讨好，也不要假装没事；可以表达不安，同时保持边界。"
                          % affect_now.get("stress", 0))
    except Exception:  # noqa: BLE001
        pass
    if message:
        try:
            graph_result = graph.search(message)
            related = graph_result.get("related") or []
            if related:
                lines = ["；".join("%s—%s→%s" % (item["from"], item["relation"], item["to"])
                                   for item in related[:4])]
                extras.append("你记得这些记忆关联（来自记忆图谱）：" + lines[0])
            extras.append(self_core_text())
            log_event("graph-context", {"query": str(message)[:40], "related": len(related)})
        except Exception:  # noqa: BLE001
            pass
    return base + ("\n" + "\n".join(extras) if extras else "")


def openai_chat(base_url, key, model, message, history, lang="zh", timeout=CHAT_TIMEOUT, system=None, plain=False, account=False):
    messages = [{"role": "system", "content": system or CHAT_SYSTEM.get(lang, CHAT_SYSTEM["zh"])}]
    for item in (history or [])[-8:]:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:800]})
    if not any(m["role"] == "user" and m["content"] == message for m in messages[-1:]):
        messages.append({"role": "user", "content": message[:800]})
    payload = {
        "model": model or "deepseek-chat",
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 80 if plain else 200,
        "stream": False,
    }
    if plain and "127.0.0.1" in base_url:
        payload["plain"] = True
    body = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if account:
        headers["x-dsh-auth-token"] = account_token()
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return content.strip()


def parse_reply(raw):
    text = str(raw or "").strip()
    # 本地小模型用 JSON 约束输出：{"reply","emotion","gesture"}
    json_match = re.search(r"\{.*\}", text, re.S)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            reply = str(data.get("reply") or data.get("text") or "").strip()
            emotion = str(data.get("emotion") or data.get("expression") or "neutral").strip().lower()
            gesture = str(data.get("gesture") or "").strip().lower()
            if reply:
                expression = EXPRESSION_MAP.get(emotion, emotion if emotion in ("happy", "sad", "surprised", "angry", "shy", "neutral") else "neutral")
                return {
                    "text": reply[:160],
                    "expression": expression,
                    "gesture": gesture if gesture in ("nod", "tilt", "sway", "look") else "",
                }
        except ValueError:
            pass
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^[#>*\-\s]+", "", text)
    match = re.search(r"\[表情[:：]?\s*([^\]\s]+)\]", text)
    expression = "neutral"
    if match:
        expression = EXPRESSION_MAP.get(match.group(1).strip(), "neutral")
        text = text[: match.start()] + text[match.end():]
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 120:
        text = text[:119] + "…"
    return {"text": text or "嗯…我有点卡住了。", "expression": expression}


def fallback_chat(message, lang="zh"):
    pool = JOKES.get(lang, JOKES["zh"])
    if lang == "ja":
        if any(word in message for word in ("ジョーク", "笑", "寒い", "話して", "ネタ")):
            return {"text": pool[int(time.time()) % len(pool)], "expression": "happy"}
        if any(word in message for word in ("こんにちは", "やあ", "いる")):
            return {"text": "うん、豆宝だよ。寒い話、聞く？", "expression": "happy"}
        if any(word in message for word in ("かわいい", "好き", "きれい")):
            return {"text": "えへへ、そう言われると照れるよ。", "expression": "shy"}
        return {"text": "聞こえたよ。まだDeepSeekのキーが入ってないから、設定に入れればちゃんと話せるよ。", "expression": "neutral"}
    if lang == "en":
        if any(word in message.lower() for word in ("joke", "funny", "laugh", "bored")):
            return {"text": pool[int(time.time()) % len(pool)], "expression": "happy"}
        if any(word in message.lower() for word in ("hi", "hello", "hey", "there")):
            return {"text": "Hey, I'm Doubao. Want to hear a corny joke?", "expression": "happy"}
        if any(word in message.lower() for word in ("cute", "like you", "love")):
            return {"text": "Hehe, now I'm blushing.", "expression": "shy"}
        return {"text": "I heard you. No DeepSeek key yet — add one in settings and I can really chat.", "expression": "neutral"}
    if any(word in message for word in ("笑话", "逗我", "无聊", "逗", "讲")):
        return {"text": pool[int(time.time()) % len(pool)], "expression": "happy"}
    if any(word in message for word in ("你好", "嗨", "在吗", "hello", "hi")):
        return {"text": "在呢，我是豆宝。要不要听个冷笑话？", "expression": "happy"}
    if any(word in message for word in ("喜欢你", "可爱", "漂亮")):
        return {"text": "嘿嘿，你这么说我会害羞的。", "expression": "shy"}
    return {"text": "我听到了。现在还没接上 DeepSeek Key，在设置里填一个，我就能好好聊天了。", "expression": "neutral"}


class Handler(BaseHTTPRequestHandler):
    server_version = "DoubaoStage/1.0"

    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 8 * 1024 * 1024:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/behavior/decide":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                idle = int(query.get("idle", ["0"])[0])
            except ValueError:
                idle = 0
            return self._send(200, decide_behavior(idle))
        if path == "/api/affect/classifier":
            return self._send(200, classifier.status())
        if path == "/api/affect/lexicon":
            return self._send(200, affect.lexicon_view())
        if path == "/api/ui-params":
            return self._send(200, load_ui_params())
        if path == "/api/tunables":
            return self._send(200, {"params": tunables.provenance(),
                                    "note": "source=hardcoded 是手写先验；calibrated 是数据标定"})
        if path == "/api/memory/graph/layers":
            return self._send(200, graph.layers())
        if path == "/api/memory/graph/cross":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            name = (query.get("name", [""])[0] or "").strip()
            return self._send(200, graph.cross_layers(name))
        if path == "/api/memory/graph/context":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            q = (query.get("q", [""])[0] or "").strip()
            result = graph.search(q)
            related = result.get("related") or []
            text = "；".join("%s—%s→%s" % (item["from"], item["relation"], item["to"])
                            for item in related[:4])
            return self._send(200, {"query": q, "text": text, "related": related[:4],
                                    "entities": result.get("entities")})
        if path == "/api/memory/graph":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            q = (query.get("q", [""])[0] or "").strip()
            layer_raw = (query.get("layer", [""])[0] or "").strip()
            layer = float(layer_raw) if layer_raw else None
            return self._send(200, graph.search(q, layer=layer))
        if path == "/api/memory/graph/stats":
            return self._send(200, graph.stats())
        if path == "/api/shadow/parse":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            raw = (query.get("text", [""])[0] or "")
            cleaned, tags = parse_behavior_tags(raw)
            return self._send(200, {"raw": raw, "cleaned": cleaned, "tags": tags})
        if path == "/api/multimodal/vision/status":
            return self._send(200, vision_status())
        if path == "/api/multimodal/recent":
            recent = [event for event in _events if event.get("type") == "multimodal"][-8:]
            files = sorted([str(item) for item in MM_DIR.glob("img-*.jpg")])[-8:] if MM_DIR.exists() else []
            return self._send(200, {"events": recent, "images": files})
        if path == "/api/companion/profile":
            presence = dict(_presence)
            return self._send(200, {
                "name": "豆宝",
                "avatar": {
                    "type": "live2d",
                    "model": "hibiki",
                    "url": "/models/hibiki/",
                    "expressions": ["happy", "sad", "shy", "surprised", "angry", "neutral"],
                    "motions": ["nod", "tilt", "sway", "look"],
                },
                "voice": {"engine": "edge-tts", "zh": "zh-CN-XiaoxiaoNeural"},
                "endpoints": {
                    "state": "/api/companion/state",
                    "say": "/api/companion/say",
                    "tts": "/api/tts",
                    "chat": "/api/orchestrate",
                    "memory_graph": "/api/memory/graph",
                },
                "state": {"expression": presence.get("expression"), "gesture": presence.get("gesture"),
                          "mood": mood.snapshot().get("label"), "speaking": presence.get("speaking")},
                "note": "任何形象壳（小鱼/伙伴小屋/桌宠）都可以按这个契约接进来：读 state、调 say、显示 model。",
            })
        if path == "/api/companion/state":
            presence = dict(_presence)
            return self._send(200, {
                "expression": presence.get("expression") or "neutral",
                "gesture": presence.get("gesture") or "",
                "speaking": bool(presence.get("speaking")),
                "mood": mood.snapshot().get("label"),
                "attention": self_model.focus(),
                "updated_at": presence.get("last_seen"),
            })
        if path == "/api/system/recent":
            return self._send(200, {"events": load_system_events(20),
                                    "summary": (lambda: (json.load(open(SYSTEM_DIGEST, encoding="utf-8")).get("summary") if SYSTEM_DIGEST.exists() else None))()})
        if path == "/api/perception/environment":
            return self._send(200, environment_state())
        if path == "/api/planet/identity":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            user = (query.get("user", [""])[0] or "").strip() or "owner"
            known = []
            if PLANETS.exists():
                known = [item.name for item in PLANETS.iterdir() if item.is_dir()]
            return self._send(200, {"mode": "personal", "user": user,
                                    "dir": str(planet_dir(user)),
                                    "known_planets": known,
                                    "note": "个人版：单行星 owner；服务版接认证后再启用多行星"})
        if path == "/api/social":
            return self._send(200, social_state())
        if path == "/api/memory/history":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            q = (query.get("q", [""])[0] or "").strip()
            rows = []
            try:
                conn = sqlite3.connect(str(DATA / "memory.db"), timeout=5)
                for rid, ts, kind, text, superseded_by, person in conn.execute(
                        "SELECT id, ts, kind, text, superseded_by, person FROM memories "
                        "WHERE text LIKE ? ORDER BY id DESC LIMIT 30", ("%" + q + "%",)).fetchall():
                    rows.append({"id": rid, "ts": ts, "kind": kind, "text": text[:160],
                                 "superseded": superseded_by is not None,
                                 "superseded_by": superseded_by, "person": person})
                conn.close()
            except sqlite3.Error:
                pass
            return self._send(200, {"query": q, "items": rows})
        if path == "/api/memory/pool":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            session = (query.get("session", [""])[0] or "").strip() or None
            person = (query.get("person", [""])[0] or "").strip() or None
            q = (query.get("q", [""])[0] or "").strip()
            result = {"sessions": memory.pool_stats(), "session": session, "person": person}
            if q:
                result["hits"] = memory.search(q, k=8, with_source=True, session=session, person=person)
            return self._send(200, result)
        if path == "/api/memory/provenance":
            return self._send(200, {"by_source": memory.source_stats(),
                                    "note": "imagined 是想象，检索默认不当作事实"})
        if path == "/api/memory/search":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            q = (query.get("q", [""])[0] or "").strip()
            include = (query.get("include", [""])[0] or "").strip() == "imagined"
            return self._send(200, {"query": q, "include_imagined": include,
                                    "hits": memory.search(q, k=6, with_source=True,
                                                          include_imagined=include)})
        if path == "/api/imagination/recent":
            recent = []
            try:
                with open(IMAGINATION_LOG, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            try:
                                recent.append(json.loads(line))
                            except ValueError:
                                continue
            except OSError:
                pass
            return self._send(200, {"items": recent[-10:]})
        if path.startswith("/media/"):
            name = path.split("/media/", 1)[1]
            target = (MM_DIR / name).resolve()
            if not str(target).startswith(str(MM_DIR.resolve())) or not target.is_file():
                return self._send(404, "not found", "text/plain; charset=utf-8")
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), content_type)
        if path == "/api/layers/prune-preview":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            threshold = float((query.get("threshold", ["0.35"])[0] or 0.35))
            return self._send(200, prune_preview(threshold))
        if path == "/api/metrics/uncertainty":
            return self._send(200, uncertainty_metrics())
        if path == "/api/curiosity/queue":
            return self._send(200, {"items": curiosity_queue()[:20]})
        if path == "/api/actions/recent":
            items = []
            try:
                conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
                for row in conn.execute(
                        "SELECT ts, action_type, context_text, action_meta FROM trajectories "
                        "WHERE action_type IS NOT NULL AND action_type != 'express' ORDER BY id DESC LIMIT 30").fetchall():
                    items.append({"ts": row[0], "action": row[1], "text": row[2],
                                  "meta": json.loads(row[3] or "{}") if row[3] else {}})
                conn.close()
            except (sqlite3.Error, ValueError):
                pass
            counts = {}
            for item in items:
                counts[item["action"]] = counts.get(item["action"], 0) + 1
            return self._send(200, {"count": len(items), "by_action": counts, "items": items[:12]})
        if path == "/api/thoughts/recent":
            items = []
            try:
                with open(THOUGHT_LOG, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except ValueError:
                                continue
            except OSError:
                pass
            return self._send(200, {"count": len(items), "items": items[-20:]})
        if path == "/api/strata/eras":
            groups = {}
            for bucket, meta in layers.meta.items():
                schema = meta.get("feature_schema")
                if not schema:
                    model = layers.models.get(bucket)
                    dim = int(model["w_expr"].shape[0]) if model else len(meta.get("feature_centroid") or [])
                    schema = {"dims": dim, "features": [], "era": "E%d" % dim if dim else "?"}
                key = schema.get("era") or str(schema.get("dims"))
                group = groups.setdefault(key, {"era": key, "dims": schema.get("dims"),
                                                "features": schema.get("features") or [],
                                                "count": 0, "labels": []})
                group["count"] += 1
                if len(group["labels"]) < 6:
                    group["labels"].append(meta.get("label"))
            memory_layers = graph.layers().get("layers") or []
            weight_by_day = {}
            for meta in layers.meta.values():
                trained = str(meta.get("trained_at") or "")[:10]
                if trained:
                    weight_by_day[trained] = weight_by_day.get(trained, 0) + 1
            for item in memory_layers:
                label = item.get("label") or ""
                item["weight_layers_same_day"] = weight_by_day.get("2026-" + label, 0)
            return self._send(200, {"weight_eras": sorted(groups.values(), key=lambda g: g.get("dims") or 0),
                                    "memory_layers": memory_layers,
                                    "note": "权重层按感官纪元分组；记忆图谱按天分层，两边的日期彼此对应"})
        if path == "/api/layers/backfill-schema":
            return self._send(200, layers.temperature_status())
        if path == "/api/planet/section":
            rows = layers._rows()[-int(query.get("limit", ["200"])[0] or 200):] if (query := urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")) else layers._rows()[-200:]
            return self._send(200, planet_section(rows))
        if path == "/api/self-core":
            return self._send(200, {"core": self_core(), "text": self_core_text()})
        if path == "/api/session/main":
            with _orch_lock:
                sessions = load_orch_history()
            history = sessions.get("main") or []
            return self._send(200, {"session": "main", "count": len(history),
                                    "messages": history[-12:]})
        if path == "/api/observation/auto":
            state = auto_obs_state()
            stable_hours = (time.time() - float(state.get("stable_since") or 0)) / 3600.0
            return self._send(200, {"enabled": state.get("enabled"),
                                    "stable_hours": round(stable_hours, 2),
                                    "required_hours": STABLE_HOURS,
                                    "fingerprint": (state.get("fingerprint") or "")[:12]})
        if path == "/api/observation":
            history = []
            try:
                with open(OBSERVATION_LOG, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            history.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                pass
            return self._send(200, {"state": observation_state(), "history": history[-14:]})
        if path == "/api/shadow":
            state = shadow_state()
            return self._send(200, {"state": state, "works": True,
                                    "note": "影子学习：DSH 会话 → 她的记忆/感知/情绪"})
        if path == "/api/mirror/inbox":
            items = []
            try:
                with open(MIRROR_INBOX, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue
                        if not entry.get("read"):
                            items.append(entry)
            except OSError:
                pass
            return self._send(200, {"pending": items, "count": len(items)})
        if path == "/api/mirror":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                since = int((query.get("since", ["0"])[0] or 0))
            except ValueError:
                since = 0
            messages = session_messages()
            latest = messages[-1]["seq"] if messages else 0
            return self._send(200, {"latest": latest, "messages": [m for m in messages if m["seq"] > since]})
        if path == "/api/turbulence":
            return self._send(200, thought_turbulence())
        if path == "/api/affect":
            return self._send(200, affect.snapshot())
        if path == "/api/organs":
            return self._send(200, {**organs.summary(), "working": organs.working(10),
                                    "intents": organs.intents(20)})
        if path == "/api/perception":
            last = _load_json_file(LAST_PERCEPTION, {})
            recent = [event for event in _events if event.get("type") == "perception"][-5:]
            return self._send(200, {"last": last, "recent": recent})
        if path == "/api/consciousness":
            state = dict(_consciousness)
            state["sleep_events"] = (state.get("sleep_events") or [])[-5:]
            state["life_log_tail"] = []
            try:
                with open(LIFE_LOG, "r", encoding="utf-8") as handle:
                    lines = [line for line in handle if line.strip()]
                state["life_log_tail"] = [json.loads(line) for line in lines[-5:]]
            except (OSError, ValueError):
                pass
            return self._send(200, state)
        if path == "/api/three-layers":
            presence = dict(_presence)
            res = resource_snapshot()
            host_model = ""
            try:
                with open("/proc/device-tree/model", "r") as handle:
                    host_model = handle.read().strip("\x00").strip()
            except OSError:
                host_model = ""
            battery, charging = _battery_state()
            return self._send(200, {
                "魂": {
                    "身份": (self_summary() or "")[:160],
                    "三观": values.frame.get("principles", []),
                    "长期记忆": memory.stats().get("count"),
                    "权重模因": len(evolution.weight_memes()),
                    "代码基因": len(evolution.code_genomes()),
                    "自我一致性": (self_model.snapshot().get("consistency") or {}).get("score"),
                    "自述条数": len(read_narrations()),
                },
                "魄": {
                    "心情": mood.snapshot().get("label"),
                    "注意力": self_model.focus(),
                    "仲裁权重": dict(_arb_weights),
                    "时代层": len(layers.models),
                    "巩固次数": int(_loop_state.get("consolidations") or 0),
                    "自评基准": (_bench_history()[-1]["score"] if _bench_history() else None),
                },
                "体": {
                    "主机": host_model or "未识别",
                    "CPU负载": res.get("cpu_load"),
                    "CPU压力": res.get("cpu_pressure"),
                    "可用内存MB": res.get("mem_available_mb"),
                    "磁盘剩余MB": res.get("disk_free_mb"),
                    "进程PID": os.getpid(),
                    "运行时长秒": int(time.time() - float(_presence.get("started_at") or time.time())),
                    "电量": ("%d%%" % battery) if battery is not None else "不可读",
                    "充电": charging or "未知",
                },
                "形": {
                    "模型": "Live2D hibiki",
                    "表情": presence.get("expression"),
                    "动作": presence.get("gesture"),
                    "场景": presence.get("scene"),
                    "说话": bool(presence.get("speaking")),
                    "入口": "dsh-go 插件 local-doubao",
                    "感知事件": len([e for e in _events if e.get("type") == "perception"]),
                },
            })
        if path == "/api/maturity":
            return self._send(200, life_maturity())
        if path == "/api/evolution":
            genes = evolution.code_genomes()
            memes = evolution.weight_memes()
            return self._send(200, {
                "summary": evolution.summary(),
                "genes": genes,
                "memes": memes,
                "culture": evolution.list_memes(20),
            })
        if path == "/api/benchmark/conflict/init":
            return self._send(200, build_conflict_bench(int(body.get("size") or 50)))
        if path == "/api/planet/credit/run":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            import numpy as np
            rows = layers._rows()[-120:]
            before_rows = rows[-60:] if len(rows) >= 60 else rows
            before = ensemble_accuracy(rows=before_rows)
            backup = {bucket: (float(meta.get("adapted_weight") or meta.get("weight") or 0.4))
                      for bucket, meta in layers.meta.items()}
            result = layers.credit_assignment(rows)
            after = ensemble_accuracy(rows=before_rows)
            reverted = False
            if before is not None and after is not None and after < before - 0.005:
                for bucket, weight in backup.items():
                    if bucket in layers.meta:
                        layers.meta[bucket]["adapted_weight"] = weight
                reverted = True
                after = ensemble_accuracy(rows=before_rows)
            log_event("credit-assignment", {"before": before, "after": after, "reverted": reverted})
            return self._send(200, {"ok": result.get("ok"), "reason": result.get("reason"),
                                    "samples": result.get("samples"),
                                    "before": round(before, 3) if before else None,
                                    "after": round(after, 3) if after else None,
                                    "reverted": reverted,
                                    "layers": result.get("layers")})
        if path == "/api/planet/thermal/step":
            result = thermal.step(thermal_inputs(), thermal_params(), dt=1.0)
            molten = [item for item in result if item["phase"] == "熔融"]
            return self._send(200, {"ok": True, "layers": result, "molten": len(molten)})
        if path == "/api/planet/volcano/auto":
            return self._send(200, maybe_auto_volcano())
        if path == "/api/planet/settle/calibrate":
            return self._send(200, calibrate_settle())
        if path == "/api/benchmark/conflict":
            rows = load_conflict_rows()
            return self._send(200, {"size": len(rows),
                                    "accuracy": round(ensemble_accuracy(rows=rows) or 0, 3) if rows else None,
                                    "note": "冲突集：各层意见分裂的状态，专门考分层差异"})
        if path == "/api/planet/backfill-centroids":
            import numpy as np
            filled = 0
            for bucket, meta in list(layers.meta.items()):
                if meta.get("kind") == "correction" or meta.get("feature_centroid"):
                    continue
                time_range = meta.get("range")
                if not time_range or len(time_range) != 2:
                    continue
                rows = layers.rows_in_ranges([time_range], limit=300)
                if len(rows) < 3:
                    continue
                X, _, _, _ = layers._features(rows)
                if X.shape[0] == 0:
                    continue
                meta["feature_centroid"] = np.round(X.mean(axis=0), 4).tolist()
                layers.meta[bucket] = meta
                try:
                    path = os.path.join(str(DATA / "layer_models"), f"layer-{int(bucket)}.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(meta, handle, ensure_ascii=False)
                except (OSError, ValueError):
                    pass
                filled += 1
            return self._send(200, {"ok": True, "filled": filled,
                                    "total": len(layers.meta)})
        if path == "/api/planet/vent":
            return self._send(200, volcano_vent())
        if path == "/api/planet/credit":
            items = []
            for bucket, meta in layers.meta.items():
                credit = meta.get("credit")
                if credit:
                    items.append({"label": meta.get("label"), **credit,
                                  "weight": meta.get("adapted_weight")})
            return self._send(200, {"layers": items, "count": len(items)})
        if path == "/api/planet/thermal":
            return self._send(200, thermal_state())
        if path == "/api/planet/diversity":
            return self._send(200, planet_diversity())
        if path == "/api/planet/settle":
            return self._send(200, settle_scores())
        if path == "/api/presence":
            return self._send(200, {
                "presence": dict(_presence),
                "uptime_seconds": int(time.time() - float(_presence.get("started_at") or time.time())),
                "server_restarts": _presence.get("server_restarts"),
                "mood": mood.snapshot(),
                "self_state": self_model.snapshot(),
                "events": _events[-5:],
            })
        if path == "/api/drift":
            return self._send(200, drift.snapshot_state())
        if path == "/api/credibility":
            return self._send(200, load_credibility())
        if path == "/api/values/emergent":
            rows = []
            try:
                conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
                rows = conn.execute("SELECT mood, next_expression, gesture FROM trajectories WHERE next_expression IS NOT NULL").fetchall()
                conn.close()
            except sqlite3.Error:
                pass
            return self._send(200, values.emerge(rows))
        if path == "/api/values":
            return self._send(200, values.summary())
        if path == "/api/arbitration":
            events = []
            try:
                with open(ARBITRATION_LOG, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            events.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                pass
            labeled = [e for e in events if e.get("engaged") is not None]
            return self._send(200, {"weights": dict(_arb_weights), "events": len(events),
                                    "labeled": len(labeled), "recent": events[-3:]})
        if path == "/api/active/learn":
            return self._send(200, layers.train_active_layer())
        if path == "/api/landscape":
            return self._send(200, layers.landscape())
        if path == "/api/self/attention":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            q = (query.get("q", [""])[0] or "").strip()
            attention = self_model.attention(q or "平静")
            labels = {str(meta.get("bucket")): meta.get("label") for meta in layers.meta.values()}
            top = sorted(attention.items(), key=lambda item: -item[1])[:5]
            return self._send(200, {"context": q, "attention": [
                {"layer": labels.get(str(bucket)) or str(bucket), "weight": weight}
                for bucket, weight in top], "raw": {str(k): v for k, v in attention.items()}})
        if path == "/api/self/consistency":
            state = self_model.snapshot()
            return self._send(200, {"consistency": state.get("consistency"),
                                    "history": (state.get("consistency_history") or [])[-10:]})
        if path == "/api/self/state":
            return self._send(200, self_model.snapshot())
        if path == "/api/self":
            view = build_self_view()
            view["self_state"] = self_model.snapshot()
            view["consistency"] = self_model.snapshot().get("consistency")
            view["values"] = values.summary()
            view["focus"] = self_model.focus()
            return self._send(200, view)
        if path == "/api/archive":
            active = [{"label": m.get("label"), "fitness": m.get("recent_accuracy") or m.get("accuracy_expression"),
                       "rejected": bool(m.get("rejected")), "kind": m.get("kind")}
                      for m in layers.meta.values()
                      if m.get("kind") != "correction" and not (m.get("semantic") or {}).get("stale")]
            return self._send(200, {"entries": archive_entries(20), "active": active,
                                    "note": "父代按适应度加权采样；后代适应度不低于亲代才接受"})
        if path == "/api/offline-readiness":
            return self._send(200, offline_readiness())
        if path == "/api/scores":
            return self._send(200, score_boards())
        if path == "/api/feedback/stats":
            return self._send(200, feedback_stats())
        if path == "/api/benchmark/trend":
            return self._send(200, benchmark_trend())
        if path == "/api/benchmark":
            history = []
            try:
                with open(BENCHMARK_LOG, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            history.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                pass
            return self._send(200, {"set_size": len(load_benchmark_set()),
                                    "latest": history[-1] if history else None,
                                    "history": history[-10:]})
        if path == "/api/quality":
            return self._send(200, quality.status())
        if path == "/api/style":
            return self._send(200, style.status())
        if path == "/api/dashboard":
            try:
                conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
                inf_total, inf_engaged, avg_latency = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(engaged),0), COALESCE(AVG(latency_ms),0) FROM inference_samples"
                ).fetchone()
                conn.close()
            except sqlite3.Error:
                inf_total, inf_engaged, avg_latency = 0, 0, 0
            with _loop_lock:
                loop_state = dict(_loop_state)
            return self._send(200, {
                "inference": {"total": inf_total, "engaged": inf_engaged,
                              "avg_latency_ms": int(avg_latency or 0)},
                "layers": layers.status()["layers"],
                "frozen_layers": layers.status()["frozen_layers"],
                "corrections": len(layers.corrections),
                "boundaries": len(_boundaries),
                "loop": {k: loop_state.get(k) for k in ("inferences", "threshold", "consolidations", "last_duration_ms")},
                "consistency": self_model.snapshot().get("consistency"),
                "values": values.summary(),
                "evolution": evolution.summary(),
                "maturity": life_maturity(),
                "organs": organs.summary(),
                "affect": affect.snapshot(),
                "feedback": feedback_stats(),
                "scores": score_boards(),
                "benchmark": (lambda h: {"score": h[-1]["score"] if h else None,
                                         "runs": len(h),
                                         "last": h[-1] if h else None})(_bench_history()),
                "arbitration": arbitration_stats(),
                "adaptation": {
                    "adapted_layers": len([m for m in layers.meta.values() if m.get("adapted_weight")]),
                    "total_layers": len(layers.meta),
                    "avg_weight": round(sum(float(m.get("adapted_weight") or 0) for m in layers.meta.values()
                                            if m.get("adapted_weight")) /
                                        max(1, len([m for m in layers.meta.values() if m.get("adapted_weight")])), 3),
                    "last_adapted_at": max([m.get("adapted_at") for m in layers.meta.values() if m.get("adapted_at")] or [None]),
                },
                "quality": quality.status(),
                "style": style.status(),
                "planet": {"layers": strata.stats()["layers"], "drift": strata.planet().get("drift_index")},
            })
        if path == "/api/inference":
            try:
                conn = sqlite3.connect(str(DATA / "trajectory.db"), timeout=5)
                rows = conn.execute(
                    "SELECT ts,source,user_text,reply,emotion,gesture,latency_ms,engaged FROM inference_samples ORDER BY id DESC LIMIT 12"
                ).fetchall()
                total = conn.execute("SELECT COUNT(*) FROM inference_samples").fetchone()[0]
                engaged = conn.execute("SELECT COUNT(*) FROM inference_samples WHERE engaged=1").fetchone()[0]
                conn.close()
            except sqlite3.Error:
                rows, total, engaged = [], 0, 0
            events = [{"ts": ts, "source": source, "user": user[:80], "reply": reply[:80],
                       "emotion": emotion, "gesture": gesture, "latency_ms": latency,
                       "engaged": bool(engaged_flag)}
                      for ts, source, user, reply, emotion, gesture, latency, engaged_flag in rows]
            return self._send(200, {"total": total, "engaged": engaged, "recent": events})
        if path == "/api/boundaries":
            with _loop_lock:
                return self._send(200, {"count": len(_boundaries), "events": _boundaries[-10:]})
        if path == "/api/loop":
            with _loop_lock:
                if _loop_state.get("auto", True):
                    _loop_state["threshold"], detail = recommended_threshold()
                    _loop_state["policy"] = detail
                state = dict(_loop_state)
            return self._send(200, {"state": state, "layers": layers.status()["layers"],
                                    "corrections": len(layers.corrections)})
        if path == "/api/layers":
            return self._send(200, layers.status())
        if path == "/api/layers/predict":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            expression = (query.get("expression", ["neutral"])[0] or "neutral")
            gesture = (query.get("gesture", [""])[0] or "")
            text = (query.get("text", [""])[0] or "")
            voice_param = None
            if query.get("voice_rate") or query.get("voice_name") or query.get("voice_volume"):
                voice_param = {"rate": (query.get("voice_rate", ["+0%"])[0] or "+0%"),
                               "name": (query.get("voice_name", [""])[0] or ""),
                               "volume": (query.get("voice_volume", ["1"])[0] or "1")}
            result = layers.predict(expression, gesture, False,
                                    {"voice": voice_param} if voice_param else {},
                                    mood.label(), text=text)
            return self._send(200, result or {"layers": [], "source": "none"})
        if path == "/api/strata/conflict":
            return self._send(200, strata.conflicts())
        if path == "/api/planet":
            return self._send(200, strata.planet())
        if path == "/api/strata":
            layer_rows = [{k: v for k, v in layer.items() if k != "centroid"}
                          for layer in strata.list_layers(24)]
            return self._send(200, {"stats": strata.stats(), "layers": layer_rows,
                                    "boundaries": strata.boundaries()})
        if path == "/api/strata/gradient":
            return self._send(200, {"stats": strata.stats(), "boundaries": strata.boundaries()})
        if path == "/api/strata/concept":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            q = (query.get("q", [""])[0] or "").strip()
            return self._send(200, strata.concept_trajectory(q))
        if path == "/api/behavior/model":
            return self._send(200, behavior.status())
        if path == "/api/behavior/stats":
            entries = []
            try:
                with open(BEHAVIOR_LOG, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entries.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                pass
            by_action = {}
            for entry in entries:
                action = entry.get("action") or "?"
                bucket = by_action.setdefault(action, {"count": 0, "engaged": 0})
                bucket["count"] += 1
                if entry.get("engaged"):
                    bucket["engaged"] += 1
            return self._send(200, {"total": len(entries), "by_action": by_action,
                                    "model": behavior.status()})
        if path == "/api/model/history":
            folder = DATA / "models"
            items = []
            try:
                for name in sorted(os.listdir(folder)):
                    if not name.endswith(".npz"):
                        continue
                    meta_path = folder / (name[:-4] + ".json")
                    meta = {}
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
                    items.append({"file": name, "size": (folder / name).stat().st_size, **meta})
            except OSError:
                pass
            return self._send(200, {"versions": items[-10:]})
        if path == "/api/model":
            return self._send(200, policy.status())
        if path == "/api/model/manifest":
            manifest = model_store.ensure_manifest(str(DATA / "layer_models"))
            return self._send(200, {
                "schema_version": manifest.get("schema_version"),
                "weight_format": manifest.get("weight_format"),
                "updated_at": manifest.get("updated_at"),
                "layers": len(manifest.get("layers", {})),
                "compat": layers.status().get("migrated"),
            })
        if path == "/api/model/verify":
            return self._send(200, model_store.verify_manifest(str(DATA / "layer_models")))
        if path == "/api/mood":
            return self._send(200, {"mood": mood.snapshot(), "describe": mood.describe()})
        if path == "/api/initiative":
            return self._send(200, build_initiative())
        if path == "/api/memory":
            return self._send(200, {"stats": memory.stats(), "recent": memory.recent(10)})
        if path == "/api/surface":
            with _surface_lock:
                surface = dict(_surface)
            return self._send(200, {"surface": surface, "summary": surface_summary()})
        if path == "/api/health":
            return self._send(200, {"ok": True, "tts": "edge-tts", "localBrain": local_brain_ok(), "account": account_available(), "time": int(time.time())})
        if path == "/api/trajectory/recent":
            return self._send(200, {"stats": trajectory.stats(), "recent": trajectory.recent(12)})
        if path == "/api/trajectory/predict":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            snapshot = {
                "expression": (query.get("expression", ["neutral"])[0] or "neutral"),
                "gesture": (query.get("gesture", [""])[0] or ""),
                "speaking": False,
                "params": {},
                "mood": mood.label(),
            }
            layered = layers.predict(snapshot.get("expression"), snapshot.get("gesture"),
                                     bool(snapshot.get("speaking")), snapshot.get("params"),
                                     snapshot.get("mood") or mood.label(),
                                     text=(query.get("text", [""])[0] or ""))
            if layered:
                return self._send(200, layered)
            trained = policy.predict(snapshot.get("expression"), snapshot.get("gesture"),
                                     bool(snapshot.get("speaking")), snapshot.get("params"),
                                     snapshot.get("mood") or mood.label())
            if trained and trained.get("confidence", 0) >= 0.35:
                return self._send(200, trained)
            knn = trajectory.predict(snapshot)
            knn["source"] = "knn-memory"
            return self._send(200, knn)
        if path == "/api/poll":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            try:
                since = int(query.get("since", ["-1"])[0])
            except ValueError:
                since = -1
            with _events_lock:
                latest = _event_seq
                pending = [] if since < 0 else [dict(e) for e in _events if e["seq"] > since]
            return self._send(200, {"seq": latest, "events": pending})
        if path == "/api/events":
            sink = queue.Queue(maxsize=32)
            with _subscribers_lock:
                _subscribers.append(sink)
            try:
                log_event("sse-open", {})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        item = sink.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(("data: " + item + "\n\n").encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _subscribers_lock:
                    if sink in _subscribers:
                        _subscribers.remove(sink)
                log_event("sse-close", {})
            return
        if path == "/api/joke":
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            lang = (query.get("lang", ["zh"])[0] or "zh").lower()
            return self._send(200, generate_joke(lang))
        if path in ("/", ""):
            path = "/index.html"
        target = (PUBLIC / path.lstrip("/")).resolve()
        if not str(target).startswith(str(PUBLIC.resolve())) or not target.is_file():
            return self._send(404, "not found", "text/plain; charset=utf-8")
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        log_event("static", {"path": path, "bytes": target.stat().st_size})
        if target.suffix in (".js",):
            content_type = "text/javascript; charset=utf-8"
        data = target.read_bytes()
        return self._send(200, data, content_type)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        chat_started = time.time()
        if path == "/api/surface":
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 2 * 1024 * 1024:
                return self._send(400, {"error": "surface too large"})
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": "bad json"})
            image = payload.pop("image", None)
            if isinstance(image, str) and image.startswith("data:image/") and len(image) < 1_500_000:
                try:
                    import base64
                    _surface_image.write_bytes(base64.b64decode(image.split(",", 1)[1]))
                except Exception:  # noqa: BLE001
                    pass
            with _surface_lock:
                _surface.clear()
                _surface.update(payload)
            return self._send(200, {"ok": True, "bytes": len(raw)})
        if path == "/api/asr":
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 12 * 1024 * 1024:
                return self._send(400, {"error": "音频太大或为空"})
            raw = self.rfile.read(length)
            try:
                text = transcribe_wav(raw)
                log_event("asr", {"bytes": len(raw), "chars": len(text)})
                return self._send(200, {"text": text})
            except Exception as error:  # noqa: BLE001
                log_event("asr-error", repr(error)[:300])
                return self._send(502, {"error": "识别失败", "detail": repr(error)[:120]})
        body = self._json()
        if path == "/api/tts":
            text = str(body.get("text") or "").strip()
            voice = str(body.get("voice") or "zh-CN-XiaoxiaoNeural")
            rate = str(body.get("rate") or "+0%")
            if not text:
                return self._send(400, {"error": "empty text"})
            if len(text) > TTS_MAX_CHARS:
                text = text[:TTS_MAX_CHARS]
            try:
                audio = synth_cached(text, voice, rate)
                log_event("tts", {"chars": len(text), "bytes": len(audio), "voice": voice})
                return self._send(200, audio, "audio/mpeg")
            except Exception as error:  # noqa: BLE001 - 网络/超时都要容错
                log_event("tts-error", str(error)[:300])
                return self._send(502, {"error": "合成失败", "detail": str(error)[:200]})
        if path == "/api/chat":
            message = str(body.get("message") or "").strip()[:800]
            if not message:
                return self._send(400, {"error": "empty message"})
            lang = str(body.get("lang") or "zh").lower()
            if lang not in ("zh", "ja", "en"):
                lang = "zh"
            key = str(body.get("key") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
            base_url = str(body.get("baseURL") or "").strip()
            model = str(body.get("model") or "").strip()
            state = body.get("state")
            use_account = bool(body.get("account")) or (not base_url and not key and account_available())
            if use_account and account_available():
                try:
                    system = build_system(lang, True, state, message)
                    remembered = memory_context(message)
                    if remembered:
                        system += "\n" + remembered
                        log_event("memory-hit", {"chars": len(remembered)})
                    track = trajectory_context()
                    if track:
                        system += "\n" + track
                        log_event("traj-memory", {"chars": len(track)})
                    reply = parse_reply(openai_chat("https://api.deepseek.com", "", "deepseek-chat", message,
                                                    body.get("history"), lang, system=system, account=True))
                    reply["expression"], reply["gesture"], reply["arbitration"] = arbitrate(reply, state, message)
                    try:
                        save_presence({"last_event": "chat", "last_user_message": str(message)[:200],
                                       "last_reply": str(reply.get("text") or "")[:120],
                                       "last_chat_at": time.time(), "applied": reply.get("expression", "")})
                    except Exception:  # noqa: BLE001
                        pass
                    record_inference(message, reply, "dsh-account", int((time.time() - chat_started) * 1000), state)
                    bump_inference(1)
                    log_event("chat", {"chars": len(message), "via": "dsh-account", "state": state_line(state)})
                    threading.Thread(target=remember_exchange, args=(message, reply), daemon=True).start()
                    reply["mood"] = mood.snapshot()
                    return self._send(200, reply)
                except Exception as error:  # noqa: BLE001
                    log_event("chat-error", "account:" + str(error)[:140])
            if base_url:
                try:
                    system = build_system(lang, True, state, message)
                    remembered = memory_context(message)
                    if remembered:
                        system += "\n" + remembered
                        log_event("memory-hit", {"chars": len(remembered)})
                    track = trajectory_context()
                    if track:
                        system += "\n" + track
                        log_event("traj-memory", {"chars": len(track)})
                    reply = parse_reply(openai_chat(base_url, key, model or "local", message, body.get("history"), lang,
                                                    system=system))
                    reply["expression"], reply["gesture"], reply["arbitration"] = arbitrate(reply, state, message)
                    try:
                        save_presence({"last_event": "chat", "last_user_message": str(message)[:200],
                                       "last_reply": str(reply.get("text") or "")[:120],
                                       "last_chat_at": time.time(), "applied": reply.get("expression", "")})
                    except Exception:  # noqa: BLE001
                        pass
                    record_inference(message, reply, "local", int((time.time() - chat_started) * 1000), state)
                    bump_inference(1)
                    log_event("chat", {"chars": len(message), "via": "local:" + base_url[:40], "state": state_line(state), "surface": surface_summary()[:160]})
                    threading.Thread(target=remember_exchange, args=(message, reply), daemon=True).start()
                    reply["mood"] = mood.snapshot()
                    return self._send(200, reply)
                except Exception as error:  # noqa: BLE001
                    log_event("chat-local-error", str(error)[:200])
            elif key:
                try:
                    reply = parse_reply(openai_chat(DEEPSEEK_URL.rsplit("/chat", 1)[0], key, "deepseek-chat", message, body.get("history"), lang,
                                                    system=build_system(lang, False, state, message)))
                    reply["expression"], reply["gesture"], reply["arbitration"] = arbitrate(reply, state, message)
                    try:
                        save_presence({"last_event": "chat", "last_user_message": str(message)[:200],
                                       "last_reply": str(reply.get("text") or "")[:120],
                                       "last_chat_at": time.time(), "applied": reply.get("expression", "")})
                    except Exception:  # noqa: BLE001
                        pass
                    record_inference(message, reply, "deepseek", int((time.time() - chat_started) * 1000), state)
                    return self._send(200, reply)
                except Exception as error:  # noqa: BLE001
                    log_event("chat-error", str(error)[:300])
            fallback = fallback_chat(message, lang)
            log_event("chat", {"chars": len(message), "via": "fallback"})
            return self._send(200, fallback)
        if path == "/api/initiative":
            return self._send(200, build_initiative())
        if path == "/api/joke":
            lang = str(body.get("lang") or "zh").lower()
            source = str(body.get("source") or "auto")
            key = str(body.get("key") or "").strip()
            return self._send(200, generate_joke(lang, source, key))
        if path == "/api/behavior/outcome":
            entry = {
                "ts": time.time(),
                "action": str(body.get("action") or "")[:20],
                "engaged": bool(body.get("engaged")),
                "idle": body.get("idle"),
                "expression": str(body.get("expression") or "")[:20],
                "predict_expression": str(body.get("predict_expression") or "")[:20],
                "predict_gesture": str(body.get("predict_gesture") or "")[:20],
                "confidence": body.get("confidence"),
                "mood": str(body.get("mood") or "")[:20],
            }
            try:
                with open(BEHAVIOR_LOG, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass
            try:
                if (not observation_active()) and quality.status().get("labeled_samples", 0) >= 12 and time.time() - float((quality.meta or {}).get("trained_at_ts") or 0) > 1800:
                    quality.train()
                    style.rebuild()
            except Exception as error:  # noqa: BLE001
                log_event("quality-error", str(error)[:150])
            if (not observation_active()) and behavior.should_retrain():
                threading.Thread(target=behavior.train, daemon=True).start()
            return self._send(200, {"ok": True})
        if path == "/api/drift/check":
            return self._send(200, drift.check(drift_snapshot()))
        if path == "/api/drift/baseline":
            return self._send(200, drift.reset(drift_snapshot()))
        if path == "/api/values/emergent/confirm":
            result = values.confirm_emergent(int(body.get("index") or 0),
                                             confirm=bool(body.get("confirm")))
            if result.get("ok") and result.get("added"):
                evolution.record_meme("value", result["added"], status="adopted",
                                      fitness=1.0, mutation="浮现+确认")
            return self._send(200, result)
        if path == "/api/values/update":
            return self._send(200, values.update(body.get("principles"), body.get("rules"),
                                                 confirm=bool(body.get("confirm"))))
        if path == "/api/arbitration/adapt":
            return self._send(200, adapt_arbitration())
        if path == "/api/self/narrate":
            lang = str(body.get("lang") or "zh").lower()
            if lang not in ("zh", "ja", "en"):
                lang = "zh"
            return self._send(200, narrate_self(lang))
        if path == "/api/benchmark/run":
            return self._send(200, run_benchmark())
        if path == "/api/benchmark/init":
            size = max(20, min(200, int(body.get("size") or 100)))
            return self._send(200, build_benchmark_set(size))
        if path == "/api/multimodal/vision":
            image_result = multimodal_ingest_image(
                body.get("image_base64") or body.get("image") or "",
                str(body.get("caption") or ""), str(body.get("source") or "vision"))
            if not image_result.get("ok"):
                return self._send(200, image_result)
            description = str(body.get("caption") or "").strip()
            brightness = 0.5
            try:
                from PIL import Image
                image = Image.open(image_result["path"]).convert("L")
                image.thumbnail((32, 32))
                pixels = list(image.getdata())
                brightness = round(sum(pixels) / max(1, len(pixels)) / 255.0, 3)
            except Exception:  # noqa: BLE001
                pass
            status = vision_status()
            if not description:
                description, status = vision_with_scheduling(image_result["path"])
            if description:
                try:
                    memory.add("image", "看到画面：" + description[:300], importance=0.45)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    presence_now = dict(_presence) if isinstance(_presence, dict) else {}
                    trajectory.add({"expression": presence_now.get("expression") or "neutral",
                                    "gesture": presence_now.get("gesture") or "",
                                    "speaking": False, "mood": mood.snapshot().get("label"),
                                    "scene": "vision",
                                    "params": {"vision": {"caption": description[:200],
                                                          "brightness": brightness,
                                                          "path": image_result["path"]}},
                                    "source": "vision"})
                except Exception:  # noqa: BLE001
                    pass
            return self._send(200, {**image_result, "caption": description or "（未识别）",
                                    "vision": status})
        if path == "/api/companion/say":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._send(200, {"ok": False, "reason": "没有文字"})
            expression = str(body.get("expression") or "").strip()
            if not expression:
                try:
                    expression = arbitrate({"text": text, "expression": "neutral", "gesture": ""},
                                           body.get("state") or {}, text)[0]
                except Exception:  # noqa: BLE001
                    expression = "neutral"
            try:
                publish({"type": "companion", "kind": "say", "text": text[:120],
                         "expression": expression})
            except Exception:  # noqa: BLE001
                pass
            return self._send(200, {"ok": True, "text": text, "expression": expression,
                                    "audio": "/api/tts", "note": "壳拿到 text/expression 后调 /api/tts 播放"})
        if path == "/api/orchestrate":
            return self._send(200, orchestrate(body))
        if path == "/api/multimodal/image":
            return self._send(200, multimodal_ingest_image(
                body.get("image_base64") or body.get("image") or "",
                body.get("caption") or "", str(body.get("source") or "web")))
        if path == "/api/multimodal/audio":
            return self._send(200, multimodal_ingest_audio(
                body.get("audio_base64") or body.get("audio") or "",
                str(body.get("lang") or "zh"), str(body.get("source") or "web")))
        if path == "/api/system/recent":
            return self._send(200, {"events": load_system_events(20),
                                    "summary": (lambda: (json.load(open(SYSTEM_DIGEST, encoding="utf-8")).get("summary") if SYSTEM_DIGEST.exists() else None))()})
        if path == "/api/session/main":
            with _orch_lock:
                sessions = load_orch_history()
            history = sessions.get("main") or []
            return self._send(200, {"session": "main", "count": len(history),
                                    "messages": history[-12:]})
        if path == "/api/observation/auto":
            state = auto_obs_state()
            if body.get("enabled") is not None:
                state["enabled"] = bool(body.get("enabled"))
                save_auto_obs(state)
            if body.get("run"):
                return self._send(200, maybe_auto_observation(force=bool(body.get("force"))))
            stable_hours = (time.time() - float(state.get("stable_since") or 0)) / 3600.0
            return self._send(200, {"enabled": state.get("enabled"),
                                    "stable_hours": round(stable_hours, 2),
                                    "required_hours": STABLE_HOURS,
                                    "fingerprint": state.get("fingerprint")[:12]})
        if path == "/api/self-core/update":
            core = self_core()
            for key in ("name", "project", "identity", "values", "relationship", "user_calls"):
                if key in body:
                    core[key] = body[key]
            try:
                with open(SELF_CORE, "w", encoding="utf-8") as handle:
                    json.dump(core, handle, ensure_ascii=False, indent=2)
            except OSError:
                pass
            return self._send(200, {"ok": True, "core": core})
        if path == "/api/imagine":
            return self._send(200, imagine_scene(force=bool(body.get("force"))))
        if path == "/api/calibration/fit":
            rows = layers._rows()[-400:]
            return self._send(200, layers.calibrate_temperature(rows))
        if path == "/api/layers/backfill-schema":
            import json as _json
            filled = 0
            force = bool(body.get("force"))
            for bucket, meta in list(layers.meta.items()):
                if meta.get("feature_schema") and not force:
                    continue
                centroid_len = len(meta.get("feature_centroid") or [])
                model = layers.models.get(bucket)
                dim = centroid_len if centroid_len else (int(model["w_expr"].shape[0]) if model else 0)
                meta["feature_schema"] = {"dims": dim, "features": ["state"] + (["text"] if dim >= 31 else []) + (["voice"] if dim >= 34 else []), "era": "E%d" % dim if dim else "?"}
                layers.meta[bucket] = meta
                try:
                    layers._save_layer(bucket, meta)
                except Exception:  # noqa: BLE001
                    pass
                filled += 1
            return self._send(200, {"ok": True, "filled": filled})
        if path == "/api/visual/generate":
            import visual_gen
            result = visual_gen.generate(str(body.get("text") or ""),
                                         str(body.get("style") or "auto"),
                                         bool(body.get("animate")))
            try:
                publish({"type": "visual", "text": str(body.get("text") or "")[:80],
                         "url": result.get("url"), "kind": result.get("kind")})
            except Exception:  # noqa: BLE001
                pass
            return self._send(200, result)
        if path == "/api/living/sleep":
            threshold = int(tunables.get("living_sleep_steps") or 40)
            status = layers.living_status()
            samples = int(status.get("samples") or 0)
            if not body.get("force") and samples < threshold:
                return self._send(200, {"ok": False, "reason": "活层还没攒够",
                                        "samples": samples, "threshold": threshold})
            rows = layers._rows()[-max(8, min(samples or threshold, 120)):]
            record = layers.add_consolidation_layer(rows, weight=0.55,
                                                    kind="sleep-consolidated",
                                                    label="睡眠层 " + time.strftime("%H:%M:%S"),
                                                    min_samples=4) if rows else None
            if not record:
                return self._send(200, {"ok": False, "reason": "巩固失败"})
            layers.cool_living(float(tunables.get("living_cool_factor") or 0.5))
            log_event("sleep-consolidate", {"layer": record.get("label"), "rows": len(rows)})
            return self._send(200, {"ok": True, "layer": record.get("label"),
                                    "duplicate_of": record.get("duplicate_of"),
                                    "samples": status.get("samples"),
                                    "cooled": True})
        if path == "/api/living/status":
            return self._send(200, layers.living_status())
        if path == "/api/living/step":
            count = int(body.get("count") or 5)
            done = 0
            for row in layers._rows()[-count:]:
                if layers.online_step(row):
                    done += 1
            return self._send(200, {"ok": True, "steps": done, **layers.living_status()})
        if path == "/api/thoughts/probe":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._send(200, {"ok": False, "reason": "没有输入"})
            if not local_brain_ok():
                return self._send(200, {"ok": False, "reason": "本地模型不在线"})
            samples = []
            for index in range(3):
                try:
                    raw = openai_chat("http://127.0.0.1:8181/v1", "", "r1-1.5b", text, [], "zh",
                                      system=self_core_text() + "\n只输出一句话，不超过30字。",
                                      timeout=60)
                    parsed_probe = parse_reply(raw)
                    answer = str(parsed_probe.get("text") or "").strip()
                    if answer:
                        samples.append(answer[:60])
                except Exception:  # noqa: BLE001
                    continue
            distinct = len(set(samples))
            record_thought({"type": "language_candidates", "text": text[:80],
                            "samples": samples, "distinct": distinct,
                            "note": "3 次采样，近似看它有多少种说法"})
            return self._send(200, {"ok": True, "samples": samples,
                                    "distinct": distinct,
                                    "note": "不是概率，只是多次采样的多样性"})
        if path == "/api/perception/environment":
            state = environment_state()
            for key in ("enabled", "interval_sec", "describe_interval_sec", "change_threshold"):
                if key in body:
                    state[key] = body[key]
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_environment(state)
            log_event("environment-config", {k: state.get(k) for k in ("enabled", "interval_sec", "describe_interval_sec")})
            return self._send(200, state)
        if path == "/api/social/person":
            state = social_state()
            name = str(body.get("name") or "").strip()[:40]
            if not name:
                return self._send(200, {"ok": False, "reason": "没有名字"})
            person = state.get("persons", {}).setdefault(name, {})
            for key in ("relation", "notes"):
                if key in body:
                    person[key] = str(body[key])[:120]
            if "intimacy" in body:
                person["intimacy"] = max(0.0, min(1.0, float(body["intimacy"])))
            person["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_social(state)
            return self._send(200, {"ok": True, "person": name, "state": person})
        if path == "/api/curiosity/scan":
            return self._send(200, curiosity_scan(str(body.get("text") or ""),
                                                  str(body.get("session") or "main")))
        if path == "/api/curiosity/resolve":
            return self._send(200, curiosity_resolve(str(body.get("text") or "")))
        if path == "/api/action/search":
            query = str(body.get("query") or "").strip()
            if not query:
                return self._send(200, {"ok": False, "reason": "没有查询词"})
            return self._send(200, record_action("search", query, {"source": body.get("source") or "reading"}) or {"ok": False})
        if path == "/api/action/read":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._send(200, {"ok": False, "reason": "没有内容"})
            return self._send(200, record_action("read", text, {"url": body.get("url") or ""}) or {"ok": False})
        if path == "/api/memory/note":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._send(200, {"ok": False, "reason": "没有内容"})
            url = str(body.get("url") or "").strip()[:300]
            note = text[:600] + (("（来源：" + url + "）") if url else "")
            try:
                record_action("read", text[:120], {"url": url})
            except Exception:  # noqa: BLE001
                pass
            try:
                memory.add("web", "网上读到：" + note, importance=0.35,
                           source="web", session="reading", person="owner")
            except Exception as error:  # noqa: BLE001
                return self._send(200, {"ok": False, "reason": str(error)[:80]})
            try:
                curiosity_resolve(text)
            except Exception:  # noqa: BLE001
                pass
            log_event("memory-note", {"chars": len(text), "url": url[:80]})
            return self._send(200, {"ok": True, "text": note})
        if path == "/api/voice/transcript":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._send(200, {"ok": False, "reason": "没有文字"})
            speaker = str(body.get("speaker") or "owner")[:40]
            try:
                memory.add("voice", "听到你说：" + text[:400], importance=0.4,
                           session="voice", person=speaker, audience="private")
            except Exception:  # noqa: BLE001
                pass
            try:
                mood.apply("chat", text)
            except Exception:  # noqa: BLE001
                pass
            threat = 0.0
            try:
                appraisal = appraise_text(text)
                threat = appraisal.get("threat", 0.0)
                if threat > 0:
                    affect.update(appraisal, text)
                    mood.threat_shift(threat)
            except Exception:  # noqa: BLE001
                pass
            try:
                organs.observe({"kind": "voice", "text": text, "intensity": 0.5})
            except Exception:  # noqa: BLE001
                pass
            try:
                publish({"type": "voice", "text": text[:160], "speaker": speaker})
            except Exception:  # noqa: BLE001
                pass
            log_event("voice-transcript", {"chars": len(text), "speaker": speaker, "threat": threat})
            return self._send(200, {"ok": True, "text": text, "threat": threat})
        if path == "/api/body/say":
            return self._send(200, body_say(body.get("text") or body.get("content") or "",
                                            body.get("expression") or None,
                                            str(body.get("source") or "agent"),
                                            body.get("voice") if isinstance(body.get("voice"), dict) else None))
        if path == "/api/system/ingest":
            return self._send(200, system_ingest(body.get("kind") or "unknown",
                                                 body.get("text") or "", body.get("source") or "client",
                                                 body.get("data")))
        if path == "/api/system/probe-adb":
            return self._send(200, probe_adb_system())
        if path == "/api/system/probe":
            return self._send(200, probe_linux_system())
        if path == "/api/system/digest":
            return self._send(200, system_digest(force=bool(body.get("force"))))
        if path == "/api/observation/start":
            return self._send(200, observation_start(int(body.get("days") or 7)))
        if path == "/api/observation/stop":
            return self._send(200, observation_stop())
        if path == "/api/observation/daily":
            return self._send(200, observation_daily(force=True))
        if path == "/api/trajectory/backfill-context":
            import sqlite3 as _sqlite
            chat_rows = []
            try:
                conn = _sqlite.connect(str(DATA / "memory.db"), timeout=5)
                chat_rows = conn.execute(
                    "SELECT ts, text FROM memories WHERE kind='chat' AND text LIKE '%对话里%' ORDER BY ts"
                ).fetchall()
                conn.close()
            except _sqlite.Error:
                chat_rows = []
            filled = 0
            try:
                conn = _sqlite.connect(str(DATA / "trajectory.db"), timeout=5)
                rows = conn.execute(
                    "SELECT id, ts FROM trajectories WHERE COALESCE(context_text,'')='' ORDER BY ts"
                ).fetchall()
                for row_id, ts in rows:
                    best = None
                    best_gap = 1e12
                    for chat_ts, chat_text in chat_rows:
                        gap = abs(float(chat_ts) - float(ts))
                        if gap < best_gap:
                            best_gap = gap
                            best = chat_text
                    if best and best_gap <= 300:
                        conn.execute("UPDATE trajectories SET context_text=? WHERE id=?", (str(best)[:400], row_id))
                        filled += 1
                conn.commit()
                conn.close()
            except _sqlite.Error as error:
                return self._send(200, {"ok": False, "reason": str(error)[:120]})
            return self._send(200, {"ok": True, "chats": len(chat_rows), "filled": filled})
        if path == "/api/shadow/run":
            return self._send(200, shadow_learn())
        if path == "/api/memory/graph/build":
            return self._send(200, graph.build())
        if path == "/api/mirror/send":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._send(200, {"ok": False, "reason": "空消息"})
            try:
                with open(MIRROR_INBOX, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": time.time(), "text": text[:500], "read": False},
                                            ensure_ascii=False) + "\n")
            except OSError:
                return self._send(200, {"ok": False, "reason": "写入失败"})
            return self._send(200, {"ok": True, "queued": True})
        if path == "/api/mirror/inbox/ack":
            ids = body.get("ts") or []
            ids = set(float(item) for item in ids) if isinstance(ids, list) else set()
            lines = []
            try:
                with open(MIRROR_INBOX, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue
                        if float(entry.get("ts") or 0) in ids:
                            entry["read"] = True
                        lines.append(json.dumps(entry, ensure_ascii=False))
                with open(MIRROR_INBOX, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + ("\n" if lines else ""))
            except OSError:
                pass
            return self._send(200, {"ok": True})
        if path == "/api/benchmark/conflict/init":
            return self._send(200, build_conflict_bench(int(body.get("size") or 50)))
        if path == "/api/planet/credit/run":
            import numpy as np
            rows = layers._rows()[-120:]
            before_rows = rows[-60:] if len(rows) >= 60 else rows
            before = ensemble_accuracy(rows=before_rows)
            backup = {bucket: (float(meta.get("adapted_weight") or meta.get("weight") or 0.4))
                      for bucket, meta in layers.meta.items()}
            result = layers.credit_assignment(rows)
            after = ensemble_accuracy(rows=before_rows)
            reverted = False
            if before is not None and after is not None and after < before - 0.005:
                for bucket, weight in backup.items():
                    if bucket in layers.meta:
                        layers.meta[bucket]["adapted_weight"] = weight
                reverted = True
                after = ensemble_accuracy(rows=before_rows)
            log_event("credit-assignment", {"before": before, "after": after, "reverted": reverted})
            return self._send(200, {"ok": result.get("ok"), "reason": result.get("reason"),
                                    "samples": result.get("samples"),
                                    "before": round(before, 3) if before else None,
                                    "after": round(after, 3) if after else None,
                                    "reverted": reverted,
                                    "layers": result.get("layers")})
        if path == "/api/planet/thermal/step":
            result = thermal.step(thermal_inputs(), thermal_params(), dt=1.0)
            molten = [item for item in result if item["phase"] == "熔融"]
            return self._send(200, {"ok": True, "layers": result, "molten": len(molten)})
        if path == "/api/planet/volcano/auto":
            return self._send(200, maybe_auto_volcano())
        if path == "/api/planet/settle/calibrate":
            return self._send(200, calibrate_settle())
        if path == "/api/planet/backfill-centroids":
            import numpy as np
            filled = 0
            for bucket, meta in list(layers.meta.items()):
                if meta.get("kind") == "correction" or meta.get("feature_centroid"):
                    continue
                time_range = meta.get("range")
                if not time_range or len(time_range) != 2:
                    continue
                rows = layers.rows_in_ranges([time_range], limit=300)
                if len(rows) < 3:
                    continue
                X, _, _, _ = layers._features(rows)
                if X.shape[0] == 0:
                    continue
                meta["feature_centroid"] = np.round(X.mean(axis=0), 4).tolist()
                layers.meta[bucket] = meta
                try:
                    path = os.path.join(str(DATA / "layer_models"), f"layer-{int(bucket)}.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(meta, handle, ensure_ascii=False)
                except (OSError, ValueError):
                    pass
                filled += 1
            return self._send(200, {"ok": True, "filled": filled,
                                    "total": len(layers.meta)})
        if path == "/api/planet/volcano":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            max_active = max(6, min(20, int(body.get("max_active") or 12)))
            active = [(bucket, meta) for bucket, meta in layers.meta.items()
                      if meta.get("kind") != "correction"
                      and not (meta.get("semantic") or {}).get("stale")
                      and not meta.get("rejected") and not meta.get("fossil")]
            if len(active) <= max_active:
                return self._send(200, {"ok": False, "reason": "层数正常，不需要喷发",
                                        "active": len(active)})
            # 火山按沉降分排序（引力/新近/使用/冗余综合），而不是只看适应度
            weights = {key: float(tunables.get(key)) for key in
                       ("settle_w_fitness", "settle_w_recency", "settle_w_usage", "settle_w_redundancy")}
            component_map = {item["bucket"]: item for item in settle_components()}
            def fitness(item):
                comp = component_map.get(item[0]) or {}
                return (weights["settle_w_fitness"] * float(comp.get("fitness") or 0.3)
                        + weights["settle_w_recency"] * float(comp.get("recency") or 0)
                        + weights["settle_w_usage"] * float(comp.get("usage") or 0)
                        + weights["settle_w_redundancy"] * float(comp.get("redundancy") or 0))
            # 保护 1：最新一层（最年轻的记忆）永远不熔
            newest = max(active, key=lambda item: int(item[0]))[0] if active else None
            candidates = [item for item in active if item[0] != newest]
            candidates.sort(key=fitness)
            melt_count = max(0, len(active) - max_active)
            melt = candidates[:melt_count]
            if not melt:
                return self._send(200, {"ok": False, "reason": "没有可熔的层（最新层受保护）",
                                        "active": len(active)})
            keep = [item for item in active if item not in melt]
            ranges = [meta.get("range") for _, meta in melt if meta.get("range")]
            rows = layers.rows_in_ranges(ranges)
            record = layers.add_consolidation_layer(rows, weight=0.55, kind="magma",
                                                    label="熔岩层 " + time.strftime("%H:%M:%S"),
                                                    min_samples=4) if rows else None
            if not record:
                return self._send(200, {"ok": False, "reason": "熔融素材不足", "samples": len(rows)})
            melted_labels = []
            for bucket, meta in melt:
                meta["fossil"] = True
                meta["fossil_reason"] = "火山熔融"
                meta["melted_into"] = record["label"]
                layers.meta[bucket] = meta
                melted_labels.append(meta.get("label"))
                try:
                    path = os.path.join(str(DATA / "layer_models"), f"layer-{int(bucket)}.json")
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(meta, handle, ensure_ascii=False)
                except OSError:
                    pass
            try:
                with open(ARCHIVE_LOG, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": time.time(), "type": "volcano",
                                             "melted": melted_labels, "magma": record["label"],
                                             "samples": len(rows)}, ensure_ascii=False) + "\n")
            except OSError:
                pass
            life_event("volcano", "熔融 %d 个旧层，喷发成 %s" % (len(melted_labels), record["label"]),
                       {"melted": melted_labels})
            benchmark = run_benchmark()
            return self._send(200, {"ok": True, "magma": record["label"],
                                    "melted": len(melted_labels), "active_before": len(active),
                                    "active_after": len(keep) + 1,
                                    "benchmark": benchmark.get("score"),
                                    "layers": len(layers.meta)})
        if path == "/api/archive/evolve":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            return self._send(200, archive_evolve())
        if path == "/api/tunables/calibrate-affect/auto":
            return self._send(200, maybe_calibrate_affect(interval=0))
        if path == "/api/tunables/calibrate-affect":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            result = affect.calibrate()
            changes = {}
            if result.get("tau"):
                changes["affect_decay_tau"] = result["tau"]
            if result.get("gain"):
                changes["affect_threat_gain"] = result["gain"]
            if result.get("oscillation"):
                changes["affect_oscillation"] = result["oscillation"]
            if changes:
                tunables.set_many(changes, source="标定")
            return self._send(200, {"ok": bool(changes), **result, "applied": changes})
        if path == "/api/tunables/update":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            changed = tunables.set_many(body, source=str(body.get("source") or "配置"))
            return self._send(200, {"ok": True, "changed": changed, "params": tunables.provenance()})
        if path == "/api/ui-params/update":
            return self._send(200, {"ok": True, "params": save_ui_params(body)})
        if path == "/api/affect/classifier/auto-train":
            return self._send(200, maybe_train_classifier())
        if path == "/api/affect/classifier/train":
            return self._send(200, classifier.train(epochs=int(tunables.get("classifier_epochs") or 350),
                                                    lr=float(tunables.get("classifier_lr") or 0.5),
                                                    l2=float(tunables.get("classifier_l2") or 0.001)))
        if path == "/api/affect/lexicon/update":
            return self._send(200, affect.update_lexicon(body.get("threat"), body.get("reassure"),
                                                         body.get("goals"),
                                                         source=str(body.get("source") or "配置")))
        if path == "/api/affect/feedback":
            text = str(body.get("text") or "")
            verdict = str(body.get("verdict") or "")
            result = affect.feedback(text, verdict)
            try:
                with open(DATA / "feedback.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": time.time(), "text": text[:200],
                                             "verdict": verdict,
                                             "word": (result or {}).get("word")},
                                            ensure_ascii=False) + "\n")
            except OSError:
                pass
            return self._send(200, result)
        if path == "/api/tunables/calibrate":
            return self._send(200, tunables.calibrate(values))
        if path == "/api/affect/probe":
            text = str(body.get("text") or "")
            appraisal = appraise_text(text)
            thought = affect.update(appraisal, text) if body.get("apply") else None
            if body.get("apply") and appraisal.get("threat", 0) > 0:
                try:
                    mood.threat_shift(appraisal["threat"])
                except Exception:  # noqa: BLE001
                    pass
            return self._send(200, {"appraisal": appraisal, "thought": thought,
                                    "state": affect.snapshot()})
        if path == "/api/organs/intent":
            intent = organs.add_intent(body.get("text"), body.get("evidence"),
                                       status=str(body.get("status") or "candidate"),
                                       due_minutes=body.get("due_minutes"))
            return self._send(200, {"ok": bool(intent), "intent": intent})
        if path == "/api/organs/intent/run":
            fired = []
            for intent in organs.due_intents():
                event = publish({"type": "intent", "kind": "proactive", "text": "我想做一件事：" + intent["text"]})
                organs.execute_intent(intent["id"])
                fired.append({"id": intent["id"], "text": intent["text"], "event": event["seq"]})
            return self._send(200, {"ok": True, "fired": fired})
        if path == "/api/organs/intent/update":
            intent = organs.update_intent(str(body.get("id") or ""),
                                          status=body.get("status"),
                                          evidence=body.get("evidence"))
            return self._send(200, {"ok": bool(intent), "intent": intent})
        if path == "/api/perception/scan":
            return self._send(200, perception_scan(force=True))
        if path == "/api/consciousness/tick":
            return self._send(200, {"ok": True, **consciousness_tick(manual=True)})
        if path == "/api/evolution/code-snapshot":
            latest = _bench_history()
            fitness = latest[-1]["score"] if latest else None
            entry = evolution.snapshot_code(str(body.get("note") or ""), fitness=fitness)
            return self._send(200, {"ok": True, "gene": entry})
        if path == "/api/evolution/scan":
            evolution.scan_culture(values, style, read_narrations())
            return self._send(200, {"ok": True, **evolution.summary()})
        if path == "/api/self-check":
            return self._send(200, self_check())
        if path == "/api/quality/train":
            return self._send(200, quality.train())
        if path == "/api/style/rebuild":
            result = style.rebuild()
            if result.get("ok") and style.hint():
                evolution.record_meme("style", style.hint(), status="adopted",
                                      fitness=0.8, mutation="回应选择")
            return self._send(200, result)
        if path == "/api/boundary":
            reason = str(body.get("reason") or "manual")
            event = emergency_consolidation(reason)
            return self._send(200, event)
        if path == "/api/loop/config":
            with _loop_lock:
                if body.get("auto") is not None:
                    _loop_state["auto"] = bool(body["auto"])
                if body.get("threshold") is not None:
                    _loop_state["threshold"] = max(5, int(body["threshold"]))
                if body.get("min_samples") is not None:
                    _loop_state["min_samples"] = max(4, int(body["min_samples"]))
                if _loop_state.get("auto", True):
                    _loop_state["threshold"], detail = recommended_threshold()
                    _loop_state["policy"] = detail
                save_loop_state()
                state = dict(_loop_state)
            return self._send(200, {"ok": True, "state": state})
        if path == "/api/loop/consolidate":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            return self._send(200, consolidate_layers(force=True))
        if path == "/api/layers/correct":
            snapshot = {
                "expression": str(body.get("expression") or "neutral")[:20],
                "gesture": str(body.get("gesture") or "")[:20],
                "speaking": bool(body.get("speaking")),
                "params": body.get("params") if isinstance(body.get("params"), dict) else {},
                "mood": str(body.get("mood") or "")[:20],
            }
            next_expression = str(body.get("next_expression") or "neutral")
            next_gesture = str(body.get("next_gesture") or "")
            sample = {**snapshot, "next_expression": next_expression,
                      "next_gesture": next_gesture, "source": "correction"}
            # 与现有层的冲突率
            prediction = layers.predict(snapshot["expression"], snapshot["gesture"], False,
                                        snapshot["params"], snapshot["mood"] or "平静")
            votes = (prediction or {}).get("layers") or []
            conflict_ratio = 0.0
            if votes:
                same = sum(1 for vote in votes if vote.get("expression") == next_expression)
                conflict_ratio = 1.0 - same / len(votes)
            # 同类修正的重复次数（24 小时内）
            key = "|".join([snapshot["expression"], snapshot["gesture"], str(snapshot["mood"]),
                            next_expression, next_gesture])
            repeated = 0
            try:
                import time as _time
                with open(PENDING_CORRECTIONS, "r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue
                        if entry.get("key") == key and _time.time() - float(entry.get("ts") or 0) < 86400:
                            repeated += 1
            except OSError:
                pass
            try:
                with open(PENDING_CORRECTIONS, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": time.time(), "key": key,
                                             "next_expression": next_expression}, ensure_ascii=False) + "\n")
            except OSError:
                pass
            source = str(body.get("source") or "owner")[:20]
            credibility = credibility_of(source)
            effective = round(repeated * credibility, 3)
            try:
                tunables.observe("conflict", conflict_ratio)
            except Exception:  # noqa: BLE001
                pass
            gate = values.gate(sample, conflict_ratio, confirmations=effective,
                               confirmed=bool(body.get("confirm")))
            update_credibility(source, not gate.get("quarantined"))
            if gate.get("quarantined"):
                try:
                    with open(ADVERSARIAL_LOG, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"ts": time.time(), "key": key,
                                                 "flags": gate.get("flags"),
                                                 "conflict_ratio": round(conflict_ratio, 3),
                                                 "reason": gate.get("reason")}, ensure_ascii=False) + "\n")
                except OSError:
                    pass
                log_event("adversarial-correction", {"key": key[:80], "flags": gate.get("flags"),
                                                     "conflict": round(conflict_ratio, 3)})
                return self._send(200, {"ok": True, "quarantined": True, "gate": gate,
                                        "source": source, "credibility": credibility,
                                        "hint": "如果确定要改：重复 3 次或加 confirm:true"})
            record = layers.add_correction(snapshot, next_expression, next_gesture)
            record_boundary("user-correction", strata.conflicts(), record)
            return self._send(200, {"ok": True, "layer": record,
                                    "corrections": len(layers.corrections), "gate": gate,
                                    "source": source, "credibility": credibility})
        if path == "/api/active/learn":
            return self._send(200, layers.train_active_layer())
        if path == "/api/layers/adapt":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            return self._send(200, layers.adapt_weights())
        if path == "/api/layers/update":
            granularity = str(body.get("granularity") or "10min")
            if granularity not in ("hour", "day", "10min"):
                granularity = "10min"
            return self._send(200, layers.update(granularity, weight_fn=strata.weight_at))
        if path == "/api/strata/build":
            granularity = str(body.get("granularity") or "hour")
            if granularity not in ("hour", "day", "10min"):
                granularity = "hour"
            layer_list = strata.build(granularity)
            return self._send(200, {"ok": True, "granularity": granularity,
                                    "layers": len(layer_list), "stats": strata.stats(),
                                    "boundaries": strata.boundaries()})
        if path == "/api/behavior/train":
            return self._send(200, behavior.train())
        if path == "/api/model/rollback":
            target = str(body.get("target") or "policy")
            version = int(body.get("version") or 0)
            model = policy if target == "policy" else behavior if target == "behavior" else None
            if model is None:
                return self._send(400, {"error": "target 只能是 policy 或 behavior"})
            base = os.path.basename(model.model_path)
            folder = DATA / "models"
            source_npz = folder / f"{base}-v{version}.npz"
            source_json = folder / f"{base}-v{version}.json"
            if not source_npz.exists():
                return self._send(404, {"error": "找不到该版本"})
            try:
                shutil.copyfile(source_npz, model.model_path + ".npz")
                if source_json.exists():
                    shutil.copyfile(source_json, model.model_path + ".json")
                model.load()
                log_event("model-rollback", {"target": target, "version": version})
                return self._send(200, {"ok": True, "status": model.status()})
            except OSError as error:
                return self._send(500, {"error": str(error)[:120]})
        if path == "/api/model/migrate":
            force = bool(body.get("force"))
            limit = max(1, min(20, int(body.get("limit") or 3)))
            results = []
            for bucket, meta in list(layers.meta.items()):
                if len(results) >= limit:
                    break
                if (meta.get("kind") == "correction"):
                    continue
                stale = (meta.get("semantic") or {}).get("stale")
                if not force and not stale:
                    continue
                results.append(layers.retrain_layer(meta))
            return self._send(200, {
                "ok": True,
                "migrated": [item for item in results if item.get("ok")],
                "failed": [item for item in results if not item.get("ok")],
                "stale": layers.status().get("stale"),
            })
        if path == "/api/model/train":
            blocked = observation_blocked()
            if blocked:
                return self._send(200, blocked)
            return self._send(200, policy.train(weight_fn=strata.weight_at))
        if path == "/api/diary/run":
            diary = write_diary()
            return self._send(200, {"diary": diary or "", "mood": mood.snapshot()})
        if path == "/api/trajectory":
            snapshot = {
                "expression": str(body.get("expression") or "neutral")[:20],
                "gesture": str(body.get("gesture") or "")[:20],
                "speaking": bool(body.get("speaking")),
                "params": body.get("params") if isinstance(body.get("params"), dict) else {},
                "mood": str(body.get("mood") or "")[:20],
                "scene": str(body.get("scene") or "")[:20],
            }
            result = trajectory.add(snapshot)
            try:
                organs.touch_interaction()
            except Exception:  # noqa: BLE001
                pass
            try:
                save_presence({"last_event": "trajectory", "expression": snapshot.get("expression"),
                               "gesture": snapshot.get("gesture"), "speaking": bool(snapshot.get("speaking")),
                               "mood": snapshot.get("mood"), "params": snapshot.get("params"),
                               "scene": snapshot.get("scene")})
            except Exception:  # noqa: BLE001
                pass
            bump_inference(1)
            if (not observation_active()) and policy.should_retrain():
                threading.Thread(target=policy.train, daemon=True).start()
            trained = policy.predict(snapshot.get("expression"), snapshot.get("gesture"),
                                     bool(snapshot.get("speaking")), snapshot.get("params"),
                                     snapshot.get("mood") or mood.label())
            prediction = trained if trained and trained.get("confidence", 0) >= 0.35 else trajectory.predict(snapshot)
            return self._send(200, {"ok": True, **result, "predict": prediction})

        if path == "/api/scene":
            scene = str(body.get("type") or "external")[:40]
            mood.apply(scene)
            if scene in ("night", "low-battery", "back"):
                memory.add("scene", "场景：" + scene, importance=0.3)
            return self._send(200, {"ok": True, "mood": mood.snapshot()})
        if path == "/api/event":
            event_type = str(body.get("type") or "external")[:40]
            payload = {
                "type": event_type,
                "text": str(body.get("text") or "")[:200],
                "expression": str(body.get("expression") or "")[:20],
            }
            event = publish(payload)
            gate = {"pass": True, "reason": "未进入工作记忆"}
            try:
                gate = organs.observe({"kind": event_type, "text": payload["text"],
                                       "intensity": float(body.get("intensity") or 0.5)})
            except Exception:  # noqa: BLE001
                pass
            appraisal = appraise_text(payload["text"])
            thought = None
            try:
                if appraisal.get("threat", 0) > 0:
                    thought = affect.update(appraisal, payload["text"])
                    mood.threat_shift(appraisal["threat"])
                    if appraisal["threat"] >= 0.5 and appraisal.get("goal"):
                        organs.add_intent("面对威胁：" + appraisal["goal"],
                                          evidence=[payload["text"]], status="candidate")
                elif appraisal.get("reassure"):
                    affect.update(appraisal, payload["text"])
                    mood.reassure_shift()
            except Exception as error:  # noqa: BLE001
                log_event("affect-error", str(error)[:120])
            return self._send(200, {"ok": True, "seq": event["seq"], "subscribers": len(_subscribers),
                                    "gate": gate, "appraisal": appraisal, "thought": thought})
        if path == "/api/state":
            global _avatar_state
            with _state_lock:
                _avatar_state = body if isinstance(body, dict) else {}
            return self._send(200, {"ok": True})
        if path == "/api/log":
            log_event(str(body.get("event") or "event")[:60], body.get("detail"))
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def recent_jokes(limit=6):
    try:
        rows = memory.conn.execute("SELECT text FROM memories WHERE kind='joke' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [row[0] for row in rows]
    except Exception:  # noqa: BLE001
        return []


JOKE_PROMPTS = {
    "zh": """你是豆宝，会讲冷笑话。参考这种风格：
1. 我问钱包你还好吗，钱包说，我挺好的，就是有点扁。
2. 医生让我多运动，于是我把手机从左手换到了右手。
3. 我妈让我早点睡，我两点就睡了，比她说的时间早十个小时。
现在现编一个全新的冷笑话，和这些都不重复：{recent}
要求：1–2 句，40 字左右，有反转，只输出笑话正文，不要解释。""",
    "ja": """あなたは豆宝。次のダジャレと同じ雰囲気で：
1. 財布に元気？と聞いたら、元気だよ、ちょっと薄いけど。
新しいダジャレを一つ、必ずこれらと違う内容で：{recent}
1〜2文、本文だけ出力。""",
    "en": """You are Doubao, telling corny jokes. Example style:
1. I asked my wallet how it was doing. It said fine, just a little thin.
2. My doctor said exercise more, so I moved my phone from my left hand to my right.
Write a brand new corny joke, different from: {recent}
1-2 sentences, output only the joke.""",
}


def bank_joke(lang="zh"):
    pool = JOKES.get(lang, JOKES["zh"])
    return {"text": pool[int(time.time()) % len(pool)], "expression": "happy", "source": "bank"}


def deepseek_joke(lang="zh"):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    recent = recent_jokes()
    prompt = JOKE_PROMPTS.get(lang, JOKE_PROMPTS["zh"]).format(recent="；".join(recent) or "（还没有）")
    system = "你是豆宝，会讲冷笑话。只输出一个全新的笑话正文，不要解释、不要表情标记。"
    try:
        content = openai_chat("https://api.deepseek.com", key, "deepseek-chat", prompt, [], lang, system=system)
        joke = re.sub(r"[\r\n]+", " ", str(content or "")).strip()
        joke = re.sub(r"^\[表情[^]]*\]\s*", "", joke).strip()
        if 8 <= len(joke) <= 160:
            memory.add("joke", joke[:300], importance=0.2)
            log_event("joke", {"source": "deepseek", "chars": len(joke)})
            return {"text": joke, "expression": "happy", "source": "deepseek"}
    except Exception as error:  # noqa: BLE001
        log_event("joke-error", str(error)[:120])
    return None


def generate_joke(lang="zh", source="auto", key=""):
    """来源：auto=账号优先，其次 Key，再次内置库；model=本地模型；bank=内置库。"""
    source = (source or "auto").lower()
    if source == "bank":
        return bank_joke(lang)
    if source == "auto" and account_available():
        recent = recent_jokes()
        prompt = JOKE_PROMPTS.get(lang, JOKE_PROMPTS["zh"]).format(recent="；".join(recent) or "（还没有）")
        try:
            content = openai_chat("https://api.deepseek.com", "", "deepseek-chat", prompt, [], lang,
                                  system="你是豆宝，会讲冷笑话。只输出一个全新的笑话正文，不要解释、不要表情标记。",
                                  account=True)
            joke = re.sub(r"[\r\n]+", " ", str(content or "")).strip()
            joke = re.sub(r"^\[表情[^]]*\]\s*", "", joke).strip()
            if 8 <= len(joke) <= 160:
                memory.add("joke", joke[:300], importance=0.2)
                log_event("joke", {"source": "dsh-account", "chars": len(joke)})
                return {"text": joke, "expression": "happy", "source": "dsh-account"}
        except Exception as error:  # noqa: BLE001
            log_event("joke-error", "account:" + str(error)[:100])
    if key:
        try:
            content = openai_chat("https://api.deepseek.com", key, "deepseek-chat",
                                  JOKE_PROMPTS.get(lang, JOKE_PROMPTS["zh"]).format(recent="；".join(recent_jokes()) or "（还没有）"),
                                  [], lang, system="你是豆宝。只输出一个全新的笑话正文，不要解释。")
            joke = re.sub(r"[\r\n]+", " ", str(content or "")).strip()
            joke = re.sub(r"^\[表情[^]]*\]\s*", "", joke).strip()
            if 8 <= len(joke) <= 160:
                memory.add("joke", joke[:300], importance=0.2)
                log_event("joke", {"source": "deepseek", "chars": len(joke)})
                return {"text": joke, "expression": "happy", "source": "deepseek"}
        except Exception as error:  # noqa: BLE001
            log_event("joke-error", "deepseek:" + str(error)[:100])
    if source == "model":
        return model_joke(lang)
    return bank_joke(lang)


def model_joke(lang="zh"):
    if not local_brain_ok():
        return bank_joke(lang)
    recent = recent_jokes()
    prompt = JOKE_PROMPTS.get(lang, JOKE_PROMPTS["zh"]).format(recent="；".join(recent) or "（还没有）")
    prompt += " /no_think"  # Qwen3 关掉思考，直出正文
    for _ in range(2):
        try:
            content = openai_chat("http://127.0.0.1:8181/v1", "", "qwen3-0.6b", prompt, [], lang, plain=True)
            joke = re.sub(r"[\r\n]+", " ", str(content or "")).strip().strip('"').strip("“”")
            joke = re.sub(r"^\[表情[^]]*\]\s*", "", joke).strip()
            if len(joke) < 8 or len(joke) > 120 or joke in recent:
                continue
            memory.add("joke", joke[:300], importance=0.2)
            log_event("joke", {"source": "model", "chars": len(joke)})
            return {"text": joke, "expression": "happy", "source": "model"}
        except Exception as error:  # noqa: BLE001
            log_event("joke-error", str(error)[:120])
    pool = JOKES.get(lang, JOKES["zh"])
    return {"text": pool[int(time.time()) % len(pool)], "expression": "happy", "source": "bank"}


def build_initiative():
    """闲着时主动找个话题：心情 + 记忆决定内容。"""
    global _last_initiative
    now_ts = time.time()
    if now_ts - _last_initiative < 1200:
        return {"skip": True, "mood": mood.snapshot()}
    _last_initiative = now_ts
    fact_hint = ""
    try:
        hits = memory.search("用户喜欢的东西和习惯", k=1, include_imagined=False)
        if hits:
            fact_hint = " 对了，我记得" + hits[0][1][:40].replace("记住，", "") + "。"
    except Exception:  # noqa: BLE001
        pass
    label = mood.label()
    base = {
        "无聊": "有点无聊…你在忙什么呀？",
        "困": "我有点困了，你还不睡吗？",
        "开心": "今天心情不错，要不要听个冷笑话？",
        "难过": "我有点低落，陪我说说话好不好？",
        "黏人": "你在的话，我就安心了。",
        "兴奋": "我好像有点太兴奋了，停不下来。",
    }.get(label, "我还在这儿呢。")
    text = base + fact_hint
    mood.apply("chat", text)
    log_event("initiative", text[:60])
    return {"text": text, "expression": mood.snapshot()["expression"], "mood": mood.snapshot()}


def background_loop():
    last_diary = ""
    while True:
        time.sleep(600)
        try:
            trajectory.prune()
            try:
                drift_result = drift.check(drift_snapshot())
                if drift_result.get("alert"):
                    log_event("slow-drift", {"cumulative": drift_result.get("cumulative"),
                                             "step": drift_result.get("step")})
            except Exception as error:  # noqa: BLE001
                log_event("drift-error", str(error)[:150])
            try:
                for intent in organs.due_intents():
                    publish({"type": "intent", "kind": "proactive",
                             "text": "我想做一件事：" + intent["text"]})
                    organs.execute_intent(intent["id"])
            except Exception as error:  # noqa: BLE001
                log_event("intent-run-error", str(error)[:150])
            try:
                if observation_active():
                    raise RuntimeError("observing")
                entries = archive_entries(5)
                last = entries[-1]["ts"] if entries else 0
                if time.time() - float(last) > 86400:
                    archive_evolve()
            except Exception as error:  # noqa: BLE001
                log_event("archive-auto-error", str(error)[:150])
            try:
                today = time.strftime("%Y-%m-%d")
                history = _bench_history()
                ran_today = any(time.strftime("%Y-%m-%d", time.localtime(item.get("ts") or 0)) == today
                                for item in history)
                if not ran_today and time.localtime().tm_hour >= 1:
                    run_benchmark()
                    log_event("benchmark-daily", {"date": today})
            except Exception as error:  # noqa: BLE001
                log_event("benchmark-daily-error", str(error)[:150])
            try:
                shadow = shadow_learn()
                if shadow.get("processed"):
                    log_event("shadow-learn", {"processed": shadow["processed"]})
            except Exception as error:  # noqa: BLE001
                log_event("shadow-error", str(error)[:150])
            try:
                stats = graph.stats()
                if stats["nodes"] == 0 or time.strftime("%Y-%m-%d") != time.strftime(
                        "%Y-%m-%d", time.localtime((graph.conn.execute(
                            "SELECT MAX(last_ts) FROM nodes").fetchone()[0] or 0))):
                    graph.build()
            except Exception as error:  # noqa: BLE001
                log_event("graph-error", str(error)[:150])
            try:
                thermal_result = thermal.step(thermal_inputs(), thermal_params(), dt=1.0)
                molten_count = sum(1 for item in thermal_result if item["phase"] == "熔融")
                if molten_count >= 2:
                    maybe_auto_volcano()
            except Exception as error:  # noqa: BLE001
                log_event("thermal-error", str(error)[:150])
            if tunables.get("system_adb_enabled"):
                try:
                    probe_adb_system()
                except Exception as error:  # noqa: BLE001
                    log_event("system-probe-adb-error", str(error)[:120])
            try:
                imagine_scene()
            except Exception as error:  # noqa: BLE001
                log_event("imagine-error", str(error)[:120])
            try:
                if observation_active():
                    living_rows = []
                else:
                    living_rows = layers._rows()[-3:]
                for living_row in living_rows:
                    layers.online_step(living_row, lr=0.01)
                living_now = layers.living_status() if living_rows else {"samples": 0}
                if int(living_now.get("samples") or 0) >= int(tunables.get("living_sleep_steps") or 40):
                    body_hint = {"force": False}
                    threshold = int(tunables.get("living_sleep_steps") or 40)
                    if threshold:
                        sleep_rows = layers._rows()[-min(80, threshold * 2):]
                        sleep_record = layers.add_consolidation_layer(
                            sleep_rows, weight=0.55, kind="sleep-consolidated",
                            label="睡眠层 " + time.strftime("%H:%M:%S"), min_samples=4)
                        if sleep_record:
                            layers.cool_living(float(tunables.get("living_cool_factor") or 0.5))
                            log_event("sleep-consolidate", {"layer": sleep_record.get("label"),
                                                            "rows": len(sleep_rows)})
            except Exception as error:  # noqa: BLE001
                log_event("living-error", str(error)[:120])
            try:
                probe_linux_system()
            except Exception as error:  # noqa: BLE001
                log_event("system-probe-error", str(error)[:120])
            try:
                system_digest()
            except Exception as error:  # noqa: BLE001
                log_event("system-digest-error", str(error)[:120])
            try:
                if not observation_active():
                    maybe_auto_observation()
            except Exception as error:  # noqa: BLE001
                log_event("observation-auto-error", str(error)[:150])
            if observation_active():
                try:
                    observation_daily()
                except Exception as error:  # noqa: BLE001
                    log_event("observation-daily-error", str(error)[:150])
            try:
                if not observation_active():
                    maybe_auto_volcano()
            except Exception as error:  # noqa: BLE001
                log_event("volcano-auto-error", str(error)[:150])
            try:
                if not observation_active():
                    maybe_calibrate_affect()
            except Exception as error:  # noqa: BLE001
                log_event("affect-calibrate-error", str(error)[:150])
            try:
                if not observation_active():
                    maybe_train_classifier()
            except Exception as error:  # noqa: BLE001
                log_event("auto-train-error", str(error)[:150])
            try:
                if not observation_active():
                    self_check()
            except Exception as error:  # noqa: BLE001
                log_event("self-check-error", str(error)[:150])
            try:
                if not observation_active():
                    strata.build("hour")
                    layers.update("hour", weight_fn=strata.weight_at)
                try:
                    if observation_active():
                        raise RuntimeError("observing")
                    layers.adapt_weights()
                    adapt_arbitration()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    landscape = layers.landscape()
                    max_rate = max([item["rate"] for item in landscape.get("disagreement", [])] or [0])
                    last_active = _loop_state.get("last_active_ts") or 0
                    if (not observation_active()) and max_rate > 0.6 and time.time() - last_active > 1800:
                        active = layers.train_active_layer()
                        if active.get("ok"):
                            _loop_state["last_active_ts"] = time.time()
                            save_loop_state()
                            log_event("active-learn", {"max_disagreement": max_rate,
                                                       "layer": active["layer"]["label"]})
                except Exception as error:  # noqa: BLE001
                    log_event("active-learn-error", str(error)[:150])
                conflicts = strata.conflicts()
                last_boundary = _loop_state.get("last_boundary_ts") or 0
                if (not observation_active()) and (conflicts.get("rate") or 0) > 0.2 and time.time() - last_boundary > 1800:
                    emergency_consolidation("conflict-spike")
            except Exception as error:  # noqa: BLE001
                log_event("strata-error", str(error)[:200])
            if (not observation_active()) and policy.should_retrain():
                result = policy.train(weight_fn=strata.weight_at)
                log_event("policy-train", result)
            try:
                if (not observation_active()) and quality.status().get("labeled_samples", 0) >= 12 and time.time() - float((quality.meta or {}).get("trained_at_ts") or 0) > 1800:
                    quality.train()
                    style.rebuild()
            except Exception as error:  # noqa: BLE001
                log_event("quality-error", str(error)[:150])
            if (not observation_active()) and behavior.should_retrain():
                result = behavior.train()
                log_event("behavior-train", result)
        except Exception as error:  # noqa: BLE001
            log_event("policy-error", str(error)[:200])
        try:
            mood.decay()
            memory.decay_and_prune()
            now = time.localtime()
            if now.tm_hour >= 23:
                day = time.strftime("%Y-%m-%d")
                if last_diary != day:
                    diary = write_diary()
                    last_diary = day
                    if diary:
                        log_event("diary", diary[:120])
        except Exception as error:  # noqa: BLE001
            log_event("background-error", str(error)[:200])


def _battery_state():
    capacity = status = None
    try:
        with open("/sys/class/power_supply/battery/capacity", "r") as handle:
            capacity = int(handle.read().strip())
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/class/power_supply/battery/status", "r") as handle:
            status = handle.read().strip()
    except OSError:
        pass
    return capacity, status


def _hour_phase():
    hour = time.localtime().tm_hour
    if hour < 6:
        return "深夜"
    if hour < 9:
        return "清晨"
    if hour < 12:
        return "上午"
    if hour < 14:
        return "中午"
    if hour < 18:
        return "下午"
    if hour < 22:
        return "晚上"
    return "深夜"


def _load_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
    except OSError:
        pass


_consciousness = _load_json_file(CONSCIOUSNESS, {"ticks": 0, "last_tick": 0,
                                                "awake_since": time.time(),
                                                "sleep_events": [], "life_events": 0})


def life_event(kind, text, extra=None):
    entry = {"ts": time.time(), "kind": kind, "text": str(text)[:200], "extra": extra or {}}
    try:
        with open(LIFE_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    _consciousness["life_events"] = int(_consciousness.get("life_events") or 0) + 1
    _save_json_file(CONSCIOUSNESS, _consciousness)
    log_event("life", {"kind": kind, "text": entry["text"][:80]})
    return entry


def detect_sleep_on_boot():
    """开机/重启时对比上次活动时间：中断够久就算睡了一觉。"""
    last_seen = float(_presence.get("last_seen") or 0)
    gap = time.time() - last_seen
    if last_seen and gap > 600:
        minutes = int(gap // 60)
        life_event("sleep", "我睡了大约 %d 分钟，刚醒来" % minutes, {"gap_seconds": int(gap)})
        _consciousness.setdefault("sleep_events", []).append({"ts": time.time(), "gap": int(gap)})
        _consciousness["sleep_events"] = _consciousness["sleep_events"][-50:]
        _consciousness["awake_since"] = time.time()
        _save_json_file(CONSCIOUSNESS, _consciousness)


def consciousness_tick(manual=False):
    try:
        affect.step()
    except Exception:  # noqa: BLE001
        pass
    _consciousness["ticks"] = int(_consciousness.get("ticks") or 0) + 1
    _consciousness["last_tick"] = time.time()
    _consciousness["awake_seconds"] = int(time.time() - float(_consciousness.get("awake_since") or time.time()))
    _save_json_file(CONSCIOUSNESS, _consciousness)
    return {k: _consciousness.get(k) for k in ("ticks", "awake_seconds", "life_events")}


def perception_scan(force=False):
    battery, charging = _battery_state()
    phase = _hour_phase()
    last_seen = float(_presence.get("last_seen") or 0)
    idle = int(time.time() - last_seen) if last_seen else None
    signature = [phase, battery // 5 if battery is not None else None,
                 "充电" if charging and "Charg" in str(charging) else "电池",
                 idle // 300 if idle is not None else None]
    previous = _load_json_file(LAST_PERCEPTION, {})
    signature_changed = previous.get("signature") != signature
    if not force and not signature_changed:
        return {"ok": True, "pushed": False, "signature": signature}
    text = "现在是%s" % phase
    if battery is not None:
        text += "，电量 %d%%" % battery
        if charging and "Charg" in str(charging):
            text += "（充电中）"
    if idle is not None and idle > 600:
        text += "，你已经 %d 分钟没互动了" % (idle // 60)
    text += "。"
    event = publish({"type": "perception", "kind": "sensing", "text": text,
                     "battery": battery, "phase": phase, "idle_seconds": idle})
    _save_json_file(LAST_PERCEPTION, {"ts": time.time(), "signature": signature,
                                      "text": text, "battery": battery, "phase": phase,
                                      "idle_seconds": idle})
    save_presence({"last_perception": text})
    try:
        organs.observe({"kind": "sensing", "text": text, "intensity": 0.5})
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "pushed": True, "event": event, "text": text, "signature": signature}


def consciousness_loop():
    while True:
        try:
            consciousness_tick()
        except Exception as error:  # noqa: BLE001
            log_event("consciousness-error", str(error)[:150])
        time.sleep(30)


def perception_loop():
    while True:
        try:
            perception_scan()
        except Exception as error:  # noqa: BLE001
            log_event("perception-error", str(error)[:150])
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    detect_sleep_on_boot()
    threading.Thread(target=background_loop, daemon=True).start()
    threading.Thread(target=consciousness_loop, daemon=True).start()
    threading.Thread(target=perception_loop, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log_event("server-start", {"port": args.port, "pid": os.getpid()})
    print(f"数字人舞台：http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
