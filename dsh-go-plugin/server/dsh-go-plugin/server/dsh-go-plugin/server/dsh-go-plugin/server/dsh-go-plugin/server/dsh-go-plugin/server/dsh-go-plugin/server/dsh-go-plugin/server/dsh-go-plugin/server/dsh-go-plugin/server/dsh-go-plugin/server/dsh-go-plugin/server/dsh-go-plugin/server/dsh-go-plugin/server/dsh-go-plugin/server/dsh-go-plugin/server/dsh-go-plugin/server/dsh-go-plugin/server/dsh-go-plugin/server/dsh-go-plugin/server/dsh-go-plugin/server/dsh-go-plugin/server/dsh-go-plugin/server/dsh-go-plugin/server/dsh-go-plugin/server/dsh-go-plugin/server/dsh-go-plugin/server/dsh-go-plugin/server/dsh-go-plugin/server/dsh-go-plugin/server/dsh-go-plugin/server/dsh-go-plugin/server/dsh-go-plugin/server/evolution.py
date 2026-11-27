# 进化视图（按用户定义）：基因 = 系统源代码；模因 = 权重参数数据
import hashlib
import json
import os
import time

import model_store

CODE_FILES = ["server.py", "layer_models.py", "policy_model.py", "model_store.py",
              "values.py", "evolution.py", "quality_model.py", "style_profile.py",
              "stratum_store.py", "self_model.py", "trajectory_store.py", "memory_store.py",
              "mood.py", "behavior_model.py", "drift_monitor.py", "guardian.sh", "mcp-server.mjs"]


def code_hash(base_dir):
    digest = hashlib.sha256()
    counts = {}
    for name in CODE_FILES:
        path = os.path.join(base_dir, name)
        try:
            content = open(path, "rb").read()
        except OSError:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(content)
        counts[name] = len(content.splitlines())
    return digest.hexdigest()[:12], counts


def _meme_id(kind, content):
    import hashlib
    return hashlib.sha256((kind + "|" + str(content)).encode("utf-8")).hexdigest()[:12]


class EvolutionStore:
    def __init__(self, path, layers, layers_dir, base_dir):
        self.path = path
        self.layers = layers
        self.layers_dir = layers_dir
        self.base_dir = base_dir
        self.memes = self._load()
        self.code_path = os.path.join(base_dir, "data", "code_genome.jsonl")
        self.codes = self._load_codes()

    # ---------- 基因：系统源代码 ----------
    def _load_codes(self):
        entries = []
        try:
            with open(self.code_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return entries

    def snapshot_code(self, note="", fitness=None):
        digest, counts = code_hash(self.base_dir)
        parent = self.codes[-1]["id"] if self.codes else None
        entry = {
            "id": digest,
            "parent": parent,
            "changed": bool(parent and parent != digest),
            "mutation": note or ("代码变更" if parent and parent != digest else "快照"),
            "files": len(counts),
            "lines": sum(counts.values()),
            "fitness": fitness,
            "birth": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.codes.append(entry)
        self.codes = self.codes[-100:]
        try:
            with open(self.code_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return entry

    def code_genomes(self):
        return self.codes[-30:]

    def maybe_snapshot_code(self, fitness=None):
        digest, _ = code_hash(self.base_dir)
        if self.codes and self.codes[-1]["id"] == digest:
            return None
        return self.snapshot_code("自动检测到代码变化", fitness=fitness)

    def _load(self):
        memes = {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if entry.get("id"):
                        memes[entry["id"]] = entry
        except OSError:
            pass
        return memes

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                for entry in self.memes.values():
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---------- 模因：权重参数数据 ----------
    def weight_memes(self):
        manifest = model_store.ensure_manifest(self.layers_dir)
        entries = manifest.get("layers", {})
        rows = []
        ordered = sorted(self.layers.meta.items(), key=lambda item: float(item[0]) if str(item[0]).replace(".", "").isdigit() else 0)
        generation = 0
        for bucket, meta in ordered:
            if meta.get("kind") == "correction":
                continue
            generation += 1
            entry = entries.get(str(bucket)) or {}
            stale = (meta.get("semantic") or {}).get("stale")
            rows.append({
                "generation": generation,
                "label": meta.get("label"),
                "kind": meta.get("kind") or "era",
                "content_hash": (entry.get("sha256") or "")[:12],
                "parents": [meta.get("source_layer")] if meta.get("source_layer") else (
                    [meta["range"][0], meta["range"][1]] if meta.get("range") else []),
                "mutation": "重锚" if meta.get("kind") == "retrained" else (
                    "巩固" if meta.get("kind") == "consolidation" else (
                        "边界" if meta.get("kind") == "boundary" else "沉积")),
                "fitness": meta.get("recent_accuracy") or meta.get("accuracy_expression"),
                "weight": meta.get("adapted_weight") or meta.get("weight"),
                "samples": meta.get("samples"),
                "status": "stale" if stale else "active",
                "birth": meta.get("trained_at"),
            })
        return rows

    # ---------- 模因：三观/自述/风格 ----------
    def record_meme(self, kind, content, status="expressed", fitness=0.5,
                    parents=None, mutation="生成"):
        content = str(content or "").strip()
        if not content:
            return None
        meme_id = _meme_id(kind, content)
        entry = self.memes.get(meme_id) or {
            "id": meme_id,
            "kind": kind,
            "content": content[:300],
            "parents": parents or [],
            "mutation": mutation,
            "birth": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fitness": fitness,
            "status": status,
        }
        entry["fitness"] = round(float(fitness), 3)
        entry["status"] = status
        entry["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.memes[meme_id] = entry
        self._save()
        return entry

    def scan_culture(self, values, style, narrations):
        for principle in values.frame.get("principles", []):
            self.record_meme("value", principle, status="adopted", fitness=1.0,
                             mutation="人写+确认")
        for candidate in getattr(values, "emergent", []) or []:
            self.record_meme("value-candidate", candidate.get("principle"),
                             status="candidate", fitness=candidate.get("share", 0.5),
                             mutation="行为浮现")
        hint = style.hint() if style else ""
        if hint:
            self.record_meme("style", hint, status="adopted", fitness=0.8,
                             mutation="回应选择")
        previous = None
        for entry in narrations[-10:]:
            text = entry.get("text")
            if not text:
                continue
            meme = self.record_meme("narrative", text, status="expressed",
                                    fitness=entry.get("benchmark") or 0.5,
                                    parents=[previous] if previous else [],
                                    mutation="自述变异")
            previous = meme["id"] if meme else previous

    def list_memes(self, limit=30):
        ordered = sorted(self.memes.values(), key=lambda item: item.get("last_seen") or "", reverse=True)
        return ordered[:limit]

    def summary(self):
        weight_memes = self.weight_memes()
        by_kind = {}
        for meme in weight_memes:
            by_kind[meme["kind"]] = by_kind.get(meme["kind"], 0) + 1
        culture = {}
        for meme in self.memes.values():
            culture[meme["kind"]] = culture.get(meme["kind"], 0) + 1
        return {
            "code_genomes": len(self.codes),
            "code_changed": bool(self.codes and self.codes[-1].get("changed")),
            "weight_memes": len(weight_memes),
            "by_kind": by_kind,
            "stale": sum(1 for meme in weight_memes if meme["status"] == "stale"),
            "culture": culture,
        }
