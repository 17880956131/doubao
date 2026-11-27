# 可标定参数：代码只负责结构，"判断阈值"从数据里标定，并记录来源（硬编码 / 标定 / 学习）
import json
import math
import os
import time

DEFAULTS = {
    "turbulence_high": 0.6,
    "turbulence_mid": 0.35,
    "values_conflict_threshold": 0.75,
    "layer_confidence_threshold": 0.35,
    "classifier_epochs": 350,
    "classifier_lr": 0.5,
    "classifier_l2": 0.001,
    "classifier_min_new": 5,
    "classifier_min_samples": 8,
    "ensemble_weight_power": 1.0,
    "ensemble_sample_power": 0.5,
    "arb_mood_bonus": 0.12,
    "arb_stress_penalty": 0.25,
    "arb_stress_threshold": 0.45,
    "affect_decay_tau": 120.0,
    "affect_oscillation": 0.04,
    "affect_threat_gain": 0.6,
    "affect_valence_gain": 0.35,
    "affect_reassure_retain": 0.4,
    "settle_w_fitness": 0.45,
    "settle_w_recency": 0.2,
    "settle_w_usage": 0.15,
    "settle_w_redundancy": -0.25,
    "active_layer_cap": 12,
    "volcano_min_interval_hours": 6,
    "thermal_ambient": 0.2,
    "thermal_conductivity": 0.15,
    "thermal_cooling": 0.05,
    "thermal_heat_fitness": 0.30,
    "thermal_heat_usage": 0.25,
    "thermal_melt_point": 0.85,
    "thermal_solid_point": 0.35,
    "layer_transmission": 0.15,
    "system_adb_enabled": False,
    "settle_w_credit": 0.15,
    "credit_low_threshold": 0.3,
}


class Tunables:
    def __init__(self, path, observations_path):
        self.path = path
        self.observations_path = observations_path
        self.params = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                self.params = json.load(handle)
        except (OSError, ValueError):
            self.params = {}
        for name, value in DEFAULTS.items():
            self.params.setdefault(name, {"value": value, "source": "hardcoded",
                                          "samples": 0, "updated_at": None})

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.params, handle, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def get(self, name, default=None):
        entry = self.params.get(name) or {}
        return entry.get("value", DEFAULTS.get(name, default))

    def provenance(self):
        return {name: {"value": entry.get("value"), "source": entry.get("source"),
                       "samples": entry.get("samples"), "updated_at": entry.get("updated_at")}
                for name, entry in self.params.items()}

    def observe(self, kind, value):
        entry = {"ts": time.time(), "kind": kind, "value": float(value)}
        try:
            with open(self.observations_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _values(self, kind, window=500):
        values = []
        try:
            with open(self.observations_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if entry.get("kind") == kind:
                        values.append(float(entry.get("value") or 0))
        except OSError:
            pass
        return values[-window:]

    @staticmethod
    def _quantile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    def set_many(self, patch, source="配置"):
        changed = {}
        for name, value in (patch or {}).items():
            if name not in DEFAULTS:
                continue
            entry = self.params.setdefault(name, {"value": DEFAULTS[name]})
            entry["value"] = value
            entry["source"] = source
            entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            changed[name] = value
        self._save()
        return changed

    def calibrate(self, values_frame=None):
        """从观测分布标定阈值：高阈值取 80 分位、中阈值取 55 分位、冲突阈值取 90 分位。"""
        changes = {}
        turbulence = self._values("turbulence")
        if len(turbulence) >= 20:
            high = round(min(0.9, max(0.3, self._quantile(turbulence, 0.8))), 3)
            mid = round(min(high - 0.05, max(0.15, self._quantile(turbulence, 0.55))), 3)
            changes["turbulence_high"] = high
            changes["turbulence_mid"] = mid
        conflicts = self._values("conflict")
        if len(conflicts) >= 20:
            threshold = round(min(0.95, max(0.4, self._quantile(conflicts, 0.9))), 3)
            changes["values_conflict_threshold"] = threshold
            if values_frame is not None:
                values_frame.frame.setdefault("rules", {})["conflict_threshold"] = threshold
                values_frame.save()
        for name, value in changes.items():
            self.params[name] = {"value": value, "source": "calibrated",
                                 "samples": len(turbulence) if name.startswith("turbulence") else len(conflicts),
                                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._save()
        return {"ok": True, "changes": changes, "turbulence_samples": len(turbulence),
                "conflict_samples": len(conflicts), "provenance": self.provenance()}
