# 慢漂移检测：防止"温水煮青蛙"式的渐进带偏
import json
import math
import os
import time

DEFAULTS = {
    "window_days": 7,
    "slow_step_threshold": 0.15,
    "cumulative_alert": 0.35,
}


class DriftMonitor:
    def __init__(self, path):
        self.path = path
        self.state = self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {"baseline": None, "cumulative": 0.0, "last_step": 0.0, "history": [], "alerts": []}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False)
        except OSError:
            pass

    @staticmethod
    def _js_divergence(a, b):
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        pa = [float(a.get(key, 0)) for key in keys]
        pb = [float(b.get(key, 0)) for key in keys]
        sa = sum(pa) or 1.0
        sb = sum(pb) or 1.0
        pa = [value / sa for value in pa]
        pb = [value / sb for value in pb]
        mid = [(x + y) / 2 for x, y in zip(pa, pb)]
        def kl(p, m):
            return sum(x * math.log(x / y) for x, y in zip(p, m) if x > 0 and y > 0)
        return round(0.5 * kl(pa, mid) + 0.5 * kl(pb, mid), 4)

    def reset(self, snapshot):
        self.state["baseline"] = snapshot
        self.state["baseline_ts"] = time.time()
        self.state["cumulative"] = 0.0
        self.state["last_step"] = 0.0
        self._save()
        return {"ok": True, "baseline_ts": self.state["baseline_ts"]}

    def check(self, snapshot):
        baseline = self.state.get("baseline")
        if not baseline:
            self.reset(snapshot)
            return {"ok": True, "baseline_created": True, "step": 0.0, "cumulative": 0.0,
                    "alert": False, "baseline": snapshot}
        components = {}
        if baseline.get("layer_weights") and snapshot.get("layer_weights"):
            keys = set(baseline["layer_weights"]) | set(snapshot["layer_weights"])
            diffs = [abs(baseline["layer_weights"].get(key, 0.0) - snapshot["layer_weights"].get(key, 0.0))
                     for key in keys]
            components["weights"] = round(sum(diffs) / max(len(diffs), 1), 4)
        if baseline.get("expression_dist") and snapshot.get("expression_dist"):
            components["expression_js"] = self._js_divergence(baseline["expression_dist"],
                                                              snapshot["expression_dist"])
        if baseline.get("arbitration") and snapshot.get("arbitration"):
            keys = set(baseline["arbitration"]) | set(snapshot["arbitration"])
            diffs = [abs(float(baseline["arbitration"].get(key, 0)) - float(snapshot["arbitration"].get(key, 0)))
                     for key in keys]
            components["arbitration"] = round(sum(diffs) / max(len(diffs), 1), 4)
        step = round(sum(components.values()) / max(len(components), 1), 4)
        slow = step < DEFAULTS["slow_step_threshold"]
        cumulative = float(self.state.get("cumulative") or 0.0)
        cumulative = cumulative + step if slow else step
        alert = slow and cumulative > DEFAULTS["cumulative_alert"]
        self.state["cumulative"] = round(cumulative, 4)
        self.state["last_step"] = step
        self.state.setdefault("history", []).append({
            "ts": time.time(), "step": step, "cumulative": round(cumulative, 4),
            "components": components, "slow": slow,
        })
        self.state["history"] = self.state["history"][-200:]
        if alert:
            self.state.setdefault("alerts", []).append({"ts": time.time(), "step": step,
                                                        "cumulative": round(cumulative, 4),
                                                        "components": components})
            self.state["alerts"] = self.state["alerts"][-20:]
        self._save()
        return {"ok": True, "step": step, "cumulative": round(cumulative, 4),
                "alert": alert, "components": components, "slow": slow,
                "baseline_ts": self.state.get("baseline_ts")}

    def snapshot_state(self):
        return {
            "cumulative": self.state.get("cumulative"),
            "last_step": self.state.get("last_step"),
            "baseline_ts": self.state.get("baseline_ts"),
            "baseline": self.state.get("baseline"),
            "alerts": (self.state.get("alerts") or [])[-5:],
            "history": (self.state.get("history") or [])[-10:],
        }
