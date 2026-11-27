# 核心热动力学：热源 → 传导 → 冷却 → 熔融/固结（火山是结果，不是开关）
import json
import os
import time

DEFAULTS = {
    "ambient": 0.2,
    "conductivity": 0.15,
    "cooling": 0.05,
    "heat_fitness": 0.30,
    "heat_usage": 0.25,
    "heat_bias": 0.05,
    "melt_point": 0.85,
    "solid_point": 0.35,
}


class ThermalModel:
    def __init__(self, path):
        self.path = path
        self.state = self._load()
        self.state.setdefault("layers", {})
        self.state.setdefault("last_ts", time.time())

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

    def step(self, layers, params, dt=1.0):
        """一次热演化：热源 + 层间传导 + 向环境冷却。"""
        ambient = float(params.get("ambient", DEFAULTS["ambient"]))
        conductivity = float(params.get("conductivity", DEFAULTS["conductivity"]))
        cooling = float(params.get("cooling", DEFAULTS["cooling"]))
        heat_fitness = float(params.get("heat_fitness", DEFAULTS["heat_fitness"]))
        heat_usage = float(params.get("heat_usage", DEFAULTS["heat_usage"]))
        heat_bias = float(params.get("heat_bias", DEFAULTS["heat_bias"]))
        melt_point = float(params.get("melt_point", DEFAULTS["melt_point"]))
        solid_point = float(params.get("solid_point", DEFAULTS["solid_point"]))

        ordered = sorted(layers, key=lambda item: int(item["bucket"]))
        temps = {}
        for item in ordered:
            key = str(item["bucket"])
            current = float(self.state["layers"].get(key) or ambient)
            source = (heat_bias + heat_fitness * float(item.get("fitness") or 0)
                      + heat_usage * float(item.get("usage") or 0))
            temps[key] = current + (source - cooling * (current - ambient)) * dt
        # 传导：和相邻层交换热量（差距越大传得越快）
        adjusted = dict(temps)
        keys = list(temps)
        for index, key in enumerate(keys):
            neighbor_values = []
            if index > 0:
                neighbor_values.append(temps[keys[index - 1]])
            if index + 1 < len(keys):
                neighbor_values.append(temps[keys[index + 1]])
            if neighbor_values:
                mean_neighbor = sum(neighbor_values) / len(neighbor_values)
                adjusted[key] = temps[key] + conductivity * (mean_neighbor - temps[key]) * dt
        result = []
        for item in ordered:
            key = str(item["bucket"])
            temperature = max(0.0, min(1.5, adjusted[key]))
            self.state["layers"][key] = round(temperature, 4)
            phase = "熔融" if temperature >= melt_point else ("固结" if temperature <= solid_point else "活跃")
            result.append({"bucket": item["bucket"], "label": item.get("label"),
                           "temperature": round(temperature, 3), "phase": phase,
                           "fitness": item.get("fitness"), "usage": item.get("usage")})
        self.state["last_ts"] = time.time()
        self._save()
        return result

    def snapshot(self, layers, params):
        melt_point = float(params.get("melt_point", DEFAULTS["melt_point"]))
        solid_point = float(params.get("solid_point", DEFAULTS["solid_point"]))
        items = []
        for layer in layers:
            key = str(layer["bucket"])
            temperature = float(self.state["layers"].get(key) or float(params.get("ambient", 0.2)))
            phase = "熔融" if temperature >= melt_point else ("固结" if temperature <= solid_point else "活跃")
            items.append({**layer, "temperature": round(temperature, 3), "phase": phase})
        items.sort(key=lambda item: -item["temperature"])
        return items
