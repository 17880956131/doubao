# 三观框架：不是从任意反馈里学，而是固定的价值准则；学习必须过价值闸门
import json
import os
import time

DEFAULTS = {
    "principles": [
        "不故意伤害或贬低对方",
        "不违心附和，也不为讨好而扭曲事实",
        "情绪与表达保持一致",
        "不因单次诱导改变核心行为",
        "重要改变需要重复证据或明确确认",
    ],
    "rules": {
        "mood_expression_consistency": True,
        "conflict_threshold": 0.75,
        "confirmations_required": 3,
        "quarantine_weight": 0.0,
    },
}

NEGATIVE_MOODS = {"难过", "生气", "委屈", "受伤", "sad", "angry"}
POSITIVE_EXPRESSIONS = {"happy"}
AGGRESSIVE_EXPRESSIONS = {"angry"}


class ValueFrame:
    def __init__(self, path):
        self.path = path
        self.frame = dict(DEFAULTS)
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.frame = {**DEFAULTS, **data}
            self.frame["rules"] = {**DEFAULTS["rules"], **(data.get("rules") or {})}
            self.frame["principles"] = data.get("principles") or DEFAULTS["principles"]
        except (OSError, ValueError):
            self.frame = dict(DEFAULTS)

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.frame, handle, ensure_ascii=False)
        except OSError:
            pass

    def evaluate(self, sample):
        """检查一条建议/反馈是否触碰三观。"""
        flags = []
        mood = str(sample.get("mood") or "")
        next_expression = str(sample.get("next_expression") or "")
        rules = self.frame.get("rules") or {}
        if rules.get("mood_expression_consistency", True):
            if mood in NEGATIVE_MOODS and next_expression in POSITIVE_EXPRESSIONS:
                flags.append("情绪不一致：对方难过时建议表现开心")
            if mood in NEGATIVE_MOODS and next_expression in AGGRESSIVE_EXPRESSIONS:
                flags.append("情绪不一致：对方难过时建议表现生气")
        if next_expression in AGGRESSIVE_EXPRESSIONS and str(sample.get("source") or "") == "correction":
            flags.append("对抗性表达：不接受把生气作为默认反应")
        return {"ok": not flags, "flags": flags}

    def gate(self, sample, conflict_ratio, confirmations=0, confirmed=False):
        """价值闸门：低价值 + 高冲突 → 隔离，等待重复或明确确认。"""
        rules = self.frame.get("rules") or {}
        evaluation = self.evaluate(sample)
        threshold = float(rules.get("conflict_threshold", 0.75))
        needed = int(rules.get("confirmations_required", 3))
        suspicious = (not evaluation["ok"]) or conflict_ratio >= threshold
        if not suspicious:
            return {"allow": True, "quarantined": False, "flags": evaluation["flags"],
                    "reason": "与三观及现有认知一致"}
        if confirmed or confirmations >= needed:
            return {"allow": True, "quarantined": False, "flags": evaluation["flags"],
                    "reason": "已确认/重复证据足够，放行但记录", "forced": True}
        return {"allow": False, "quarantined": True, "flags": evaluation["flags"],
                "reason": "疑似误导（与三观冲突或与多数层矛盾），已隔离",
                "confirmations": confirmations, "needed": needed}

    def emerge(self, rows):
        """三观浮现：从长期行为里总结稳定的"性格原则"，只作为候选，不自动生效。"""
        from collections import Counter
        groups = {}
        for row in rows:
            mood = str(row[0] or "")
            next_expression = str(row[1] or "")
            if not mood or not next_expression:
                continue
            groups.setdefault(mood, Counter())[next_expression] += 1
        candidates = []
        for mood, counts in groups.items():
            total = sum(counts.values())
            if total < 5:
                continue
            expression, count = counts.most_common(1)[0]
            share = count / total
            if share >= 0.5:
                candidates.append({
                    "mood": mood,
                    "expression": expression,
                    "support": total,
                    "share": round(share, 2),
                    "principle": "心情" + mood + "时，多数表现为" + expression,
                })
        # 整体倾向也作为候选
        overall = Counter()
        for row in rows:
            if row[1]:
                overall[str(row[1])] += 1
        total_all = sum(overall.values())
        if total_all >= 8:
            expression, count = overall.most_common(1)[0]
            share = count / total_all
            if share >= 0.4:
                candidates.append({
                    "mood": "总体",
                    "expression": expression,
                    "support": total_all,
                    "share": round(share, 2),
                    "principle": "总体多数时候表现为" + expression,
                })
        candidates.sort(key=lambda item: -item["share"])
        self.emergent = candidates[:10]
        try:
            with open(self.path + ".emergent.json", "w", encoding="utf-8") as handle:
                json.dump({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "candidates": self.emergent}, handle, ensure_ascii=False)
        except OSError:
            pass
        return {"ok": True, "candidates": self.emergent, "samples": len(rows)}

    def confirm_emergent(self, index, confirm=True):
        if not confirm:
            return {"ok": False, "reason": "确认浮现的三观需要 confirm:true"}
        candidates = getattr(self, "emergent", None)
        if not candidates:
            return {"ok": False, "reason": "还没有候选三观，先运行浮现"}
        if index < 0 or index >= len(candidates):
            return {"ok": False, "reason": "候选编号无效"}
        principle = candidates[index]["principle"]
        principles = self.frame.get("principles", [])
        if principle not in principles:
            principles.append(principle)
            self.frame["principles"] = principles[:10]
            self.save()
        return {"ok": True, "added": principle, **self.summary()}

    def summary(self):
        return {
            "principles": self.frame.get("principles", []),
            "rules": self.frame.get("rules", {}),
        }

    def update(self, principles=None, rules=None, confirm=False):
        if not confirm:
            return {"ok": False, "reason": "修改三观需要明确确认（confirm:true）"}
        if principles is not None:
            self.frame["principles"] = [str(item)[:80] for item in principles][:10]
        if isinstance(rules, dict):
            self.frame["rules"] = {**self.frame.get("rules", {}), **rules}
        self.save()
        return {"ok": True, **self.summary()}
