# 模型存储：权重（npz）与元数据/契约（json+manifest）分离，并负责跨版本兼容加载
import hashlib
import json
import os
import time

import numpy as np

WEIGHT_FORMAT = "npz-float32"
SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def manifest_path(layers_dir):
    return os.path.join(layers_dir, MANIFEST_NAME)


def load_manifest(layers_dir):
    try:
        with open(manifest_path(layers_dir), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "layers" in data:
            return data
    except (OSError, ValueError):
        pass
    return {"schema_version": SCHEMA_VERSION, "weight_format": WEIGHT_FORMAT,
            "updated_at": None, "layers": {}}


def save_manifest(layers_dir, manifest):
    manifest["schema_version"] = manifest.get("schema_version") or SCHEMA_VERSION
    manifest["weight_format"] = manifest.get("weight_format") or WEIGHT_FORMAT
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = manifest_path(layers_dir)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass
    return manifest


def register_layer(layers_dir, bucket, record, npz_path, meta_path, features=(), labels=None):
    """写入 manifest：权重文件的契约与校验和，与权重本身分开存。"""
    manifest = load_manifest(layers_dir)
    manifest["layers"][str(bucket)] = {
        "bucket": bucket,
        "label": record.get("label"),
        "kind": record.get("kind"),
        "weight_file": os.path.basename(npz_path),
        "meta_file": os.path.basename(meta_path) if meta_path else None,
        "sha256": sha256_file(npz_path),
        "bytes": os.path.getsize(npz_path) if os.path.exists(npz_path) else None,
        "schema_version": record.get("schema_version") or SCHEMA_VERSION,
        "weight_format": WEIGHT_FORMAT,
        "feature_dim": len(features),
        "labels": labels,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_manifest(layers_dir, manifest)
    return manifest["layers"][str(bucket)]


def ensure_manifest(layers_dir):
    """扫描目录，给旧文件补登记（兼容之前没有 manifest 的层）。"""
    manifest = load_manifest(layers_dir)
    changed = False
    try:
        names = sorted(os.listdir(layers_dir))
    except OSError:
        names = []
    for name in names:
        if not name.endswith(".npz") or name.startswith("layer-corr"):
            continue
        bucket = name[len("layer-"):-len(".npz")]
        if bucket in manifest["layers"]:
            continue
        npz_path = os.path.join(layers_dir, name)
        meta_path = npz_path[:-4] + ".json"
        record = {}
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            pass
        register_layer(layers_dir, int(bucket), record, npz_path,
                       meta_path if os.path.exists(meta_path) else None)
        changed = True
    if changed:
        manifest = load_manifest(layers_dir)  # register_layer 已逐条写入，重新读回
    return manifest


def adapt_weights(weights, meta, feature_dim, expressions, gestures):
    """旧权重兼容：按特征维度补零/截断，按标签名重排类别列。返回 (适配后权重, 说明)。"""
    notes = []
    adapted = {}
    feature_dim = int(feature_dim)
    old_labels = (meta.get("schema") or {}).get("labels") or {}
    label_map = {
        "w_expr": list(old_labels.get("expressions") or expressions),
        "b_expr": list(old_labels.get("expressions") or expressions),
        "w_gest": list(old_labels.get("gestures") or gestures),
        "b_gest": list(old_labels.get("gestures") or gestures),
    }
    for key, value in weights.items():
        if key not in ("w_expr", "b_expr", "w_gest", "b_gest"):
            continue
        array = np.asarray(value, dtype=np.float32)
        if key.startswith("w_"):
            rows, cols = array.shape
            if rows != feature_dim:
                fixed = np.zeros((feature_dim, cols), dtype=np.float32)
                keep = min(rows, feature_dim)
                fixed[:keep, :] = array[:keep, :]
                array = fixed
                notes.append("%s 特征维度 %d → %d（补零/截断）" % (key, rows, feature_dim))
            expected = expressions if key == "w_expr" else gestures
            if cols != len(expected):
                fixed = np.zeros((array.shape[0], len(expected)), dtype=np.float32)
                src_labels = label_map[key]
                for old_index, label in enumerate(src_labels[:cols]):
                    if label in expected:
                        fixed[:, expected.index(label)] = array[:, old_index]
                array = fixed
                notes.append("%s 类别 %d → %d（按标签名重排）" % (key, cols, len(expected)))
        else:
            expected = expressions if key == "b_expr" else gestures
            if array.shape[0] != len(expected):
                fixed = np.zeros(len(expected), dtype=np.float32)
                keep = min(array.shape[0], len(expected))
                fixed[:keep] = array[:keep]
                array = fixed
                notes.append("%s 偏置 %d → %d" % (key, array.shape[0], len(expected)))
        adapted[key] = array
    return adapted, notes


def verify_manifest(layers_dir):
    """校验 manifest 里每个权重文件是否存在、校验和是否一致。"""
    manifest = ensure_manifest(layers_dir)
    results = []
    for bucket, entry in manifest.get("layers", {}).items():
        path = os.path.join(layers_dir, entry.get("weight_file") or "")
        exists = os.path.exists(path)
        checksum = sha256_file(path) if exists else None
        ok = exists and (not entry.get("sha256") or checksum == entry.get("sha256"))
        results.append({
            "bucket": bucket,
            "label": entry.get("label"),
            "exists": exists,
            "checksum_ok": (checksum == entry.get("sha256")) if entry.get("sha256") and checksum else bool(exists),
            "schema_version": entry.get("schema_version"),
            "feature_dim": entry.get("feature_dim"),
            "ok": ok,
        })
    bad = [item for item in results if not item["ok"]]
    return {
        "layers": len(results),
        "ok": not bad,
        "bad": bad,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": manifest.get("schema_version"),
        "weight_format": manifest.get("weight_format"),
    }
