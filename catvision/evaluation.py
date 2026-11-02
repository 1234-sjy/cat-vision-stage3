from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


class InputIntegrityError(ValueError):
    def __init__(self, image_id, expected, actual):
        self.actual_sha256 = actual
        super().__init__(f"Input SHA256 mismatch for {image_id}: expected {expected}, actual {actual}")


def read_verified_input(record: dict, data_root: Path) -> tuple[bytes, str]:
    path = (data_root / record["relative_path"]).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError("Input path leaves the selected data directory")
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != record["sha256"]:
        raise InputIntegrityError(record["image_id"], record["sha256"], actual)
    return content, actual


def prediction_cache_matches(prediction: dict, image_id: str, measured_sha: str, fingerprint: str) -> bool:
    return (prediction.get("image_id") == image_id and prediction.get("input_sha256") == measured_sha
            and prediction.get("run_fingerprint") == fingerprint)


def inference_fingerprint(root: Path, route: str) -> str:
    digest = hashlib.sha256(route.encode())
    files = [root / "catvision/common.py"]
    if route == "traditional":
        files += [root / "catvision" / name for name in ["traditional.py", "geometry.py"]]
        files += [root / "weights" / name for name in ["yolo26n.pt", "sam2.1_t.pt"]]
        active = root / 'configs/active-classifier.json'
        if active.exists():
            config = json.loads(active.read_text(encoding='utf-8'))
            checkpoint = config['checkpoint']
            if Path(checkpoint).name != checkpoint:
                raise ValueError('Checkpoint must be a filename inside weights')
            files += [active,root/'catvision/breed_transformer.py',root/'weights'/checkpoint]
        else:
            files += [root/'weights/breed-best.pt']
    else:
        files += [root / "catvision/vlm.py"]
        base = root / "weights/qwen3-vl-4b"
        index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        if any(Path(name).name != name for name in shards):
            raise ValueError("Unexpected model shard path")
        files += [base / name for name in shards]
        files += [p for p in base.iterdir() if p.is_file() and p.suffix in {".json", ".txt", ".jinja"}]
    for path in sorted(set(files)):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_comparison(truth: list[dict], *routes: list[dict]) -> None:
    expected = {r["image_id"]: r["sha256"] for r in truth}
    for predictions in routes:
        ids = [p["image_id"] for p in predictions]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError("两条路线必须提供固定清单的完整逐图结果")
        for prediction in predictions:
            if prediction.get("input_sha256") != expected[prediction["image_id"]]:
                raise ValueError("两条路线的原始输入SHA256与固定清单不一致")


def score_predictions(truth: list[dict], predictions: list[dict]) -> dict:
    lookup = {p["image_id"]: p for p in predictions}
    if len(lookup) != len(predictions):
        raise ValueError("Duplicate prediction image IDs")
    labels = sorted({r["breed"] for r in truth})
    indexes = {name: i for i, name in enumerate(labels)}
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    correct = 0
    statuses, times = Counter(), []
    for record in truth:
        prediction = lookup.get(record["image_id"], {"status": "missing"})
        status = prediction.get("status", "error")
        statuses[status] += 1
        breed = prediction.get("breed") if status == "ok" else None
        matrix[indexes[record["breed"]]][indexes.get(breed, len(labels))] += 1
        correct += int(breed == record["breed"])
        if isinstance(prediction.get("inference_seconds"), (int, float)):
            times.append(prediction["inference_seconds"])
    per_class = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = true_positive / predicted if predicted else 0
        recall = true_positive / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {"count": len(truth), "correct": correct,
            "accuracy": correct / len(truth) if truth else 0,
            "macro_f1": statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0,
            "status_counts": dict(statuses), "per_class": per_class,
            "row_labels": labels, "column_labels": labels + ["no_result"],
            "confusion_matrix": matrix,
            "mean_inference_seconds": statistics.mean(times) if times else None,
            "median_inference_seconds": statistics.median(times) if times else None}
