from __future__ import annotations

from collections import Counter
import statistics


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
