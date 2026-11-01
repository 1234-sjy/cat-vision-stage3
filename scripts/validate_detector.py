"""Validation-only detector comparison with small batches and no automatic retries."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask, gpu_checkpoint, set_low_load_threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    set_low_load_threads()
    from ultralytics import YOLO
    import torch

    policy = json.loads((ROOT / "configs/runtime-policy.json").read_text(encoding="utf-8"))
    batch = policy["gpu_validation_batch"]
    pause = policy["pause_seconds_after_sync"]
    print(f"CONFIG batch={batch}, workers=0, synchronize=True, pause={pause}, retry=False", flush=True)
    records = [r for r in json.loads((ROOT / "artifacts/data/manifest.json").read_text(encoding="utf-8"))
               if r["split"] == "val"]
    negative = defaultdict(list)
    for line in (args.data_root / "annotations/trainval.txt").read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#") and int(fields[2]) == 2:
            negative[fields[0].rsplit("_", 1)[0]].append(fields[0])
    rng = random.Random(17)
    for breed in sorted(negative):
        for image_id in rng.sample(sorted(negative[breed]), 2):
            records.append({"image_id": image_id, "relative_path": f"images/{image_id}.jpg",
                            "breed": "negative_dog"})
    if args.limit:
        records = records[:args.limit]
    report = {}
    old_log = ROOT / "runtime-detector-validation.log"
    if old_log.exists():
        for line in old_log.read_text(encoding="utf-8").splitlines():
            if line.startswith("yolo26n.pt {"):
                report["yolo26n.pt"] = {"metrics": ast.literal_eval(line.split(" ", 1)[1]),
                    "origin": "earlier completed validation; older batch/timing settings"}
    model = YOLO(str(ROOT / "weights/yolo26m.pt"))
    started = time.perf_counter()
    scores = []
    for first in range(0, len(records), batch):
        group = records[first:first + batch]
        results = model.predict([str(args.data_root / r["relative_path"]) for r in group],
                                batch=batch, workers=0, conf=0.05, imgsz=640,
                                device=0, verbose=False, stream=False)
        gpu_checkpoint(pause_seconds=pause)
        for record, result in zip(group, results):
            cat_id = next(key for key, value in result.names.items() if value == "cat")
            values = result.boxes.conf[result.boxes.cls == cat_id]
            scores.append({"image_id": record["image_id"], "is_cat": record["breed"] != "negative_dog",
                           "cat_score": float(values.max()) if len(values) else 0})
        save_json(ROOT / "runtime/detector-candidate-progress.json", scores)
        print(f"BATCH {first//batch+1}; {len(scores)}/{len(records)}; "
              f"elapsed={time.perf_counter()-started:.2f}s; "
              f"allocated_MB={torch.cuda.memory_allocated()/1024**2:.1f}", flush=True)
    metrics = {}
    for threshold in [0.05, 0.10, 0.15, 0.25]:
        metrics[str(threshold)] = {
            "localized_cats": sum(s["is_cat"] and s["cat_score"] >= threshold for s in scores),
            "cat_count": sum(s["is_cat"] for s in scores),
            "false_cat_on_dog": sum(not s["is_cat"] and s["cat_score"] >= threshold for s in scores),
            "dog_count": sum(not s["is_cat"] for s in scores)}
    report["yolo26m.pt"] = {"metrics": metrics, "seconds": time.perf_counter() - started,
                            "records": scores, "batch": batch, "workers": 0,
                            "pause_seconds_after_sync": pause, "partial": bool(args.limit)}
    destination = "detector-validation-partial.json" if args.limit else "detector-validation.json"
    save_json(ROOT / "artifacts" / destination, report)
    print("DETECTOR VALIDATION FINISHED", metrics, flush=True)


if __name__ == "__main__":
    with ModelTask("validate-detector"):
        main()
