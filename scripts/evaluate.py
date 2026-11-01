from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import score_predictions
from catvision.run_control import ModelTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, choices=["traditional", "vlm"])
    parser.add_argument("--set", default="comparison", choices=["comparison", "test", "val"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.set == "comparison":
        records = json.loads((ROOT / "artifacts/data/comparison.json").read_text(encoding="utf-8"))
    else:
        records = [r for r in json.loads((ROOT / "artifacts/data/manifest.json").read_text(encoding="utf-8"))
                   if r["split"] == args.set]
    if args.limit:
        records = records[:args.limit]
    directory = ROOT / "artifacts/evaluation" / args.set / args.route
    directory.mkdir(parents=True, exist_ok=True)
    output_file = directory / "predictions.json"
    existing = {} if args.force or not output_file.exists() else {
        r["image_id"]: r for r in json.loads(output_file.read_text(encoding="utf-8"))}
    if args.route == "traditional":
        from catvision.traditional import TraditionalVision, visualization
        model = TraditionalVision()
    else:
        from catvision.vlm import LocalVLM, PROMPT, PROMPT_VERSION
        model = LocalVLM()
        save_json(directory / "prompt.json", {"version": PROMPT_VERSION, "text": PROMPT,
                                               "do_sample": False, "max_new_tokens": 192})
    save_json(directory / "run-config.json", {"route": args.route, "set": args.set,
        "image_ids": [r["image_id"] for r in records], "count": len(records),
        "load_seconds": model.load_seconds, "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "inference_seconds includes preprocessing, forward/generation, postprocessing; excludes weight loading"})
    started = time.perf_counter()
    for index, record in enumerate(records):
        key = record["image_id"]
        if key in existing:
            continue
        try:
            image = load_image(args.data_root / record["relative_path"])
            result = model.predict(image)
            if args.route == "traditional":
                visualization(image, result).save(directory / f"{key}.jpg", quality=90)
            result.pop("mask", None)
            result.pop("crop", None)
        except Exception as exc:
            result = {"status": "error", "breed": None, "error": f"{type(exc).__name__}: {exc}"}
            result.update(image_id=key, input_sha256=record["sha256"], true_breed=record["breed"])
            existing[key] = result
            save_json(output_file, list(existing.values()))
            model.close()
            raise  # Full traceback is logged by ModelTask; do not continue or restart.
        result.update(image_id=key, input_sha256=record["sha256"], true_breed=record["breed"],
                      created_utc=datetime.now(timezone.utc).isoformat())
        existing[key] = result
        save_json(output_file, list(existing.values()))
        print(f"PREDICT {index+1}/{len(records)} {key} {result['status']} {result.get('breed')} "
              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    model.close()
    requested = {r["image_id"] for r in records}
    metrics = score_predictions(records, [p for key, p in existing.items() if key in requested])
    save_json(directory / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with ModelTask("evaluation"):
        main()
