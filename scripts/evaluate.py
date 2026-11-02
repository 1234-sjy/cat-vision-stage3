from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.evaluation import (score_predictions, read_verified_input,
                                 inference_fingerprint, prediction_cache_matches)
from catvision.run_control import ModelTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, choices=["traditional", "vlm"])
    parser.add_argument("--set", default="comparison", choices=["comparison", "comparison-v2", "test", "val"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Audit compatible cached results without loading models")
    args = parser.parse_args()
    if args.set == "comparison":
        records = json.loads((ROOT / "artifacts/data/comparison.json").read_text(encoding="utf-8"))
    elif args.set == 'comparison-v2':
        records = json.loads((ROOT/'artifacts/optimization-v2/comparison-v2.json').read_text(encoding='utf-8'))
    else:
        records = [r for r in json.loads((ROOT / "artifacts/data/manifest.json").read_text(encoding="utf-8"))
                   if r["split"] == args.set]
    if args.limit:
        records = records[:args.limit]
    directory = ROOT / "artifacts/evaluation" / args.set / args.route
    directory.mkdir(parents=True, exist_ok=True)
    output_file = directory / "predictions.json"
    if output_file.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        history = directory / "history" / stamp
        history.mkdir(parents=True)
        for filename in ["predictions.json", "run-config.json", "prompt.json", "metrics.json"]:
            path = directory / filename
            if path.exists():
                shutil.copy2(path, history / filename)
    existing = {} if args.force or not output_file.exists() else {
        r["image_id"]: r for r in json.loads(output_file.read_text(encoding="utf-8"))}
    fingerprint = inference_fingerprint(ROOT, args.route)
    model = None
    run_config = {"route": args.route, "set": args.set,
        "image_ids": [r["image_id"] for r in records], "count": len(records),
        "load_seconds": 0, "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": fingerprint, "fresh_count": 0, "reused_count": 0,
        "timing": "original inference_seconds retained for reused results; excludes weight loading and requested pauses"}
    save_json(directory / "run-config.json", run_config)
    started = time.perf_counter()
    for index, record in enumerate(records):
        key = record["image_id"]
        actual_sha = None
        inference_attempted = False
        try:
            content, actual_sha = read_verified_input(record, args.data_root)
            if key in existing and prediction_cache_matches(existing[key], key, actual_sha, fingerprint):
                existing[key].update(prediction_origin="reused", fresh_run=False,
                                     last_reused_utc=datetime.now(timezone.utc).isoformat())
                run_config["reused_count"] += 1
                save_json(output_file, list(existing.values()))
                print(f"REUSE {index+1}/{len(records)} {key}; measured input and model identity match", flush=True)
                continue
            if args.cache_only:
                raise RuntimeError(f"Compatible cache unavailable for {key}; --cache-only refused model execution")
            if model is None:
                if args.route == "traditional":
                    from catvision.traditional import TraditionalVision, visualization
                    model = TraditionalVision()
                else:
                    from catvision.vlm import LocalVLM, PROMPT, PROMPT_VERSION
                    model = LocalVLM()
                    save_json(directory / "prompt.json", {"version": PROMPT_VERSION, "text": PROMPT,
                                                           "do_sample": False, "max_new_tokens": 192})
                run_config["load_seconds"] = model.load_seconds
            image = load_image(content)
            inference_attempted = True
            result = model.predict(image)
            if args.route == "traditional":
                visualization(image, result).save(directory / f"{key}.jpg", quality=90)
            result.pop("mask", None)
            result.pop("crop", None)
        except Exception as exc:
            result = {"status": "error", "breed": None, "error": f"{type(exc).__name__}: {exc}"}
            result.update(image_id=key, input_sha256=getattr(exc, "actual_sha256", actual_sha),
                          expected_input_sha256=record["sha256"], true_breed=record["breed"],
                          run_fingerprint=fingerprint,
                          prediction_origin="fresh_failure" if inference_attempted else "input_or_cache_failure",
                          fresh_run=inference_attempted)
            existing[key] = result
            save_json(output_file, list(existing.values()))
            if model is not None:
                model.close()
            raise  # Full traceback is logged by ModelTask; do not continue or restart.
        result.update(image_id=key, input_sha256=actual_sha, true_breed=record["breed"],
                      run_fingerprint=fingerprint, prediction_origin="fresh", fresh_run=True,
                      created_utc=datetime.now(timezone.utc).isoformat())
        run_config["fresh_count"] += 1
        existing[key] = result
        save_json(output_file, list(existing.values()))
        print(f"PREDICT {index+1}/{len(records)} {key} {result['status']} {result.get('breed')} "
              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if model is not None:
        model.close()
    run_config["model_loaded"] = model is not None
    save_json(directory / "run-config.json", run_config)
    requested = {r["image_id"] for r in records}
    metrics = score_predictions(records, [p for key, p in existing.items() if key in requested])
    save_json(directory / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with ModelTask("evaluation"):
        main()
