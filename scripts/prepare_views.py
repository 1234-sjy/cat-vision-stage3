from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, load_image, save_json
from catvision.data import make_view_record
from catvision.traditional import TraditionalVision
from catvision.run_control import ModelTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = json.loads((ROOT / "artifacts/data/manifest.json").read_text(encoding="utf-8"))
    # Never generate a training view from test images.
    records = [r for r in records if r["split"] in {"train", "val"}]
    if args.limit:
        records = records[:args.limit]
    vision = TraditionalVision(classifier=False)
    log_path = ROOT / "artifacts/views/records.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if log_path.exists():
        existing = {r["image_id"]: r for r in map(json.loads, log_path.read_text(encoding="utf-8").splitlines())}
    for index, record in enumerate(records):
        directory = ROOT / "data/classification" / record["split"] / record["breed"]
        directory.mkdir(parents=True, exist_ok=True)
        crop_path = directory / (record["image_id"] + "-subject.jpg")
        original_path = directory / (record["image_id"] + "-original.jpg")
        if record["image_id"] in existing and crop_path.exists() and (record["split"] != "train" or original_path.exists()):
            continue
        image = load_image(args.data_root / record["relative_path"])
        result = vision.extract_subject(image)
        crop = result.pop("crop", image)
        result.pop("mask", None)
        crop.save(crop_path, quality=95)
        if record["split"] == "train":
            image.save(original_path, quality=95)
        entry = make_view_record(record, result)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if index % 20 == 0:
            print(f"VIEWS {index+1}/{len(records)} {record['image_id']} {result['status']}", flush=True)
    vision.close()
    save_json(ROOT / "artifacts/views/config.json", {
        "detector": "yolo26n.pt", "detector_conf": 0.15, "segmenter": "sam2.1_t.pt",
        "crop_margin": 0.06, "training_views_per_source": 2, "validation_views_per_source": 1,
        "source_count": len(records), "note": "Views share original source split; test images excluded."})
    print("VIEW PREPARATION FINISHED", flush=True)


if __name__ == "__main__":
    with ModelTask("prepare-views"):
        main()
