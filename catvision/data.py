from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from .common import BREEDS, load_image, save_json


def audit_records(records: list[dict]) -> None:
    seen = {}
    for record in records:
        digest = record["sha256"]
        if digest in seen and seen[digest] != record["split"]:
            raise ValueError(f"Cross-split duplicate image: {record['image_id']}")
        seen[digest] = record["split"]


def choose_comparison(records: list[dict], per_class: int = 10, seed: int = 20261101) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if record["split"] == "test":
            groups[record["breed"]].append(record)
    rng = random.Random(seed)
    result = []
    for breed in sorted(groups):
        options = sorted(groups[breed], key=lambda x: x["image_id"])
        if len(options) < per_class:
            raise ValueError(f"Insufficient test images for {breed}")
        result.extend(rng.sample(options, per_class))
    if not result:
        raise ValueError("No test images")
    return result


def prepare_manifest(data_root: Path, out_dir: Path) -> dict:
    candidates = []
    for split, list_name in [("trainval", "trainval.txt"), ("test", "test.txt")]:
        for line in (data_root / "annotations" / list_name).read_text().splitlines():
            fields = line.split()
            if not fields or fields[0].startswith("#") or int(fields[2]) != 1:
                continue
            image_id = fields[0]
            breed = image_id.rsplit("_", 1)[0]
            if breed not in BREEDS:
                continue
            relative_path = f"images/{image_id}.jpg"
            path = data_root / relative_path
            image = load_image(path)
            candidates.append({"image_id": image_id, "relative_path": relative_path,
                               "breed": breed, "breed_zh": BREEDS[breed], "split": split,
                               "width": image.width, "height": image.height,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    # Preserve official test priority; never copy a test duplicate into training.
    unique, removed = {}, []
    for record in sorted(candidates, key=lambda x: (x["split"] != "test", x["image_id"])):
        digest = record["sha256"]
        if digest in unique:
            removed.append({"removed": record["image_id"], "kept": unique[digest]["image_id"]})
        else:
            unique[digest] = record
    records = list(unique.values())
    rng = random.Random(314159)
    for breed in sorted(BREEDS):
        trainval = sorted((r for r in records if r["split"] == "trainval" and r["breed"] == breed),
                          key=lambda x: x["image_id"])
        rng.shuffle(trainval)
        val_count = max(1, round(len(trainval) * 0.2))
        for index, record in enumerate(trainval):
            record["split"] = "val" if index < val_count else "train"
    records.sort(key=lambda x: x["image_id"])
    audit_records(records)
    comparison = choose_comparison(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "manifest.json", records)
    save_json(out_dir / "comparison.json", comparison)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"source": "Oxford-IIIT Pet", "split_seed": 314159, "comparison_seed": 20261101,
               "counts": dict(Counter(r["split"] for r in records)),
               "comparison_count": len(comparison), "removed_duplicates": removed,
               "class_counts": {b: dict(Counter(r["split"] for r in records if r["breed"] == b))
                                for b in BREEDS}}
    save_json(out_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data"))
    arguments = parser.parse_args()
    print(json.dumps(prepare_manifest(arguments.data_root, arguments.out_dir), ensure_ascii=False, indent=2))
