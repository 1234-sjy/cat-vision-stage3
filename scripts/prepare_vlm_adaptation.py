"""Freeze source-label VLM adaptation data and a historically unexposed test subset.

This is CPU-only and never imports a model, reads a classifier answer, changes a
v3 split, or selects examples by prediction correctness. Existing outputs are
refused. Historical model outputs contribute identity metadata only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261103


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def split_source_labels(records: list[dict], split: str) -> list[dict]:
    return [dict(row, target_text=row["breed"] if row["is_cat"] else "no_cat")
            for row in records if row["split"] == split]


def validation_selection(records: list[dict], classes: list[dict]) -> list[dict]:
    if any(row["split"] != "val" for row in records):
        raise ValueError("Validation selection may only receive fixed validation rows")
    rng = random.Random(SEED)
    selected = []
    for item in classes:
        candidates = sorted((row for row in records if row["is_cat"] and row["breed"] == item["breed"]),
                            key=lambda row: row["image_id"])
        if len(candidates) < 10:
            raise ValueError(f"Too few validation cats for {item['breed']}")
        rng.shuffle(candidates)
        selected.extend(candidates[:10])
    dogs = sorted((row for row in records if not row["is_cat"]), key=lambda row: row["image_id"])
    if len(dogs) < 50:
        raise ValueError("Too few validation dogs")
    rng.shuffle(dogs)
    return selected + dogs[:50]


class IdentityGraph:
    def __init__(self):
        self.parents = {}
        self.sizes = {}
        self.sha_index = defaultdict(set)
        self.records = {}
        self.edge_kinds = Counter()

    def add(self, image_id: str) -> None:
        if image_id not in self.parents:
            self.parents[image_id] = image_id
            self.sizes[image_id] = 1

    def find(self, image_id: str) -> str:
        while self.parents[image_id] != image_id:
            self.parents[image_id] = self.parents[self.parents[image_id]]
            image_id = self.parents[image_id]
        return image_id

    def union(self, left: str, right: str, kind: str) -> None:
        self.edge_kinds[kind] += 1
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parents[right] = left
        self.sizes[left] += self.sizes[right]


def build_identity_graph(records: list[dict], edges: list[dict]) -> IdentityGraph:
    """Reconstruct v3 components, including removed and reference-only images."""
    graph = IdentityGraph()
    values = {"sha256": {}, "animal_key": {}, "group": {}}
    for row in records:
        image_id = row["image_id"]
        graph.add(image_id)
        graph.records.setdefault(image_id, row)
        if row.get("sha256"):
            graph.sha_index[row["sha256"]].add(image_id)
        for key, seen in values.items():
            value = row.get(key)
            if not value:
                continue
            if value in seen:
                graph.union(image_id, seen[value], key)
            else:
                seen[value] = image_id
    for edge in edges:
        if edge["left"] not in graph.parents or edge["right"] not in graph.parents:
            raise ValueError(f"Unresolved frozen identity edge: {edge}")
        graph.union(edge["left"], edge["right"], edge["kind"])
    return graph


def is_vlm_result(value) -> bool:
    if not isinstance(value, dict):
        return False
    name = str(value.get("route", "")) + " " + str(value.get("model", ""))
    return "qwen" in name.lower() or "vlm" in name.lower()


def exposure_records(value, source_path: str) -> list[dict]:
    """Read identities only; prediction labels, scores, and correctness are ignored."""
    explicit_predictions = "vlm" in source_path.lower() and "prediction" in Path(source_path).name.lower()
    result = []

    def visit(node):
        if isinstance(node, dict):
            if node.get("image_id") and (explicit_predictions or is_vlm_result(node)
                                         or is_vlm_result(node.get("result"))
                                         or is_vlm_result(node.get("vlm"))):
                result.append({"image_id": node["image_id"],
                               "sha256": node.get("input_sha256") or node.get("sha256"),
                               "source_path": source_path})
            for child in node.values():
                if isinstance(child, (list, dict)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return result


def historical_exposure(project: Path, out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Scan every artifact JSON, including generic paths and profiling histories."""
    observations, inventory = [], []
    for path in sorted((project / "artifacts").rglob("*.json")):
        if path.is_relative_to(out_dir):
            continue
        relative = path.relative_to(project).as_posix()
        rows = exposure_records(read_json(path), relative)
        if rows:
            observations.extend(rows)
            inventory.append({"path": relative, "sha256": sha256(path), "observation_count": len(rows),
                              "unique_image_count": len({row["image_id"] for row in rows})})
    if not observations:
        raise ValueError("No historical VLM observations found; cannot certify an unseen subset")
    return observations, inventory


def select_fresh_test(records: list[dict], graph: IdentityGraph, observations: list[dict]) -> tuple[list[dict], list[dict]]:
    if any(row["split"] != "test" for row in records):
        raise ValueError("Fresh final test must be a subset of the existing fixed test split")
    exposure_by_component = defaultdict(list)
    unresolved = []
    historical_ids = {row["image_id"] for row in observations}
    historical_shas = {row["sha256"] for row in observations if row.get("sha256")}
    for observation in observations:
        candidates = set()
        if observation["image_id"] in graph.parents:
            candidates.add(observation["image_id"])
        candidates.update(graph.sha_index.get(observation.get("sha256"), ()))
        if not candidates:
            unresolved.append(observation)
        for component in {graph.find(image_id) for image_id in candidates}:
            exposure_by_component[component].append(observation)
    if unresolved:
        raise ValueError(f"Historical VLM inputs lack a resolved identity; cannot certify freshness: {unresolved}")
    kept, removed = [], []
    for row in records:
        linked = exposure_by_component.get(graph.find(row["image_id"]))
        if not linked:
            kept.append(dict(row))
            continue
        reasons = ["linked_frozen_identity_component"]
        if row["image_id"] in historical_ids:
            reasons.append("direct_historical_image_id")
        if row["sha256"] in historical_shas:
            reasons.append("exact_historical_input_sha256")
        removed.append({"image_id": row["image_id"], "relative_path": row["relative_path"],
                        "group": row["group"], "sha256": row["sha256"], "breed": row["breed"],
                        "original_split": "test", "reasons": reasons,
                        "historical_image_ids": sorted({item["image_id"] for item in linked}),
                        "historical_source_paths": sorted({item["source_path"] for item in linked})})
    return kept, removed


def describe_manifest(records: list[dict], classes: list[dict]) -> dict:
    cats = Counter(row["breed"] for row in records if row["is_cat"])
    return {"count": len(records), "cat_count": sum(cats.values()),
            "dog_count": sum(not row["is_cat"] for row in records),
            "per_class": {item["breed"]: cats[item["breed"]] for item in classes}}


def prepare(project: Path, source_dir: Path, out_dir: Path, data_root: Path) -> dict:
    output_names = ["train.json", "val-full.json", "val-selection.json", "fresh-test.json",
                    "shared-regression.json", "classes.json", "historical-exposure.json",
                    "excluded-previous-test.json", "freeze.json"]
    existing = [str(out_dir / name) for name in output_names if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Preserve frozen outputs; preparation refused: {existing}")
    source_names = ["train-manifest.json", "test-manifest.json", "classes.json", "decoded-candidates.json",
                    "identity-references.json", "duplicate-edges.json", "photographic-sessions.json", "freeze.json"]
    source_files = [{"path": (source_dir / name).relative_to(project).as_posix(),
                     "sha256": sha256(source_dir / name)} for name in source_names]
    source_hashes = {item["path"]: item["sha256"] for item in source_files}
    source = read_json(source_dir / "train-manifest.json")
    fixed_test = read_json(source_dir / "test-manifest.json")
    classes = read_json(source_dir / "classes.json")
    if len(classes) != 20 or [row["class_index"] for row in classes] != list(range(20)):
        raise ValueError("Require the unchanged ordered 20-class source mapping")
    if {row["split"] for row in source} != {"train", "val"}:
        raise ValueError("Unexpected source training/validation splits")
    train = split_source_labels(source, "train")
    val = split_source_labels(source, "val")
    val_selection = validation_selection(val, classes)
    partitions = {"train": train, "val": val, "test": fixed_test}
    for key in ("image_id", "sha256", "group"):
        seen = set()
        for name, rows in partitions.items():
            current = {row[key] for row in rows}
            if seen & current:
                raise ValueError(f"Existing fixed splits overlap by {key}: {name}")
            seen.update(current)
    known = read_json(source_dir / "decoded-candidates.json") + read_json(source_dir / "identity-references.json")
    # Current manifest groups are additionally unioned; no original split is reassigned.
    known += source + fixed_test
    edges = read_json(source_dir / "duplicate-edges.json")
    edges += read_json(source_dir / "photographic-sessions.json")["edges"]
    graph = build_identity_graph(known, edges)
    observations, historical_inventory = historical_exposure(project, out_dir)
    fresh, excluded = select_fresh_test(fixed_test, graph, observations)
    if {row["breed"] for row in fresh} != {item["breed"] for item in classes}:
        raise ValueError("Fresh test no longer covers all 20 classes")
    shared_path = project / "artifacts/optimization-v3/shared-test/manifest.json"
    shared = read_json(shared_path)
    fixed_ids = {row["image_id"] for row in fixed_test}
    if len(shared) != 120 or not {row["image_id"] for row in shared} <= fixed_ids:
        raise ValueError("Expected the existing 120-image shared regression subset")
    shared = [dict(row, evaluation_purpose="previously_evaluated_regression_not_blind_test") for row in shared]
    source_files.append({"path": shared_path.relative_to(project).as_posix(), "sha256": sha256(shared_path)})
    manifests = {"train.json": train, "val-full.json": val, "val-selection.json": val_selection,
                 "fresh-test.json": fresh, "shared-regression.json": shared}
    exposure_audit = {
        "scope": "All project artifact JSONs; VLM prediction paths, VLM/Qwen result markers, and nested profiling runs",
        "selection_uses": ["image_id", "input_sha256", "frozen identity relationships"],
        "prediction_correctness_used": False, "files": historical_inventory,
        "observation_count": len(observations), "unique_image_count": len({row["image_id"] for row in observations}),
        "observations": sorted(observations, key=lambda row: (row["image_id"], row["source_path"])),
        "identity_graph": {"nodes": len(graph.parents), "edge_kinds": dict(graph.edge_kinds),
                           "components": len({graph.find(image_id) for image_id in graph.parents}),
                           "unresolved_historical_inputs": 0,
                           "phash_policy": "All frozen v3 pHash edges at Hamming distance <=6, SHA256, animal keys, photographic-session edges, and current manifest groups are unioned transitively"}}
    # Check source hashes immediately before publishing a new, exclusive freeze.
    for relative, expected in source_hashes.items():
        if sha256(project / relative) != expected:
            raise ValueError(f"Source changed during preparation: {relative}")
    for name, rows in manifests.items():
        write_new_json(out_dir / name, rows)
    with (out_dir / "classes.json").open("xb") as stream:
        stream.write((source_dir / "classes.json").read_bytes())
    write_new_json(out_dir / "historical-exposure.json", exposure_audit)
    write_new_json(out_dir / "excluded-previous-test.json", excluded)
    manifest_metadata = {name: dict(describe_manifest(rows, classes), sha256=sha256(out_dir / name))
                         for name, rows in manifests.items()}
    manifest_metadata["classes.json"] = {"count": len(classes), "sha256": sha256(out_dir / "classes.json")}
    freeze = {"schema_version": "vlm-source-label-adaptation-v4", "created_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(data_root.resolve()), "seed": SEED,
              "preparation_script": {"path": Path(__file__).resolve().relative_to(project).as_posix(),
                                     "sha256": sha256(Path(__file__))},
              "source_files": source_files, "historical_vlm_files": historical_inventory,
              "manifests": manifest_metadata,
              "files": {name: sha256(out_dir / name) for name in output_names if name != "freeze.json"},
              "fresh_test_audit": {"original_test_count": len(fixed_test), "excluded_count": len(excluded),
                                   "fresh_test_count": len(fresh), "unresolved_historical_inputs": 0,
                                   "excluded_per_class": dict(Counter(row["breed"] for row in excluded))},
              "policy": {"targets": "Original source breed for cats; no_cat for dogs",
                         "teacher_predictions_used": False, "test_labels_used_for_training_or_selection": False,
                         "existing_splits_unchanged": True, "train_split": "train only",
                         "validation_selection": "10 cats per class plus 50 dogs from fixed validation, seeded sampling",
                         "fresh_test_selection": "Remove all previously exposed VLM identity components, without using prediction outcomes",
                         "shared_regression": "Previously evaluated 120 images, never a new blind test",
                         "manifest_schema": "Plain JSON lists; source fields retained; train/val have target_text",
                         "historical_scope_limit": "Freshness is certified against retained local VLM inference artifacts and frozen identity graph; upstream pretrained-model exposure is unknowable"}}
    write_new_json(out_dir / "freeze.json", freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/optimization-v3/data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/optimization-v4/data")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.project.resolve(), args.source_dir.resolve(), args.out_dir.resolve(), args.data_root.resolve())
    print(json.dumps({"manifests": result["manifests"], "fresh_test_audit": result["fresh_test_audit"]}, indent=2))


if __name__ == "__main__":
    main()
