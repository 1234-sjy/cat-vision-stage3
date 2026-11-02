"""CPU-only, source-aware dataset preparation with group and pHash isolation.

Oxford splits are inherited. Kaggle Refined v2 supplies the additional classes.
The older, noisier ma7555 copy is an identity reference only, never training data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .common import ROOT, save_json

SEED = 20261102
PHASH_DISTANCE = 6
MIN_EDGE = 96
CLASS_NAMES = [
    ("Abyssinian", "阿比西尼亚猫"), ("Bengal", "孟加拉猫"),
    ("Birman", "伯曼猫"), ("Bombay", "孟买猫"),
    ("British_Shorthair", "英国短毛猫"), ("Egyptian_Mau", "埃及猫"),
    ("Maine_Coon", "缅因猫"), ("Persian", "波斯猫"),
    ("Ragdoll", "布偶猫"), ("Russian_Blue", "俄罗斯蓝猫"),
    ("Siamese", "暹罗猫"), ("Sphynx", "斯芬克斯猫"),
    ("American_Bobtail", "美国短尾猫"), ("American_Curl", "美国卷耳猫"),
    ("American_Shorthair", "美国短毛猫"), ("Exotic_Shorthair", "异国短毛猫"),
    ("Manx", "曼岛猫"), ("Norwegian_Forest", "挪威森林猫"),
    ("Scottish_Fold", "苏格兰折耳猫"), ("Turkish_Angora", "土耳其安哥拉猫"),
]
CLASSES = dict(CLASS_NAMES)
CLASS_INDEX = {name: index for index, (name, _) in enumerate(CLASS_NAMES)}
_FREQ = np.arange(32, dtype=np.float64)[:, None]
_POS = np.arange(32, dtype=np.float64)[None, :] + 0.5
_DCT = np.cos(np.pi * _FREQ * _POS / 32) * np.sqrt(2 / 32)
_DCT[0] /= np.sqrt(2)


def perceptual_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    values = (_DCT @ np.asarray(gray, dtype=np.float64) @ _DCT.T)[:8, :8].ravel()
    bits = values > np.median(values[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big")


def inspect_image(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.load()
        return {"sha256": digest, "width": image.width, "height": image.height,
                "gray_std": round(float(np.asarray(image.convert("L"), dtype=np.float32).std()), 6),
                "phash": f"{perceptual_hash(image):016x}"}


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


class HashIndex:
    """Exact Hamming metric BK-tree. All radius matches are returned."""
    def __init__(self):
        self.nodes: list[tuple[int, int, dict[int, int]]] = []

    def add(self, value: int, row: int) -> None:
        if not self.nodes:
            self.nodes.append((value, row, {}))
            return
        position = 0
        while True:
            previous, _, children = self.nodes[position]
            distance = (value ^ previous).bit_count()
            if distance in children:
                position = children[distance]
            else:
                children[distance] = len(self.nodes)
                self.nodes.append((value, row, {}))
                return

    def matches(self, value: int, radius: int) -> list[tuple[int, int]]:
        result, stack = [], [0] if self.nodes else []
        while stack:
            position = stack.pop()
            previous, row, children = self.nodes[position]
            distance = (value ^ previous).bit_count()
            if distance <= radius:
                result.append((row, distance))
            stack.extend(child for edge, child in children.items()
                         if distance - radius <= edge <= distance + radius)
        return result


def original_animal_key(source_name: str) -> str | None:
    """Use the original Petfinder animal id, not the added ZIP-offset prefix."""
    match = re.match(r"(\d+)_", Path(source_name).name)
    return f"ma7555:{match.group(1)}" if match else None


def extract_archive(data_root: Path, out: Path) -> list[dict]:
    archive = data_root / "expanded-v3-catbreedsrefined-7k.zip"
    destination = data_root / "expanded-v3"
    inventory = []
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir():
                continue
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read(entry)  # zipfile verifies each image's CRC32.
            if target.exists():
                if target.read_bytes() != raw:
                    raise FileExistsError(f"Existing extracted file differs: {target}")
            else:
                target.write_bytes(raw)
            label = Path(entry.filename).parent.name
            breed = label.replace(" ", "_")
            inventory.append({"image_id": "refined_" + Path(entry.filename).stem,
                              "relative_path": target.relative_to(data_root).as_posix(),
                              "breed": breed, "source_label": label,
                              "source": "kaggle_doctrinek_catbreedsrefined_v2",
                              "original_relative_path": entry.filename,
                              "crc32": f"{entry.CRC:08x}", "archive_bytes": entry.file_size,
                              "is_cat": True, "fixed_split": None,
                              "identity_origin": "renamed_by_refined_publisher"})
    save_json(out / "archive-inventory.json", inventory)
    return inventory


def identity_references(data_root: Path, out: Path) -> list[dict]:
    workspace = data_root.parent.parent
    source_index = workspace / "work/cat-extra/selected-images/source-index.csv"
    cleaning = workspace / "outputs/cat-breed-recognition/artifacts/data-expanded/extra_cleaning_log.csv"
    original_names = {}
    with source_index.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            original_names[row["relative_path"]] = row["source_name"]
    records = []
    missing_identity = 0
    with cleaning.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["action"] != "added":
                continue
            relative = f"extra-cats/{row['breed']}/{row['sha256']}{Path(row['source_relative_path']).suffix.lower()}"
            path = data_root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("rb") as raw:
                digest = hashlib.file_digest(raw, "sha256").hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Changed identity reference: {path}")
            source_name = original_names.get(row["source_relative_path"])
            animal = original_animal_key(source_name) if source_name else None
            missing_identity += animal is None
            records.append({"image_id": "identity_" + digest, "relative_path": relative,
                            "breed": row["breed"], "source": "ma7555_v2_identity_reference_only",
                            "sha256": digest, "phash": row["phash"],
                            "width": int(row["width"]), "height": int(row["height"]),
                            "gray_std": float(row["gray_std"]),
                            "animal_key": animal, "original_relative_path": source_name,
                            "reference_only": True,
                            "inspection": "historical_verified_decode_current_sha256_rechecked"})
    save_json(out / "identity-references.json", records)
    print(f"identity_references={len(records)} missing_original_animal_id={missing_identity}", flush=True)
    return records


def fixed_oxford() -> list[dict]:
    original = json.loads((ROOT / "artifacts/data/manifest.json").read_text(encoding="utf-8"))
    v2 = json.loads((ROOT / "artifacts/optimization-v2/train-manifest.json").read_text(encoding="utf-8"))
    records = []
    for raw in original + [r for r in v2 if not r["is_cat"]]:
        row = dict(raw)
        row.update(source="oxford_iiit_pet_official", fixed_split=raw["split"],
                   is_cat=raw.get("is_cat", True),
                   relative_path="oxford-iiit-pet/" + raw["relative_path"],
                   identity_origin="official_image_id")
        records.append(row)
    return records


def connect(records: list[dict], out: Path) -> tuple[UnionFind, UnionFind, list[dict]]:
    duplicates, groups = UnionFind(len(records)), UnionFind(len(records))
    sha_seen, animal_seen, index = {}, {}, HashIndex()
    edges = []
    for number, row in enumerate(records):
        digest = row["sha256"]
        if digest in sha_seen:
            previous = sha_seen[digest]
            duplicates.union(number, previous)
            groups.union(number, previous)
            edges.append({"left": records[previous]["image_id"], "right": row["image_id"], "kind": "sha256"})
        else:
            sha_seen[digest] = number
        animal = row.get("animal_key")
        if animal:
            if animal in animal_seen:
                groups.union(number, animal_seen[animal])
            else:
                animal_seen[animal] = number
        value = int(row["phash"], 16)
        matches = index.matches(value, PHASH_DISTANCE)
        for previous, distance in matches:
            duplicates.union(number, previous)
            groups.union(number, previous)
            edges.append({"left": records[previous]["image_id"], "right": row["image_id"],
                          "kind": "phash", "distance": distance})
        index.add(value, number)
        if (number + 1) % 2000 == 0:
            print(f"phash_audit={number+1}/{len(records)} duplicate_edges={len(edges)}", flush=True)
    save_json(out / "duplicate-edges.json", edges)
    return duplicates, groups, edges


def select_and_split(records: list[dict], duplicates: UnionFind, groups: UnionFind) -> tuple[list[dict], list[dict]]:
    """Deduplicate first; retain fixed Oxford assignment and keep all group views together."""
    active = [i for i, row in enumerate(records) if not row.get("reference_only")]
    by_group = defaultdict(list)
    for index in active:
        by_group[groups.find(index)].append(index)
    removed, selected = [], []
    split_rank = {"test": 0, "val": 1, "train": 2, None: 3}
    group_names = {}
    for root, members in by_group.items():
        members = sorted(members, key=lambda i: (split_rank[records[i].get("fixed_split")], records[i]["image_id"]))
        group_name = "group_" + hashlib.sha256("\n".join(sorted(records[i]["sha256"] for i in members)).encode()).hexdigest()[:20]
        group_names[root] = group_name
        official_labels = {records[i]["breed"] for i in members if records[i].get("fixed_split")}
        all_labels = {records[i]["breed"] for i in members}
        fixed = next((records[i]["fixed_split"] for i in members if records[i].get("fixed_split")), None)
        seen_duplicate = {}
        for i in members:
            row = records[i]
            reason = None
            if len(official_labels) > 1:
                reason = "conflicting_official_labels_in_group"
            elif official_labels and row["breed"] not in official_labels:
                reason = "label_conflicts_with_official_group"
            elif not official_labels and len(all_labels) > 1:
                reason = "conflicting_refined_labels_in_group"
            elif row.get("fixed_split") and row["fixed_split"] != fixed:
                reason = "official_cross_split_duplicate_or_identity_group"
            elif duplicates.find(i) in seen_duplicate:
                reason = "duplicate_or_near_duplicate"
            if reason:
                removed.append({"image_id": row["image_id"], "reason": reason,
                                "group": group_name, "source": row["source"],
                                "original_split": row.get("fixed_split"),
                                "matched_kept": seen_duplicate.get(duplicates.find(i))})
                continue
            seen_duplicate[duplicates.find(i)] = row["image_id"]
            new = dict(row, group=group_name, split=fixed)
            new["breed_zh"] = CLASSES.get(row["breed"], "非猫：犬类")
            new["class_index"] = CLASS_INDEX.get(row["breed"], -1)
            new["evaluation_scope"] = ("oxford_test" if fixed == "test" and row["source"] == "oxford_iiit_pet_official"
                                       else "expanded_test" if fixed == "test" else "training_validation")
            selected.append(new)
    # A source-animal component is assigned once, before model training.
    unassigned = defaultdict(lambda: defaultdict(list))
    for row in selected:
        if row["split"] is None:
            unassigned[row["breed"]][row["group"]].append(row)
    rng = random.Random(SEED)
    for breed in CLASS_INDEX:
        options = sorted(unassigned[breed].items())
        rng.shuffle(options)
        total = sum(len(rows) for _, rows in options)
        targets = {"test": round(total * 0.15), "val": round(total * 0.15)}
        counts = Counter()
        for _, rows in options:
            split = next((s for s in ("test", "val") if counts[s] < targets[s]), "train")
            for row in rows:
                row["split"] = split
                row["evaluation_scope"] = "expanded_test" if split == "test" else "training_validation"
            counts[split] += len(rows)
    selected.sort(key=lambda r: (r["class_index"], r["source"], r["image_id"]))
    return selected, removed


def audit_manifest(records: list[dict]) -> dict:
    splits_by_sha, splits_by_group = defaultdict(set), defaultdict(set)
    for row in records:
        splits_by_sha[row["sha256"]].add(row["split"])
        splits_by_group[row["group"]].add(row["split"])
        if min(row["width"], row["height"]) < MIN_EDGE:
            raise ValueError(f"Too small: {row['image_id']}")
    cross_sha = sum(len(splits) > 1 for splits in splits_by_sha.values())
    cross_group = sum(len(splits) > 1 for splits in splits_by_group.values())
    if cross_sha or cross_group:
        raise ValueError(f"Cross-split leakage: sha={cross_sha}, group={cross_group}")
    return {"cross_split_sha256": cross_sha, "cross_split_groups": cross_group,
            "unique_sha256": len(splits_by_sha), "unique_groups": len(splits_by_group),
            "minimum_image_edge": min(min(r["width"], r["height"]) for r in records)}


def contact_sheets(records: list[dict], data_root: Path, out: Path) -> None:
    for breed, _ in CLASS_NAMES[12:]:
        rows = [r for r in records if r["breed"] == breed]
        # Review a deterministic spread across the source order; never select by model score.
        options = sorted(rows, key=lambda r: r["image_id"])
        sample = random.Random(SEED).sample(options, min(20, len(options)))
        canvas = Image.new("RGB", (5 * 176, 4 * 156 + 35), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"{breed} | deterministic source review", fill="black")
        for position, row in enumerate(sample):
            with Image.open(data_root / row["relative_path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((168, 128))
            x, y = (position % 5) * 176 + 4, (position // 5) * 156 + 35
            canvas.paste(image, (x, y))
            draw.text((x, y + 130), row["image_id"].removeprefix("refined_")[:26], fill="black")
        destination = out / "contact-sheets" / f"{breed}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, quality=90)


def prepare(data_root: Path, out: Path) -> dict:
    if (out / "manifest.json").exists():
        raise FileExistsError("Frozen manifest exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    raw = fixed_oxford() + extract_archive(data_root, out)
    accepted, exclusions = [], []
    for number, row in enumerate(raw, 1):
        if row["is_cat"] and row["breed"] not in CLASSES:
            exclusions.append({"image_id": row["image_id"], "reason": "non_breed_or_unsupported_label"})
            continue
        try:
            row.update(inspect_image(data_root / row["relative_path"]))
        except (OSError, ValueError, SyntaxError) as error:
            exclusions.append({"image_id": row["image_id"], "reason": "decode_error", "error": str(error)})
            continue
        if min(row["width"], row["height"]) < MIN_EDGE or row["gray_std"] < 5:
            exclusions.append({"image_id": row["image_id"], "reason": "too_small_or_low_variation",
                               "width": row["width"], "height": row["height"], "gray_std": row["gray_std"]})
            continue
        accepted.append(row)
        if number % 500 == 0:
            print(f"decoded={number}/{len(raw)} accepted={len(accepted)}", flush=True)
    save_json(out / "decoded-candidates.json", accepted)
    references = identity_references(data_root, out)
    candidates = accepted + references
    duplicates, groups, edges = connect(candidates, out)
    selected, removed = select_and_split(candidates, duplicates, groups)
    exclusions.extend(removed)
    audit = audit_manifest(selected)
    active_by_id = {r["image_id"]: r for r in selected}
    cross_phash = [edge for edge in edges if edge["kind"] == "phash"
                   and edge["left"] in active_by_id and edge["right"] in active_by_id
                   and active_by_id[edge["left"]]["split"] != active_by_id[edge["right"]]["split"]]
    if cross_phash:
        raise ValueError("Cross-split pHash duplicate remains")
    audit["cross_split_phash_pairs"] = len(cross_phash)
    classes = [{"class_index": i, "breed": breed, "breed_zh": zh} for i, (breed, zh) in enumerate(CLASS_NAMES)]
    save_json(out / "classes.json", classes)
    save_json(out / "excluded.json", exclusions)
    save_json(out / "manifest.json", selected)
    save_json(out / "train-manifest.json", [r for r in selected if r["split"] in {"train", "val"}])
    save_json(out / "test-manifest.json", [r for r in selected if r["split"] == "test"])
    contact_sheets(selected, data_root, out)
    source_counts = {source: dict(Counter(r["split"] for r in selected if r["source"] == source))
                     for source in sorted({r["source"] for r in selected})}
    summary = {
        "created_utc": created, "frozen_before_training": True, "seed": SEED,
        "data_root": "work/cat-data", "relative_path_base": "--data-root",
        "class_count": len(classes), "raw_refined_class_counts": dict(Counter(r["breed"] for r in raw if r["source"].startswith("kaggle"))),
        "counts": dict(Counter(r["split"] for r in selected)),
        "cat_counts": dict(Counter(r["split"] for r in selected if r["is_cat"])),
        "dog_counts": dict(Counter(r["split"] for r in selected if not r["is_cat"])),
        "class_counts": {b: dict(Counter(r["split"] for r in selected if r["breed"] == b)) for b in CLASSES},
        "source_counts": source_counts,
        "evaluation_counts": dict(Counter(r["evaluation_scope"] for r in selected if r["split"] == "test")),
        "excluded_counts": dict(Counter(r["reason"] for r in exclusions)),
        "old_extra_identity_references": len(references), "audit": audit,
        "identity_policy": "All known original ma7555 animal ids and pHash/SHA256 connected components remain in one split. Identity reference images never enter training.",
        "identity_limitation": "The refined publisher renamed and reencoded all images. Original animal ids can only be recovered through near-duplicate links to local ma7555 references. Unmatched refined images lack animal identity; distinct unrecognized views of the same animal may remain.",
        "split_policy": "Existing Oxford train/val/official test assignments are retained; lower-priority cross-split duplicates are excluded, never moved. Additional independent groups use seed20261102 and approximate70/15/15 train/val/test. Groups linked to Oxford inherit its split.",
        "duplicate_policy": "SHA256 equality or63-bit DCT pHash Hamming distance<=6 forms duplicate components; keep one active representative. Known animal ids union grouping, retaining distinct views only within one split.",
        "quality_policy": "Active images require complete PIL verify/decode, min edge96, gray std>=5; label-conflicting identity groups excluded.20 source labels all correspond to named breeds; no coat-colour classes.",
        "sources": [
            {"name": "Oxford-IIIT Pet", "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/", "license": "CC BY-SA4.0 per Oxford dataset page", "role": "official cats plus existing dog train/val", "provenance_file": "work/cat-data/download_source.json"},
            {"name": "doctrinek/catbreedsrefined-7k", "version": 2, "url": "https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k", "license": "CC BY-SA4.0", "role": "expanded20-class cleaned images", "download_record": "download-attempt.json"},
            {"name": "ma7555/cat-breeds-dataset", "version": 2, "url": "https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset", "role": "identity/duplicate reference only; no direct training samples", "license_note": "Historical local acquisition; source metadata should be retained. No new raw download."},
        ],
        "raw_files_deleted": 0,
    }
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def photographic_session_edges(records: list[dict], data_root: Path, out: Path) -> list[dict]:
    """One conservative, CPU-only round of local-feature photographic-session grouping.

    Neighbours are proposed by colour histogram and pHash within each source label.
    A geometrically consistent ORB match can connect different poses photographed
    in the same setting. Such edges affect grouping, not duplicate removal.
    """
    import cv2
    cv2.setNumThreads(2)
    orb = cv2.ORB_create(nfeatures=500)
    features, histograms, shapes = {}, {}, {}
    cats = [i for i, row in enumerate(records) if row.get("is_cat") and not row.get("reference_only")]
    for count, index in enumerate(cats, 1):
        row = records[index]
        with Image.open(data_root / row["relative_path"]) as raw:
            image = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
        h, w = image.shape[:2]
        scale = min(1, 512 / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        histogram /= max(float(np.linalg.norm(histogram)), 1e-6)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        features[index] = (np.float32([point.pt for point in keypoints]), descriptors)
        histograms[index], shapes[index] = histogram, gray.shape
        if count % 500 == 0:
            print(f"session_features={count}/{len(cats)}", flush=True)
    by_breed = defaultdict(list)
    for index in cats:
        by_breed[records[index]["breed"]].append(index)
    proposed = set()
    for members in by_breed.values():
        matrix = np.stack([histograms[i] for i in members])
        similarities = matrix @ matrix.T
        hashes = [int(records[i]["phash"], 16) for i in members]
        for position, index in enumerate(members):
            nearest_colour = np.argsort(-similarities[position], kind="stable")[:11]
            nearest_hash = sorted(range(len(members)), key=lambda j: ((hashes[position] ^ hashes[j]).bit_count(), j))[:7]
            for candidate in set(nearest_colour.tolist() + nearest_hash):
                other = members[candidate]
                if index != other:
                    proposed.add(tuple(sorted((index, other))))
    # This pair was identified in the deterministic contact sheet, before any training.
    lookup = {row["image_id"]: i for i, row in enumerate(records)}
    manual_pair = tuple(sorted((lookup["refined_Scottish Fold_119"], lookup["refined_Scottish Fold_92"])))
    proposed.add(manual_pair)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    edges = []
    for count, (left, right) in enumerate(sorted(proposed), 1):
        if records[left]["sha256"] == records[right]["sha256"]:
            continue
        if (int(records[left]["phash"], 16) ^ int(records[right]["phash"], 16)).bit_count() <= PHASH_DISTANCE:
            continue
        key_left, descriptor_left = features[left]
        key_right, descriptor_right = features[right]
        if descriptor_left is None or descriptor_right is None or min(len(descriptor_left), len(descriptor_right)) < 16:
            continue
        pairs = matcher.knnMatch(descriptor_left, descriptor_right, k=2)
        good = [first for pair in pairs if len(pair) == 2 for first, second in [pair]
                if first.distance < 0.70 * second.distance]
        if len(good) < 16:
            continue
        points_left = key_left[[match.queryIdx for match in good]]
        points_right = key_right[[match.trainIdx for match in good]]
        _, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, 4.0)
        if mask is None:
            continue
        inliers = mask.ravel().astype(bool)
        count_inliers = int(inliers.sum())
        cover_left = cv2.contourArea(cv2.convexHull(points_left[inliers])) / np.prod(shapes[left]) if count_inliers >= 3 else 0
        cover_right = cv2.contourArea(cv2.convexHull(points_right[inliers])) / np.prod(shapes[right]) if count_inliers >= 3 else 0
        if count_inliers >= 16 and float(inliers.mean()) >= 0.65 and min(cover_left, cover_right) >= 0.05:
            edges.append({"left": records[left]["image_id"], "right": records[right]["image_id"],
                          "kind": "orb_photographic_session", "inliers": count_inliers,
                          "ratio": round(float(inliers.mean()), 5),
                          "coverage_left": round(float(cover_left), 5),
                          "coverage_right": round(float(cover_right), 5),
                          "manual_confirmation": (left, right) == manual_pair})
        if count % 5000 == 0:
            print(f"session_candidates={count}/{len(proposed)} accepted={len(edges)}", flush=True)
    if not any(edge.get("manual_confirmation") for edge in edges):
        edges.append({"left": records[manual_pair[0]]["image_id"], "right": records[manual_pair[1]]["image_id"],
                      "kind": "manual_same_animal", "manual_confirmation": True,
                      "evidence": "Same white/grey folded-ear cat, wood-block pedestal, black studio background, photographer watermark; different poses."})
    save_json(out / "photographic-sessions.json", {"proposal_count": len(proposed), "edges": edges,
              "algorithm": "Same-label top10 HSV-colour and top6 pHash neighbours; ORB500,maxside512,Lowe0.70,homographyRANSAC4px,>=16inliers,ratio>=0.65,hull coverage>=0.05 in each image.",
              "scope": "Single pretraining CPU audit; shared photographic-session edges conservatively stay within one split."})
    return edges


def refine_sessions(data_root: Path, out: Path) -> dict:
    """Preserve the first freeze, then publish one reviewed identity-group revision."""
    history = out / "history/v1-phash-only"
    if history.exists():
        raise FileExistsError("Session revision already attempted; inspect logs before another revision")
    history.mkdir(parents=True)
    preserved = ["manifest.json", "train-manifest.json", "test-manifest.json", "summary.json", "excluded.json"]
    for name in preserved:
        shutil.copy2(out / name, history / name)
    shutil.copytree(out / "contact-sheets", history / "contact-sheets")
    old_summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    records = json.loads((out / "decoded-candidates.json").read_text(encoding="utf-8"))
    records += json.loads((out / "identity-references.json").read_text(encoding="utf-8"))
    ids = {row["image_id"]: i for i, row in enumerate(records)}
    duplicates, groups = UnionFind(len(records)), UnionFind(len(records))
    animals = {}
    for index, row in enumerate(records):
        animal = row.get("animal_key")
        if animal:
            if animal in animals:
                groups.union(index, animals[animal])
            else:
                animals[animal] = index
    old_edges = json.loads((out / "duplicate-edges.json").read_text(encoding="utf-8"))
    for edge in old_edges:
        left, right = ids[edge["left"]], ids[edge["right"]]
        duplicates.union(left, right)
        groups.union(left, right)
    session_edges = photographic_session_edges(records, data_root, out)
    for edge in session_edges:
        groups.union(ids[edge["left"]], ids[edge["right"]])
    selected, removed = select_and_split(records, duplicates, groups)
    audit = audit_manifest(selected)
    selected_by_id = {row["image_id"]: row for row in selected}
    for kind, edges in (("phash", old_edges), ("photographic_session", session_edges)):
        crossed = [edge for edge in edges if edge["left"] in selected_by_id and edge["right"] in selected_by_id
                   and selected_by_id[edge["left"]]["split"] != selected_by_id[edge["right"]]["split"]]
        if crossed:
            raise ValueError(f"Cross-split {kind} edges remain")
        audit[f"cross_split_{kind}_pairs"] = 0
    previous_excluded = json.loads((history / "excluded.json").read_text(encoding="utf-8"))
    group_reasons = {"duplicate_or_near_duplicate", "official_cross_split_duplicate_or_identity_group", "label_conflicts_with_official_group", "conflicting_official_labels_in_group", "conflicting_refined_labels_in_group"}
    excluded = [row for row in previous_excluded if row["reason"] not in group_reasons] + removed
    save_json(out / "manifest.json", selected)
    save_json(out / "train-manifest.json", [row for row in selected if row["split"] in {"train", "val"}])
    save_json(out / "test-manifest.json", [row for row in selected if row["split"] == "test"])
    save_json(out / "excluded.json", excluded)
    summary = dict(old_summary, revision="v2-photographic-session-groups", audit=audit,
                   frozen_utc=datetime.now(timezone.utc).isoformat(), photographic_session_edges=len(session_edges),
                   counts=dict(Counter(r["split"] for r in selected)),
                   cat_counts=dict(Counter(r["split"] for r in selected if r["is_cat"])),
                   dog_counts=dict(Counter(r["split"] for r in selected if not r["is_cat"])),
                   class_counts={b: dict(Counter(r["split"] for r in selected if r["breed"] == b)) for b in CLASSES},
                   source_counts={s: dict(Counter(r["split"] for r in selected if r["source"] == s)) for s in sorted({r["source"] for r in selected})},
                   evaluation_counts=dict(Counter(r["evaluation_scope"] for r in selected if r["split"] == "test")),
                   excluded_counts=dict(Counter(r["reason"] for r in excluded)))
    summary["identity_policy"] += " A single conservative ORB photographic-session round and the documented manual same-animal pair also union groups before splitting."
    summary["visual_review_fact"] = "Deterministic20-image samples per new class include several ordinary-looking adoption cats without the label's distinctive ear/tail traits; the publisher's breed labels are retained as source labels. Scottish Fold_119 and_92 show the same cat in different studio poses and were grouped together."
    save_json(out / "summary.json", summary)
    contact_sheets(selected, data_root, out)
    freeze = {"revision": summary["revision"], "seed": SEED, "frozen_utc": summary["frozen_utc"],
              "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in
                        ["manifest.json", "train-manifest.json", "test-manifest.json", "classes.json"]},
              "previous_revision": "history/v1-phash-only", "training_started": False,
              "manual_pair": ["refined_Scottish Fold_119", "refined_Scottish Fold_92"]}
    save_json(out / "freeze.json", freeze)
    print(json.dumps({"summary": summary, "freeze": freeze}, ensure_ascii=False, indent=2), flush=True)
    return summary
