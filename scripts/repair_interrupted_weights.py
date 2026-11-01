"""One user-authorized repair of identified blank ranges, without automatic retries."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask
from scripts.download_models import checksum, fetch_range_once


def main():
    base = ROOT / "weights/qwen3-vl-4b"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repair = base / "interrupted-repair" / stamp
    repair.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((ROOT / "runtime/modelscope-files.json").read_text(encoding="utf-8"))
    files = {f["Path"]: f for f in metadata["Data"]["Files"]}
    audit1 = json.loads((ROOT / "artifacts/models/chunk-local-audit.json").read_text())
    audit2 = json.loads((ROOT / "artifacts/models/shard2-blank-audit.json").read_text())
    plans = [
        ("model-00001-of-00002.safetensors", [r["index"] for r in audit1 if r["first_64KiB_zero_run"] >= 0]),
        ("model-00002-of-00002.safetensors", [r["index"] for r in audit2["suspect_blocks"]]),
    ]
    records = []
    chunk_size = 16 * 1024 * 1024
    for name, indexes in plans:
        info = files[name]
        url = ("https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct/repo?Revision="
               + quote(info["Revision"]) + "&FilePath=" + quote(name))
        recovered = repair / name
        if name.startswith("model-00001"):
            source = base / (name + ".assembled")
        else:
            source = base / name
        print(f"REPAIR PLAN {name}; identified blank blocks={indexes}; preserve={source.name}", flush=True)
        shutil.copyfile(source, recovered)
        corrections = []
        for index in indexes:
            first = index * chunk_size
            last = min(info["Size"], first + chunk_size) - 1
            patch = repair / f"{name}.{index:05d}.bin"
            fetch_range_once(url, patch, first, last, info["Size"])
            new = patch.read_bytes()
            with source.open("rb") as old_stream:
                old_stream.seek(first)
                old = old_stream.read(len(new))
            difference = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None)
            corrections.append({"block": index, "bytes": len(new), "first_difference": difference,
                                "download_sha256": checksum(patch)})
            with recovered.open("r+b") as target:
                target.seek(first)
                target.write(new)
                target.flush()
                os.fsync(target.fileno())
            print(f"PATCH {name} block={index}; first_difference={difference}", flush=True)
        actual = checksum(recovered)
        entry = {"file": name, "expected_sha256": info["Sha256"], "actual_sha256": actual,
                 "corrections": corrections, "verified": actual == info["Sha256"]}
        records.append(entry)
        save_json(ROOT / "artifacts/models/interrupted-repair.json", records)
        if not entry["verified"]:
            raise RuntimeError(f"Repair checksum mismatch; stopped without another attempt: {name}")
        # Preserve the failed formal file before promoting a verified replacement.
        formal = base / name
        if formal.exists():
            preserved = repair / (name + ".original-interrupted")
            formal.rename(preserved)
        recovered.rename(formal)
        print(f"VERIFIED AND READY {name} {actual}", flush=True)
    print("IDENTIFIED BLANK RANGES REPAIRED; no GPU used", flush=True)


if __name__ == "__main__":
    with ModelTask("repair-interrupted-download"):
        main()
