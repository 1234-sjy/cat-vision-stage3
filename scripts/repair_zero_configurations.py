"""Repair only the audited all-zero configuration files, preserving originals."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask
from scripts.download_models import download


def main():
    base = ROOT / "weights/qwen3-vl-4b"
    audit = json.loads((ROOT / "artifacts/models/configuration-audit.json").read_text())
    info = json.loads((ROOT / "runtime/modelscope-files.json").read_text())
    files = {f["Path"]: f for f in info["Data"]["Files"]}
    history = base / "interrupted-repair" / datetime.now(timezone.utc).strftime("configs-%Y%m%dT%H%M%SZ")
    history.mkdir(parents=True, exist_ok=False)
    records = []
    for record in audit:
        if record.get("sha_matches"):
            continue
        name = record["file"]
        formal = base / name
        original = formal.read_bytes()
        if original.count(0) != len(original):
            raise RuntimeError(f"Unexpected corruption pattern, stopped: {name}")
        item = files[name]
        url = ("https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct/repo?Revision="
               + quote(item["Revision"]) + "&FilePath=" + quote(name))
        verified = history / (name + ".verified")
        entry = download(url, verified, item["Size"], item["Sha256"])
        with verified.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        formal.rename(history / (name + ".original-zero"))
        verified.rename(formal)
        records.append(entry)
        print("REPAIRED ALL-ZERO CONFIG", name, flush=True)
    save_json(ROOT / "artifacts/models/configuration-repair.json", records)


if __name__ == "__main__":
    with ModelTask("repair-zero-configurations"):
        main()
