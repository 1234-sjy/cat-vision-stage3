"""Acquire one Kaggle stream, then prepare a frozen CPU-only expanded dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json

SOURCE = "https://www.kaggle.com/api/v1/datasets/download/doctrinek/catbreedsrefined-7k"


def download(data_root: Path, out: Path) -> None:
    from catvision.run_control import ModelTask
    archive = data_root / "expanded-v3-catbreedsrefined-7k.zip"
    attempt_path = out / "download-attempt.json"
    if archive.exists() or attempt_path.exists():
        raise FileExistsError("Existing download attempt: inspect it; automatic retry is forbidden")
    data_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    attempt = {"url": SOURCE, "started_utc": datetime.now(timezone.utc).isoformat(),
               "state": "started", "automatic_retry": False, "range_requests": False,
               "streams": 1, "archive": archive.name,
               "advertised_version": 2, "advertised_zip_bytes": 463820141,
               "license": "CC BY-SA 4.0", "dataset_owner": "doctrinek"}
    save_json(attempt_path, attempt)
    try:
        with ModelTask("expanded-data-download-v3"):
            request = urllib.request.Request(SOURCE, headers={"User-Agent": "CatVisionDatasetAudit/3.0"})
            digest = hashlib.sha256()
            start = time.monotonic()
            written = 0
            next_report = 16 * 1024 * 1024
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as target:
                attempt.update(http_status=response.status, final_url=response.url,
                               headers=dict(response.headers.items()))
                save_json(attempt_path, attempt)
                if response.status != 200:
                    raise RuntimeError(f"Unexpected download status {response.status}")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
                    digest.update(block)
                    written += len(block)
                    if written >= next_report:
                        print(f"download_bytes={written} elapsed_seconds={time.monotonic()-start:.1f}", flush=True)
                        next_report += 16 * 1024 * 1024
            expected = attempt["headers"].get("Content-Length")
            if expected is not None and int(expected) != written:
                raise IOError(f"Incomplete response: got {written}, expected {expected}")
            with zipfile.ZipFile(archive) as source:
                entries = source.infolist()
            attempt.update(state="completed", bytes=written, sha256=digest.hexdigest(),
                           zip_entries=len(entries), elapsed_seconds=time.monotonic()-start,
                           finished_utc=datetime.now(timezone.utc).isoformat())
            save_json(attempt_path, attempt)
            print(json.dumps({k: v for k, v in attempt.items() if k not in {"headers", "final_url"}}, indent=2), flush=True)
    except BaseException as error:
        attempt.update(state="failed", error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc(),
                       finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(attempt_path, attempt)
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/optimization-v3/data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--refine-sessions", action="store_true")
    args = p.parse_args()
    if args.download:
        download(args.data_root.resolve(), args.out_dir.resolve())
    elif args.refine_sessions:
        from catvision.dataset_expanded import refine_sessions
        refine_sessions(args.data_root.resolve(), args.out_dir.resolve())
    else:
        from catvision.dataset_expanded import prepare
        prepare(args.data_root.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
