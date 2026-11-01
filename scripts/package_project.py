"""Collect the full local handover, including data omitted from Git.

Run only after model work has stopped. ZIP_STORED avoids a CPU-heavy compression pass.
No input file is deleted. Partial model chunks and historical logs are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from zipfile import ZIP_STORED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requirements-root", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [(ROOT, "项目"), (args.data_root.resolve(), "原始数据/oxford-iiit-pet"),
               (args.requirements_root.resolve(), "要求依据"), (args.videos_root.resolve(), "课程视频")]
    output = args.output.resolve()
    # Never include the output in itself; never silently overwrite an earlier handover.
    if any(output.is_relative_to(source) for source, _ in sources) or output.exists():
        raise ValueError("成果包须放在所有来源目录之外，并使用新的文件名")
    required = [ROOT / "artifacts/data/manifest.json", ROOT / "artifacts/data/comparison.json",
                ROOT / "data/classification", ROOT / "weights", ROOT / "artifacts/views/records.jsonl"]
    for path in required + [p for p, _ in sources]:
        if not path.exists():
            raise FileNotFoundError(f"Required source missing; packaging stopped: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with ZipFile(output, "x", compression=ZIP_STORED, allowZip64=True) as archive:
        for source, prefix in sources:
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                if any(part in {".venv", "__pycache__", ".pytest_cache"} for part in relative.parts):
                    continue
                # The OS lock itself is transient. Preserve its recorded state and logs.
                if relative.as_posix() == "runtime/model-task.lock":
                    continue
                name = f"{prefix}/{relative.as_posix()}"
                archive.write(path, name)
                files.append({"file": name, "bytes": path.stat().st_size})
        archive.writestr("成果包文件清单.json", json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
            "contains_original_requirements": True, "contains_stage2_data": True,
            "note": "阶段性保留包也可生成；完成状态以项目文档和实际记录为准。"}, ensure_ascii=False, indent=2))
    print(f"PACKAGE SAVED {output}; files={len(files)}", flush=True)


if __name__ == "__main__":
    main()
