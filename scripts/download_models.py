"""Download official weights; large files are resumable and checksum-verified."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT, save_json
from catvision.run_control import ModelTask


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_ranges(url: str, path: Path, size: int, sha256: str,
                    chunk_size: int = 16 * 1024 * 1024) -> None:
    chunks_dir = path.with_name(path.name + ".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    count = (size + chunk_size - 1) // chunk_size

    def fetch(index):
        first, last = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        chunk = chunks_dir / f"{index:05d}.bin"
        wanted = last - first + 1
        if chunk.exists() and chunk.stat().st_size == wanted:
            return chunk
        part = chunk.with_suffix(".part")
        result = subprocess.run([
            "curl.exe" if sys.platform == "win32" else "curl", "-sS", "--fail", "-L",
            "--connect-timeout", "10", "--max-time", "180", "-r", f"{first}-{last}",
            "-o", str(part), "-w", "%{http_code}", url], capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        if result.returncode != 0 or not part.exists() or part.stat().st_size != wanted:
            raise RuntimeError(f"Range download failed, no retry: {path.name} chunk {index}; "
                               f"exit={result.returncode}, HTTP={result.stdout}, "
                               f"bytes={part.stat().st_size if part.exists() else 0}/{wanted}")
        if result.stdout.strip() != "206" and not (first == 0 and last == size - 1):
            raise RuntimeError("Server did not honor bounded Range request")
        part.replace(chunk)
        print("RANGE", path.name, f"{index+1}/{count}", flush=True)
        return chunk

    # Only one HTTP request is active; no implicit retry or background pool.
    parts = [fetch(index) for index in range(count)]
    assembled = path.with_name(path.name + ".assembled")
    with assembled.open("wb") as target:
        for chunk in parts:
            with chunk.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
    if assembled.stat().st_size != size or checksum(assembled) != sha256:
        raise RuntimeError(f"Assembled model checksum mismatch: {path.name}")
    assembled.replace(path)
    # Keep download chunks until a separately authorized cleanup; user requested preservation.


def download(url: str, path: Path, size: int | None = None, sha256: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if (size is not None and path.stat().st_size != size) or (sha256 and checksum(path) != sha256):
            raise RuntimeError(f"Existing model size/checksum mismatch: {path.name}; stopped without replacement")
        print("CACHED", path.name, flush=True)
        return {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha256 or checksum(path), "source": url}
    if size and sha256 and size >= 8 * 1024 * 1024:
        download_ranges(url, path, size, sha256)
        print("READY", path.name, size, flush=True)
        return {"file": str(path.relative_to(ROOT)), "bytes": size, "sha256": sha256, "source": url}
    partial = path.with_name(path.name + ".part")
    print("DOWNLOAD ONCE", path.name, "resume", partial.stat().st_size if partial.exists() else 0, flush=True)
    result = subprocess.run(
        ["curl.exe" if sys.platform == "win32" else "curl", "-sS", "--fail", "-L",
         "--connect-timeout", "15", "--max-time", "600", "--speed-limit", "1024",
         "--speed-time", "60", "-C", "-", "-o", str(partial), url],
        capture_output=True, text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    if result.returncode != 0 or not partial.is_file():
        raise RuntimeError(f"Download failed, no retry: {path.name}; exit={result.returncode}")
    if size is not None and partial.stat().st_size != size:
        raise RuntimeError(f"Unexpected file size for {path.name}")
    actual = checksum(partial)
    if sha256 and actual != sha256:
        raise RuntimeError(f"Checksum mismatch for {path.name}; partial file retained for manual handling")
    partial.replace(path)
    print("READY", path.name, path.stat().st_size, flush=True)
    return {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": actual, "source": url}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["traditional", "vlm", "all"], default="all")
    args = parser.parse_args()
    records = []
    if args.only in {"traditional", "all"}:
        metadata = ROOT / "runtime/ultralytics-release.json"
        download("https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0", metadata)
        assets = {a["name"]: a for a in json.loads(metadata.read_text(encoding="utf-8"))["assets"]}
        mirrors = {
            "yolo26s-cls.pt": "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26s-cls.pt",
            "sam2.1_t.pt": "https://hf-mirror.com/spaces/ArctionO/sam2-mask-replacement/resolve/main/sam2.1_t.pt",
        }
        for name in ["yolo26n.pt", "yolo26s-cls.pt", "sam2.1_t.pt"]:
            asset = assets[name]
            record = download(mirrors.get(name, asset["browser_download_url"]), ROOT / "weights" / name,
                              asset["size"], asset["digest"].removeprefix("sha256:"))
            record["upstream_source"] = asset["browser_download_url"]
            records.append(record)
        save_json(ROOT / "artifacts/models/traditional-downloads.json", records)
    if args.only in {"vlm", "all"}:
        metadata = ROOT / "runtime/modelscope-files.json"
        download("https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct/repo/files?Revision=master&Recursive=true",
                 metadata)
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("Code") != 200:
            raise RuntimeError("ModelScope metadata request failed")
        files = [f for f in info["Data"]["Files"] if f["Type"] == "blob" and
                 (f["Path"].endswith((".json", ".txt", ".jinja", ".safetensors")))]
        def fetch(item):
            name = item["Path"]
            if Path(name).name != name:
                raise ValueError("Unexpected nested model file")
            url = ("https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct/repo?Revision="
                   + quote(item["Revision"]) + "&FilePath=" + quote(name))
            return download(url, ROOT / "weights/qwen3-vl-4b" / name, item["Size"], item["Sha256"])
        records = [fetch(item) for item in files]
        save_json(ROOT / "artifacts/models/qwen-downloads.json", records)


if __name__ == "__main__":
    with ModelTask("download-models"):
        main()
