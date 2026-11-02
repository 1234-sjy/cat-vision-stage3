from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .common import BREEDS, ROOT, image_hash, load_image, save_json
from .run_control import ModelTask


def model_files_ready(base_dir: Path) -> bool:
    try:
        weights = base_dir / "weights"
        vlm = weights / "qwen3-vl-4b"
        index = json.loads((vlm / "model.safetensors.index.json").read_text(encoding="utf-8"))
        names = set(index["weight_map"].values())
        if not names or any(Path(name).name != name for name in names):
            return False
        required = [weights / name for name in ["yolo26n.pt", "sam2.1_t.pt", "breed-best.pt"]]
        required += [vlm / name for name in names | {"config.json", "preprocessor_config.json",
                                                    "tokenizer.json", "tokenizer_config.json"}]
        return all(path.is_file() and path.stat().st_size > 0 for path in required)
    except (OSError, ValueError, KeyError, TypeError):
        return False


class LocalComparison:
    def __init__(self):
        self.engine = None
        self.engine_route = None

    def close(self):
        if self.engine:
            self.engine.close()
        self.engine = self.engine_route = None

    def __call__(self, image, directory, progress):
        with ModelTask("web-comparison"):
            try:
                return self._run(image, directory, progress)
            finally:
                # Every request starts with the traditional route, so keeping the
                # final VLM loaded cannot serve the next request and wastes VRAM.
                self.close()

    def _run(self, image, directory, progress):
        result = {"input_sha256": image_hash(image), "fresh_run": True}
        started = time.perf_counter()
        for route, title in [("traditional", "传统视觉模型"), ("vlm", "本地视觉大模型")]:
            try:
                progress(f"正在加载{title}")
                if self.engine_route != route:
                    self.close()
                    if route == "traditional":
                        from .traditional import TraditionalVision
                        self.engine = TraditionalVision()
                    else:
                        from .vlm import LocalVLM
                        self.engine = LocalVLM()
                    self.engine_route = route
                progress(f"{title}正在识别")
                prediction = self.engine.predict(image.copy())
                prediction["load_seconds"] = self.engine.load_seconds
                if route == "traditional":
                    from .traditional import visualization
                    visualization(image, prediction).save(directory / "traditional.png")
                    crop = prediction.pop("crop", None)
                    if crop is not None:
                        crop.save(directory / "subject.png")
                    prediction.pop("mask", None)
                result[route] = prediction
            except Exception as exc:
                result[route] = {"status": "error", "breed": None,
                                 "error": f"{type(exc).__name__}: {exc}"}
                self.close()
                save_json(directory / "partial-result.json", result)
                raise  # Do not load the next model after a runtime failure.
        result["total_seconds"] = time.perf_counter() - started
        return result


def create_app(base_dir: Path = ROOT, runner=None) -> FastAPI:
    results_dir = base_dir / "runtime/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = base_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    pipeline = runner if runner is not None else LocalComparison()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cat-gpu")
    jobs = {}

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True, cancel_futures=True)
        if hasattr(pipeline, "close"):
            pipeline.close()

    app = FastAPI(title="看图识猫 · 第二阶段", lifespan=lifespan)
    app.mount("/files", StaticFiles(directory=results_dir), name="results")
    app.mount("/examples", StaticFiles(directory=examples_dir), name="examples")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/api/info")
    def info():
        return {"breeds": list(BREEDS.values()), "mode": "local_gpu",
                "models_ready": model_files_ready(base_dir)}

    @app.get("/api/examples")
    def examples():
        path = examples_dir / "index.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @app.get("/api/report")
    def report():
        result = {}
        for route in ["traditional", "vlm"]:
            path = base_dir / f"artifacts/evaluation/comparison/{route}/metrics.json"
            result[route] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return result

    def execute(job_id, image, directory):
        job = jobs[job_id]
        def progress(text):
            job.update(state="running", message=text)
        try:
            job["result"] = pipeline(image, directory, progress)
            job.update(state="completed", message="识别完成")
        except Exception as exc:
            job.update(state="error", message=f"识别未完成：{type(exc).__name__}: {exc}")
        save_json(directory / "result.json", job)

    @app.post("/api/recognize", status_code=202)
    async def recognize(file: UploadFile):
        content = await file.read(12 * 1024 * 1024 + 1)
        if len(content) > 12 * 1024 * 1024:
            raise HTTPException(413, "图片超过12MB，请缩小后重试")
        try:
            image = load_image(content, max_side=1600)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if sum(j["state"] in {"queued", "running"} for j in jobs.values()) >= 4:
            raise HTTPException(429, "已有图片等待识别，请稍后再试")
        job_id = uuid4().hex
        directory = results_dir / job_id
        directory.mkdir()
        image.save(directory / "input.png")
        jobs[job_id] = {"id": job_id, "state": "queued", "message": "等待显卡处理",
                        "input_url": f"/files/{job_id}/input.png",
                        "visualization_url": f"/files/{job_id}/traditional.png",
                        "created_utc": datetime.now(timezone.utc).isoformat()}
        executor.submit(execute, job_id, image, directory)
        return JSONResponse({"id": job_id, "state": "queued"}, status_code=202)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise HTTPException(404, "没有找到这次识别")
        if job_id in jobs:
            return jobs[job_id]
        path = results_dir / job_id / "result.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise HTTPException(404, "没有找到这次识别")

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8765)
