from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
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
        active = base_dir/'configs/active-classifier.json'
        checkpoint = json.loads(active.read_text(encoding='utf-8'))['checkpoint'] if active.exists() else 'breed-best.pt'
        if Path(checkpoint).name != checkpoint:
            return False
        required = [weights / name for name in ["yolo26n.pt", "sam2.1_t.pt", checkpoint]]
        required += [vlm / name for name in names | {"config.json", "preprocessor_config.json",
                                                    "tokenizer.json", "tokenizer_config.json"}]
        return all(path.is_file() and path.stat().st_size > 0 for path in required)
    except (OSError, ValueError, KeyError, TypeError):
        return False


class LocalComparison:
    def __init__(self, keep_loaded=False):
        self.engine = None
        self.engine_route = None
        self.keep_loaded = keep_loaded
        self.engines = {}

    @staticmethod
    def _make_engine(route):
        if route == 'traditional':
            from .traditional import TraditionalVision
            return TraditionalVision()
        from .vlm import LocalVLM
        return LocalVLM()

    def _engine_for(self, route):
        if self.keep_loaded and route in self.engines:
            self.engine, self.engine_route = self.engines[route], route
            return self.engine, True
        if not self.keep_loaded:
            self.close()
        self.engine = self._make_engine(route)
        self.engine_route = route
        if self.keep_loaded:
            self.engines[route] = self.engine
        return self.engine, False

    def preload(self):
        if not self.keep_loaded:
            raise ValueError('Preloading requires keep_loaded mode')
        with ModelTask('web-model-preload'):
            try:
                for route in ['traditional','vlm']:
                    self._engine_for(route)
                    print(f'MODEL READY {route}',flush=True)
            except BaseException:
                self.close()
                raise

    def close(self):
        unique = {id(engine):engine for engine in self.engines.values()}
        if self.engine:
            unique[id(self.engine)] = self.engine
        for engine in unique.values():
            engine.close()
        self.engine = self.engine_route = None
        self.engines = {}

    def __call__(self, image, directory, progress):
        with ModelTask("web-comparison"):
            try:
                return self._run(image, directory, progress)
            except BaseException:
                self.close()
                raise
            finally:
                # Every request starts with the traditional route, so keeping the
                # final VLM loaded cannot serve the next request and wastes VRAM.
                if not self.keep_loaded:
                    self.close()

    def _run(self, image, directory, progress):
        result = {"input_sha256": image_hash(image), "fresh_run": True}
        started = time.perf_counter()
        for route, title in [("traditional", "传统视觉模型"), ("vlm", "本地视觉大模型")]:
            try:
                progress(f"正在准备{title}")
                self.engine, reused = self._engine_for(route)
                progress(f"{title}正在识别")
                prediction = self.engine.predict(image.copy())
                prediction["load_seconds"] = 0.0 if reused else self.engine.load_seconds
                prediction['model_reused'] = reused
                prediction['initial_load_seconds'] = self.engine.load_seconds
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
                "models_preloaded": bool(getattr(pipeline,'engines',{})),
                "models_ready": model_files_ready(base_dir)}

    @app.get("/api/examples")
    def examples():
        path = examples_dir / "index.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @app.get("/api/report")
    def report():
        result = {}
        active_report = base_dir/'artifacts/evaluation/comparison-v2'
        experiment = 'comparison-v2' if all((active_report/route/'metrics.json').exists() for route in ['traditional','vlm']) else 'comparison'
        for route in ["traditional", "vlm"]:
            path = base_dir / f"artifacts/evaluation/{experiment}/{route}/metrics.json"
            result[route] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        result['experiment'] = experiment
        full = base_dir/'artifacts/optimization-v2/full-original-test/metrics.json'
        result['full_classifier'] = json.loads(full.read_text(encoding='utf-8')) if full.exists() else None
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
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser()
    parser.add_argument('--fast-demo',action='store_true',help='Preload models; remove inference pauses; fresh predictions remain serial')
    args=parser.parse_args()
    if args.fast_demo:
        os.environ['CATVISION_FAST_INFERENCE']='1'
        runner=LocalComparison(keep_loaded=True)
        runner.preload()
        application=create_app(runner=runner)
    else:
        application=create_app()
    uvicorn.run(application, host="127.0.0.1", port=8765)
