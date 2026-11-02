import io
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from catvision.server import LocalComparison, create_app


class ApiTests(unittest.TestCase):
    def test_completed_request_releases_last_model_from_gpu(self):
        closed = []

        class Engine:
            def close(self):
                closed.append(True)

        pipeline = LocalComparison()
        pipeline.engine = Engine()
        pipeline.engine_route = "vlm"
        pipeline._run = lambda *args: {"result": "saved"}
        with patch("catvision.server.ModelTask", return_value=nullcontext()):
            self.assertEqual(pipeline(None, None, None), {"result": "saved"})
        self.assertEqual(closed, [True])
        self.assertIsNone(pipeline.engine)

    def test_model_index_without_weight_shards_is_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["yolo26n.pt", "sam2.1_t.pt", "breed-best.pt"]:
                path = root / "weights" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not empty")
            index = root / "weights/qwen3-vl-4b/model.safetensors.index.json"
            index.parent.mkdir(parents=True)
            index.write_text(json.dumps({"weight_map": {"tensor": "missing-shard.safetensors"}}))
            with TestClient(create_app(root, lambda *args: {})) as client:
                self.assertFalse(client.get("/api/info").json()["models_ready"])

    def test_corrupt_upload_is_rejected_before_model_execution(self):
        def should_not_run(image, folder, progress):
            raise AssertionError("Corrupt input reached model")
        with tempfile.TemporaryDirectory() as folder, TestClient(create_app(Path(folder), should_not_run)) as client:
            response = client.post("/api/recognize", files={"file": ("cat.jpg", b"bad image", "image/jpeg")})
            self.assertEqual(response.status_code, 400)

    def test_real_upload_pixels_are_delivered_and_job_result_can_be_read(self):
        def runner(image, folder, progress):
            self.assertEqual(image.size, (3, 2))
            self.assertEqual(image.getpixel((0, 0)), (7, 31, 53))
            progress("识别中")
            return {"traditional": {"status": "ok", "breed": "Ragdoll", "breed_zh": "布偶猫"},
                    "vlm": {"status": "ok", "breed": "Ragdoll", "breed_zh": "布偶猫"}}
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), (7, 31, 53)).save(buffer, "PNG")
        with tempfile.TemporaryDirectory() as folder, TestClient(create_app(Path(folder), runner)) as client:
            response = client.post("/api/recognize", files={"file": ("中文图片.png", buffer.getvalue(), "image/png")})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["id"]
            for _ in range(50):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["state"] in {"completed", "error"}:
                    break
                time.sleep(0.01)
            self.assertEqual(job["state"], "completed")
            self.assertEqual(job["result"]["vlm"]["breed_zh"], "布偶猫")
            self.assertEqual(client.get(job["input_url"]).status_code, 200)


if __name__ == "__main__":
    unittest.main()
