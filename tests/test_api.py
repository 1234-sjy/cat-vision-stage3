import io
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from catvision.server import create_app


class ApiTests(unittest.TestCase):
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
