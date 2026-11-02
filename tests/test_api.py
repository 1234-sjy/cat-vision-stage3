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

from catvision.server import LocalComparison, create_app, model_files_ready


class ApiTests(unittest.TestCase):
    def test_preloaded_models_are_reused_without_reusing_predictions(self):
        created=[];closed=[]
        class Engine:
            def close(self):closed.append(self)
        pipeline=LocalComparison(keep_loaded=True)
        def factory(route):
            created.append(route);return Engine()
        with patch.object(pipeline,'_make_engine',side_effect=factory),patch('catvision.server.ModelTask',return_value=nullcontext()):
            pipeline.preload()
            for _ in range(2):
                a,reused=pipeline._engine_for('traditional');self.assertTrue(reused)
                b,reused=pipeline._engine_for('vlm');self.assertTrue(reused)
            self.assertEqual(created,['traditional','vlm'])
            pipeline._run=lambda *args:{'fresh_run':True}
            self.assertTrue(pipeline(None,None,None)['fresh_run'])
            self.assertEqual(closed,[])
            pipeline.close()
            self.assertEqual(len(closed),2)

    def test_preloaded_models_are_released_after_failure(self):
        closed=[]
        class Engine:
            def close(self):closed.append(True)
        pipeline=LocalComparison(keep_loaded=True)
        with patch.object(pipeline,'_make_engine',side_effect=lambda route:Engine()),patch('catvision.server.ModelTask',return_value=nullcontext()):
            pipeline.preload()
            def fail(*args):raise RuntimeError('stop now')
            pipeline._run=fail
            with self.assertRaisesRegex(RuntimeError,'stop now'):pipeline(None,None,None)
        self.assertEqual(len(closed),2)
        self.assertEqual(pipeline.engines,{})

    def test_report_keeps_full_classifier_and_common_test_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for route in ['traditional','vlm']:
                p=root/f'artifacts/evaluation/comparison-v2/{route}/metrics.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'count':120,'correct':106 if route=='traditional' else 87}))
            p=root/'artifacts/optimization-v2/full-original-test/metrics.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'count':1183,'correct':1067,'scope':'original classifier only'}))
            with TestClient(create_app(root,lambda *args:{})) as client:
                report=client.get('/api/report').json()
            self.assertEqual(report['traditional']['count'],120)
            self.assertEqual(report['full_classifier']['count'],1183)

    def test_ready_checks_active_classifier_instead_of_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            base=root/'weights/qwen3-vl-4b';base.mkdir(parents=True)
            for name in ['model.safetensors','config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json']:
                (base/name).write_bytes(b'x')
            (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'tensor':'model.safetensors'}}))
            for name in ['yolo26n.pt','sam2.1_t.pt','breed-best.pt']:
                (root/'weights'/name).write_bytes(b'x')
            active=root/'configs/active-classifier.json';active.parent.mkdir()
            active.write_text(json.dumps({'checkpoint':'swin.pt','architecture':'swin_t'}))
            self.assertFalse(model_files_ready(root))
            (root/'weights/swin.pt').write_bytes(b'x')
            self.assertTrue(model_files_ready(root))

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
