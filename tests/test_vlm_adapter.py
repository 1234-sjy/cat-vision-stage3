import json
from pathlib import Path
import tempfile
import unittest

from catvision.evaluation import inference_fingerprint
from catvision.server import model_files_ready


class AdapterDeploymentTests(unittest.TestCase):
    def make_project(self, root):
        for name in ['catvision/common.py', 'catvision/vlm.py', 'catvision/vlm_adapter.py',
                     'weights/yolo26n.pt', 'weights/sam2.1_t.pt', 'weights/breed-best.pt',
                     'weights/qwen3-vl-4b/model.safetensors', 'weights/qwen3-vl-4b/config.json',
                     'weights/qwen3-vl-4b/preprocessor_config.json',
                     'weights/qwen3-vl-4b/tokenizer.json', 'weights/qwen3-vl-4b/tokenizer_config.json']:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'base')
        (root/'weights/qwen3-vl-4b/model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'x': 'model.safetensors'}}))
        (root/'configs').mkdir()
        (root/'configs/active-vlm.json').write_text(json.dumps({'adapter': 'weights/cat-adapter'}))

    def test_missing_adapter_cannot_report_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_project(root)
            self.assertFalse(model_files_ready(root))
            adapter = root/'weights/cat-adapter'
            adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'trained adapter')
            self.assertTrue(model_files_ready(root))

    def test_adapter_weights_change_prediction_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_project(root)
            adapter = root/'weights/cat-adapter'
            adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            weights = adapter/'adapter_model.safetensors'
            weights.write_bytes(b'epoch one')
            first = inference_fingerprint(root, 'vlm')
            weights.write_bytes(b'epoch two')
            self.assertNotEqual(first, inference_fingerprint(root, 'vlm'))

    def test_external_adapter_path_is_rejected(self):
        from catvision.vlm_adapter import resolve_adapter
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in ['../outside', 'C:/outside', '/outside', 'weights/../outside']:
                with self.assertRaises(ValueError):
                    resolve_adapter(root, {'adapter': path})


if __name__ == '__main__':
    unittest.main()
