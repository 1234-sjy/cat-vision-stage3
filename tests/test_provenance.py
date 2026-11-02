import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from catvision.evaluation import prediction_cache_matches, read_verified_input, inference_fingerprint


class InputProvenanceTests(unittest.TestCase):
    def test_vlm_prompt_and_response_schema_invalidate_saved_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            files=['catvision/common.py','catvision/vlm.py','catvision/vlm_prompts.py',
                   'catvision/vlm_format.py','catvision/vlm_acceleration.py',
                   'weights/qwen3-vl-4b/model.safetensors']
            for name in files:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original')
            base=root/'weights/qwen3-vl-4b'
            (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'tensor':'model.safetensors'}}))
            active=root/'configs/active-vlm.json';active.parent.mkdir()
            active.write_text('{"output_format":"json"}')
            first=inference_fingerprint(root,'vlm')
            active.write_text('{"output_format":"lines"}')
            second=inference_fingerprint(root,'vlm')
            self.assertNotEqual(first,second)
            (root/'catvision/vlm_format.py').write_bytes(b'changed parser')
            self.assertNotEqual(second,inference_fingerprint(root,'vlm'))

    def test_input_hash_is_measured_from_actual_file_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cat.jpg").write_bytes(b"original-image")
            digest = hashlib.sha256(b"original-image").hexdigest()
            content, measured = read_verified_input({"image_id": "a", "relative_path": "cat.jpg", "sha256": digest}, root)
            self.assertEqual(content, b"original-image")
            self.assertEqual(measured, digest)

    def test_replaced_image_is_rejected_even_with_same_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cat.jpg").write_bytes(b"replaced-image")
            record = {"image_id": "a", "relative_path": "cat.jpg", "sha256": hashlib.sha256(b"original-image").hexdigest()}
            with self.assertRaisesRegex(ValueError, "SHA256"):
                read_verified_input(record, root)

    def test_changed_model_or_input_invalidates_cached_prediction(self):
        prediction = {"image_id": "a", "input_sha256": "image-one", "run_fingerprint": "model-one"}
        self.assertTrue(prediction_cache_matches(prediction, "a", "image-one", "model-one"))
        self.assertFalse(prediction_cache_matches(prediction, "a", "image-two", "model-one"))
        self.assertFalse(prediction_cache_matches(prediction, "a", "image-one", "model-two"))

    def test_legacy_cache_without_model_identity_is_not_reused(self):
        self.assertFalse(prediction_cache_matches({"image_id": "a", "input_sha256": "image-one"},
                                                  "a", "image-one", "model-one"))

    def test_selected_transformer_and_policy_change_cache_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            names=['catvision/common.py','catvision/traditional.py','catvision/geometry.py',
                   'catvision/breed_transformer.py','weights/yolo26n.pt','weights/sam2.1_t.pt',
                   'weights/breed-best.pt','weights/custom.pt']
            for name in names:
                path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'old')
            active=root/'configs/active-classifier.json';active.parent.mkdir()
            active.write_text(json.dumps({'architecture':'swin_t','checkpoint':'custom.pt','cat_threshold':0.5}))
            first=inference_fingerprint(root,'traditional')
            (root/'weights/custom.pt').write_bytes(b'new')
            second=inference_fingerprint(root,'traditional')
            self.assertNotEqual(first,second)
            active.write_text(json.dumps({'architecture':'swin_t','checkpoint':'custom.pt','cat_threshold':0.8}))
            self.assertNotEqual(second,inference_fingerprint(root,'traditional'))


if __name__ == "__main__":
    unittest.main()
