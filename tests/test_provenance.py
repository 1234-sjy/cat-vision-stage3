import hashlib
from pathlib import Path
import tempfile
import unittest

from catvision.evaluation import prediction_cache_matches, read_verified_input


class InputProvenanceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
