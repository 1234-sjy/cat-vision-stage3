"""CPU-only data isolation checks for the source-label VLM adaptation."""
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_vlm_adaptation.py"


def module():
    spec = importlib.util.spec_from_file_location("prepare_vlm_adaptation", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def row(image_id, split="test", group=None, sha=None, breed="Cat", is_cat=True):
    return {"image_id": image_id, "split": split, "group": group or image_id,
            "sha256": sha or image_id, "breed": breed, "is_cat": is_cat,
            "relative_path": image_id + ".jpg", "breed_zh": "source label"}


class VLMAdaptationDataTests(unittest.TestCase):
    def test_training_uses_only_train_rows_and_source_labels(self):
        data = module()
        rows = [row("train", "train"), row("val", "val"), row("test"),
                row("dog", "train", breed="negative_dog", is_cat=False)]
        train = data.split_source_labels(rows, "train")
        self.assertEqual([r["image_id"] for r in train], ["train", "dog"])
        self.assertEqual([r["target_text"] for r in train], ["Cat", "no_cat"])
        self.assertNotIn("target_text", rows[0])

    def test_all_linked_group_members_are_excluded_without_using_answers(self):
        data = module()
        known = [row("old"), row("bridge"), row("new", group="shared"),
                 row("same_group", group="shared"), row("same_sha", sha="old"), row("fresh")]
        graph = data.build_identity_graph(known, [
            {"left": "old", "right": "bridge", "kind": "phash", "distance": 4},
            {"left": "bridge", "right": "new", "kind": "orb_photographic_session"}])
        observations = [{"image_id": "old", "sha256": "old", "source_path": "old.json"}]
        kept, excluded = data.select_fresh_test(known[2:], graph, observations)
        self.assertEqual([r["image_id"] for r in kept], ["fresh"])
        self.assertEqual({r["image_id"] for r in excluded}, {"new", "same_group", "same_sha"})
        self.assertTrue(all(r["historical_image_ids"] == ["old"] for r in excluded))

    def test_selection_is_deterministic_and_drawn_only_from_validation(self):
        data = module()
        classes = [{"breed": "A"}, {"breed": "B"}]
        rows = [row(f"{breed}{i}", "val", breed=breed)
                for breed in ("A", "B") for i in range(14)]
        rows += [row(f"dog{i}", "val", is_cat=False) for i in range(60)]
        first = data.validation_selection(rows, classes)
        second = data.validation_selection(list(reversed(rows)), classes)
        self.assertEqual([r["image_id"] for r in first], [r["image_id"] for r in second])
        self.assertEqual(Counter(r["breed"] for r in first if r["is_cat"]), {"A": 10, "B": 10})
        self.assertEqual(sum(not r["is_cat"] for r in first), 50)
        with self.assertRaises(ValueError):
            data.validation_selection(rows + [row("leaked", "test")], classes)

    def test_profile_and_failed_prediction_inputs_count_as_exposed(self):
        data = module()
        profile = {"runs": [{"image_id": "profile", "result": {"route": "qwen3_vl_4b"}}]}
        self.assertEqual(data.exposure_records(profile, "profile.json")[0]["image_id"], "profile")
        failed = [{"image_id": "failed", "status": "error", "input_sha256": "abc"}]
        self.assertEqual(data.exposure_records(failed, "vlm/predictions.json")[0]["sha256"], "abc")

    def test_existing_output_is_never_overwritten(self):
        data = module()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frozen.json"
            data.write_new_json(path, ["original"])
            with self.assertRaises(FileExistsError):
                data.write_new_json(path, ["changed"])
            self.assertIn("original", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
