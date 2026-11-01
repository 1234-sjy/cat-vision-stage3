import unittest

from catvision.data import audit_records, choose_comparison, make_view_record


class DataTests(unittest.TestCase):
    def test_detection_failure_cannot_overwrite_source_label_or_split(self):
        source = {"image_id": "r", "breed": "Ragdoll", "split": "train", "sha256": "source-hash"}
        prediction = {"status": "no_cat", "breed": None, "split": "test"}
        view = make_view_record(source, prediction)
        self.assertEqual(view["breed"], "Ragdoll")
        self.assertEqual(view["split"], "train")
        self.assertEqual(view["status"], "no_cat")
        self.assertTrue(view["classifier_view_fallback"])

    def test_same_photo_in_train_and_test_is_rejected(self):
        records = [
            {"image_id": "a", "sha256": "same", "split": "train", "breed": "Ragdoll"},
            {"image_id": "b", "sha256": "same", "split": "test", "breed": "Ragdoll"},
        ]
        with self.assertRaises(ValueError):
            audit_records(records)

    def test_comparison_is_balanced_and_only_from_test(self):
        records = [
            {"image_id": f"{breed}-{split}-{i}", "split": split, "breed": breed}
            for breed in ["Ragdoll", "Persian"]
            for split in ["train", "val", "test"] for i in range(5)
        ]
        selected = choose_comparison(records, per_class=2, seed=17)
        self.assertEqual(len(selected), 4)
        self.assertEqual({r["split"] for r in selected}, {"test"})
        self.assertEqual(sum(r["breed"] == "Ragdoll" for r in selected), 2)
        self.assertEqual(selected, choose_comparison(list(reversed(records)), 2, 17))

    def test_too_few_test_images_does_not_silently_make_unbalanced_subset(self):
        with self.assertRaises(ValueError):
            choose_comparison([{"image_id": "one", "split": "test", "breed": "Ragdoll"}], 2, 17)


if __name__ == "__main__":
    unittest.main()
