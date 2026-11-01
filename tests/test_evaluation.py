import unittest

from catvision.evaluation import score_predictions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = [{"image_id": "a", "breed": "Ragdoll"}, {"image_id": "b", "breed": "Persian"}]

    def test_missing_prediction_counts_as_failure_and_stays_in_confusion_matrix(self):
        result = score_predictions(self.truth, [{"image_id": "a", "status": "ok", "breed": "Ragdoll"}])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 2)
        self.assertEqual(result["status_counts"]["missing"], 1)

    def test_failed_response_cannot_get_credit_for_a_stale_label(self):
        records = [{"image_id": "a", "status": "ok", "breed": "Ragdoll"},
                   {"image_id": "b", "status": "error", "breed": "Persian"}]
        self.assertEqual(score_predictions(self.truth, records)["correct"], 1)

    def test_duplicate_prediction_is_rejected(self):
        records = [{"image_id": "a", "status": "ok", "breed": "Ragdoll"}] * 2
        with self.assertRaises(ValueError):
            score_predictions(self.truth, records)


if __name__ == "__main__":
    unittest.main()
