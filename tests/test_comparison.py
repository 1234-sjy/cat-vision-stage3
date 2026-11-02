import unittest

from catvision.evaluation import validate_comparison


class ComparisonInputTests(unittest.TestCase):
    def setUp(self):
        self.truth = [{"image_id": "cat-a", "breed": "Ragdoll", "sha256": "original"}]
        self.prediction = [{"image_id": "cat-a", "input_sha256": "original"}]

    def test_two_complete_routes_for_identical_source_are_accepted(self):
        validate_comparison(self.truth, self.prediction, self.prediction)

    def test_changed_image_is_rejected_before_comparison(self):
        other = [{"image_id": "cat-a", "input_sha256": "modified"}]
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_comparison(self.truth, self.prediction, other)

    def test_partial_route_cannot_be_presented_as_complete_comparison(self):
        with self.assertRaisesRegex(ValueError, "完整"):
            validate_comparison(self.truth, self.prediction, [])


if __name__ == "__main__":
    unittest.main()
