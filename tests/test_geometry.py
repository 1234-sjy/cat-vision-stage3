import unittest

import numpy as np

from catvision.geometry import crop_box, main_cat_index, mask_to_box


class GeometryTests(unittest.TestCase):
    def test_box_is_clipped_and_not_empty(self):
        self.assertEqual(crop_box([-5, -8, 12, 20], (10, 15), margin=0), (0, 0, 10, 15))
        with self.assertRaises(ValueError):
            crop_box([9, 0, 2, 3], (10, 15), margin=0)

    def test_primary_subject_is_largest_cat_not_highest_score_or_dog(self):
        boxes = [[0, 0, 100, 100], [1, 1, 4, 4], [2, 2, 10, 10]]
        self.assertEqual(main_cat_index(boxes, [16, 15, 15]), 2)
        self.assertIsNone(main_cat_index(boxes, [16, 16, 16]))

    def test_mask_uses_exclusive_bottom_and_right_edges(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:4, 2:7] = True
        self.assertEqual(mask_to_box(mask), (2, 1, 7, 4))
        self.assertIsNone(mask_to_box(np.zeros((6, 8), dtype=bool)))


if __name__ == "__main__":
    unittest.main()
