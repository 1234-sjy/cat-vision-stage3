import unittest
from catvision.vlm_features import parse_visible_features


class VisibleFeatureTests(unittest.TestCase):
    def test_keeps_two_observations_without_turning_breed_into_a_feature(self):
        self.assertEqual(parse_visible_features('1. 毛发较短，呈浅米色条纹。\n2. 耳朵向前折，眼睛圆。'),
            ['毛发较短，呈浅米色条纹', '耳朵向前折，眼睛圆'])
        self.assertEqual(parse_visible_features('苏格兰折耳猫\n短毛、浅色毛发'), ['短毛、浅色毛发'])

    def test_rejects_scores_and_preserves_missing_description(self):
        self.assertEqual(parse_visible_features('98.8%\nScottish_Fold'), [])
        self.assertEqual(parse_visible_features('特征：\n黑色短毛'), ['黑色短毛'])
        self.assertEqual(parse_visible_features(''), [])


if __name__ == '__main__':
    unittest.main()
