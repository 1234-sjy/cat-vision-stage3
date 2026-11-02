import unittest
from catvision.vlm_format import parse_breed_lines


class LineFormatTests(unittest.TestCase):
    def test_one_breed_and_observed_features(self):
        r = parse_breed_lines('American_Curl\n耳尖向后卷、短毛')
        self.assertEqual(r['breed'], 'American_Curl')
        self.assertEqual(r['features'], ['耳尖向后卷','短毛'])

    def test_no_cat_and_unknown_have_no_invented_breed(self):
        self.assertEqual(parse_breed_lines('no_cat')['status'], 'no_cat')
        self.assertEqual(parse_breed_lines('other\n耳朵被挡住')['status'], 'unsupported')

    def test_multiple_candidates_or_unrecognized_code_is_an_error(self):
        for text in ['Bengal / Ragdoll\n长毛','Unicorn\n三只耳朵','']:
            self.assertEqual(parse_breed_lines(text)['status'], 'parse_error')


if __name__=='__main__': unittest.main()
