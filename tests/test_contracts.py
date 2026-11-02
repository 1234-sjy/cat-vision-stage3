import io
import unittest

from PIL import Image

from catvision.common import load_image, parse_vlm_output


class ContractTests(unittest.TestCase):
    def test_nonimage_upload_is_rejected(self):
        with self.assertRaises(ValueError):
            load_image(b"this is not a photograph")

    def test_alpha_image_is_composited_on_white(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (2, 1), (0, 0, 0, 0)).save(buffer, format="PNG")
        image = load_image(buffer.getvalue())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_single_breed_is_normalized_from_json_fence(self):
        result = parse_vlm_output('```json\n{"has_cat":true,"breed":"Ragdoll","features":["blue eyes"]}\n```')
        self.assertEqual(result["breed"], "Ragdoll")
        self.assertEqual(result["breed_zh"], "布偶猫")
        self.assertEqual(result["status"], "ok")

    def test_no_cat_cannot_keep_a_breed(self):
        result = parse_vlm_output('{"has_cat":false,"breed":"Persian","features":[]}')
        self.assertEqual(result["status"], "no_cat")
        self.assertIsNone(result["breed"])

    def test_invalid_model_response_is_a_parse_failure(self):
        for text in ['{"has_cat":"false","breed":"Ragdoll"}', '{"has_cat":true,"breed":"unicorn"}', 'no json here']:
            with self.subTest(text=text):
                self.assertEqual(parse_vlm_output(text)["status"], "parse_error")

    def test_missing_required_features_is_retained_as_parse_error(self):
        text = '{"has_cat":true,"breed":"Ragdoll"}'
        result = parse_vlm_output(text)
        self.assertEqual(result["status"], "parse_error")
        self.assertEqual(result["raw_text"], text)


if __name__ == "__main__":
    unittest.main()
