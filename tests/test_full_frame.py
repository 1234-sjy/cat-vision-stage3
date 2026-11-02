import unittest
from PIL import Image
from catvision.breed_transformer import Letterbox, eval_transform


class FullFrameTests(unittest.TestCase):
    def test_wide_image_preserves_both_edge_features(self):
        image=Image.new('RGB',(400,100),(30,30,30))
        image.paste((255,0,0),(0,0,40,100));image.paste((0,0,255),(360,0,400,100))
        result=Letterbox(200)(image)
        self.assertEqual(result.size,(200,200))
        self.assertEqual(result.getpixel((2,100)),(255,0,0))
        self.assertEqual(result.getpixel((197,100)),(0,0,255))
        self.assertEqual(eval_transform(320,'letterbox')(image).shape,(3,320,320))


if __name__=='__main__': unittest.main()
