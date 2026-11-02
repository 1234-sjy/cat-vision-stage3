import unittest
from PIL import Image

from catvision.breed_transformer import combine_views, decision
from catvision.traditional import TraditionalVision


class BreedDecisionTests(unittest.TestCase):
    def test_missing_detection_keeps_original_classification(self):
        original={'breed_probs':[0.8,0.2], 'cat_probability':0.95}
        p=combine_views(original,None,0.5)
        self.assertEqual(decision(p,['Sphynx','Siamese'],0.5)['breed'],'Sphynx')

    def test_dog_head_overrides_confident_cat_breed(self):
        p={'breed_probs':[0.999,0.001], 'cat_probability':0.02}
        self.assertEqual(decision(p,['Sphynx','Siamese'],0.5)['status'],'no_cat')
        self.assertIsNone(decision(p,['Sphynx','Siamese'],0.5)['breed'])

    def test_subject_cannot_override_original_species_head(self):
        a={'breed_probs':[0.8,0.2], 'cat_probability':0.01}
        b={'breed_probs':[0.2,0.8], 'cat_probability':0.99}
        p=combine_views(a,b,0.5)
        self.assertEqual(decision(p,['Sphynx','Siamese'],0.5)['status'],'no_cat')

    def test_invalid_fusion_weight_rejected(self):
        with self.assertRaises(ValueError):
            combine_views({'breed_probs':[1.0], 'cat_probability':1},None,1.5)

    def test_pipeline_calls_transformer_when_detector_misses(self):
        class StubClassifier:
            last_throttle_seconds=0
            config={'subject_weight':0.5}
            def predict(self,image,subject=None):
                self.called=True
                return {'status':'ok','has_cat':True,'breed':'Sphynx','breed_zh':'斯芬克斯猫','breed_score':0.9}
        pipeline=object.__new__(TraditionalVision)
        pipeline.classifier_type='swin_t'
        pipeline.classifier=StubClassifier()
        pipeline.extract_subject=lambda image:{'status':'no_cat','has_cat':False,'breed':None,'throttle_seconds':0.2}
        result=pipeline.predict(Image.new('RGB',(30,30)))
        self.assertTrue(pipeline.classifier.called)
        self.assertEqual(result['breed'],'Sphynx')
        self.assertEqual(result['localization_status'],'no_cat')


if __name__=='__main__':
    unittest.main()
