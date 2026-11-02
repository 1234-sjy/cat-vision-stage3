import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torchvision.transforms.functional import pil_to_tensor

from catvision.breed_transformer import BreedTransformer
from catvision.evaluation import inference_fingerprint
from catvision.server import model_files_ready


class TinyHeads(nn.Module):
    """CPU heads with real input-dependent logits and tiny checkpoint state."""
    def __init__(self, classes):
        super().__init__()
        self.register_buffer('breed_bias', torch.zeros(len(classes)))
        self.register_buffer('species_bias', torch.zeros(2))

    def forward(self, tensor):
        edges = torch.stack((tensor[:, 0, 0, 0], tensor[:, 0, 0, -1]), dim=1)
        return edges + self.breed_bias, self.species_bias.expand(len(tensor), -1)


def tiny_payload(breeds=(0.8, 0.2), presence=0.4, names=None):
    return {'classes': names or ['Bengal', 'Siamese'],
            'state_dict': {'breed_bias': torch.tensor(breeds).log(),
                           'species_bias': torch.tensor([1-presence, presence]).log()}}


class EnsembleRuntimeTests(unittest.TestCase):
    def make_runtime(self, *, alpha=0.25, primary=None, secondary=None, primary_tta=True,
                     secondary_tta=True, secondary_name='detail.pt', ensemble=True):
        config = {'classes': ['Bengal', 'Siamese'], 'image_size': 2, 'preprocess': 'letterbox',
                  'tta': primary_tta, 'subject_weight': 0, 'cat_threshold': 0.5}
        if ensemble:
            config['secondary_classifier'] = {'checkpoint': secondary_name, 'weight': alpha,
                                             'image_size': 2, 'preprocess': 'center',
                                             'tta': secondary_tta}
        payloads = [primary or tiny_payload(), secondary or tiny_payload((0.4, 0.6), 0.8)]
        def transform(size, preprocess):
            # Different transforms still consume the same original PIL pixels.
            factor = 1 if preprocess == 'letterbox' else 2
            return lambda image: pil_to_tensor(image).float() / 255 * factor
        with patch('catvision.breed_transformer.BreedSwin', TinyHeads), \
             patch('catvision.breed_transformer.torch.load', side_effect=payloads), \
             patch('catvision.breed_transformer.torch.cuda.is_available', return_value=False), \
             patch('catvision.breed_transformer.eval_transform', side_effect=transform), \
             patch('catvision.breed_transformer.set_low_load_threads'):
            return BreedTransformer(Path('weights/main.pt'), config)

    def setUp(self):
        # Even a CPU checkpoint would query CUDA and sleep without this stub.
        self.checkpoint = patch('catvision.breed_transformer.gpu_checkpoint', return_value=0)
        self.checkpoint.start()
        self.addCleanup(self.checkpoint.stop)

    def test_batch_matches_single_images_and_weights_both_heads(self):
        runtime = self.make_runtime()
        images = [Image.new('RGB', (2, 2), color) for color in [0, (255, 0, 0), (64, 0, 0)]]
        batch = runtime.probabilities_batch(images)
        self.assertAlmostEqual(batch['breed_probs'][0][0], 0.7, places=6)
        self.assertAlmostEqual(batch['breed_probs'][0][1], 0.3, places=6)
        self.assertAlmostEqual(batch['cat_probability'][0], 0.5, places=6)
        for index, image in enumerate(images):
            single = runtime.probabilities(image)
            torch.testing.assert_close(torch.tensor(batch['breed_probs'][index]),
                                       torch.tensor(single['breed_probs']))
            self.assertAlmostEqual(batch['cat_probability'][index], single['cat_probability'])
        self.assertIs(runtime.components[0][0], runtime.model)
        self.assertIs(runtime.components[0][1], runtime.transform)
        self.assertEqual([component[2] for component in runtime.components], [0.75, 0.25])

    def test_each_component_uses_its_own_tta_unless_explicitly_overridden(self):
        runtime = self.make_runtime(primary=tiny_payload((0.5, 0.5)),
                                    secondary=tiny_payload((0.5, 0.5)), primary_tta=False)
        image = Image.new('RGB', (2, 2))
        image.putpixel((0, 0), (255, 0, 0))
        self.assertAlmostEqual(runtime.probabilities(image)['breed_probs'][0],
                               0.673293934, places=6)
        self.assertAlmostEqual(runtime.probabilities(image, tta=False)['breed_probs'][0],
                               0.768493203, places=6)
        self.assertAlmostEqual(runtime.probabilities(image, tta=True)['breed_probs'][0],
                               0.5, places=6)

    def test_weighted_presence_controls_single_breed_decision_at_half_threshold(self):
        image = Image.new('RGB', (2, 2))
        accepted = self.make_runtime().predict(image)
        self.assertEqual(accepted['status'], 'ok')
        self.assertEqual(accepted['breed'], 'Bengal')
        self.assertNotIn('top3', accepted)
        rejected = self.make_runtime(primary=tiny_payload(presence=0.39)).predict(image)
        self.assertEqual(rejected['status'], 'no_cat')
        self.assertIsNone(rejected['breed'])

    def test_secondary_class_order_and_primary_config_order_must_match(self):
        for kwargs in [dict(secondary=tiny_payload(names=['Siamese', 'Bengal'])),
                       dict(primary=tiny_payload(names=['Siamese', 'Bengal']))]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'configuration'):
                self.make_runtime(**kwargs)

    def test_invalid_secondary_weights_are_rejected(self):
        for alpha in [-0.01, 1.01, math.nan, math.inf]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.make_runtime(alpha=alpha)

    def test_secondary_checkpoint_must_be_a_plain_filename(self):
        for name in ['../detail.pt', '..\\detail.pt', '/tmp/detail.pt', '', '.', '..']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.make_runtime(secondary_name=name)

    def test_legacy_single_model_and_empty_batch_are_supported(self):
        runtime = self.make_runtime(ensemble=False)
        result = runtime.probabilities(Image.new('RGB', (2, 2)))
        self.assertAlmostEqual(result['breed_probs'][0], 0.8, places=6)
        self.assertAlmostEqual(result['cat_probability'], 0.4, places=6)
        self.assertEqual(runtime.probabilities_batch([]), {'breed_probs': [], 'cat_probability': []})

    def test_throttle_time_tracks_actual_checkpoint_pause(self):
        runtime = self.make_runtime()
        with patch('catvision.breed_transformer.gpu_checkpoint', return_value=0.125):
            runtime.predict(Image.new('RGB', (2, 2)))
        self.assertAlmostEqual(runtime.last_throttle_seconds, 0.5)
        runtime.predict(Image.new('RGB', (2, 2)))
        self.assertEqual(runtime.last_throttle_seconds, 0)


class EnsembleFilesTests(unittest.TestCase):
    def prepare(self, root):
        files = ['catvision/common.py', 'catvision/traditional.py', 'catvision/geometry.py',
                 'catvision/breed_transformer.py', 'weights/yolo26n.pt', 'weights/sam2.1_t.pt',
                 'weights/main.pt', 'weights/qwen3-vl-4b/model.safetensors',
                 'weights/qwen3-vl-4b/config.json', 'weights/qwen3-vl-4b/preprocessor_config.json',
                 'weights/qwen3-vl-4b/tokenizer.json', 'weights/qwen3-vl-4b/tokenizer_config.json']
        for name in files:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original')
        (root/'weights/qwen3-vl-4b/model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'tensor': 'model.safetensors'}}))
        active = root/'configs/active-classifier.json'
        active.parent.mkdir()
        active.write_text(json.dumps({'checkpoint': 'main.pt', 'architecture': 'swin_t',
                                     'secondary_classifier': {'checkpoint': 'detail.pt', 'weight': 0.25}}))
        return active

    def test_readiness_requires_secondary_checkpoint_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = self.prepare(root)
            self.assertFalse(model_files_ready(root))
            (root/'weights/detail.pt').write_bytes(b'detail')
            self.assertTrue(model_files_ready(root))
            config = json.loads(active.read_text())
            config['secondary_classifier']['checkpoint'] = '../weights/detail.pt'
            active.write_text(json.dumps(config))
            self.assertFalse(model_files_ready(root))

    def test_secondary_checkpoint_contents_change_prediction_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.prepare(root)
            detail = root/'weights/detail.pt'
            detail.write_bytes(b'first detail model')
            before = inference_fingerprint(root, 'traditional')
            detail.write_bytes(b'updated detail model')
            self.assertNotEqual(before, inference_fingerprint(root, 'traditional'))


if __name__ == '__main__':
    unittest.main()
