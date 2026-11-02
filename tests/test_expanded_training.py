"""CPU-only contracts for expanded training; no real models or image data."""
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import assess_expanded as assess
from scripts import train_expanded as train


class ExpandedTrainingTests(unittest.TestCase):
    def test_balancing_gives_equal_cat_class_mass_and_twenty_percent_dogs(self):
        rows = [{'breed': 'A', 'is_cat': True}] * 2
        rows += [{'breed': 'B', 'is_cat': True}] * 8
        rows += [{'breed': 'negative_dog', 'is_cat': False}] * 5
        weights = train.balanced_sample_weights(rows, ['A', 'B'])
        self.assertAlmostEqual(sum(weights[:2]), .4)
        self.assertAlmostEqual(sum(weights[2:10]), .4)
        self.assertAlmostEqual(sum(weights[10:]), .2)

    def test_validation_is_not_allowed_to_share_identity_with_train(self):
        rows = [dict(image_id='a', relative_path='a.jpg', breed='A', is_cat=True,
                     split='train', sha256='a' * 64, group='same'),
                dict(image_id='b', relative_path='b.jpg', breed='A', is_cat=True,
                     split='val', sha256='b' * 64, group='same')]
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            train.validate_records(rows, ['A'], {'train', 'val'})

    def test_old_head_rows_follow_names_and_new_rows_keep_seeded_initialization(self):
        class Tiny(torch.nn.Module):
            def __init__(self, names):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.breed_head = torch.nn.Linear(2, len(names))
                self.species_head = torch.nn.Linear(2, 2)
        old_names = list(train.OLD_CLASSES)
        new_names = ['new'] + list(reversed(old_names))
        old = Tiny(old_names)
        with torch.no_grad():
            old.breed_head.weight.copy_(torch.arange(24).reshape(12, 2))
            old.breed_head.bias.copy_(torch.arange(12))
        new = Tiny(new_names)
        fresh_row = new.breed_head.weight[0].detach().clone()
        details = train.initialize_from_v2(new, {'classes': old_names,
                                                'state_dict': old.state_dict()}, new_names)
        self.assertTrue(torch.equal(new.breed_head.weight[0], fresh_row))
        self.assertTrue(torch.equal(new.breed_head.weight[12], old.breed_head.weight[0]))
        self.assertTrue(torch.equal(new.species_head.weight, old.species_head.weight))
        self.assertEqual(len(details['copied_breed_rows']), 12)
        self.assertEqual(new.breed_head.weight.dtype, torch.float32)

    def test_checkpoint_epoch_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'epoch01.pt'
            train.save_checkpoint(path, {'epoch': 1}, immutable=True)
            with self.assertRaises(FileExistsError):
                train.save_checkpoint(path, {'epoch': 2}, immutable=True)
            self.assertEqual(torch.load(path, weights_only=True)['epoch'], 1)

    def test_no_cat_does_not_receive_credit_for_raw_head_label(self):
        predictions = [dict(true_breed='A', is_cat=True, raw_breed='A',
                            breed=None, status='no_cat'),
                       dict(true_breed='B', is_cat=True, raw_breed='A',
                            breed='A', status='ok')]
        metrics = assess.score_group(predictions, ['A', 'B'])
        self.assertEqual(metrics['correct'], 0)
        self.assertEqual(metrics['raw_head_accuracy'], .5)
        self.assertEqual(metrics['macro_f1'], 0)
        self.assertEqual(metrics['status_counts']['no_cat'], 1)
        self.assertIsNone(metrics['dog_false_cat_rate'])

    def test_ambiguity_summary_does_not_change_species_only_decision(self):
        result = assess.classify_probabilities([.55, .45], .7, ['A', 'B'], ['甲', '乙'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['breed'], 'A')
        self.assertAlmostEqual(result['margin'], .1)
        self.assertGreater(result['normalized_entropy'], .9)
        rejected = assess.classify_probabilities([.99, .01], .49, ['A', 'B'], ['甲', '乙'])
        self.assertEqual(rejected['status'], 'no_cat')
        self.assertIsNone(rejected['breed'])


if __name__ == '__main__':
    unittest.main()
