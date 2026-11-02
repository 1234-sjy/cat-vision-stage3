"""CPU-only LoRA contracts: no tokenizer/model downloads or CUDA initialization."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from catvision import vlm_training as train
from scripts import train_vlm_lora as cli


class FakeSelectedLogitsModel(torch.nn.Module):
    """A differentiable CPU vocabulary projection with Qwen's indexing contract."""
    def __init__(self, logits):
        super().__init__()
        self.full_logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids, attention_mask, logits_to_keep, use_cache):
        assert use_cache is False
        return SimpleNamespace(logits=self.full_logits[:, logits_to_keep, :])


class VlmTrainingTests(unittest.TestCase):
    def record(self, breed='A', cat=True, split='train', suffix='1'):
        return {'image_id': suffix, 'breed': breed if cat else 'negative_dog',
                'is_cat': cat, 'class_index': 0 if cat else -1, 'split': split,
                'relative_path': f'images/{suffix}.jpg', 'sha256': suffix * 64,
                'group': f'group-{suffix}', 'target_text': breed if cat else 'no_cat'}

    def test_mask_uses_last_assistant_header_after_expanded_image_tokens(self):
        # Realistic Qwen IDs: im_start=151644, im_end=151645, image_pad=151655.
        header = [151644, 77091, 198]
        prefix = [151644, 872, 198, 151652, *([151655] * 9), 151653, 190, 151645, 198]
        earlier = [*header, 25, 151645, 198]
        target = [*header, 33, 612, 151645, 198]
        tokens = prefix + earlier + prefix + target
        ids = torch.tensor([tokens + [151643] * 4])
        attention = torch.tensor([[1] * len(tokens) + [0] * 4])
        labels = train.assistant_labels(ids, attention, header, 151645)
        self.assertEqual(labels[labels != -100].tolist(), [33, 612, 151645])
        start = len(prefix + earlier + prefix) + len(header)
        self.assertTrue(bool((labels[0, :start] == -100).all()))
        self.assertTrue(bool((labels[0, start + 3:] == -100).all()))

    def test_mask_handles_unequal_lengths_and_padding_on_either_side(self):
        header = [151644, 77091, 198]
        ids = torch.tensor([[99, *header, 10, 11, 151645, 198, 0],
                            [0, 0, 99, *header, 19, 151645, 198]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 1, 1, 1, 1, 1]])
        labels = train.assistant_labels(ids, attention, header, 151645)
        self.assertEqual(labels[0][labels[0] != -100].tolist(), [10, 11, 151645])
        self.assertEqual(labels[1][labels[1] != -100].tolist(), [19, 151645])
        for bad in ([1, 2, 3], [*header, 151645], [*header, 88]):
            tensor = torch.tensor([bad])
            with self.assertRaises(ValueError):
                train.assistant_labels(tensor, torch.ones_like(tensor), header, 151645)

    def test_selected_causal_ce_and_gradients_equal_full_masked_causal_ce(self):
        generator = torch.Generator().manual_seed(8)
        logits = torch.randn(2, 11, 29, generator=generator, dtype=torch.float64)
        labels = torch.full((2, 11), -100, dtype=torch.long)
        labels[0, 7:10] = torch.tensor([2, 4, 7])
        labels[1, 4:6] = torch.tensor([5, 7])
        full = torch.nn.Parameter(logits.clone())
        full_loss = F.cross_entropy(full[:, :-1].float().reshape(-1, 29),
                                    labels[:, 1:].reshape(-1), ignore_index=-100)
        full_loss.backward()
        selected = FakeSelectedLogitsModel(logits)
        inputs = {'input_ids': torch.zeros((2, 11), dtype=torch.long),
                  'attention_mask': torch.ones((2, 11), dtype=torch.long), 'labels': labels}
        loss = train.assistant_only_loss(selected, inputs)
        loss.backward()
        self.assertTrue(torch.allclose(loss, full_loss, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(selected.full_logits.grad, full.grad, atol=1e-7, rtol=1e-7))
        positions, targets = train.supervised_positions(labels)
        self.assertEqual(positions.tolist(), [3, 4, 6, 7, 8])
        self.assertEqual(int((targets != -100).sum()), 5)

    def test_source_labels_and_frozen_class_indices_cannot_be_replaced(self):
        records = [self.record(), self.record(cat=False, suffix='2')]
        train.validate_records(records, ['A'], 'train')
        for changes in ({'target_text': 'B'}, {'class_index': 1}, {'breed': 'B'},
                        {'split': 'test'}, {'is_cat': 1}, {'relative_path': '../x.jpg'}):
            bad = deepcopy(records)
            bad[0].update(changes)
            with self.assertRaises(ValueError):
                train.validate_records(bad, ['A'], 'train')
        bad_dog = deepcopy(records)
        bad_dog[1]['target_text'] = 'A'
        with self.assertRaises(ValueError):
            train.validate_records(bad_dog, ['A'], 'train')

    def test_train_and_validation_cannot_share_identity_group_hash_or_path(self):
        first = [self.record()]
        other = self.record(split='val', suffix='2')
        train.validate_disjoint(first, [other])
        for key in ('sha256', 'group', 'relative_path'):
            bad = deepcopy(other)
            bad[key] = first[0][key]
            with self.assertRaisesRegex(ValueError, key):
                train.validate_disjoint(first, [bad])

    def test_sampling_assigns_80_percent_to_equal_cat_classes(self):
        rows = [self.record(), self.record(suffix='2'),
                self.record(breed='B', suffix='3'), self.record(cat=False, suffix='4')]
        weights = train.balanced_sample_weights(rows, ['A', 'B'])
        self.assertAlmostEqual(sum(weights[:2]), .4)
        self.assertAlmostEqual(weights[2], .4)
        self.assertAlmostEqual(weights[3], .2)
        self.assertAlmostEqual(sum(weights), 1.)

    def test_probe_only_chooses_measured_fitting_fastest_batch(self):
        rows = [{'batch': 1, 'status': 'measured', 'images_per_second': 3.,
                 'fits_training_memory_budget': True},
                {'batch': 2, 'status': 'measured', 'images_per_second': 5.,
                 'fits_training_memory_budget': True},
                {'batch': 4, 'status': 'measured', 'images_per_second': 7.,
                 'fits_training_memory_budget': False},
                {'batch': 8, 'status': 'ramp_stopped_memory_forecast'}]
        self.assertEqual(train.choose_batch(rows), 2)
        self.assertAlmostEqual(train.forecast_peak({'batch': 1, 'peak_allocated_bytes': 300},
                                                  2, 100, 50), 610.)
        with self.assertRaises(RuntimeError):
            train.choose_batch([])

    def test_frozen_manifest_requires_exact_hash_and_never_accepts_test(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.json'
            path.write_text('[]', encoding='utf-8')
            sha = train.file_sha256(path)
            freeze = {'manifests': {'train.json': {'sha256': sha}}}
            self.assertEqual(train.verify_frozen_manifest(path, freeze), sha)
            path.write_text('[{}]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                train.verify_frozen_manifest(path, freeze)
            with self.assertRaises(ValueError):
                train.verify_frozen_manifest(Path(folder) / 'fresh-test.json', freeze)

    def test_data_image_sha_and_resolved_root_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'x.jpg'
            path.write_bytes(b'unchanged image bytes')
            row = {'relative_path': 'x.jpg', 'sha256': train.file_sha256(path)}
            self.assertEqual(train.verified_image_paths([row], root), [path.resolve()])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                train.verified_image_paths([row], root)

    def test_dataset_and_collate_are_windows_spawn_picklable(self):
        dataset = train.VlmDataset([], [])
        self.assertEqual(len(pickle.loads(pickle.dumps(dataset))), 0)
        self.assertIs(pickle.loads(pickle.dumps(train.identity_collate)), train.identity_collate)

    def test_probe_rng_restores_state_even_when_forward_fails(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        expected = random.random(), np.random.rand(), torch.rand(2)
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                with train.preserve_rng():
                    random.random(), np.random.rand(), torch.rand(4)
                    raise RuntimeError('failed')
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(2), expected[2]))

    def test_durable_state_is_immutable_and_weights_only_loadable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            with patch.object(torch.cuda, 'is_available', return_value=False):
                state = train.rng_state()
            train.atomic_checkpoint(path, {'rng_state': state, 'optimizer_state_dict': {}})
            restored = torch.load(path, weights_only=True)
            self.assertIn('numpy', restored['rng_state'])
            with self.assertRaises(FileExistsError):
                train.atomic_checkpoint(path, {})

    def test_validation_counts_every_cat_and_separates_dog_false_cats(self):
        rows = [{'is_cat': True, 'target_text': 'A', 'prediction': {'status': 'ok', 'breed': 'A', 'has_cat': True}},
                {'is_cat': True, 'target_text': 'B', 'prediction': {'status': 'parse_error', 'breed': None, 'has_cat': None}},
                {'is_cat': False, 'prediction': {'status': 'no_cat', 'breed': None, 'has_cat': False}},
                {'is_cat': False, 'prediction': {'status': 'ok', 'breed': 'A', 'has_cat': True}}]
        metrics = cli.validation_metrics(rows, ['A', 'B'])
        self.assertEqual((metrics['cat_correct'], metrics['cat_total']), (1, 2))
        self.assertEqual(metrics['cat_macro_accuracy'], .5)
        self.assertEqual(metrics['dog']['false_cat_rate'], .5)
        self.assertFalse(cli.is_better(metrics, metrics))
        better = deepcopy(metrics)
        better['dog']['false_cat_rate'] = 0.
        self.assertTrue(cli.is_better(better, metrics))

    def test_cli_defaults_never_point_to_test_or_existing_classifier(self):
        args = cli.parse_args([])
        self.assertEqual(args.validation_manifest.name, 'val-selection.json')
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.microbatch, 'auto')
        self.assertEqual(args.eval_batch_size, 2)
        self.assertEqual(args.adapter_output.name, 'qwen-cat-lora-v4')


if __name__ == '__main__':
    unittest.main()
