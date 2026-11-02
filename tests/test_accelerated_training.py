"""CPU-only tests: no checkpoint/data downloads, CUDA initialization, or GPU work."""
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_accelerated as train


class AcceleratedTrainingTests(unittest.TestCase):
    def test_mixup_never_mixes_cat_targets_with_dog_sentinel(self):
        images=torch.tensor([0.,1.,10.,11.]).reshape(4,1,1,1)
        breeds=torch.tensor([2,4,-1,-1]);species=torch.tensor([1,1,0,0])
        mixed,secondary=train.mix_within_species(images,breeds,species,.5)
        self.assertTrue(bool((mixed[:2]<=1).all()))
        self.assertTrue(bool((mixed[2:]>=10).all()))
        self.assertTrue(bool((secondary[:2]>=0).all()))
        self.assertEqual(secondary[2:].tolist(),[-1,-1])

    def source(self):
        return {'classes': ['A'], 'epoch': 18, 'config': {
            'classes_sha256': 'classes', 'train_manifest_sha256': 'manifest',
            'initialization': {'no_project_trained_initialization': True},
            'initialization_sha256': '704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b',
            'no_test_input': True}}

    def test_source_requires_exact_epoch_class_manifest_and_generic_provenance(self):
        train.validate_source(self.source(), ['A'], 'classes', 'manifest', 18)
        for key, value in [('epoch', 17), ('classes', ['B'])]:
            payload = self.source()
            payload[key] = value
            with self.assertRaises(ValueError):
                train.validate_source(payload, ['A'], 'classes', 'manifest', 18)
        for key, value in [('train_manifest_sha256', 'changed'), ('classes_sha256', 'changed'),
                           ('no_test_input', False), ('initialization', {})]:
            payload = self.source()
            payload['config'][key] = value
            with self.assertRaises(ValueError):
                train.validate_source(payload, ['A'], 'classes', 'manifest', 18)

    def test_fastest_demonstrated_batch_excludes_forecast_only_entry(self):
        rows = [{'batch': 32, 'status': 'measured', 'images_per_second': 120},
                {'batch': 64, 'status': 'measured', 'images_per_second': 150},
                {'batch': 128, 'status': 'measured', 'images_per_second': 140},
                {'batch': 160, 'status': 'measured', 'images_per_second': 200,
                 'fits_training_memory_budget': False},
                {'batch': 192, 'status': 'ramp_stopped_memory_forecast'}]
        self.assertEqual(train.choose_batch(rows), 64)
        with self.assertRaises(RuntimeError):
            train.choose_batch([])

    def test_memory_projection_includes_optimizer_and_margin(self):
        result = train.forecast_peak({'batch': 32, 'peak_allocated_bytes': 300}, 64, 100, 50)
        self.assertAlmostEqual(result, 610)

    def test_probe_rng_context_restores_even_on_failure(self):
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                with train.preserve_rng():
                    random.random(), np.random.rand(), torch.rand(100)
                    raise RuntimeError('stop')
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(1), expected[2]))

    def test_dataset_can_be_pickled_for_windows_spawn(self):
        dataset = train.PetDataset([], {}, ['A'], None)
        self.assertEqual(len(pickle.loads(pickle.dumps(dataset))), 0)

    def test_checkpoint_contains_optimizer_state_and_is_weights_only_loadable(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        payload = {'state_dict': train.cpu_tree(model.state_dict()),
                   'optimizer_state_dict': train.cpu_tree(optimizer.state_dict()),
                   'scaler_state_dict': None, 'python_rng_state': random.getstate()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'epoch19.pt'
            train.save_checkpoint(path, payload, immutable=True)
            restored = torch.load(path, weights_only=True)
            self.assertTrue(restored['optimizer_state_dict']['state'])
            with self.assertRaises(FileExistsError):
                train.save_checkpoint(path, payload, immutable=True)

    def test_telemetry_error_is_diagnostic_and_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gpu.jsonl'
            with patch.object(train.subprocess, 'run', side_effect=OSError('missing')) as command:
                telemetry = train.GpuTelemetry(path)
                telemetry._run()
            self.assertEqual(command.call_count, 1)
            row = json.loads(path.read_text())
            self.assertTrue(row['monitoring_stopped'])
            self.assertIn('missing', row['diagnostic'])


if __name__ == '__main__':
    unittest.main()
