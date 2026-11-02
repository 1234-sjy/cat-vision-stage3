from copy import deepcopy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from catvision.vlm_training import rng_state, restore_rng_state, validate_resume_config


class VlmResumeTests(unittest.TestCase):
    def test_restore_replays_next_random_draws_without_starting_a_new_seed(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            random.seed(17)
            np.random.seed(18)
            torch.manual_seed(19)
            random.random(), np.random.rand(), torch.rand(3)
            saved = rng_state()
            expected = random.random(), np.random.rand(3), torch.rand(3)
            random.random(), np.random.rand(7), torch.rand(9)
            restore_rng_state(saved)
            self.assertEqual(random.random(), expected[0])
            self.assertTrue(np.array_equal(np.random.rand(3), expected[1]))
            self.assertTrue(torch.equal(torch.rand(3), expected[2]))

    def test_resume_rejects_changed_data_prompt_or_learning_schedule(self):
        original = {'source': {'aggregate_sha256': 'base'}, 'manifest_sha256': {'train.json': 'train'},
            'classes': ['A'], 'prompt': 'prompt', 'input_image_pixels_max': 262144,
            'epochs_planned': 2, 'effective_batch': 16, 'learning_rate': .0001,
            'lora': {'r': 8, 'alpha': 16, 'dropout': .05, 'target_modules': ['q_proj']}}
        validate_resume_config(original, deepcopy(original))
        for key, value in [('manifest_sha256', {'train.json': 'different'}),
                           ('prompt', 'changed'), ('epochs_planned', 3), ('classes', ['B'])]:
            changed = deepcopy(original)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, key):
                validate_resume_config(original, changed)


if __name__ == '__main__':
    unittest.main()
