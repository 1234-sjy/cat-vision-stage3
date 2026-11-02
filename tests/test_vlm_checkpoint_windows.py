import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from catvision.vlm_training import save_adapter_snapshot, BASE_MODEL_ID


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([1.0]))

    def save_pretrained(self, path, **kwargs):
        (Path(path)/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': 'C:/local/base'}))
        (Path(path)/'adapter_model.safetensors').write_bytes(b'tiny test adapter')


class CheckpointFileTests(unittest.TestCase):
    def test_snapshot_syncs_real_files_and_keeps_optimizer_state(self):
        with tempfile.TemporaryDirectory() as folder:
            model = TinyAdapter()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            model.adapter.sum().backward()
            optimizer.step()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            with patch.object(torch.cuda, 'is_available', return_value=False):
                path = save_adapter_snapshot(model, Path(folder)/'step01', optimizer, scheduler,
                    epoch=1, global_step=1, config={'versions': {'torch': torch.__version__}},
                    data_rng_state={'completed': 1})
            self.assertTrue((path/'adapter_model.safetensors').is_file())
            state = torch.load(path/'training-state.pt', weights_only=True)
            self.assertEqual(state['global_step'], 1)
            self.assertTrue(state['optimizer_state_dict']['state'])
            config = json.loads((path/'adapter_config.json').read_text())
            self.assertEqual(config['base_model_name_or_path'], BASE_MODEL_ID)


if __name__ == '__main__':
    unittest.main()
