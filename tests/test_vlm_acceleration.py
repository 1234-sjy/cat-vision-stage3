import unittest
import torch
from torch import nn

from catvision.vlm_acceleration import LinearPatchProjection


class PatchProjectionTests(unittest.TestCase):
    def test_nonoverlapping_3d_patch_projection_equals_flattened_linear(self):
        torch.manual_seed(5)
        conv = nn.Conv3d(3, 11, kernel_size=(2, 4, 4), stride=(2, 4, 4)).double()
        x = torch.randn(9, 3 * 2 * 4 * 4, dtype=torch.float64)
        expected = conv(x.reshape(-1, 3, 2, 4, 4)).reshape(-1, 11)
        actual = LinearPatchProjection(conv)(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(LinearPatchProjection(conv).weight.data_ptr(), conv.weight.data_ptr())

    def test_overlapping_or_padded_projection_is_rejected(self):
        for conv in [nn.Conv3d(3, 7, 2, stride=1), nn.Conv3d(3, 7, 2, stride=2, padding=1)]:
            with self.assertRaises(ValueError): LinearPatchProjection(conv)


if __name__ == '__main__': unittest.main()
