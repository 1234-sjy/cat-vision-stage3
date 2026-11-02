"""Equivalent GEMM for Qwen's one-kernel-per-patch Conv3d projection.

The processor already packs every nonoverlapping image patch as one sample.
Its convolution has exactly one output voxel. Flattening the same kernel
therefore computes the same affine map, using a fast matrix multiplication.
"""
from torch import nn
from torch.nn import functional as F


class LinearPatchProjection(nn.Module):
    def __init__(self, convolution: nn.Conv3d):
        super().__init__()
        if (convolution.stride != convolution.kernel_size or any(convolution.padding)
                or convolution.groups != 1 or any(v != 1 for v in convolution.dilation)):
            raise ValueError('Only nonoverlapping, unpadded, ungrouped patch kernels are equivalent')
        self.weight = convolution.weight
        self.bias = convolution.bias
        self.input_features = self.weight[0].numel()

    def forward(self, hidden_states):
        return F.linear(hidden_states.reshape(-1, self.input_features).to(self.weight.dtype),
                        self.weight.flatten(1), self.bias)
