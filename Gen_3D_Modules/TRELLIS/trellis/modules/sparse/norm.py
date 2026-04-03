import torch
import torch.nn as nn
from . import SparseTensor
from . import DEBUG

__all__ = [
    'SparseGroupNorm',
    'SparseLayerNorm',
    'SparseGroupNorm32',
    'SparseLayerNorm32',
]


def _cast_sparse_feats(x: SparseTensor, dtype):
    # Safe helper (no assumptions about SparseTensor internals beyond feats)
    return x.replace(x.feats.to(dtype))


class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        dtype = input.feats.dtype
        device = input.feats.device

        # Force params to float32 (IMPORTANT)
        self.weight.data = self.weight.data.to(torch.float32)
        self.bias.data   = self.bias.data.to(torch.float32)

        nfeats = torch.zeros_like(input.feats, dtype=dtype, device=device)

        for k in range(input.shape[0]):
            if DEBUG:
                assert (input.coords[input.layout[k], 0] == k).all()

            bfeats = input.feats[input.layout[k]]

            # convert to float32 for norm
            bfeats = bfeats.to(torch.float32)

            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)

            # restore dtype
            bfeats = bfeats.to(dtype)

            nfeats[input.layout[k]] = bfeats

        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        dtype = input.feats.dtype
        device = input.feats.device

        self.weight.data = self.weight.data.to(torch.float32)
        self.bias.data   = self.bias.data.to(torch.float32)

        nfeats = torch.zeros_like(input.feats, dtype=dtype, device=device)

        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]

            bfeats = bfeats.to(torch.float32)

            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)

            bfeats = bfeats.to(dtype)

            nfeats[input.layout[k]] = bfeats

        return input.replace(nfeats)


# These now become NO-OP wrappers (important)
class SparseGroupNorm32(SparseGroupNorm):
    pass


class SparseLayerNorm32(SparseLayerNorm):
    pass