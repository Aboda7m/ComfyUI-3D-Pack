import torch
import torch.nn as nn


def _align_input_dtype(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    # Prefer module weight dtype if available
    if hasattr(module, "weight") and module.weight is not None:
        target_dtype = module.weight.dtype
    else:
        target_dtype = x.dtype

    if x.dtype != target_dtype:
        x = x.to(target_dtype)

    return x


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _align_input_dtype(x, self)
        return super().forward(x)


class GroupNorm32(nn.GroupNorm):
    """
    GroupNorm that aligns input dtype to weight dtype
    instead of forcing float32.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _align_input_dtype(x, self)
        return super().forward(x)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()

        # Move channels to last dim
        x = x.permute(0, *range(2, DIM), 1).contiguous()

        # Apply LayerNorm (already dtype-safe)
        x = super().forward(x)

        # Restore original layout
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()

        return x