from typing import *
import torch
import torch.nn.functional as F
import math
from .. import SparseTensor
from .. import DEBUG, ATTN

# Safe backend detection for RTX 5080 / PyTorch 2.11
if ATTN == 'xformers':
    try:
        import xformers.ops as xops
    except ImportError:
        ATTN = 'sdpa'
elif ATTN == 'flash_attn':
    try:
        import flash_attn
    except ImportError:
        ATTN = 'sdpa'

__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
]

def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32) // OFFSET[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices

def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'window_partition_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = serialization_spatial_cache

    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]
    H, C = qkv.feats.shape[2], qkv.feats.shape[3]

    # Path A: Fixed Window Size (Optimized Batch)
    if all([seq_len == window_size for seq_len in seq_lens]):
        B, N = len(seq_lens), window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)
            out = xops.memory_efficient_attention(q, k, v)
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)
        else: # SDPA Path
            q, k, v = qkv_feats.unbind(dim=2)
            # Transpose to [B, H, N, C]
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        out = out.reshape(-1, H, C)

    # Path B: Variable Sequence Lengths
    else:
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=1)
            mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
            out = xops.memory_efficient_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask)[0]
        elif ATTN == 'flash_attn':
            cu_seqlens = torch.cat([torch.tensor([0], device=qkv.device), torch.cumsum(torch.tensor(seq_lens, device=qkv.device), dim=0)]).int()
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens))
        else: # SDPA Varlen Fallback
            q_all, k_all, v_all = qkv_feats.unbind(dim=1)
            out_list = []
            curr = 0
            for sl in seq_lens:
                q = q_all[curr:curr+sl].transpose(0, 1).unsqueeze(0) # [1, H, L, C]
                k = k_all[curr:curr+sl].transpose(0, 1).unsqueeze(0)
                v = v_all[curr:curr+sl].transpose(0, 1).unsqueeze(0)
                res = F.scaled_dot_product_attention(q, k, v)
                out_list.append(res.squeeze(0).transpose(0, 1))
                curr += sl
            out = torch.cat(out_list, dim=0)

    return qkv.replace(out[bwd_indices])