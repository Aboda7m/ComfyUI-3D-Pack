from typing import *
from enum import Enum
import torch
import torch.nn.functional as F
import math
from .. import SparseTensor
from .. import DEBUG, ATTN

# Safe backend detection
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
    'sparse_serialized_scaled_dot_product_self_attention',
]

class SerializeMode(Enum):
    Z_ORDER = 0
    Z_ORDER_TRANSPOSED = 1
    HILBERT = 2
    HILBERT_TRANSPOSED = 3

# --- Keep your existing calc_serialization function exactly as it is ---
def calc_serialization(
    tensor: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    fwd_indices = []
    bwd_indices = []
    seq_lens = []
    seq_batch_indices = []
    offsets = [0]
    
    if 'vox2seq' not in globals():
        import vox2seq

    serialize_coords = tensor.coords[:, 1:].clone()
    serialize_coords += torch.tensor(shift_window, dtype=torch.int32, device=tensor.device).reshape(1, 3)
    
    if serialize_mode == SerializeMode.Z_ORDER:
        code = vox2seq.encode(serialize_coords, mode='z_order', permute=[0, 1, 2])
    elif serialize_mode == SerializeMode.Z_ORDER_TRANSPOSED:
        code = vox2seq.encode(serialize_coords, mode='z_order', permute=[1, 0, 2])
    elif serialize_mode == SerializeMode.HILBERT:
        code = vox2seq.encode(serialize_coords, mode='hilbert', permute=[0, 1, 2])
    elif serialize_mode == SerializeMode.HILBERT_TRANSPOSED:
        code = vox2seq.encode(serialize_coords, mode='hilbert', permute=[1, 0, 2])
    else:
        raise ValueError(f"Unknown serialize mode: {serialize_mode}")
    
    for bi, s in enumerate(tensor.layout):
        num_points = s.stop - s.start
        num_windows = (num_points + window_size - 1) // window_size
        valid_window_size = num_points / num_windows
        to_ordered = torch.argsort(code[s.start:s.stop])
        if num_windows == 1:
            fwd_indices.append(to_ordered + s.start)
            bwd_idx = torch.zeros_like(to_ordered).scatter_(0, to_ordered, torch.arange(num_points, device=tensor.device))
            bwd_indices.append(bwd_idx + offsets[-1])
            seq_lens.append(num_points)
            seq_batch_indices.append(bi)
            offsets.append(offsets[-1] + num_points)
        else:
            offset = 0
            mids = [(i + 0.5) * valid_window_size + shift_sequence for i in range(num_windows)]
            split = [math.floor(i * valid_window_size + shift_sequence) for i in range(num_windows + 1)]
            bwd_index = torch.zeros((num_points,), dtype=torch.int64, device=tensor.device)
            for i in range(num_windows):
                mid = mids[i]
                v_start, v_end = split[i], split[i+1]
                p_start = math.floor(mid - 0.5 * window_size)
                p_end = p_start + window_size
                fwd = to_ordered[torch.arange(p_start, p_end, device=tensor.device) % num_points]
                fwd_indices.append(fwd + s.start)
                offset += v_start - p_start
                bwd_index.scatter_(0, fwd[v_start-p_start:v_end-p_start], torch.arange(offset, offset + v_end - v_start, device=tensor.device))
                offset += p_end - v_start
            seq_lens.extend([window_size] * num_windows)
            seq_batch_indices.extend([bi] * num_windows)
            bwd_indices.append(bwd_index + offsets[-1])
            offsets.append(offsets[-1] + num_windows * window_size)

    return torch.cat(fwd_indices), torch.cat(bwd_indices), seq_lens, seq_batch_indices

def sparse_serialized_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    serialization_cache_name = f'serialization_{serialize_mode}_{window_size}_{shift_sequence}_{shift_window}'
    cache = qkv.get_spatial_cache(serialization_cache_name)
    if cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_serialization(qkv, window_size, serialize_mode, shift_sequence, shift_window)
        qkv.register_spatial_cache(serialization_cache_name, (fwd_indices, bwd_indices, seq_lens, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = cache

    qkv_feats = qkv.feats[fwd_indices] # [M, 3, H, C]
    H, C = qkv_feats.shape[2], qkv_feats.shape[3]

    # Path A: All windows are the same size (Optimized Batch Path)
    if all([sl == window_size for sl in seq_lens]):
        B, N = len(seq_lens), window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)
            out = xops.memory_efficient_attention(q, k, v)
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)
        else: # SDPA
            q, k, v = qkv_feats.unbind(dim=2)
            # Reshape to [B, H, N, C] for SDPA
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        out = out.reshape(-1, H, C)

    # Path B: Variable sequence lengths
    else:
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=1)
            mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
            out = xops.memory_efficient_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask)[0]
        elif ATTN == 'flash_attn':
            cu_seqlens = torch.cat([torch.zeros(1, device=qkv.device), torch.cumsum(torch.tensor(seq_lens, device=qkv.device), 0)]).int()
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