from typing import *
import torch
import torch.nn.functional as F
from .. import SparseTensor
from .. import DEBUG, ATTN

# Setup imports based on backend
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

if ATTN == 'sdpa':
    # No extra import needed, F.scaled_dot_product_attention is built-in
    pass

__all__ = [
    'sparse_scaled_dot_product_attention',
]

@overload
def sparse_scaled_dot_product_attention(qkv: SparseTensor) -> SparseTensor: ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]) -> SparseTensor: ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: SparseTensor) -> torch.Tensor: ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: SparseTensor, v: SparseTensor) -> SparseTensor: ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {1: ['qkv'], 2: ['q', 'kv'], 3: ['q', 'k', 'v']}
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}"
    
    # --- Data Extraction Logic (Keep your existing extraction block) ---
    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        device = qkv.device
        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv_feats = qkv.feats  # [T, 3, H, C]
    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        device = q.device
        if isinstance(q, SparseTensor):
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q_feats = q.feats
        else:
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q_feats = q.reshape(N * L, H, C)
        
        if isinstance(kv, SparseTensor):
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv_feats = kv.feats
        else:
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv_feats = kv.reshape(N * L, 2, H, C)
    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        device = q.device
        if isinstance(q, SparseTensor):
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q_feats = q.feats
        else:
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q_feats = q.reshape(N * L, H, CI)
        
        if isinstance(k, SparseTensor):
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k_feats, v_feats = k.feats, v.feats
        else:
            N, L, H, CI = k.shape
            kv_seqlen = [L] * N
            k_feats, v_feats = k.reshape(N * L, H, CI), v.reshape(N * L, v.shape[2], v.shape[3])

    # --- Execution Logic ---
    if ATTN == 'xformers':
        if num_all_args == 1: q_f, k_f, v_f = qkv_feats.unbind(dim=1)
        elif num_all_args == 2: q_f, (k_f, v_f) = q_feats, kv_feats.unbind(dim=1)
        else: q_f, k_f, v_f = q_feats, k_feats, v_feats
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q_f.unsqueeze(0), k_f.unsqueeze(0), v_f.unsqueeze(0), mask)[0]

    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1: out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2: out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        else: out = flash_attn.flash_attn_varlen_func(q_feats, k_feats, v_feats, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))

    elif ATTN == 'sdpa':
        # SDPA requires batching. We process sequences individually to maintain sparse layout integrity.
        if num_all_args == 1: q_all, k_all, v_all = qkv_feats.unbind(dim=1)
        elif num_all_args == 2: q_all, (k_all, v_all) = q_feats, kv_feats.unbind(dim=1)
        else: q_all, k_all, v_all = q_feats, k_feats, v_feats
        
        out_list = []
        q_start, kv_start = 0, 0
        for q_len, kv_len in zip(q_seqlen, kv_seqlen):
            q_s = q_all[q_start:q_start+q_len].transpose(0, 1).unsqueeze(0) # [1, H, L, C]
            k_s = k_all[kv_start:kv_start+kv_len].transpose(0, 1).unsqueeze(0)
            v_s = v_all[kv_start:kv_start+kv_len].transpose(0, 1).unsqueeze(0)
            
            res = F.scaled_dot_product_attention(q_s, k_s, v_s)
            out_list.append(res.squeeze(0).transpose(0, 1)) # [L, H, C]
            q_start += q_len
            kv_start += kv_len
        out = torch.cat(out_list, dim=0)

    # --- Post-Processing ---
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(len(q_seqlen), q_seqlen[0], out.shape[-2], out.shape[-1])