import torch
from ...utils import log

from comfy.ldm.modules.attention import optimized_attention

def attention_func_error(*args, **kwargs):
    raise ImportError("Selected attention mode not available. Please ensure required packages are installed correctly.")

from .attention_flash import flash_attention

# Sage Attention imports
# using custom ops to avoid graph breaks with torch.compile
try:
    from sageattention import sageattn

    @torch.library.custom_op("wanvideo::sageattn", mutates_args=())
    def sageattn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, tensor_layout: str = "HND"
    ) -> torch.Tensor:
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
        elif q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)

    @sageattn_func.register_fake
    def _(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, tensor_layout="HND"):
        # Return tensor with same shape as q
        return q.clone()

    sageattn_func = torch.ops.wanvideo.sageattn

    def sageattn_func_compiled(q, k, v, attn_mask=None, dropout_p=0, is_causal=False, tensor_layout="HND"):
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
        elif q.dtype == torch.float32:
            return sageattn(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout).to(torch.float32)
        else:
            return sageattn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, tensor_layout=tensor_layout)
except Exception as e:
    log.warning(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        log.warning("sageattention package is not installed, sageattention will not be available")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        log.warning("sageattention DLL loading error, sageattention will not be available")
    sageattn_func = attention_func_error

try:
    from sageattention import sageattn_varlen
    from typing import List

    @torch.library.custom_op("wanvideo::sageattn_varlen", mutates_args=())
    def sageattn_varlen_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_lens: List[int], k_lens: List[int], max_seqlen_q: int, max_seqlen_k: int, dropout_p: float = 0.0, is_causal: bool = False) -> torch.Tensor:
        cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), dim=0)), device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(k_lens), dim=0)), device=q.device, dtype=torch.int32)
        if not (q.dtype == k.dtype == v.dtype):
            return sageattn_varlen(q, k.to(q.dtype), v.to(q.dtype), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)
        elif q.dtype == torch.float32:
            return sageattn_varlen(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal).to(torch.float32)
        else:
            return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=dropout_p, is_causal=is_causal)

    @sageattn_varlen_func.register_fake
    def _(q, k, v, q_lens, k_lens, max_seqlen_q, max_seqlen_k, dropout_p=0.0, is_causal=False):
        # Return tensor with same shape as q
        return q.clone()
    sageattn_varlen_func = torch.ops.wanvideo.sageattn_varlen
except Exception:
    sageattn_varlen_func = attention_func_error

# sage3
try:
    from sageattn3 import sageattn3_blackwell as sageattn_blackwell
except Exception:
    try:
        from sageattn import sageattn_blackwell
    except Exception:
        sageattn_blackwell = attention_func_error

try:
    from ...ultravico.sageattn.core import sage_attention as sageattn_ultravico
    @torch.library.custom_op("wanvideo::sageattn_ultravico", mutates_args=())
    def sageattn_func_ultravico(qkv: List[torch.Tensor], attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, multi_factor: float = 0.9, frame_tokens: int = 1536
    ) -> torch.Tensor:
        return sageattn_ultravico(qkv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, multi_factor=multi_factor, frame_tokens=frame_tokens)

    @sageattn_func_ultravico.register_fake
    def _(qkv, attn_mask=None, dropout_p=0.0, is_causal=False, multi_factor=0.9):
        return torch.empty_like(qkv[0]).contiguous()
    sageattn_func_ultravico = torch.ops.wanvideo.sageattn_ultravico
except Exception:
    sageattn_func_ultravico = attention_func_error


def attention(q, k, v, q_lens=None, k_lens=None, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.,
    softmax_scale=None, q_scale=None, causal=False,  window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
    attention_mode='sdpa', attn_mask=None, transformer_options={}, frame_tokens=1536, heads=128):
    if "flash" in attention_mode:
        return flash_attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic, dtype=dtype, version=2 if attention_mode == 'flash_attn_2' else 3,
        )
    elif attention_mode == 'sageattn_3':
        return sageattn_blackwell(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), per_block_mean=False).transpose(1,2).contiguous()
    elif attention_mode == 'sageattn_varlen':
        return sageattn_varlen_func(q,k,v, q_lens=q_lens, k_lens=k_lens, max_seqlen_k=max_seqlen_k, max_seqlen_q=max_seqlen_q)
    elif attention_mode == 'sageattn_compiled': # for sage versions that allow torch.compile, may be redundant now as other sageattn ops are wrapper in custom ops
        return sageattn_func_compiled(q, k, v, tensor_layout="NHD").contiguous()
    elif attention_mode == 'sageattn':
        return sageattn_func(q, k, v, tensor_layout="NHD").contiguous()
    elif attention_mode == 'sageattn_ultravico':
        return sageattn_func_ultravico([q, k, v], multi_factor=transformer_options.get("ultravico_alpha", 0.9), frame_tokens=frame_tokens).contiguous()
    elif attention_mode == 'comfy':
        return optimized_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), heads=heads, skip_reshape=True)
    else: # sdpa
        if not (q.dtype == k.dtype == v.dtype):
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).to(q.dtype), v.transpose(1, 2).to(q.dtype), attn_mask=attn_mask).transpose(1, 2).contiguous()
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask).transpose(1, 2).contiguous()
