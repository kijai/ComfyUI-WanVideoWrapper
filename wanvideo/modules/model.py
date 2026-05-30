# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from ...enhance_a_video.enhance import get_feta_scores
import time
from contextlib import nullcontext

try:
    from ..radial_attention.attn_mask import RadialSpargeSageAttn, RadialSpargeSageAttnDense, MaskMap
except Exception:
    pass

from .attention import attention
import numpy as np
from tqdm import tqdm
import gc

from ...utils import log, get_module_memory_mb
from ...cache_methods.cache_methods import TeaCacheState, MagCacheState, EasyCacheState, relative_l1_distance
from ...multitalk.multitalk import get_attn_map_with_target
from ...echoshot.echoshot import rope_apply_z, rope_apply_c, rope_apply_echoshot
from ...custom_linear import update_lora_step

from ...MTV.mtv import apply_rotary_emb
from comfy.ldm.flux.math import apply_rope1 as apply_rope_comfy1
from comfy.ldm.flux.math import apply_rope as apply_rope_comfy
from comfy import model_management as mm

__all__ = ['WanModel']

def apply_rotary_emb_split(hidden_states, freqs_cis, t_dim):
    """Apply rotary embedding only to the spatial (H/W) dimensions, leaving temporal (T) unchanged."""
    t_part, hw_part = torch.split(hidden_states, [t_dim, hidden_states.shape[-1] - t_dim], dim=-1)
    hw_freqs = freqs_cis[..., t_dim//2:, :, :]

    x_ = hw_part.to(dtype=hw_freqs.dtype).reshape(*hw_part.shape[:-1], -1, 1, 2)
    x_out = hw_freqs[..., 0] * x_[..., 0]
    x_out.addcmul_(hw_freqs[..., 1], x_[..., 1])
    out_hw = x_out.reshape(*hw_part.shape).type_as(hidden_states)

    return torch.cat([t_part, out_hw], dim=-1)

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim, output_dim=None, norm_elementwise_affine=False, norm_eps=1e-5):
        super().__init__()

        output_dim = output_dim or embedding_dim * 2

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(self, x, temb):
        temb = self.linear(self.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x

class FramePackMotioner(nn.Module):#from comfy.ldm.wan.model
    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[1, 2, 16],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            ):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = zip_frame_buckets

        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.drop_mode = drop_mode

    def forward(self, motion_latents, rope_embedder, add_last_motion=2):
        lat_height, lat_width = motion_latents.shape[3], motion_latents.shape[4]
        padd_lat = torch.zeros(motion_latents.shape[0], 16, sum(self.zip_frame_buckets), lat_height, lat_width).to(device=motion_latents.device, dtype=motion_latents.dtype)
        overlap_frame = min(padd_lat.shape[2], motion_latents.shape[2])
        if overlap_frame > 0:
            padd_lat[:, :, -overlap_frame:] = motion_latents[:, :, -overlap_frame:]

        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = sum(self.zip_frame_buckets[:len(self.zip_frame_buckets) - add_last_motion - 1])
            padd_lat[:, :, -zero_end_frame:] = 0

        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -sum(self.zip_frame_buckets):, :, :].split(self.zip_frame_buckets[::-1], dim=2)  # 16, 2 ,1

        # patchfy
        clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
        clean_latents_2x = self.proj_2x(clean_latents_2x)
        l_2x_shape = clean_latents_2x.shape
        clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)
        clean_latents_4x = self.proj_4x(clean_latents_4x)
        l_4x_shape = clean_latents_4x.shape
        clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

        if add_last_motion < 2 and self.drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
            clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        rope_post = rope_embedder.rope_encode_comfy(1, lat_height, lat_width, t_start=-1, device=motion_latents.device, dtype=motion_latents.dtype)
        rope_2x = rope_embedder.rope_encode_comfy(1, lat_height, lat_width, t_start=-3, steps_h=l_2x_shape[-2], steps_w=l_2x_shape[-1], device=motion_latents.device, dtype=motion_latents.dtype)
        rope_4x = rope_embedder.rope_encode_comfy(4, lat_height, lat_width, t_start=-19, steps_h=l_4x_shape[-2], steps_w=l_4x_shape[-1], device=motion_latents.device, dtype=motion_latents.dtype)

        rope = torch.cat([rope_post, rope_2x, rope_4x], dim=1)
        return motion_lat, rope

def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names

def rope_riflex(pos, dim, i, theta, L_test, k, ntk_factor=1.0):
    assert dim % 2 == 0
    if mm.is_device_mps(pos.device) or mm.is_intel_xpu() or mm.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    if ntk_factor != 1.0:
        theta *= ntk_factor

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)

    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if i==0 and k > 0 and L_test:
        omega[k-1] = 0.9 * 2 * torch.pi / L_test

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

class EmbedND_RifleX(nn.Module):
    def __init__(self, dim, theta, axes_dim, num_frames, k):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids, ntk_factor=[1.0,1.0,1.0]):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(
                ids[..., i], 
                self.axes_dim[i], 
                i, #f h w
                self.theta, 
                self.num_frames, 
                self.k,
                ntk_factor[i])
            for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result.abs()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def rope_params(max_seq_len, dim, theta=10000, L_test=25, k=0, freqs_scaling=1.0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
    
    inv_theta_pow *= freqs_scaling
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.autocast(device_type=mm.get_autocast_device(mm.get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs, reverse_time=False):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs, reverse_time=reverse_time)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)

def rope_apply_3d(x, grid_sizes, freqs, reverse_time=False):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        if reverse_time:
            time_freqs = freqs[0][:f].view(f, 1, 1, -1)
            time_freqs = torch.flip(time_freqs, dims=[0])
            time_freqs = time_freqs.expand(f, h, w, -1)
            
            spatial_freqs = torch.cat([
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1)
            
            freqs_i = torch.cat([time_freqs, spatial_freqs], dim=-1).reshape(seq_len, 1, -1)
        else:
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                                dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, num_chunks=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return self._norm(x.to(self.weight.dtype)) * self.weight

    def _norm(self, x):
        return x * (torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            
            chunk = x[:, start_idx:end_idx, :]
            
            norm_factor = torch.rsqrt(chunk.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            output[:, start_idx:end_idx, :] = chunk * norm_factor.to(chunk.dtype) * self.weight

            start_idx = end_idx
            
        return output
    
class WanFusedRMSNorm(nn.RMSNorm):
    def forward(self, x, num_chunks=1):
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return super().forward(x)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = x[:, start_idx:end_idx, :]
            output[:, start_idx:end_idx, :] = super().forward(chunk)
            start_idx = end_idx
            
        return output

class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)

#region selfattn
class WanSelfAttention(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode="sdpa",
                 rms_norm_function="default",
                 kv_dim=None,
                 head_norm=False):
        assert out_features % num_heads == 0
        super().__init__()
        self.dim = min(in_features, out_features)
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        #radial attention
        self.mask_map = None
        self.decay_factor = 0.2
        self.cond_size = None
        self.ref_adapter = None

        # layers
        self.q = nn.Linear(in_features, out_features)
        if kv_dim is not None:
            self.k = nn.Linear(kv_dim, out_features)
            self.v = nn.Linear(kv_dim, out_features)
        else:
            self.k = nn.Linear(in_features, out_features)
            self.v = nn.Linear(in_features, out_features)
        self.o = nn.Linear(in_features, out_features)

        norm_dim = self.head_dim if head_norm else self.dim

        if rms_norm_function=="pytorch":
            self.norm_q = WanFusedRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanFusedRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()
        else:
            self.norm_q = WanRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanRMSNorm(norm_dim, eps=eps) if qk_norm else nn.Identity()

    def qkv_fn(self, x, is_longcat=False):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if is_longcat:
            q = self.q(x).view(b, s, n, d)
            q = self.norm_q(q.float()).to(x.dtype)
            k = self.k(x).view(b, s, n, d)
            k = self.norm_k(k.float()).to(x.dtype)
        else:
            q = self.norm_q(self.q(x).to(self.norm_q.weight.dtype)).to(x.dtype).view(b, s, n, d)
            k = self.norm_k(self.k(x).to(self.norm_k.weight.dtype)).to(x.dtype).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def _qkv_fn_with_rope(self, x, linear_layer, norm_layer, freqs, num_chunks=1, is_longcat=False):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        use_chunked = num_chunks > 1
        if use_chunked:
            out = torch.empty(b, s, n, d, dtype=x.dtype, device=x.device)

            for i, x_chunk in enumerate(x.chunk(num_chunks, dim=1)):
                chunk_size = x_chunk.size(1)
                start_idx = i * (s // num_chunks + (1 if i < s % num_chunks else 0))

                if is_longcat:
                    chunk = linear_layer(x_chunk).view(b, chunk_size, n, d)
                    chunk = norm_layer(chunk.float()).to(x.dtype)
                else:
                    chunk = norm_layer(linear_layer(x_chunk).to(norm_layer.weight.dtype)).to(x.dtype).view(b, chunk_size, n, d)

                freqs_chunk = freqs[:, start_idx:start_idx + chunk_size] if freqs.shape[1] > 1 else freqs
                out[:, start_idx:start_idx + chunk_size] = apply_rope_comfy1(chunk, freqs_chunk)

            return out
        else:
            if is_longcat:
                result = linear_layer(x).view(b, s, n, d)
                result = norm_layer(result.float()).to(x.dtype)
            else:
                result = norm_layer(linear_layer(x).to(norm_layer.weight.dtype)).to(x.dtype).view(b, s, n, d)
            return apply_rope_comfy1(result, freqs)

    def qkv_fn_q_with_rope(self, x, freqs, num_chunks=1, is_longcat=False):
        return self._qkv_fn_with_rope(x, self.q, self.norm_q, freqs, num_chunks, is_longcat)

    def qkv_fn_k_with_rope(self, x, freqs, num_chunks=1, is_longcat=False):
        return self._qkv_fn_with_rope(x, self.k, self.norm_k, freqs, num_chunks, is_longcat)

    def qkv_fn_v(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        return self.v(x).view(b, s, n, d)

    def qkv_fn_ip(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x) + self.q_loras(x).to(self.norm_q.weight.dtype)).to(x.dtype).view(b, s, n, d)
        k = self.norm_k(self.k(x) + self.k_loras(x).to(self.norm_k.weight.dtype)).to(x.dtype).view(b, s, n, d)
        v = (self.v(x) + self.v_loras(x)).view(b, s, n, d)
        return q, k, v

    def forward(self, q, k, v, seq_lens, transformer_options={}, attention_mode_override=None, lynx_ref_feature=None, lynx_ref_scale=1.0, onetoall_ref=None, onetoall_ref_scale=1.0, frame_tokens=1536):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        attention_mode = self.attention_mode
        if attention_mode_override is not None:
            attention_mode = attention_mode_override

        if self.ref_adapter is not None and lynx_ref_feature is not None:
            ref_x = self.ref_adapter(self, q, lynx_ref_feature)

        x = attention(q, k, v, k_lens=seq_lens, attention_mode=attention_mode, heads=self.num_heads, frame_tokens=frame_tokens, transformer_options=transformer_options)

        if self.ref_adapter is not None and lynx_ref_feature is not None:
            x = x.add(ref_x, alpha=lynx_ref_scale)

        if onetoall_ref is not None:
            x = x.add(onetoall_ref, alpha=onetoall_ref_scale)

        # output
        return self.o(x.flatten(2))

    def forward_ip(self, q, k, v, q_ip, k_ip, v_ip, seq_lens, attention_mode_override=None):
        attention_mode = self.attention_mode
        if attention_mode_override is not None:
            attention_mode = attention_mode_override

        # Concatenate main and IP keys/values for main attention
        full_k = torch.cat([k, k_ip], dim=1)
        full_v = torch.cat([v, v_ip], dim=1)
        main_out = attention(q, full_k, full_v, k_lens=seq_lens, attention_mode=attention_mode, heads=self.num_heads)

        cond_out = attention(q_ip, k_ip, v_ip, k_lens=seq_lens, attention_mode=attention_mode, heads=self.num_heads)
        x = torch.cat([main_out, cond_out], dim=1)

        return self.o(x.flatten(2))


    def forward_radial(self, q, k, v, dense_step=False):
        if dense_step:
            x = RadialSpargeSageAttnDense(q, k, v, self.mask_map)
        else:
            x = RadialSpargeSageAttn(q, k, v, self.mask_map, decay_factor=self.decay_factor)
        return self.o(x.flatten(2))


    def forward_multitalk(self, q, k, v, seq_lens, grid_sizes, ref_target_masks):
        x = attention(q, k, v, k_lens=seq_lens, attention_mode=self.attention_mode, heads=self.num_heads)
        x = self.o(x.flatten(2))
        x_ref_attn_map = get_attn_map_with_target(q.type_as(x), k.type_as(x), grid_sizes[0], ref_target_masks=ref_target_masks)
        return x, x_ref_attn_map


    def forward_split(self, q, k, v, seq_lens, grid_sizes, seq_chunks):
        # Split by frames if multiple prompts are provided
        frames, height, width = grid_sizes[0]
        tokens_per_frame = height * width

        seq_chunks_tensor = torch.tensor(seq_chunks, device=q.device, dtype=frames.dtype)
        actual_chunks = torch.minimum(seq_chunks_tensor, frames)
        base_frames_per_chunk = frames // actual_chunks
        extra_frames = frames % actual_chunks

        chunk_indices = torch.arange(actual_chunks, device=q.device)
        chunk_sizes = base_frames_per_chunk + (chunk_indices < extra_frames)
        chunk_starts = torch.cumsum(torch.cat([torch.zeros(1, device=q.device, dtype=torch.long), chunk_sizes[:-1]]), dim=0)
        chunk_ends = chunk_starts + chunk_sizes

        outputs = []
        for i in chunk_indices:
            start_idx = chunk_starts[i] * tokens_per_frame
            end_idx = chunk_ends[i] * tokens_per_frame

            chunk_out = attention(
                q[:, start_idx:end_idx, :, :],
                k[:, start_idx:end_idx, :, :],
                v[:, start_idx:end_idx, :, :],
                k_lens=seq_lens,
                attention_mode=self.attention_mode,
                heads=self.num_heads
            )
            outputs.append(chunk_out)
        x = torch.cat(outputs, dim=1)

        # output
        return self.o(x.flatten(2))

    def nag_attention(self, b, n, d, q, context, nag_context=None):
        k_positive = self.norm_k(self.k(context).to(self.norm_k.weight.dtype)).view(b, -1, n, d).to(q.dtype)
        v_positive = self.v(context).view(b, -1, n, d)
        x_positive = attention(q, k_positive, v_positive, attention_mode=self.attention_mode, heads=self.num_heads)
        del k_positive, v_positive

        k_negative = self.norm_k(self.k(nag_context).to(self.norm_k.weight.dtype)).view(b, -1, n, d).to(q.dtype)
        v_negative = self.v(nag_context).view(b, -1, n, d)
        x_negative = attention(q, k_negative, v_negative, attention_mode=self.attention_mode, heads=self.num_heads)
        del k_negative, v_negative

        return x_positive.flatten(2), x_negative.flatten(2)

    def normalized_attention_guidance(self, x_positive, x_negative,nag_params={}):
        # NAG text attention
        nag_scale = nag_params['nag_scale']
        nag_alpha = nag_params['nag_alpha']
        nag_tau = nag_params['nag_tau']
        inplace = nag_params.get('inplace', True)

        if inplace:
            nag_guidance = x_negative.mul_(nag_scale - 1).neg_().add_(x_positive, alpha=nag_scale)
        else:
            nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
        del x_negative

        norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

        scale = norm_guidance / norm_positive
        torch.nan_to_num_(scale, nan=10.0)
        mask = scale > nag_tau
        del scale

        adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        del norm_positive, norm_guidance

        nag_guidance.mul_(torch.where(mask, adjustment, 1.0))
        del mask, adjustment

        if inplace:
            nag_guidance.sub_(x_positive).mul_(nag_alpha).add_(x_positive)
        else:
            nag_guidance = nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)
        del x_positive

        return nag_guidance

class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device=torch.device("cuda"),
        dtype=torch.float32,
        strength: float = 1.0
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features
        self.strength = strength

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states) * self.strength
        return up_hidden_states.to(orig_dtype)

#region crossattn
class WanT2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, kv_dim=None, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default", head_norm=False):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, kv_dim=kv_dim, rms_norm_function=rms_norm_function, head_norm=head_norm)
        self.attention_mode = attention_mode
        self.ip_adapter = None
        self.k_fusion = None

    def forward(self, x, context, grid_sizes=None, clip_embed=None, audio_proj=None, audio_scale=1.0,
                num_latent_frames=21, nag_params={}, nag_context=None, rope_func="comfy",
                inner_t=None, inner_c=None, cross_freqs=None,
                adapter_proj=None, ip_scale=1.0, orig_seq_len=None, lynx_x_ip=None, lynx_ip_scale=1.0, longcat_num_cond_latents=None, **kwargs):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        s = x.size(1)
        # compute query
        is_longcat = x.shape[-1] == 4096

        if is_longcat:
            if longcat_num_cond_latents is not None and longcat_num_cond_latents > 0:
                num_cond_latents_thw = longcat_num_cond_latents * (s // num_latent_frames)
                x = x[:, num_cond_latents_thw:]
            q = self.norm_q(self.q(x).view(b, -1, n, d))
        else:
            q = self.norm_q(self.q(x).to(self.norm_q.weight.dtype),num_chunks=2 if rope_func == "comfy_chunked" else 1).to(x.dtype).view(b, -1, n, d)

        if nag_context is not None:
            x_positive, x_negative = self.nag_attention(b, n, d, q, context, nag_context)
            del q
            x = self.normalized_attention_guidance(x_positive, x_negative, nag_params)
            del x_positive, x_negative
        else:
            if is_longcat:
                k = self.norm_k(self.k(context).to(self.norm_k.weight.dtype).view(b, -1, n, d)).to(x.dtype)
            else:
                k = self.norm_k(self.k(context).to(self.norm_k.weight.dtype)).to(x.dtype).view(b, -1, n, d)

            v = self.v(context).view(b, -1, n, d)

            #EchoShot rope
            if inner_t is not None and cross_freqs is not None:
                q = rope_apply_z(q, grid_sizes, cross_freqs, inner_t).to(q)
                k = rope_apply_c(k, cross_freqs, inner_c).to(q)

            x = attention(q, k, v, attention_mode=self.attention_mode, heads=self.num_heads).flatten(2)

        if lynx_x_ip is not None and self.ip_adapter is not None and ip_scale !=0:
            lynx_x_ip = self.ip_adapter(self, q, lynx_x_ip)
            x = x.add(lynx_x_ip, alpha=lynx_ip_scale)

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                audio_x = attention(audio_q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads)
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads).flatten(2)
            x = x + audio_x * audio_scale

        # FantasyPortrait adapter attention
        if adapter_proj is not None:
            if len(adapter_proj.shape) == 4:
                q_in = q[:, :orig_seq_len]
                adapter_q = q_in.view(b * num_latent_frames, -1, n, d)
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)

                adapter_x = attention(adapter_q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads)
                adapter_x = adapter_x.view(b, q_in.size(1), n, d)
                adapter_x = adapter_x.flatten(2)
            elif len(adapter_proj.shape) == 3:
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b, -1, n, d)
                adapter_x = attention(q_in, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads)
                adapter_x = adapter_x.flatten(2)
            x[:, :orig_seq_len] = x[:, :orig_seq_len] + adapter_x * ip_scale

        if self.k_fusion is not None:
            # compute target attention
            target_seq = self.pre_attn_norm_fusion(kwargs["target_seq"])
            k_target = self.norm_k_fusion(self.k_fusion(target_seq)).view(b, -1, n, d)
            v_target = self.v_fusion(target_seq).view(b, -1, n, d)

            q = rope_apply(q, grid_sizes, kwargs["src_freqs"])
            k_target = rope_apply(k_target, kwargs["target_grid_sizes"], kwargs["target_freqs"])
            target_x = attention(q, k_target, v_target, k_lens=kwargs["target_seq_lens"], heads=self.num_heads).flatten(2)

            x = x.add(target_x)

        if is_longcat and longcat_num_cond_latents > 0:
            return torch.cat([torch.zeros((b, num_cond_latents_thw, x.shape[-1]), dtype=x.dtype, device=x.device), self.o(x)], dim=1).contiguous()

        return self.o(x)

class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default", **kwargs):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function)
        self.k_img = nn.Linear(in_features, out_features)
        self.v_img = nn.Linear(in_features, out_features)
        self.norm_k_img = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, grid_sizes=None, clip_embed=None, audio_proj=None,
                audio_scale=1.0, num_latent_frames=21, nag_params={}, nag_context=None, rope_func="comfy",
                adapter_proj=None, ip_scale=1.0, orig_seq_len=None, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query
        q = self.norm_q(self.q(x).to(self.norm_q.weight.dtype),num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d).to(x.dtype)

        if nag_context is not None:
            x_positive, x_negative = self.nag_attention(b, n, d, q, context, nag_context)
            x = self.normalized_attention_guidance(x_positive, x_negative, nag_params)
            del x_positive, x_negative
        else:
            # text attention
            k = self.norm_k(self.k(context).to(self.norm_k.weight.dtype)).view(b, -1, n, d).to(x.dtype)
            v = self.v(context).view(b, -1, n, d)
            x = attention(q, k, v, attention_mode=self.attention_mode, heads=self.num_heads).flatten(2)
            del k, v

        #img attention
        if clip_embed is not None:
            k_img = self.norm_k_img(self.k_img(clip_embed).to(self.norm_k_img.weight.dtype)).view(b, -1, n, d).to(x.dtype)
            v_img = self.v_img(clip_embed).view(b, -1, n, d)
            x.add_(attention(q, k_img, v_img, attention_mode=self.attention_mode, heads=self.num_heads).flatten(2))
            del k_img, v_img

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)

                audio_x = attention(audio_q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads)
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads).flatten(2)
            x = x + audio_x * audio_scale

        # FantasyPortrait adapter attention
        if adapter_proj is not None:
            if len(adapter_proj.shape) == 4:
                adapter_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)

                adapter_x = attention(adapter_q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads)
                adapter_x = adapter_x.view(b, q.size(1), n, d)
                adapter_x = adapter_x.flatten(2)
            elif len(adapter_proj.shape) == 3:
                ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(b, -1, n, d)
                ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(b, -1, n, d)
                adapter_x = attention(q, ip_key, ip_value, attention_mode=self.attention_mode, heads=self.num_heads)
                adapter_x = adapter_x.flatten(2)
            x = x + adapter_x * ip_scale
        del q
        return self.o(x)

class WanHuMoCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, kv_dim=None, qk_norm=True, eps=1e-6, attention_mode='sdpa', rms_norm_function="default"):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps, kv_dim=kv_dim, rms_norm_function=rms_norm_function)
        self.attention_mode = attention_mode

    def forward(self, x, context, grid_sizes, **kwargs):

        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x).to(self.norm_q.weight.dtype).to(x.dtype)).view(b, -1, n, d)
        k = self.norm_k(self.k(context).to(self.norm_k.weight.dtype).to(context.dtype)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Handle video spatial structure
        hlen_wlen = grid_sizes[0][1] * grid_sizes[0][2]
        q = q.reshape(-1, hlen_wlen, n, d)

        # Handle audio temporal structure (16 tokens per frame)
        k = k.reshape(-1, 16, n, d)
        v = v.reshape(-1, 16, n, d)

        x_text = attention(q, k, v, attention_mode=self.attention_mode, heads=self.num_heads)
        x_text = x_text.view(b, -1, n, d).flatten(2)

        x = x_text

        return self.o(x)

class AudioCrossAttentionWrapper(nn.Module):
    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, kv_dim=None):
        super().__init__()

        self.audio_cross_attn = WanHuMoCrossAttention(in_features, out_features, num_heads, kv_dim=kv_dim)
        self.norm1_audio = WanLayerNorm(out_features, eps, elementwise_affine=True)

    def forward(self, x, audio, grid_sizes, humo_audio_scale=1.0):
        x = x.to(self.norm1_audio.weight.dtype)
        x = x + self.audio_cross_attn(self.norm1_audio(x), audio, grid_sizes) * humo_audio_scale
        return x

class MTVCrafterMotionAttention(WanSelfAttention):

    def forward(self, x, mo, pe, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            mo: Motion tokens
            pe: 4D RoPE
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(mo)).view(b, n, -1, d)
        v = self.v(mo).view(b, -1, n, d)

        # compute attention
        x = attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=apply_rotary_emb(k, pe).transpose(1, 2),
            v=v,
            heads=self.num_heads,
        )

        return self.o(x.flatten(2))
    

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads,
                qk_norm=True, cross_attn_norm=False, eps=1e-6, attention_mode="sdpa", rope_func="comfy", rms_norm_function="default",
                use_motion_attn=False, use_humo_audio_attn=False, face_fuser_block=False, lynx_ip_layers=None, lynx_ref_layers=None,
                block_idx=0, is_longcat=False):
        super().__init__()
        self.dim = out_features
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        #radial attn
        self.dense_timesteps = 10
        self.dense_block = False
        self.dense_attention_mode = "sageattn"
        self.block_idx = block_idx

        self.kv_cache = None
        self.use_motion_attn = use_motion_attn
        self.has_face_fuser_block = face_fuser_block
        self.ref_attn_k_img = None
        self.ref_attn_v_img = None

        # layers
        self.norm1 = WanLayerNorm(self.dim, eps)
        self.self_attn = WanSelfAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode, rms_norm_function=rms_norm_function,
                                          head_norm=is_longcat)

        # MTV Crafter motion attn
        if self.use_motion_attn:
            self.norm4 = WanLayerNorm(out_features, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.motion_attn = MTVCrafterMotionAttention(in_features, out_features, num_heads, qk_norm, eps, self.attention_mode)

        if cross_attn_type != "no_cross_attn":
            self.norm3 = WanLayerNorm(out_features, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](in_features, out_features, num_heads, qk_norm, eps, rms_norm_function=rms_norm_function,
                                                                          head_norm=is_longcat)
        self.norm2 = WanLayerNorm(self.dim, eps)

        if not is_longcat:
            self.ffn = nn.Sequential(nn.Linear(in_features, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn2_dim, out_features))
        else:
            from ...LongCat.layers import FeedForwardSwiGLU
            mlp_ratio = 4
            self.ffn = FeedForwardSwiGLU(dim=self.dim, hidden_dim=int(self.dim * mlp_ratio))

        # modulation
        if not is_longcat:
            self.modulation = nn.Parameter(torch.randn(1, 6, out_features) / in_features**0.5)
        else:
            adaln_tembed_dim = 512
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * self.dim, bias=True))

        self.seg_idx = None

        # HuMo audio cross-attn
        if use_humo_audio_attn:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(in_features, out_features, num_heads, qk_norm, eps, kv_dim=1536)

        if face_fuser_block:
            from .wananimate.face_blocks import FaceBlock
            self.fuser_block = FaceBlock(self.dim, num_heads)

        # Lynx
        self.ref_adapter = None
        if lynx_ref_layers == "full":
            from ...lynx.modules import WanLynxRefAttention
            self.self_attn.ref_adapter = WanLynxRefAttention(dim=self.dim)
        if lynx_ip_layers == "full":
            from ...lynx.modules import WanLynxIPCrossAttention
            self.cross_attn.ip_adapter = WanLynxIPCrossAttention(cross_attention_dim=self.dim, dim=self.dim, n_registers=16)
        elif lynx_ip_layers == "lite":
            from ...lynx.modules import WanLynxIPCrossAttention
            if self.block_idx % 2 == 0:
                self.cross_attn.ip_adapter = WanLynxIPCrossAttention(cross_attention_dim=2048, dim=self.dim, n_registers=0, bias=False)

    def get_mod(self, e, modulation):
        if e.dim() == 3:
            if e.shape[-1] == 512:
                e = self.modulation(e)
                return e.unsqueeze(2).chunk(6, dim=-1)
            return (modulation + e).chunk(6, dim=1) # 1, 6, dim
        elif e.dim() == 4:
            e_mod = modulation.unsqueeze(2) + e
            return [ei.squeeze(1) for ei in e_mod.unbind(dim=1)]


    def modulate(self, norm_x, shift_msa, scale_msa, seg_idx=None):
        """
        Modulate x with shift and scale. If seg_idx is provided, apply segmented modulation.
        """
        if seg_idx is not None:
            parts = []
            for i in range(2):
                part = torch.addcmul(
                    shift_msa[:, i:i + 1],
                    norm_x[:, seg_idx[i]:seg_idx[i + 1]],
                    1 + scale_msa[:, i:i + 1]
                )
                parts.append(part)
            norm_x = torch.cat(parts, dim=1)
            return norm_x
        else:
            return torch.addcmul(shift_msa, norm_x, 1 + scale_msa)

    def ffn_chunked(self, mod_x, num_chunks=4):
        seq_len = mod_x.shape[1]
        if seq_len <= 8192 or num_chunks <= 1:
            return self.ffn(mod_x)
        return torch.cat([self.ffn(chunk.contiguous()) for chunk in mod_x.chunk(num_chunks, dim=1)], dim=1)

    #region attention forward
    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, current_step,
        last_step=False,
        clip_embed=None,
        seq_chunks=0, #comfy chunked cross-attn
        chunked_self_attention=False,
        camera_embed=None, #ReCamMaster
        audio_proj=None, audio_scale=1.0, #fantasytalking
        num_latent_frames=21,
        original_seq_len=None,
        enhance_enabled=False, #feta
        nag_params={}, nag_context=None, #normalized attention guidance
        multitalk_audio_embedding=None, ref_target_masks=None, human_num=0, #multitalk
        inner_t=None, inner_c=None, cross_freqs=None, #echoshot
        x_ip=None, e_ip=None, freqs_ip=None, ip_scale=1.0, #stand-in
        adapter_proj=None, #fantasyportrait
        reverse_time=False,
        zero_timestep=False, #s2v zero timestep
        mtv_motion_tokens=None, mtv_motion_rotary_emb=None, mtv_strength=1.0, mtv_freqs=None, #mtv crafter
        humo_audio_input=None, humo_audio_scale=1.0, #humo audio
        lynx_x_ip=None, lynx_ref_feature=None, lynx_ip_scale=1.0, lynx_ref_scale=1.0, #lynx
        x_ovi=None, e_ovi=None, freqs_ovi=None, context_ovi=None, seq_lens_ovi=None, grid_sizes_ovi=None,
        longcat_num_cond_latents=0, longcat_avatar_options=None, #longcat image cond amount
        x_onetoall_ref=None, onetoall_freqs=None, onetoall_ref=None, onetoall_ref_scale=1.0, #one-to-all
        e_tr=None, tr_num=0, tr_start=0, #token replacement
        attention_mode_override=None, frame_tokens=None, transformer_options={}
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        input_dtype = x.dtype
        B, N, C = x.shape
        T = num_latent_frames
        is_longcat = C == 4096

        zero_timestep = len(e) == 2
        if zero_timestep: #s2v zero timestep
            self.seg_idx = e[1]
            self.seg_idx = min(max(0, self.seg_idx), x.size(1))
            self.seg_idx = [0, self.seg_idx, x.size(1)]
            e = e[0]

        use_token_replace = False
        if e_tr is not None and tr_num > 0:
            tr_shift_msa, tr_scale_msa, tr_gate_msa, tr_shift_mlp, tr_scale_mlp, tr_gate_mlp = self.get_mod(e_tr.to(x.device), self.modulation)
            use_token_replace = True
            tr_start = tr_start or 0
            tr_end = tr_start + (tr_num or 0)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod(e.to(x.device), self.modulation)
        if multitalk_audio_embedding is not None and is_longcat:
            audio_shift_mca, audio_scale_mca, audio_gate_mca = self.audio_modulation(e[:, longcat_num_cond_latents:]).unsqueeze(2).chunk(3, dim=-1)
        del e

        if is_longcat:
            input_x = self.modulate(self.norm1(x.view(B, T, -1, C).to(shift_msa.dtype)), shift_msa, scale_msa, seg_idx=self.seg_idx).to(input_dtype).view(B, N, C)
        elif use_token_replace:
            norm_x = self.norm1(x.to(shift_msa.dtype))
            input_x = torch.cat([
                torch.addcmul(shift_msa, norm_x[:, :tr_start], 1 + scale_msa),           # before replace → T
                torch.addcmul(tr_shift_msa, norm_x[:, tr_start:tr_end], 1 + tr_scale_msa), # replace segment → t=0
                torch.addcmul(shift_msa, norm_x[:, tr_end:], 1 + scale_msa)              # after replace → T
            ], dim=1).to(input_dtype)
        else:
            input_x = self.modulate(self.norm1(x.to(shift_msa.dtype)), shift_msa, scale_msa, seg_idx=self.seg_idx).to(input_dtype)

        del shift_msa, scale_msa

        if x_ip is not None:
            shift_msa_ip, scale_msa_ip, gate_msa_ip, shift_mlp_ip, scale_mlp_ip, gate_mlp_ip = self.get_mod(e_ip.to(x.device), self.modulation)
            input_x_ip = self.modulate(self.norm1(x_ip), shift_msa_ip, scale_msa_ip)
            self.cond_size = input_x_ip.shape[1]
            input_x = torch.concat([input_x, input_x_ip], dim=1)
            self.kv_cache = None

        if x_ovi is not None:
            shift_msa_ovi, scale_msa_ovi, gate_msa_ovi, shift_mlp_ovi, scale_mlp_ovi, gate_mlp_ovi = self.get_mod(e_ovi.to(x.device), self.audio_block.modulation)
            input_x_ovi = self.modulate(self.audio_block.norm1(x_ovi), shift_msa_ovi, scale_msa_ovi)

        if camera_embed is not None:
            # encode ReCamMaster camera
            camera_embed = self.cam_encoder(camera_embed.to(x))
            camera_embed = camera_embed.repeat(1, 2, 1)
            camera_embed = camera_embed.unsqueeze(2).unsqueeze(3).repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1)
            camera_embed = rearrange(camera_embed, 'b f h w d -> b (f h w) d')
            input_x += camera_embed

        # self-attention
        x_ref_attn_map = None

        # self-attention variables
        q_ip = k_ip = v_ip = None

        if lynx_ref_feature is None and self.self_attn.ref_adapter is not None:
            lynx_ref_feature = input_x

        onetoall_ref = None
        if x_onetoall_ref is not None:
            b, s, n, d = *x_onetoall_ref.shape[:2], self.self_attn.num_heads, self.self_attn.head_dim
            h_dim = w_dim = 2 * (self.head_dim  // 6)
            t_dim = self.head_dim - h_dim - w_dim

            q_ref = self.self_attn.norm_q(self.self_attn.q(input_x)).to(input_x.dtype).view(b, N, n, d)
            q_ref = apply_rotary_emb_split(q_ref, freqs, t_dim) # Apply split rotary embedding (only to H/W dimensions, leaving T unchanged)

            k_ref = self.ref_attn_norm_k_img(self.ref_attn_k_img(x_onetoall_ref).to(self.ref_attn_norm_k_img.weight.dtype)).to(x_onetoall_ref.dtype).view(b, s, n, d)
            k_ref = apply_rotary_emb_split(k_ref, onetoall_freqs, t_dim)

            v_ref = self.ref_attn_v_img(x_onetoall_ref).view(b, s, n, d)

            onetoall_ref = attention(q_ref, k_ref, v_ref, k_lens=seq_lens, attention_mode=self.attention_mode, heads=self.num_heads)
            del q_ref, k_ref, v_ref

        #RoPE and QKV computation
        if inner_t is not None:
            #query, key, value
            q, k, v = self.self_attn.qkv_fn(input_x)
            q=rope_apply_echoshot(q, grid_sizes, freqs, inner_t).to(q)
            k=rope_apply_echoshot(k, grid_sizes, freqs, inner_t).to(k)
        elif x_ip is not None and self.kv_cache is None:
            # First pass - separate main and IP components
            x_main, x_ip_input = input_x[:, : -self.cond_size], input_x[:, -self.cond_size :]
            # Compute QKV for main content
            if self.rope_func == "comfy":
                q = self.self_attn.qkv_fn_q_with_rope(x_main, freqs)
                k = self.self_attn.qkv_fn_k_with_rope(x_main, freqs)
                v = self.self_attn.qkv_fn_v(x_main)
            elif self.rope_func == "comfy_chunked":
                q = self.self_attn.qkv_fn_q_with_rope(x_main, freqs, num_chunks=2)
                k = self.self_attn.qkv_fn_k_with_rope(x_main, freqs, num_chunks=2)
                v = self.self_attn.qkv_fn_v(x_main)
            # Compute QKV for IP content
            if "comfy" in self.rope_func:
                q_ip, k_ip, v_ip = self.self_attn.qkv_fn_ip(x_ip_input)
                q_ip, k_ip = apply_rope_comfy(q_ip, k_ip, freqs_ip)
        else:
            if "comfy" in self.rope_func:
                num_chunks = 2 if self.rope_func == "comfy_chunked" else 1
                q = self.self_attn.qkv_fn_q_with_rope(input_x, freqs, num_chunks=num_chunks, is_longcat=is_longcat)
                k = self.self_attn.qkv_fn_k_with_rope(input_x, freqs, num_chunks=num_chunks, is_longcat=is_longcat)
                v = self.self_attn.qkv_fn_v(input_x)
            else:
                q, k, v = self.self_attn.qkv_fn(input_x)
                if self.rope_func == "mocha":
                    from ...mocha.nodes import rope_apply_mocha
                    q = rope_apply_mocha(q, grid_sizes, freqs)
                    k = rope_apply_mocha(k, grid_sizes, freqs)
                else:
                    q = rope_apply(q, grid_sizes, freqs, reverse_time=reverse_time)
                    k = rope_apply(k, grid_sizes, freqs, reverse_time=reverse_time)

        del input_x

        if x_ovi is not None:
            q_ovi, k_ovi, v_ovi = self.audio_block.self_attn.qkv_fn(input_x_ovi)
            q_ovi = rope_apply(q_ovi, grid_sizes_ovi, freqs_ovi)
            k_ovi = rope_apply(k_ovi, grid_sizes_ovi, freqs_ovi)
            y_ovi = self.audio_block.self_attn.forward(q_ovi, k_ovi, v_ovi, seq_lens_ovi)
            x_ovi = x_ovi.addcmul(y_ovi, gate_msa_ovi)
            del input_x_ovi, y_ovi, gate_msa_ovi

        # FETA
        if enhance_enabled:
            feta_scores = get_feta_scores(q, k)

        if self.attention_mode == "sageattn_3" and attention_mode_override is None:
            if current_step != 0 and not last_step:
                attention_mode_override = "sageattn"

        #self-attention
        split_attn = (context is not None
                      and (context.shape[0] > 1 or (clip_embed is not None and clip_embed.shape[0] > 1))
                      and x.shape[0] == 1
                      and inner_t is None
                      and x_ip is None  # Don't split when using IP-Adapter
                      )
        if split_attn and chunked_self_attention:
            y = self.self_attn.forward_split(q, k, v, seq_lens, grid_sizes, seq_chunks)
        elif ref_target_masks is not None: #multi/infinite talk
            y, x_ref_attn_map = self.self_attn.forward_multitalk(q, k, v, seq_lens, grid_sizes, ref_target_masks)
        elif self.attention_mode == "radial_sage_attention" or attention_mode_override is not None and attention_mode_override == "radial_sage_attention":
            if self.dense_block or self.dense_timesteps is not None and current_step < self.dense_timesteps:
                if self.dense_attention_mode == "sparse_sage_attn":
                    y = self.self_attn.forward_radial(q, k, v, dense_step=True)
                else:
                    y = self.self_attn.forward(q, k, v, seq_lens, attention_mode_override=attention_mode_override)
            else:
                y = self.self_attn.forward_radial(q, k, v, dense_step=False)
        elif x_ip is not None and self.kv_cache is None: #stand-in
            # First pass: cache IP keys/values and compute attention
            self.kv_cache = {"k_ip": k_ip.detach(), "v_ip": v_ip.detach()}
            y = self.self_attn.forward_ip(q, k, v, q_ip, k_ip, v_ip, seq_lens)
        elif self.kv_cache is not None:
            # Subsequent passes: use cached IP keys/values
            k_ip = self.kv_cache["k_ip"]
            v_ip = self.kv_cache["v_ip"]
            full_k = torch.cat([k, k_ip], dim=1)
            full_v = torch.cat([v, v_ip], dim=1)
            y = self.self_attn.forward(q, full_k, full_v, seq_lens, attention_mode_override=attention_mode_override)
        elif is_longcat and longcat_num_cond_latents > 0:
            if longcat_num_cond_latents == 1:
                num_cond_latents_thw = longcat_num_cond_latents * (N // num_latent_frames)
                # process the noise tokens
                x_noise = self.self_attn.forward(q[:, num_cond_latents_thw:].contiguous(), k, v, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options)
                # process the condition tokens
                x_cond = self.self_attn.forward(
                    q[:, :num_cond_latents_thw].contiguous(),
                    k[:, :num_cond_latents_thw].contiguous(),
                    v[:, :num_cond_latents_thw].contiguous(),
                    seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options)
                # merge x_cond and x_noise
                y = torch.cat([x_cond, x_noise], dim=1).contiguous()
            elif longcat_num_cond_latents > 1: # video continuation
                num_ref_latents_thw = (N // num_latent_frames)
                num_cond_latents_thw = longcat_num_cond_latents * (N // num_latent_frames)
                if not longcat_num_cond_latents == num_latent_frames:
                    # process the noise tokens
                    q_noise = q[:, num_cond_latents_thw:].contiguous()
                    start_noise, end_noise, num_noisy_frames = 0, 0, num_latent_frames - longcat_num_cond_latents
                    mask_frame_range = longcat_avatar_options["ref_mask_frame_range"]
                    ref_img_index = longcat_avatar_options["ref_frame_index"]
                    num_ref_latents = 1
                    if mask_frame_range is not None and mask_frame_range > 0:
                        start_noise = ref_img_index - mask_frame_range - longcat_num_cond_latents + num_ref_latents
                        end_noise   = ref_img_index + mask_frame_range - longcat_num_cond_latents + num_ref_latents + 1

                    if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                        # remove attention with the reference image in the target range, preventing repeated actions.

                        start_pos = start_noise * (N // num_latent_frames)
                        end_pos   = end_noise * (N // num_latent_frames)

                        q_noise_front = q_noise[:, :start_pos].contiguous()
                        q_noise_maskref = q_noise[:, start_pos:end_pos].contiguous()
                        q_noise_back = q_noise[:, end_pos:].contiguous()
                        k_non_ref = k[:, num_ref_latents_thw:].contiguous()
                        v_non_ref = v[:, num_ref_latents_thw:].contiguous()

                        x_noise_front = self.self_attn.forward(q_noise_front, k, v, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options) # q_front has attention with ref + cond + noisy
                        x_noise_back = self.self_attn.forward(q_noise_back, k, v, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options) # q_back has attention with ref + cond + noisy
                        x_noise_maskref = self.self_attn.forward(q_noise_maskref, k_non_ref, v_non_ref, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options) # q_mask has attention with cond+noisy
                        x_noise = torch.cat([x_noise_front, x_noise_maskref, x_noise_back], dim=1).contiguous()
                    else:
                        x_noise = self.self_attn.forward(q_noise, k, v, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options)
                # process the condition tokens
                q_ref = q[:, :num_ref_latents_thw].contiguous()
                k_ref = k[:, :num_ref_latents_thw].contiguous()
                v_ref = v[:, :num_ref_latents_thw].contiguous()
                q_cond = q[:, num_ref_latents_thw:num_cond_latents_thw].contiguous()
                k_cond = k[:, num_ref_latents_thw:num_cond_latents_thw].contiguous()
                v_cond = v[:, num_ref_latents_thw:num_cond_latents_thw].contiguous()
                x_ref = self.self_attn.forward(q_ref, k_ref, v_ref, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options)
                x_cond = self.self_attn.forward(q_cond, k_cond, v_cond, seq_lens, attention_mode_override=attention_mode_override, transformer_options=transformer_options)

                # merge x_cond and x_noise
                y = torch.cat([x_ref, x_cond, x_noise], dim=1).contiguous()
        else:
            y = self.self_attn.forward(q, k, v, seq_lens, lynx_ref_feature=lynx_ref_feature, lynx_ref_scale=lynx_ref_scale,
                                       onetoall_ref=onetoall_ref, onetoall_ref_scale=onetoall_ref_scale, attention_mode_override=attention_mode_override, transformer_options=transformer_options, frame_tokens=frame_tokens)

        del q, k, v

        # FETA
        if enhance_enabled:
            y.mul_(feta_scores)

        # ReCamMaster
        if camera_embed is not None:
            y = self.projector(y)

        # Stand-in
        if x_ip is not None:
            y, y_ip = (
                y[:, : -self.cond_size],
                y[:, -self.cond_size :],
            )

        # S2V
        if zero_timestep:
            z = []
            for i in range(2):
                z.append(y[:, self.seg_idx[i]:self.seg_idx[i + 1]] * gate_msa[:, i:i + 1])
            y = torch.cat(z, dim=1)
            x = x.add(y)
        else:
            if is_longcat:
                x = x + (y.view(B, -1, N//T, C).float() * gate_msa).to(input_dtype).view(B, -1, C)
            elif use_token_replace:
                x = x + torch.cat([
                    y[:, :tr_start] * gate_msa,
                    y[:, tr_start:tr_end] * tr_gate_msa,
                    y[:, tr_end:] * gate_msa
                ], dim=1).to(input_dtype)
            else:
                x.addcmul_(y, gate_msa)
        del y, gate_msa

        # cross-attention & ffn function
        if context is not None:
            if x_ovi is not None:
                #audio
                og_ovi_x = x_ovi
                x_ovi = x_ovi + self.audio_block.cross_attn(self.audio_block.norm3(x_ovi), context_ovi, grid_sizes_ovi, 
                                        src_freqs=freqs_ovi,
                                        target_seq=x, 
                                        target_seq_lens=seq_lens, 
                                        target_grid_sizes=grid_sizes, 
                                        target_freqs=freqs)
                y = self.audio_block.ffn(torch.addcmul(shift_mlp_ovi, self.audio_block.norm2(x_ovi), 1 + scale_mlp_ovi))
                x_ovi = x_ovi.addcmul(y, gate_mlp_ovi)

                # video
                x = x + self.cross_attn(self.norm3(x), context, grid_sizes,
                                        src_freqs=freqs,
                                        target_seq=og_ovi_x, 
                                        target_seq_lens=seq_lens_ovi, 
                                        target_grid_sizes=grid_sizes_ovi, 
                                        target_freqs=freqs_ovi)
            elif split_attn:
                if nag_context is not None:
                    raise NotImplementedError("nag_context is not supported in split_cross_attn_ffn")
                x = self.split_cross_attn_ffn(x, context, shift_mlp, scale_mlp, gate_mlp, clip_embed, grid_sizes)
                return x, x_ip, lynx_ref_feature, x_ovi
            else:
                x += self.cross_attn(self.norm3(x.to(self.norm3.weight.dtype)).to(input_dtype), context, grid_sizes, clip_embed=clip_embed, audio_proj=audio_proj, audio_scale=audio_scale,
                                    num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context,
                                    rope_func=self.rope_func, inner_t=inner_t, inner_c=inner_c, cross_freqs=cross_freqs,
                                    adapter_proj=adapter_proj, ip_scale=ip_scale, orig_seq_len=original_seq_len, lynx_x_ip=lynx_x_ip, lynx_ip_scale=lynx_ip_scale, longcat_num_cond_latents=longcat_num_cond_latents).to(input_dtype)
                # MultiTalk
                if multitalk_audio_embedding is not None and not isinstance(self, VaceWanAttentionBlock):

                    if is_longcat:
                        audio_output_cond, x_audio = self.audio_cross_attn(self.norm_x(x.to(self.norm_x.weight.dtype)).to(input_dtype), multitalk_audio_embedding, num_latent_frames=num_latent_frames,
                                                        num_cond_latents=longcat_num_cond_latents, x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                        x_audio = self.modulate(self.norm1(x_audio.view(B, T-longcat_num_cond_latents, -1, C).to(audio_shift_mca.dtype)), audio_shift_mca, audio_scale_mca, seg_idx=self.seg_idx).to(input_dtype).view(B, -1, C)
                        x_audio = (x_audio.view(B, T-longcat_num_cond_latents, -1, C).float() * audio_gate_mca).to(input_dtype).view(B, -1, C)
                        if audio_output_cond is not None:
                            x_audio = torch.cat([audio_output_cond, x_audio], dim=1).contiguous()
                    else:
                        x_audio = self.audio_cross_attn(self.norm_x(x.to(self.norm_x.weight.dtype)).to(input_dtype), encoder_hidden_states=multitalk_audio_embedding,
                                                    shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                    x.add_(x_audio, alpha=audio_scale)
                    del x_audio

                # MTV-Crafter Motion Attention
                if self.use_motion_attn and mtv_motion_tokens is not None and mtv_motion_rotary_emb is not None:
                    x_motion = self.motion_attn(self.norm4(x), mtv_motion_tokens, mtv_motion_rotary_emb, grid_sizes, mtv_freqs)
                    x = x.add(x_motion, alpha=mtv_strength)

                # HuMo Audio Cross-Attention
                if humo_audio_input is not None:
                    x = self.audio_cross_attn_wrapper(x, humo_audio_input, grid_sizes, humo_audio_scale)


        # ffn
        if self.rope_func == "comfy_chunked" and not is_longcat and not use_token_replace and not zero_timestep:
            mod_x = torch.addcmul(shift_mlp, self.norm2(x.to(shift_mlp.dtype)), 1 + scale_mlp)
            x_ffn = self.ffn_chunked(mod_x)
        else:
            if zero_timestep:
                norm2_x = self.norm2(x)
                parts = []
                for i in range(2):
                    parts.append(norm2_x[:, self.seg_idx[i]:self.seg_idx[i + 1]] *
                                (1 + scale_mlp[:, i:i + 1]) + shift_mlp[:, i:i + 1])
                norm2_x = torch.cat(parts, dim=1)
                x_ffn = self.ffn(norm2_x)
            else:
                if is_longcat:
                    mod_x = torch.addcmul(shift_mlp, self.norm2(x.view(B, -1, N//T, C).float()), 1 + scale_mlp).view(B, -1, C)
                elif use_token_replace:
                    norm2_x = self.norm2(x.to(shift_mlp.dtype))
                    mod_x = torch.cat([
                        torch.addcmul(shift_mlp, norm2_x[:, :tr_start], 1 + scale_mlp),
                        torch.addcmul(tr_shift_mlp, norm2_x[:, tr_start:tr_end], 1 + tr_scale_mlp),
                        torch.addcmul(shift_mlp, norm2_x[:, tr_end:], 1 + scale_mlp)
                    ], dim=1)
                else:
                    mod_x = torch.addcmul(shift_mlp, self.norm2(x.to(shift_mlp.dtype)), 1 + scale_mlp)

                del shift_mlp, scale_mlp
                x_ffn = self.ffn_chunked(mod_x.to(input_dtype), num_chunks=2 if is_longcat else 1)
                del mod_x

        # gate_mlp
        if zero_timestep:
            z = []
            for i in range(2):
                z.append(x_ffn[:, self.seg_idx[i]:self.seg_idx[i + 1]] * gate_mlp[:, i:i + 1])
            x_ffn = torch.cat(z, dim=1)
            x = x.add(x_ffn)
        else:
            if is_longcat:
                x = x + (gate_mlp * x_ffn.view(B, -1, N//T, C).float()).to(input_dtype).view(B, -1, C)
            elif use_token_replace:
                x = x + torch.cat([
                    x_ffn[:, :tr_start] * gate_mlp,
                    x_ffn[:, tr_start:tr_end] * tr_gate_mlp,
                    x_ffn[:, tr_end:] * gate_mlp
                ], dim=1).to(input_dtype)
            else:
                x = x.addcmul(x_ffn.to(gate_mlp.dtype), gate_mlp).to(input_dtype)
        del gate_mlp

        if x_ip is not None: #stand-in
            x_ip = x_ip.addcmul(y_ip, gate_msa_ip)
            y_ip = self.ffn(torch.addcmul(shift_mlp_ip, self.norm2(x_ip), 1 + scale_mlp_ip))
            x_ip = x_ip.addcmul(y_ip, gate_mlp_ip)
        return x, x_ip, lynx_ref_feature, x_ovi


    def split_cross_attn_ffn(self, x, context, shift_mlp, scale_mlp, gate_mlp, clip_embed=None, grid_sizes=None):
        # Get number of prompts
        num_prompts = context.shape[0]
        num_clip_embeds = 0 if clip_embed is None else clip_embed.shape[0]
        num_segments = max(num_prompts, num_clip_embeds)

        # Extract spatial dimensions
        frames, height, width = grid_sizes[0]  # Assuming batch size 1
        tokens_per_frame = height * width

        # Distribute frames across prompts
        frames_per_segment = max(1, frames // num_segments)

        # Process each prompt segment
        x_combined = torch.zeros_like(x)

        for i in range(num_segments):
            # Calculate frame boundaries for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i+1) * frames_per_segment, frames) if i < num_segments-1 else frames

            # Convert frame indices to token indices
            start_idx = start_frame * tokens_per_frame
            end_idx = end_frame * tokens_per_frame
            segment_indices = torch.arange(start_idx, end_idx, device=x.device, dtype=torch.long)

            # Get prompt segment (cycle through available prompts if needed)
            prompt_idx = i % num_prompts
            segment_context = context[prompt_idx:prompt_idx+1]

            # Handle clip_embed for this segment (cycle through available embeddings)
            segment_clip_embed = None
            if clip_embed is not None:
                clip_idx = i % num_clip_embeds
                segment_clip_embed = clip_embed[clip_idx:clip_idx+1]

            # Get tensor segment
            x_segment = x[:, segment_indices, :].to(self.norm3.weight.dtype)

            # Process segment with its prompt and clip embedding
            processed_segment = self.cross_attn(self.norm3(x_segment), segment_context, clip_embed=segment_clip_embed)
            processed_segment = processed_segment.to(x.dtype)

            # Add to combined result
            x_combined[:, segment_indices, :] = processed_segment

        # Continue with FFN
        x = x + x_combined
        mod_x = torch.addcmul(shift_mlp, self.norm2(x.to(shift_mlp.dtype)), 1 + scale_mlp)
        y = self.ffn_chunked(mod_x, num_chunks=1)
        return x.addcmul(y, gate_mlp)

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            in_features,
            out_features,
            ffn_dim,
            ffn2_dim,
            num_heads,
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
            attention_mode='sdpa',
            rope_func="comfy",
            rms_norm_function="default"
    ):
        super().__init__(cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm, cross_attn_norm, eps, attention_mode, rope_func, rms_norm_function=rms_norm_function)

        self.register_buffer('block_id', torch.tensor(block_id, dtype=torch.long))

        if torch.equal(self.block_id, torch.tensor(0)):
            self.before_proj = nn.Linear(in_features, out_features)
        self.after_proj = nn.Linear(in_features, out_features)

    def forward(self, c, **kwargs):
        return super().forward(c, **kwargs)

class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        in_features,
        out_features,
        ffn_dim,
        ffn2_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
        block_idx=0,
        attention_mode='sdpa',
        rope_func="comfy",
        rms_norm_function="default",
        lynx_ip_layers=None,
        lynx_ref_layers=None,
    ):
        super().__init__(cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm, 
                         cross_attn_norm, eps, attention_mode, rope_func, rms_norm_function=rms_norm_function,
                         block_idx=block_idx, lynx_ip_layers=lynx_ip_layers, lynx_ref_layers=lynx_ref_layers)
        if block_id is not None:
            self.register_buffer('block_id', torch.tensor(block_id, dtype=torch.long))
        else:
            self.block_id = None

    def forward(self, x, vace_hints=None, vace_context_scale=[1.0], **kwargs):
        x, x_ip, lynx_ref_feature, x_ovi = super().forward(x, **kwargs)
        if vace_hints is None:
            return x, x_ip, lynx_ref_feature, x_ovi

        if self.block_id is not None:
            for i in range(len(vace_hints)):
                x.add_(vace_hints[i][self.block_id].to(x.device), alpha=vace_context_scale[i])
        return x, x_ip, lynx_ref_feature, x_ovi

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def get_mod(self, e):
        if e.dim() == 2:
            return (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            e = (self.modulation.unsqueeze(2) + e.unsqueeze(1)).chunk(2, dim=1)
            return [ei.squeeze(1) for ei in e]

    def forward(self, x, e, e_tr=None, tr_start=0, tr_num=0, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = self.get_mod(e.to(x.device))
        if tr_num > 0 and e_tr is not None:
            e_tr = self.get_mod(e_tr.to(x.device))
            tr_end = tr_start + tr_num
            norm_x = self.norm(x.float()).to(x.dtype)
            x = self.head(torch.cat([
                norm_x[:, :tr_start].mul(1 + e[1]).add(e[0]),
                norm_x[:, tr_start:tr_end].mul(1 + e_tr[1]).add(e_tr[0]),
                norm_x[:, tr_end:].mul(1 + e[1]).add(e[0])
            ], dim=1))
        else:
            x = self.head(self.norm(x.float()).to(x.dtype).mul_(1 + e[1]).add_(e[0]))
        return x

class Head_adaLN(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, adaln_tembed_dim=512):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.adaln_tembed_dim = adaln_tembed_dim    

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * self.dim, bias=True))

    def forward(self, x, e, temp_length, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        B, N, C = x.shape
        T = temp_length
        self.modulation.to(torch.float32)
        shift, scale = self.modulation(e).unsqueeze(2).chunk(2, dim=-1) # [B, T, 1, C]
        return self.head(self.norm(x.view(B, T, -1, C).float()).mul_(1 + scale).add_(shift).view(B, N, C).to(x.dtype))



class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, fl_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if fl_pos_emb:  # NOTE: we only use this for `fl2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, 257 * 2, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            image_embeds = image_embeds + self.emb_pos.to(image_embeds.device)
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

from .s2v.auxi_blocks import MotionEncoder_tc


class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_token=4,
                 need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(self.weights)
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(
            dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioCrossAttention(WanT2VCrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AudioInjector_WAN(nn.Module):

    def __init__(self,
                 all_modules,
                 all_modules_names,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 need_adain_ont=False,
                 attention_mode='sdpa'):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([
            AudioCrossAttention(
                in_features=dim,
                out_features=dim,
                num_heads=num_heads,
                qk_norm=True,
                attention_mode=attention_mode
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim)
                for _ in range(audio_injector_id)
            ])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)])

class WanModel(torch.nn.Module):
    def __init__(self,
                model_type='t2v',
                patch_size=(1, 2, 2), text_len=512,
                in_dim=16, dim=2048, in_features=5120, out_features=5120, ffn_dim=8192, ffn2_dim=8192,
                freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, eps=1e-6,
                qk_norm=True, cross_attn_norm=True,
                attention_mode='sdpa', rope_func='comfy', rms_norm_function='default',
                main_device=torch.device('cuda'), offload_device=torch.device('cpu'), dtype=torch.float16,
                teacache_coefficients=[], magcache_ratios=[], vace_layers=None, vace_in_dim=None,
                inject_sample_info=False, add_ref_conv=False, in_dim_ref_conv=16, add_control_adapter=False,
                in_dim_control_adapter=24,  use_motion_attn=False,
                #s2v
                cond_dim=0, audio_dim=1024, num_audio_token=4, enable_adain=False, zero_timestep=False,  humo_audio=False,
                adain_mode="attn_norm", audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
                # WanAnimate
                is_wananimate=False,  motion_encoder_dim=512,
                # lynx
                lynx_ip_layers=None, lynx_ref_layers=None,
                # LongCat
                is_longcat=False,
                ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.ffn_dim = ffn_dim
        self.ffn2_dim = ffn2_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        self.main_device = main_device
        self.offload_device = offload_device
        self.vace_layers = vace_layers
        self.device = main_device
        self.patched_linear = False

        self.blocks_to_swap = -1
        self.offload_txt_emb = False
        self.offload_img_emb = False
        self.vace_blocks_to_swap = -1

        self.cache_device = offload_device

        #init TeaCache variables
        self.enable_teacache = False
        self.rel_l1_thresh = 0.15
        self.teacache_start_step= 0
        self.teacache_end_step = -1
        self.teacache_state = TeaCacheState(cache_device=self.cache_device)
        self.teacache_coefficients = teacache_coefficients
        self.teacache_use_coefficients = False
        self.teacache_mode = 'e'

        #init MagCache variables
        self.enable_magcache = False
        self.magcache_state = MagCacheState(cache_device=self.cache_device)
        self.magcache_thresh = 0.24
        self.magcache_K = 4
        self.magcache_start_step = 0
        self.magcache_end_step = -1
        self.magcache_ratios = magcache_ratios

        #init EasyCache variables
        self.enable_easycache = False
        self.easycache_thresh = 0.1
        self.easycache_start_step = 0
        self.easycache_end_step = -1
        self.easycache_state = EasyCacheState(cache_device=self.cache_device)

        self.slg_blocks = None
        self.slg_start_percent = 0.0
        self.slg_end_percent = 1.0

        self.use_non_blocking = False
        self.prefetch_blocks = 0
        self.block_swap_debug = False

        self.video_attention_split_steps = []
        self.lora_scheduling_enabled = False

        self.multitalk_model_type = "none"

        self.lynx_ip_layers = lynx_ip_layers
        self.lynx_ref_layers = lynx_ref_layers

        self.humo_audio = humo_audio

        self.motion_encoder_dim = motion_encoder_dim

        self.base_dtype = dtype

        self.is_ovi_audio_model = patch_size == [1]

        self.audio_model = None

        self.is_longcat = is_longcat

        # embeddings
        if not self.is_ovi_audio_model:
            self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        else:
            from ...Ovi.audio_model_layers import ChannelLastConv1d, ConvMLP
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )

        self.original_patch_embedding = self.patch_embedding
        self.expanded_patch_embedding = self.patch_embedding

        if model_type != 'no_cross_attn':
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))

        if not is_longcat:
            self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        else:
            from ...LongCat.layers import TimestepEmbedder
            adaln_tembed_dim = 512
            self.time_embedding = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=freq_dim)


        if vace_layers is not None:
            self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
            self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

            self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.in_features, self.out_features, self.ffn_dim, self.ffn2_dim,self.num_heads, self.qk_norm,
                                        self.cross_attn_norm, self.eps, block_id=i, attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function)
                for i in self.vace_layers
            ])

            # vace patch embeddings
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )
            self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.in_features, self.out_features, ffn_dim, self.ffn2_dim, num_heads,
                              qk_norm, cross_attn_norm, eps,
                              attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function,
                              block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None, lynx_ip_layers=lynx_ip_layers, lynx_ref_layers=lynx_ref_layers, block_idx=i)
            for i in range(num_layers)
            ])
        else:
            # blocks
            if model_type == 't2v' or model_type == 's2v':
                cross_attn_type = 't2v_cross_attn'
            elif model_type == 'i2v' or model_type == 'fl2v':
                cross_attn_type = 'i2v_cross_attn'
            else:
                cross_attn_type = 'no_cross_attn'

            self.blocks = nn.ModuleList([
                WanAttentionBlock(cross_attn_type, self.in_features, self.out_features, ffn_dim, ffn2_dim, num_heads,
                                qk_norm, cross_attn_norm, eps,
                                attention_mode=self.attention_mode, rope_func=self.rope_func, rms_norm_function=rms_norm_function, 
                                use_motion_attn=(i % 4 == 0 and use_motion_attn), use_humo_audio_attn=self.humo_audio,
                                face_fuser_block = (i % 5 == 0 and is_wananimate), lynx_ip_layers=lynx_ip_layers, lynx_ref_layers=lynx_ref_layers,
                                block_idx=i, is_longcat=is_longcat)
                for i in range(num_layers)
            ])
        #MTV Crafter
        if use_motion_attn:
            self.pad_motion_tokens = torch.zeros(1, 1, 2048)

        # head
        if not is_longcat:
            self.head = Head(dim, out_dim, patch_size, eps)
        else:
            self.head = Head_adaLN(dim, out_dim, patch_size, eps, adaln_tembed_dim=512)

        d = self.dim // self.num_heads
        self.rope_embedder = EmbedND_RifleX(d, 10000.0, [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)], num_frames=None, k=None)
        self.cached_freqs = self.cached_shape = self.cached_cond = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0

        if model_type == 'i2v' or model_type == 'fl2v':
            self.img_emb = MLPProj(1280, dim, fl_pos_emb=model_type == 'fl2v')

        #skyreels v2
        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        #fun 1.1
        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        if add_control_adapter:
            from .wan_camera_adapter import SimpleAdapter
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        #S2V
        self.zero_timestep = self.audio_injector = self.trainable_cond_mask =None
        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size)
        if self.model_type == 's2v':
            self.enable_adain = enable_adain
            self.casual_audio_encoder = CausalAudioEncoder(
                dim=audio_dim,
                out_dim=self.dim,
                num_token=num_audio_token,
                need_global=enable_adain)
            all_modules, all_modules_names = torch_dfs(
                self.blocks, parent_name="root.transformer_blocks")
            self.audio_injector = AudioInjector_WAN(
                all_modules,
                all_modules_names,
                dim=self.dim,
                num_heads=self.num_heads,
                inject_layer=audio_inject_layers,
                root_net=self,
                enable_adain=enable_adain,
                adain_dim=self.dim,
                need_adain_ont=adain_mode != "attn_norm",
                attention_mode=attention_mode
            )
            self.trainable_cond_mask = nn.Embedding(3, self.dim)

            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode='padd')
        self.adain_mode = adain_mode
        self.zero_timestep = zero_timestep

        # HuMo Audio
        if self.humo_audio:
            from ...HuMo.audio_proj import AudioProjModel
            self.audio_proj = AudioProjModel(seq_len=8, blocks=5, channels=1280, 
                intermediate_dim=512, output_dim=1536, context_tokens=16)
        # WanAnimate
        self.motion_encoder = self.pose_patch_embedding = self.face_encoder = self.face_adapter = None
        if is_wananimate:
            from .wananimate.motion_encoder import MotionExtractor
            from .wananimate.face_blocks import FaceEncoder
            self.pose_patch_embedding = nn.Conv3d(16, dim, kernel_size=patch_size, stride=patch_size)
            self.motion_encoder = MotionExtractor()

            self.face_encoder = FaceEncoder(
                in_dim=motion_encoder_dim,
                out_dim=self.dim,
                num_heads=4,
                dtype=dtype
            )

    def block_swap(self, blocks_to_swap, offload_txt_emb=False, offload_img_emb=False, vace_blocks_to_swap=None, prefetch_blocks=0, block_swap_debug=False):
        # Clamp blocks_to_swap to valid range
        blocks_to_swap = max(0, min(blocks_to_swap, len(self.blocks)))

        log.info(f"Swapping {blocks_to_swap} transformer blocks")
        self.blocks_to_swap = blocks_to_swap
        self.prefetch_blocks = prefetch_blocks
        self.block_swap_debug = block_swap_debug

        self.offload_img_emb = offload_img_emb
        self.offload_txt_emb = offload_txt_emb

        total_offload_memory = 0
        total_main_memory = 0

        # Calculate the index where swapping starts
        swap_start_idx = len(self.blocks) - blocks_to_swap

        for b, block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Initializing block swap"):
            block_memory = get_module_memory_mb(block)

            if b < swap_start_idx:
                block.to(self.main_device)
                total_main_memory += block_memory
            else:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)
                total_offload_memory += block_memory

        if blocks_to_swap != -1 and vace_blocks_to_swap == 0:
            vace_blocks_to_swap = 1

        if vace_blocks_to_swap > 0 and self.vace_layers is not None:
            # Clamp vace_blocks_to_swap to valid range
            vace_blocks_to_swap = max(0, min(vace_blocks_to_swap, len(self.vace_blocks)))
            self.vace_blocks_to_swap = vace_blocks_to_swap

            # Calculate the index where VACE swapping starts
            vace_swap_start_idx = len(self.vace_blocks) - vace_blocks_to_swap

            for b, block in tqdm(enumerate(self.vace_blocks), total=len(self.vace_blocks), desc="Initializing vace block swap"):
                block_memory = get_module_memory_mb(block)

                if b < vace_swap_start_idx:
                    block.to(self.main_device)
                    total_main_memory += block_memory
                else:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
                    total_offload_memory += block_memory

        mm.soft_empty_cache()
        gc.collect()

        log.info("-" * 25)
        log.info("Block swap memory summary:")
        log.info(f"Transformer blocks on {self.offload_device}: {total_offload_memory:.2f}MB")
        log.info(f"Transformer blocks on {self.main_device}: {total_main_memory:.2f}MB")
        log.info(f"Total memory used by transformer blocks: {(total_offload_memory + total_main_memory):.2f}MB")
        log.info(f"Non-blocking memory transfer: {self.use_non_blocking}")
        log.info("-" * 25)

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0).float()).to(x.dtype) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        if x.shape[1] > c.shape[1]:
            c = torch.cat([c.new_zeros(x.shape[0], x.shape[1] - c.shape[1], c.shape[2]), c], dim=1)
        if c.shape[1] > x.shape[1]:
            c = c[:, :x.shape[1]]
        
        hints = []
        current_c = c
        vace_swap_start_idx = len(self.vace_blocks) - self.vace_blocks_to_swap if self.vace_blocks_to_swap > 0 else len(self.vace_blocks)
        
        for b, block in enumerate(self.vace_blocks):
            if b >= vace_swap_start_idx and self.vace_blocks_to_swap > 0:
                block.to(self.main_device)
                
            if b == 0:
                c_processed = block.before_proj(current_c) + x
            else:
                c_processed = current_c
                
            c_processed = block.forward(c_processed, **kwargs)[0]
            
            # Store skip connection
            c_skip = block.after_proj(c_processed)
            hints.append(c_skip.to(
                self.offload_device if self.vace_blocks_to_swap > 0 else self.main_device, 
                non_blocking=self.use_non_blocking
            ))
            
            current_c = c_processed
            
            if b >= vace_swap_start_idx and self.vace_blocks_to_swap > 0:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)

        return hints
    
    def audio_injector_forward(self, block_idx, x, audio_emb, scale=1.0):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            num_frames = audio_emb.shape[1]# b f n c

            input_x = x[:, :self.original_seq_len].clone()  # b (f h w) c
            input_x = rearrange(input_x, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enable_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global,"b t n c -> (b t) n c")
                attn_x = self.audio_injector.injector_adain_layers[audio_attn_id](input_x, temb=audio_emb_global[:, 0])
            else:
                attn_x = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_x)

            attn_audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_x ,
                context=attn_audio_emb * scale,
            )
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            x[:, :self.original_seq_len].add_(residual_out)

        return x

    def wananimate_pose_embedding(self, x, pose_latents, strength=1.0):
        pose_latents = [self.pose_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in pose_latents]
        for x_, pose_latents_ in zip(x, pose_latents):
            x_[:, :, 1:].add_(pose_latents_, alpha=strength)
        return x


    def wananimate_face_embedding(self, face_pixel_values):
        b,c,T,h,w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        encode_bs = 8
        face_pixel_values_tmp = []
        self.motion_encoder.to(self.main_device)
        for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))
        del face_pixel_values
        self.motion_encoder.to(self.offload_device)

        motion_vec = rearrange(torch.cat(face_pixel_values_tmp), "(b t) c -> b t c", t=T)
        del face_pixel_values_tmp
        self.face_encoder.to(self.main_device)
        motion_vec = self.face_encoder(motion_vec.to(self.face_encoder.dtype))
        self.face_encoder.to(self.offload_device)

        B, L, H, C = motion_vec.shape
        pad_face = torch.zeros(B, 1, H, C, device=motion_vec.device, dtype=motion_vec.dtype)
        return torch.cat([pad_face, motion_vec], dim=1)


    def wananimate_forward(self, block, x, motion_vec, strength=1.0, motion_masks=None):
            adapter_args = [x, motion_vec, motion_masks]
            residual_out = block.fuser_block(*adapter_args)
            return x.add(residual_out, alpha=strength)


    def rope_encode_comfy(self, t, h, w, freq_offset=0, t_start=0, ref_frame_shape=None, pose_frame_shape=None,
                          steps_t=None, steps_h=None, steps_w=None, ntk_alphas=[1,1,1], device=None, dtype=None,
                          ref_frame_index=10, longcat_num_ref_latents=0, num_memory_frames=3, rope_negative_offset=0):

        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        # Main frames position IDs
        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)

        if longcat_num_ref_latents > 0:
            # Create temporal grid with ref_frame_index prepended, followed by sequential frames
            grid_t = torch.cat([
                torch.tensor([ref_frame_index], dtype=dtype, device=device),
                torch.arange(0, steps_t - longcat_num_ref_latents, dtype=dtype, device=device)
            ], dim=0)
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + grid_t.reshape(-1, 1, 1)
        elif num_memory_frames > 0 and rope_negative_offset > 0:
            # Negative RoPE shift for memory frames
            # Memory frames get negative indices: {-f_m*S, -(f_m-1)*S, ..., -S}
            # Current video frames start from 0: {0, 1, ..., f-1}
            memory_indices = torch.arange(-num_memory_frames * rope_negative_offset, 0, rope_negative_offset, dtype=dtype, device=device)
            current_indices = torch.arange(0, steps_t - num_memory_frames, dtype=dtype, device=device)
            grid_t = torch.cat([memory_indices, current_indices], dim=0)
            log.info(f"{num_memory_frames} memory frames, temporal rope positions: {grid_t}")
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + grid_t.reshape(-1, 1, 1)
        else:
            # Standard temporal encoding
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start+freq_offset, t_start+freq_offset + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)

        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(freq_offset, freq_offset + (h_len - 1), steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(freq_offset, freq_offset + (w_len - 1), steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        segments = [img_ids]  # Start with main frames

        # Reference frames position IDs
        if ref_frame_shape is not None:
            F_cond, H_cond, W_cond = ref_frame_shape[-3], ref_frame_shape[-2], ref_frame_shape[-1]
            cond_f_len = ((F_cond + (self.patch_size[0] // 2)) // self.patch_size[0])
            cond_h_len = ((H_cond + (self.patch_size[1] // 2)) // self.patch_size[1])
            cond_w_len = ((W_cond + (self.patch_size[2] // 2)) // self.patch_size[2])
            cond_img_ids = torch.zeros((cond_f_len, cond_h_len, cond_w_len, 3), device=device, dtype=dtype)

            cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(0, cond_f_len - 1, steps=cond_f_len, device=device, dtype=dtype).reshape(-1, 1, 1)
            cond_img_ids[:, :, :, 1] = cond_img_ids[:, :, :, 1] + torch.linspace(h_len, h_len + cond_h_len - 1, steps=cond_h_len, device=device, dtype=dtype).reshape(1, -1, 1)
            cond_img_ids[:, :, :, 2] = cond_img_ids[:, :, :, 2] + torch.linspace(w_len, w_len + cond_w_len - 1, steps=cond_w_len, device=device, dtype=dtype).reshape(1, 1, -1)

            segments.insert(0, cond_img_ids.reshape(1, -1, cond_img_ids.shape[-1]))  # Ref frames come first

        # Pose frames position IDs
        if pose_frame_shape is not None:
            F_pose, H_pose, W_pose = pose_frame_shape[-3], pose_frame_shape[-2], pose_frame_shape[-1]

            downscale = H_pose != h
            pose_f_len_full = ((F_pose + (self.patch_size[0] // 2)) // self.patch_size[0])
            pose_h_len_full = (((H_pose * (2 if downscale else 1)) + (self.patch_size[1] // 2)) // self.patch_size[1])  # 2x height
            pose_w_len_full = (((W_pose * (2 if downscale else 1)) + (self.patch_size[2] // 2)) // self.patch_size[2])  # 2x width

            pose_img_ids = torch.zeros((pose_f_len_full, pose_h_len_full, pose_w_len_full, 3), device=device, dtype=dtype)
            global_h_offset, global_w_offset = 0, 120  # global spatial offset to separate pose from main frames spatially (SCAIL uses 120 as offset)
            pose_img_ids[:, :, :, 0] = pose_img_ids[:, :, :, 0] + torch.linspace(t_start+freq_offset, t_start + (pose_f_len_full - 1), steps=pose_f_len_full, device=device, dtype=dtype).reshape(-1, 1, 1)
            pose_img_ids[:, :, :, 1] = pose_img_ids[:, :, :, 1] + torch.linspace(global_h_offset + freq_offset, global_h_offset + pose_h_len_full - 1, steps=pose_h_len_full, device=device, dtype=dtype).reshape(1, -1, 1)
            pose_img_ids[:, :, :, 2] = pose_img_ids[:, :, :, 2] + torch.linspace(global_w_offset + freq_offset, global_w_offset + pose_w_len_full - 1, steps=pose_w_len_full, device=device, dtype=dtype).reshape(1, 1, -1)

            segments.append(pose_img_ids.reshape(1, -1, pose_img_ids.shape[-1]))

        combined_img_ids = torch.cat(segments, dim=1)
        freqs = self.rope_embedder(combined_img_ids, ntk_alphas).movedim(1, 2)

        # Downsample pose frequencies to match actual pose input resolution
        if pose_frame_shape is not None and downscale:
            pose_h_len_actual = ((H_pose + (self.patch_size[1] // 2)) // self.patch_size[1])
            pose_w_len_actual = ((W_pose + (self.patch_size[2] // 2)) // self.patch_size[2])

            pose_start_idx = freqs.shape[1] - pose_f_len_full * pose_h_len_full * pose_w_len_full
            main_freqs, pose_freqs = freqs[:, :pose_start_idx], freqs[:, pose_start_idx:]

            B, _, heads, dim, _, _ = pose_freqs.shape
            # Reshape and pool: (B, L, heads, dim, 2, 2) -> pool H,W -> (B, L', heads, dim, 2, 2)
            pose_freqs = pose_freqs.reshape(B, pose_f_len_full, pose_h_len_full, pose_w_len_full, heads, dim, 2, 2)
            pose_freqs = pose_freqs.permute(0, 1, 4, 5, 6, 7, 2, 3).reshape(-1, pose_h_len_full, pose_w_len_full)
            pose_freqs = F.avg_pool2d(pose_freqs, kernel_size=2, stride=2)
            pose_freqs = pose_freqs.reshape(B, pose_f_len_full, heads, dim, 2, 2, pose_h_len_actual, pose_w_len_actual)
            pose_freqs = pose_freqs.permute(0, 1, 6, 7, 2, 3, 4, 5).reshape(B, -1, heads, dim, 2, 2)

            freqs = torch.cat([main_freqs, pose_freqs], dim=1)

        return freqs


    def forward(
        self, x, t, context, seq_len,
        is_uncond=False,
        current_step_percentage=0.0, current_step=0, last_step=0, total_steps=50,
        clip_fea=None, y=None,
        device=torch.device('cuda'),
        freqs=None,
        enhance_enabled=False,
        pred_id=None,
        control_lora_enabled=False,
        vace_data=None,
        camera_embed=None,
        unianim_data=None,
        fps_embeds=None,
        fun_ref=None, fun_camera=None,
        audio_proj=None, audio_scale=1.0,
        uni3c_data=None, controlnet=None,
        add_cond=None, attn_cond=None,
        nag_params={}, nag_context=None,
        multitalk_audio=None,
        ref_target_masks=None,
        inner_t=None,
        standin_input=None,
        fantasy_portrait_input=None,
        phantom_ref=None,
        reverse_time=False,
        ntk_alphas = [1.0, 1.0, 1.0],
        mtv_motion_tokens=None, mtv_motion_rotary_emb=None,
        mtv_freqs=None, mtv_strength=1.0,
        s2v_audio_input=None, s2v_ref_latent=None, s2v_audio_scale=1.0,
        s2v_ref_motion=None, s2v_pose=None, s2v_motion_frames=[1, 0],
        humo_audio=None, humo_audio_scale=1.0,
        wananim_pose_latents=None, wananim_face_pixel_values=None,
        wananim_pose_strength=1.0, wananim_face_strength=1.0,
        lynx_embeds=None,
        x_ovi=None, seq_len_ovi=None, ovi_negative_text_embeds=None,
        flashvsr_LQ_latent=None, flashvsr_strength=1.0,
        longcat_num_cond_latents=0, longcat_num_ref_latents=0, longcat_avatar_options=None, # for LongCat
        add_text_emb=None,
        sdancer_input=None,  # SteadyDancer
        one_to_all_input=None, one_to_all_controlnet_strength=0.0, # One-to-All
        scail_input=None,  # SCAIL pose
        dual_control_input=None,  # LongVie2 dual controlnet
        transformer_options={},
        rope_negative_offset=0,
        num_memory_frames=0,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Stand-In only used on first positive pass, then cached in kv_cache
        if is_uncond or current_step > 0:
            standin_input = None

        # MTV Crafter motion projection
        if mtv_motion_tokens is not None:
            bs, motion_seq_len =  mtv_motion_tokens.shape[0], mtv_motion_tokens.shape[1]
            mtv_motion_tokens = torch.cat([mtv_motion_tokens, self.pad_motion_tokens.to(mtv_motion_tokens).expand(bs, motion_seq_len, -1)], dim=-1)

        # Fantasy Portrait
        adapter_proj = ip_scale = None
        if fantasy_portrait_input is not None:
            if fantasy_portrait_input['start_percent'] <= current_step_percentage <= fantasy_portrait_input['end_percent']:
                adapter_proj = fantasy_portrait_input.get("adapter_proj", None)
                ip_scale = fantasy_portrait_input.get("strength", 1.0)

        if self.lora_scheduling_enabled:
            update_lora_step(self, current_step)

        # lynx
        lynx_x_ip = lynx_ref_feature = lynx_ref_buffer = lynx_ref_feature_extractor = None
        lynx_ip_scale = lynx_ref_scale = 1.0
        if lynx_embeds is not None:
            lynx_ref_feature_extractor = lynx_embeds.get("ref_feature_extractor", False)
            lynx_ref_blocks_to_use = lynx_embeds.get("ref_blocks_to_use", None)
            if lynx_ref_blocks_to_use is None:
                lynx_ref_blocks_to_use = list(range(len(self.blocks)))
            if (lynx_embeds['start_percent'] <= current_step_percentage <= lynx_embeds['end_percent']) and not lynx_ref_feature_extractor:
                if not is_uncond:
                    lynx_x_ip = lynx_embeds.get("ip_x", None)
                    lynx_ref_buffer = lynx_embeds.get("ref_buffer", None)
                else:
                    lynx_x_ip = lynx_embeds.get("ip_x_uncond", None)
                    lynx_ref_buffer = lynx_embeds.get("ref_buffer_uncond", None)
                lynx_x_ip = lynx_x_ip.to(self.main_device) if lynx_x_ip is not None else None

                lynx_ip_scale = lynx_embeds.get("ip_scale", 1.0)
                lynx_ref_scale = lynx_embeds.get("ref_scale", 1.0)


        #s2v
        if self.model_type == 's2v' and s2v_audio_input is not None:
            if is_uncond:
                s2v_audio_input = s2v_audio_input * 0 # to match original code
            s2v_audio_input = torch.cat([s2v_audio_input[..., 0:1].repeat(1, 1, 1, s2v_motion_frames[0]), s2v_audio_input], dim=-1)

            audio_emb_res = self.casual_audio_encoder(s2v_audio_input)
            if self.enable_adain:
                audio_emb_global, audio_emb = audio_emb_res
                self.audio_emb_global = audio_emb_global[:, s2v_motion_frames[1]:].clone()
            else:
                audio_emb = audio_emb_res
            merged_audio_emb = audio_emb[:, s2v_motion_frames[1]:, :]

        # params
        device = self.main_device

        if freqs is not None and freqs.device != device:
           freqs = freqs.to(device)

        _, F, H, W = x[0].shape
        ref_frame_shape = pose_frame_shape = None

        sdancer_enabled = False
        if sdancer_input is not None and sdancer_input['start_percent'] <= current_step_percentage <= sdancer_input['end_percent']:
            sdancer_enabled = True
            x_noise_clone = torch.stack(x)

        # I2V
        if y is not None:
            if hasattr(self, "randomref_embedding_pose") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    random_ref_emb = unianim_data["random_ref"]
                    if random_ref_emb is not None:
                        y[0].add_(random_ref_emb, alpha=unianim_data["strength"])
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        suffix_frames = x[0].shape[1]
        prefix_frames = 0

        # One-to-all-Animation
        onetoall_ref_block_samples = onetoall_freqs = prev_x = prev_control = None
        onetoall_ref_scale = 1.0
        onetoall_control_enabled = use_token_replace = False
        e0_token_replace = token_replace_start = None
        replace_token_num = token_replace_start = 0
        if one_to_all_input is not None:
            # reference condition
            ref_cond_latent = one_to_all_input.get("ref_latent_pos", None) if not is_uncond else one_to_all_input.get("ref_latent_neg", None)
            if ref_cond_latent is not None and one_to_all_input['ref_start_percent'] <= current_step_percentage <= one_to_all_input['ref_end_percent']:
                onetoall_ref_scale = one_to_all_input.get("ref_strength", 1.0)
                self.image_to_cond.to(self.main_device)
                image_cond = self.image_to_cond(ref_cond_latent.to(self.main_device, self.base_dtype))[0]
                self.image_to_cond.to(self.offload_device)
                x = [torch.cat([v, u], dim=1) for v, u in zip([image_cond], x)]
                seq_len += math.ceil((image_cond.shape[-1] * image_cond.shape[-2]) / 4 * image_cond.shape[-3])
                F += 1
                prefix_frames = 1
                suffix_frames += 1
                self.refextractor.to(self.main_device)
                onetoall_ref_block_samples, onetoall_freqs = self.refextractor(ref_cond_latent, timestep=t)
                self.refextractor.to(self.offload_device)
            # pose controlnet
            controlnet_tokens = one_to_all_input.get("controlnet_tokens", None)
            if controlnet_tokens is not None and one_to_all_input['controlnet_start_percent'] <= current_step_percentage <= one_to_all_input['controlnet_end_percent']:
                onetoall_control_enabled = one_to_all_controlnet_strength != 0.0
            # token replace
            if one_to_all_input.get("token_replace", False):
                use_token_replace = True
                num_latent_frames_to_replace = one_to_all_input.get("num_latent_frames_to_replace", 2)
                t_token_replace = torch.zeros_like(t)
                token_replace_start = (H // self.patch_size[1]) * (W // self.patch_size[2]) # skip first (ref) frame
                replace_token_num = num_latent_frames_to_replace * token_replace_start # zero next frames

        # SCAIL ref
        if scail_input is not None:
            ref_latent = scail_input.get("ref_latent_pos", None) if not is_uncond else scail_input.get("ref_latent_neg", None)
            if ref_latent is not None and scail_input['ref_start_percent'] <= current_step_percentage <= scail_input['ref_end_percent']:
                x = [torch.cat([v, u], dim=1) for v, u in zip([ref_latent], x)]
                seq_len += math.ceil((ref_latent.shape[-1] * ref_latent.shape[-2]) / 4 * ref_latent.shape[-3])
                F += 1
                prefix_frames = 1
                suffix_frames += 1

        #uni3c controlnet
        if uni3c_data is not None:
            render_latent = uni3c_data["render_latent"].to(self.base_dtype)
            hidden_states = x[0].unsqueeze(0).clone().float()
            if hidden_states.shape[1] == 16: #T2V work around
                hidden_states = torch.cat([hidden_states, torch.zeros_like(hidden_states[:, :4])], dim=1)
            if hidden_states.shape[2] != render_latent.shape[2]: # temporal resample
                render_latent = nn.functional.interpolate(render_latent, size=(hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]), mode='trilinear', align_corners=False)
            render_latent = torch.cat([hidden_states[:, :20], render_latent], dim=1)

        # SteadyDancer
        if sdancer_enabled:
            sdancer_cond = sdancer_input["cond_pos"] if not is_uncond else sdancer_input["cond_neg"]
            condition_temporal = [self.condition_embedding_temporal(c.unsqueeze(0).float()).to(self.base_dtype) for c in [sdancer_cond]] # Temporal Motion Coherence Module.
            sdancer_cond = sdancer_cond.unsqueeze(0)
            bs, _, time_steps, _, _ = sdancer_cond.shape
            condition_reshape = rearrange(sdancer_cond, 'b c t h w -> (b t) c h w')
            condition_spatial = self.condition_embedding_spatial(condition_reshape.float()).to(self.base_dtype) # Spatial Structure Adaptive Extractor.
            condition_spatial = rearrange(condition_spatial, '(b t) c h w -> b c t h w', t=time_steps, b=bs)
            condition_fused = sdancer_cond + condition_temporal[0] * sdancer_input["pose_strength_temporal"] + condition_spatial * sdancer_input["pose_strength_spatial"] # Hierarchical Aggregation (1): condition, temporal condition, spatial condition
            condition_aligned = self.condition_embedding_align(condition_fused.float(), x_noise_clone).to(self.base_dtype) # Frame-wise Attention Alignment Unit.
        else:
            # patch embed
            if control_lora_enabled:
                self.expanded_patch_embedding.to(self.main_device)
                x = [self.expanded_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in x]
            else:
                self.original_patch_embedding.to(self.main_device)
                x = [self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in x]

        # ovi audio model
        if self.audio_model is not None:
            x_ovi = [self.audio_model.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x_ovi[0].dtype) for u in x_ovi]
            grid_sizes_ovi = torch.stack([torch.tensor(u.shape[1:2], dtype=torch.long) for u in x_ovi])
            seq_lens_ovi = torch.tensor([u.size(1) for u in x_ovi], dtype=torch.int32)
            x_ovi = torch.cat([torch.cat([u, u.new_zeros(1, seq_len_ovi - u.size(1), u.size(2))], dim=1) for u in x_ovi])
            d = self.dim // self.num_heads
            freqs_ovi = rope_params(1024, d - 4 * (d // 6), freqs_scaling=0.19676).to(self.main_device)
            x_ovi = x_ovi.to(self.main_device, self.base_dtype)

        # WanAnimate
        motion_vec = None
        if wananim_face_pixel_values is not None:
            motion_vec = self.wananimate_face_embedding(wananim_face_pixel_values).to(self.base_dtype)

        if wananim_pose_latents is not None:
            x = self.wananimate_pose_embedding(x, wananim_pose_latents, strength=wananim_pose_strength)

        # s2v pose embedding
        if s2v_pose is not None:
            x[0] = x[0] + self.cond_encoder(s2v_pose.to(self.cond_encoder.weight.dtype)).to(self.base_dtype)

        # Fun camera
        if self.control_adapter is not None and fun_camera is not None:
            fun_camera = self.control_adapter(fun_camera)
            x = [u + v for u, v in zip(x, fun_camera)]

        # SteadyDancer
        if sdancer_enabled:
            ref_x = y[0][4:, :1] # reuse I2V input as reference, slice mask off
            msk = torch.ones(4, 1, H, W, device=ref_x.device) # new mask goes in middle
            ref_x = [torch.concat([ref_x, msk, ref_x])]
            ref_c = sdancer_cond[0][:, :1]
            ref_c = [torch.concat([ref_c, msk * 0, ref_c])] # zero mask for cond ref
            # Condition Fusion/Injection, Hierarchical Aggregation (2): x, fused condition, aligned condition
            x = [self.patch_embedding_fuse(torch.cat([u[None], c[None], a[None]], 1)) for u, c, a in zip(x, condition_fused, condition_aligned)]
            # Condition Augmentation: x_cond, ref_x, ref_c
            ref_x = [self.patch_embedding(r.unsqueeze(0).float()).to(self.base_dtype) for r in ref_x]
            ref_c = [self.patch_embedding_ref_c(r[:16].unsqueeze(0).float()).to(self.base_dtype) for r in ref_c]
            F += ref_x[0].shape[2] + ref_c[0].shape[2] # update frame count for rope
            x = [torch.cat([r, u, v], dim=2) for r, u, v in zip(x, ref_x, ref_c)]
            seq_len = torch.tensor([u.flatten(2).transpose(1, 2).size(1) for u in x], dtype=torch.int32).max() # update seq len

        # grid sizes and seq len
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], device=device, dtype=torch.long) for u in x])
        original_grid_sizes = grid_sizes.clone()
        f, h, w = x[0].shape[2:]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        self.original_seq_len = x[0].shape[1]

        prev_latent = None
        if dual_control_input is not None:
            prev_latent = dual_control_input.get("prev_latent", None)
            if prev_latent is not None:
                F += prev_latent.shape[2]
                prev_x = [self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in prev_latent]
                prev_x = [u.flatten(2).transpose(1, 2).to(self.base_dtype) for u in prev_x]
                seq_len += prev_x[0].shape[1]
                x = [torch.cat([u, v], dim=1) for u, v in zip(prev_x, x)]

        # SCAIL pose
        if scail_input is not None:
            scail_pose_latents = scail_input.get("pose_latent", None)
            if scail_pose_latents is not None and scail_input['pose_start_percent'] <= current_step_percentage <= scail_input['pose_end_percent']:
                scail_x = [self.patch_embedding_pose(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in [scail_pose_latents]]
                scail_x = [u.flatten(2).transpose(1, 2) * scail_input.get("pose_strength", 1) for u in scail_x]
                x = [torch.cat([u, v], dim=1) for u, v in zip(x, scail_x)]
                seq_len += scail_x[0].shape[1]
                del scail_x
                pose_frame_shape = scail_pose_latents.shape

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.int32)
        assert seq_lens.max() <= seq_len, f"max seq len {seq_lens.max()} exceeds provided seq_len {seq_len}"

        cond_mask_weight = None
        if self.trainable_cond_mask is not None:
            cond_mask_weight = self.trainable_cond_mask.weight.to(x[0]).unsqueeze(1).unsqueeze(1)

        if add_cond is not None:
            add_cond = self.add_conv_in(add_cond.to(self.add_conv_in.weight.dtype)).to(x[0].dtype)
            add_cond = add_cond.flatten(2).transpose(1, 2)
            x[0] = x[0] + self.add_proj(add_cond)
        if attn_cond is not None:
            ref_frame_shape = attn_cond.shape
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            attn_cond = self.attn_conv_in(attn_cond.to(self.attn_conv_in.weight.dtype)).to(x[0].dtype)
            attn_cond = attn_cond.flatten(2).transpose(1, 2)
            x[0] = torch.cat([x[0], attn_cond], dim=1)
            seq_len += attn_cond.size(1)
            for block in self.blocks:
                block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=F+1)

        if self.ref_conv is not None and fun_ref is not None:
            fun_ref = self.ref_conv(fun_ref.to(self.ref_conv.weight.dtype)).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += fun_ref.size(1)
            F += 1
            x = [torch.cat([_fun_ref.unsqueeze(0), u], dim=1) for _fun_ref, u in zip(fun_ref, x)]

        end_ref_latent=None
        if s2v_ref_latent is not None:
            end_ref_latent = s2v_ref_latent.squeeze(0)
        elif phantom_ref is not None:
            end_ref_latent = phantom_ref
            F += end_ref_latent.size(1)
        if end_ref_latent is not None:
            end_ref_latent_frames = end_ref_latent.size(1)
            end_ref_latent = self.original_patch_embedding(end_ref_latent.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            end_ref_latent = end_ref_latent.flatten(2).transpose(1, 2)
            if cond_mask_weight is not None:
                end_ref_latent = end_ref_latent + cond_mask_weight[1]
            grid_sizes = torch.stack([torch.tensor([u[0] + end_ref_latent_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            end_ref_latent_seq_len = end_ref_latent.size(1)
            seq_len += end_ref_latent_seq_len
            x = [torch.cat([u, end_ref_latent.unsqueeze(0)], dim=1) for end_ref_latent, u in zip(end_ref_latent, x)]


        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        if self.trainable_cond_mask is not None:
            x = x + cond_mask_weight[0]

        # StandIn LoRA input
        x_ip = None
        freq_offset = 0
        if standin_input is not None:
            ip_image = standin_input["ip_image_latent"]

            if ip_image.dim() == 6 and ip_image.shape[3] == 1:
                ip_image = ip_image.squeeze(1)

            ip_image_patch = self.original_patch_embedding(ip_image.to(x.device).float()).to(self.base_dtype)
            f_ip, h_ip, w_ip = ip_image_patch.shape[2:]
            x_ip = ip_image_patch.flatten(2).transpose(1, 2)  # [B, N, D]
            freq_offset = standin_input["freq_offset"]

        # region rope freqs
        if freqs is None and "comfy" in self.rope_func: #comfy rope
            # Create cache key from all relevant parameters
            cache_key = (
                F, H, W,
                attn_cond is not None,
                tuple(ref_frame_shape) if ref_frame_shape is not None else None,
                tuple(pose_frame_shape) if pose_frame_shape is not None else None,
                self.rope_embedder.k,
                tuple(ntk_alphas),
                longcat_num_ref_latents,
                rope_negative_offset,
                num_memory_frames,
            )

            # Check cache using key comparison
            if (self.cached_freqs is not None and
                hasattr(self, 'cached_key') and
                self.cached_key == cache_key):
                freqs = self.cached_freqs
            else:
                freqs = self.rope_encode_comfy(
                    F, H, W,
                    freq_offset=freq_offset,
                    ntk_alphas=ntk_alphas,
                    ref_frame_shape=ref_frame_shape,
                    pose_frame_shape=pose_frame_shape,
                    longcat_num_ref_latents=longcat_num_ref_latents,
                    rope_negative_offset=rope_negative_offset,
                    num_memory_frames=num_memory_frames,
                    device=x.device,
                    dtype=x.dtype
                )
                tqdm.write("Generated new RoPE frequencies")

                if s2v_ref_latent is not None:
                    freqs_ref = self.rope_encode_comfy(
                        s2v_ref_latent.shape[2],
                        s2v_ref_latent.shape[3],
                        s2v_ref_latent.shape[4],
                        t_start=max(30, F + 9),
                        device=x.device,
                        dtype=x.dtype
                    )
                    freqs = torch.cat([freqs, freqs_ref], dim=1)

                # Store cache with key
                self.cached_freqs = freqs
                self.cached_key = cache_key

        # Stand-In RoPE frequencies
        if x_ip is not None:
            # Generate RoPE frequencies for x_ip
            ip_img_ids = torch.zeros((f_ip, h_ip, w_ip, 3), device=x.device, dtype=x.dtype)
            ip_img_ids[:, :, :, 0] = -1
            ip_img_ids[:, :, :, 1] = ip_img_ids[:, :, :, 1] + torch.linspace(h + freq_offset, h + freq_offset + (h_ip - 1), steps=h_ip, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            ip_img_ids[:, :, :, 2] = ip_img_ids[:, :, :, 2] + torch.linspace(w + freq_offset, w + freq_offset + (w_ip - 1), steps=w_ip, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            ip_img_ids = repeat(ip_img_ids, "t h w c -> b (t h w) c", b=1)
            freqs_ip = self.rope_embedder(ip_img_ids).movedim(1, 2)

        # EchoShot cross attn freqs
        inner_c = None
        if inner_t is not None:
            d = self.dim // self.num_heads
            self.cross_freqs = rope_params(100, d).to(device=x.device)

        if s2v_ref_motion is not None:
            motion_encoded, freqs_motion = self.frame_packer(s2v_ref_motion, self)
            motion_encoded = motion_encoded + cond_mask_weight[2]
            x = torch.cat([x, motion_encoded], dim=1)
            freqs = torch.cat([freqs, freqs_motion], dim=1)

        # time embeddings
        if t.dim() == 2 and not self.is_longcat:
            b, f = t.shape
            expanded_timesteps = True
        else:
            expanded_timesteps = False

        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])

        if hasattr(self, "time_projection"):
            time_embed_dtype = self.time_embedding[0].weight.dtype
            if time_embed_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
                time_embed_dtype = self.base_dtype
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(time_embed_dtype))  # b, dim
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim
            if use_token_replace:
                e_token_replace = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_token_replace.flatten()).to(time_embed_dtype))  # b, dim
                e0_token_replace = self.time_projection(e_token_replace).unflatten(1, (6, self.dim))  # b, 6, dim
        else:
            time_embed_dtype = self.time_embedding.mlp[0].weight.dtype
            if time_embed_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
                time_embed_dtype = self.base_dtype
            if len(t.shape) == 1:
                t = t.unsqueeze(1).expand(-1, F) # [B, T]
            self.time_embedding.to(torch.float32)
            e = e0 = self.time_embedding(t.float().flatten(), dtype=torch.float32)#.reshape(1, F, -1)
            e = e0 = e0.reshape(1, F, -1)

        if self.audio_model is not None:
            #if t.dim() == 1:
            #    t_ovi = t.unsqueeze(1).expand(t.size(0), seq_len_ovi)
            if t.dim() == 2:
                last_timestep = t[:, -1:]
                padding = last_timestep.expand(t.size(0), seq_len_ovi - t.size(1))
                t_ovi = torch.cat([t, padding], dim=1)

                e_ovi = self.audio_model.time_embedding(sinusoidal_embedding_1d(self.audio_model.freq_dim, t_ovi.flatten()).to(time_embed_dtype)).unsqueeze(0)  # b, dim
                e0_ovi = self.audio_model.time_projection(e_ovi).unflatten(2, (6, self.dim)).movedim(1, 2)  # B, seq_len, 6, dim
            else:
                e_ovi = self.audio_model.time_embedding(sinusoidal_embedding_1d(self.audio_model.freq_dim, t.flatten()).to(time_embed_dtype))  # b, dim
                e0_ovi = self.audio_model.time_projection(e_ovi).unflatten(1, (6, self.dim))  # b, 6, dim


        #S2V zero timestep
        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat([
                e0.unsqueeze(2),
                zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)
            ], dim=2)
            e0 = [e0, self.original_seq_len]

        if x_ip is not None:
            timestep_ip = torch.zeros_like(t)  # [B] with 0s
            t_ip = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_ip.flatten()).to(time_embed_dtype))  # b, dim )
            e0_ip = self.time_projection(t_ip).unflatten(1, (6, self.dim))

        if fps_embeds is not None:
            fps_embeds = torch.tensor(fps_embeds, dtype=torch.long, device=device)

            fps_emb = self.fps_embedding(fps_embeds).to(e0.dtype)
            if expanded_timesteps:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
            else:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if expanded_timesteps:
            e = e.view(b, f, 1, 1, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], self.dim)
            e0 = e0.view(b, f, 1, 1, 6, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], 6, self.dim)

            e = e.flatten(1, 3)
            e0 = e0.flatten(1, 3)

            e0 = e0.transpose(1, 2)
            if not e0.is_contiguous():
                e0 = e0.contiguous()

            e = e.to(self.offload_device, non_blocking=self.use_non_blocking)

        # clip vision embedding
        clip_embed = None
        if clip_fea is not None and hasattr(self, "img_emb"):
            if self.offload_img_emb:
                self.img_emb.to(self.main_device)
            clip_embed = self.img_emb(clip_fea.to(self.main_device))  # bs x 257 x dim
            if sdancer_input is not None:
                clip_fea_c = sdancer_input.get("clip_fea_c", None)
                if clip_fea_c is not None:
                    clip_embed += self.img_emb(clip_fea_c.to(self.main_device))
            if self.offload_img_emb:
                self.img_emb.to(self.offload_device, non_blocking=self.use_non_blocking)

        #context (text embedding)
        if hasattr(self, "text_embedding") and context != []:
            text_embed_dtype = self.text_embedding[0].weight.dtype
            if text_embed_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
                text_embed_dtype = self.base_dtype
            if self.offload_txt_emb:
                self.text_embedding.to(self.main_device)

            if inner_t is not None:
                if nag_context is not None:
                    raise NotImplementedError("nag_context is not supported with EchoShot")
                inner_c = [[u.shape[0] for u in context]]

            if self.audio_model is not None:
                if is_uncond and ovi_negative_text_embeds is not None:
                    context_ovi = ovi_negative_text_embeds
                else:
                    context_ovi = context
                context_ovi = self.audio_model.text_embedding(
                    torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context_ovi]).to(text_embed_dtype))

            tokens = context[0].shape[0]
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]).to(text_embed_dtype)

            if add_text_emb is not None:
                self.text_projection.to(self.main_device)
                add_text_emb = self.text_projection(add_text_emb.to(self.text_projection[0].weight.dtype)).to(text_embed_dtype)
                context = torch.cat([add_text_emb, context], dim=1)
            context = self.text_embedding(context)

            if self.is_longcat:
                context[:, tokens:] = 0

            # NAG
            if nag_context is not None:
                nag_context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in nag_context
                ]).to(text_embed_dtype))

            if self.offload_txt_emb:
                self.text_embedding.to(self.offload_device, non_blocking=self.use_non_blocking)

            seq_chunks = max(context.shape[0], clip_embed.shape[0] if clip_embed is not None else 0)
            chunked_self_attention = seq_chunks > 1 and current_step in self.video_attention_split_steps
        else:
            context = None
            chunked_self_attention = False
            seq_chunks = 0

        # dual control
        if dual_control_input is not None and dual_control_input["start_percent"] <= current_step_percentage <= dual_control_input["end_percent"]:
            dense_latent = dual_control_input["dense_input_latent"]
            print("dense_latent shape:", dense_latent.shape)
            sparse_latent = dual_control_input["sparse_input_latent"]
            if dense_latent is None and sparse_latent is None:
                raise ValueError("At least one of dense_input_latent or sparse_input_latent must be provided in dual_control_input")

            if dense_latent is not None:
                dense_x = [self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in dense_latent]
                dense_x = [u.flatten(2).transpose(1, 2).to(self.base_dtype) for u in dense_x]
                dense = self.dual_controller.control_initial_combine_linear_dense(dense_x[0])

            if sparse_latent is not None:
                sparse_x = [self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype) for u in sparse_latent]
                sparse_x = [u.flatten(2).transpose(1, 2).to(self.base_dtype) for u in sparse_x]
                sparse = self.dual_controller.control_initial_combine_linear_sparse(sparse_x[0])

            if dense_latent is None:
                dense = torch.zeros_like(sparse)
            elif sparse_latent is None:
                sparse = torch.zeros_like(dense)

            control_context = clip_fea_control = None
            if context != []:
                control_context = self.dual_controller.control_text_linear(context)
                if clip_embed is not None:
                    clip_fea_control = self.dual_controller.control_text_linear(clip_embed)
            control_t_mod = self.dual_controller.control_t_mod(e0)

            control_freqs = torch.cat([
                self.dual_controller_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.dual_controller_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.dual_controller_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        else:
            dual_control_input = None

        # MultiTalk
        if multitalk_audio is not None:
            self.multitalk_audio_proj.to(self.main_device)
            audio_cond = multitalk_audio.to(device=x.device, dtype=self.base_dtype)
            first_frame_audio_emb_s = audio_cond[:, :1, ...] 
            latter_frame_audio_emb = audio_cond[:, 1:, ...] 
            latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=4) 
            middle_index = self.multitalk_audio_proj.seq_len // 2
            latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
            latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
            latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
            latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
            multitalk_audio_embedding = self.multitalk_audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
            self.multitalk_audio_proj.to(self.offload_device)
            human_num = len(multitalk_audio_embedding)

            # LongCat-Avatar specific
            if longcat_num_ref_latents > 0:
                audio_start_ref = multitalk_audio_embedding[:, [0], :, :] # padding
                multitalk_audio_embedding = torch.cat([audio_start_ref, multitalk_audio_embedding], dim=1).contiguous()

            if longcat_num_cond_latents > 0:
                multitalk_audio_embedding = multitalk_audio_embedding[:, (-F // self.patch_size[0]):]

            if ref_target_masks is not None:
                multitalk_audio_embedding = torch.concat(multitalk_audio_embedding.split(1), dim=2).to(self.base_dtype)
                multitalk_audio_embedding = multitalk_audio_embedding.squeeze(0)
            else:
                multitalk_audio_embedding = rearrange(multitalk_audio_embedding, "b t n c -> (b t) n c")


        # convert ref_target_masks to token_ref_target_masks
        token_ref_target_masks = None
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(H // 2, W // 2), mode='nearest') 
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
            token_ref_target_masks = token_ref_target_masks.to(device, self.base_dtype)

        humo_audio_input = None
        if humo_audio is not None:
            humo_audio_input = self.audio_proj(humo_audio.unsqueeze(0)).permute(0, 3, 1, 2)

            humo_audio_seq_len = torch.tensor(humo_audio.shape[2] * humo_audio_input.shape[3], device=device)
            humo_audio_input = humo_audio_input.flatten(2).transpose(1, 2) # 1, t*32, 1536
            pad_len = int(humo_audio_seq_len - humo_audio_input.size(1))
            if pad_len > 0:
                humo_audio_input = torch.nn.functional.pad(humo_audio_input, (0, 0, 0, pad_len))

        should_calc = True
        #TeaCache
        if self.enable_teacache and self.teacache_start_step <= current_step <= self.teacache_end_step:
            accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
            if pred_id is None:
                pred_id = self.teacache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                previous_modulated_input = self.teacache_state.get(pred_id)['previous_modulated_input']
                previous_modulated_input = previous_modulated_input.to(device)
                previous_residual = self.teacache_state.get(pred_id)['previous_residual']
                accumulated_rel_l1_distance = self.teacache_state.get(pred_id)['accumulated_rel_l1_distance']

                if self.teacache_use_coefficients:
                    rescale_func = np.poly1d(self.teacache_coefficients[self.teacache_mode])
                    temb = e if self.teacache_mode == 'e' else e0
                    accumulated_rel_l1_distance += rescale_func((
                        (temb.to(device) - previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()
                        ).cpu().item())
                    del temb
                else:
                    temb_relative_l1 = relative_l1_distance(previous_modulated_input, e0)
                    accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(e0.device) + temb_relative_l1
                    del temb_relative_l1


                if accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
                accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(self.cache_device)

            previous_modulated_input = e.to(self.cache_device).clone() if (self.teacache_use_coefficients and self.teacache_mode == 'e') else e0.to(self.cache_device).clone()

            if not should_calc:
                x = x.to(previous_residual.dtype) + previous_residual.to(x.device)
                self.teacache_state.update(
                    pred_id,
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                )
                self.teacache_state.get(pred_id)['skipped_steps'].append(current_step)

        # MagCache
        if self.enable_magcache and self.magcache_start_step <= current_step <= self.magcache_end_step:
            if pred_id is None:
                pred_id = self.magcache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                accumulated_ratio = self.magcache_state.get(pred_id)['accumulated_ratio']
                accumulated_err = self.magcache_state.get(pred_id)['accumulated_err']
                accumulated_steps = self.magcache_state.get(pred_id)['accumulated_steps']

                calibration_len = len(self.magcache_ratios) // 2
                cur_mag_ratio = self.magcache_ratios[int((current_step*(calibration_len/total_steps)))]

                accumulated_ratio *= cur_mag_ratio
                accumulated_err += np.abs(1-accumulated_ratio)
                accumulated_steps += 1

                self.magcache_state.update(
                    pred_id,
                    accumulated_ratio=accumulated_ratio,
                    accumulated_steps=accumulated_steps,
                    accumulated_err=accumulated_err
                )

                if accumulated_err<=self.magcache_thresh and accumulated_steps<=self.magcache_K:
                    should_calc = False
                    x += self.magcache_state.get(pred_id)['residual_cache'].to(x.device)
                    self.magcache_state.get(pred_id)['skipped_steps'].append(current_step)
                else:
                    should_calc = True
                    self.magcache_state.update(
                        pred_id,
                        accumulated_ratio=1.0,
                        accumulated_steps=0,
                        accumulated_err=0
                    )

        # EasyCache
        if self.enable_easycache and self.easycache_start_step <= current_step <= self.easycache_end_step:
            if pred_id is None:
                pred_id = self.easycache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                state = self.easycache_state.get(pred_id)
                previous_raw_input = state.get('previous_raw_input')
                previous_raw_output = state.get('previous_raw_output')
                cache = state.get('cache')
                cache_ovi = state.get('cache_ovi') if self.audio_model is not None else None
                accumulated_error = state.get('accumulated_error')
                k = state.get('k', 1)

                if previous_raw_input is not None and previous_raw_output is not None:
                    raw_input = x.clone()
                    # Calculate input change
                    raw_input_change = (raw_input - previous_raw_input.to(raw_input.device)).abs().mean()

                    output_norm = (previous_raw_output.to(x.device)).abs().mean()

                    combined_pred_change = (raw_input_change / output_norm) * k

                    accumulated_error += combined_pred_change

                    # Predict output change
                    if accumulated_error < self.easycache_thresh:
                        should_calc = False
                        x = raw_input + cache.to(x.device)
                        if cache_ovi is not None:
                            x_ovi = x_ovi + cache_ovi.to(x_ovi.device)
                        state['skipped_steps'].append(current_step)
                    else:
                        should_calc = True
                else:
                    should_calc = True

        x = x.to(self.base_dtype)
        if isinstance(e0, list):
            e0 = [item.to(self.base_dtype) if torch.is_tensor(item) else item for item in e0]
        else:
            e0 = e0.to(self.base_dtype)

        if self.enable_easycache:
            original_x = x.clone().to(self.cache_device)
            if x_ovi is not None:
                original_x_ovi = x_ovi.clone().to(self.cache_device)
        if should_calc:
            if self.enable_teacache or self.enable_magcache:
                original_x = x.clone().to(self.cache_device)

            if hasattr(self, "dwpose_embedding") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    dwpose_emb = rearrange(unianim_data['dwpose'], 'b c f h w -> b (f h w) c').contiguous()
                    x.add_(dwpose_emb, alpha=unianim_data['strength'])

            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                clip_embed=clip_embed,
                current_step=torch.tensor(current_step),
                last_step=torch.tensor(last_step, dtype=torch.bool),
                chunked_self_attention=chunked_self_attention,
                seq_chunks=seq_chunks,
                camera_embed=camera_embed,
                audio_proj=audio_proj,
                num_latent_frames = F,
                frame_tokens=x.shape[1] // F,
                original_seq_len=self.original_seq_len,
                enhance_enabled=enhance_enabled,
                audio_scale=audio_scale,
                nag_params=nag_params,
                nag_context=nag_context if not is_uncond else None,
                multitalk_audio_embedding=multitalk_audio_embedding if multitalk_audio is not None else None,
                ref_target_masks=token_ref_target_masks if multitalk_audio is not None else None,
                human_num=human_num if multitalk_audio is not None else 0,
                inner_t=inner_t, inner_c=inner_c,
                cross_freqs=self.cross_freqs if inner_t is not None and not is_uncond else None,
                freqs_ip=freqs_ip if x_ip is not None else None,
                e_ip=e0_ip if x_ip is not None else None,
                adapter_proj=adapter_proj,
                ip_scale=ip_scale,
                reverse_time=reverse_time,
                mtv_motion_tokens=mtv_motion_tokens, mtv_motion_rotary_emb=mtv_motion_rotary_emb, mtv_strength=mtv_strength, mtv_freqs=mtv_freqs,
                humo_audio_input=humo_audio_input,
                humo_audio_scale=humo_audio_scale,
                lynx_x_ip=lynx_x_ip,
                lynx_ip_scale=lynx_ip_scale,
                lynx_ref_scale=lynx_ref_scale,
                longcat_num_cond_latents=longcat_num_cond_latents,
                longcat_avatar_options=longcat_avatar_options,
                onetoall_ref_scale=onetoall_ref_scale,
                e_tr=e0_token_replace if use_token_replace else None,
                tr_start=token_replace_start,
                tr_num=replace_token_num,
                transformer_options=transformer_options
            )
            if self.audio_model is not None:
                kwargs['e_ovi'] = e0_ovi.to(self.base_dtype)
                kwargs['context_ovi'] = context_ovi
                kwargs['grid_sizes_ovi'] = grid_sizes_ovi
                kwargs['seq_lens_ovi'] = seq_lens_ovi
                kwargs['freqs_ovi'] = freqs_ovi


            if vace_data is not None:
                vace_hint_list = []
                vace_scale_list = []
                if isinstance(vace_data[0], dict):
                    for data in vace_data:
                        if (data["start"] <= current_step_percentage <= data["end"]) or \
                            (data["end"] > 0 and current_step == 0 and current_step_percentage >= data["start"]):

                            vace_hints = self.forward_vace(x, data["context"], data["seq_len"], kwargs)
                            vace_hint_list.append(vace_hints)
                            vace_scale_list.append(data["scale"][current_step])
                else:
                    vace_hints = self.forward_vace(x, vace_data, seq_len, kwargs)
                    vace_hint_list.append(vace_hints)
                    vace_scale_list.append(1.0)

                kwargs['vace_hints'] = vace_hint_list
                kwargs['vace_context_scale'] = vace_scale_list

            #uni3c controlnet
            uni3c_controlnet_states = None
            if uni3c_data is not None:
                if (uni3c_data["start"] <= current_step_percentage <= uni3c_data["end"]) or \
                            (uni3c_data["end"] > 0 and current_step == 0 and current_step_percentage >= uni3c_data["start"]):
                    if uni3c_data["offload"] or self.uni3c_controlnet.device != self.main_device:
                        self.uni3c_controlnet.to(self.main_device)
                    with torch.autocast(device_type=mm.get_autocast_device(device), dtype=self.base_dtype, enabled=self.uni3c_controlnet.quantized):
                        uni3c_controlnet_states = self.uni3c_controlnet(
                            render_latent=render_latent.to(self.main_device, self.uni3c_controlnet.dtype),
                            render_mask=uni3c_data["render_mask"],
                            camera_embedding=uni3c_data["camera_embedding"],
                            temb=e.to(self.main_device),
                            out_device=self.offload_device if uni3c_data["offload"] else device)
                    if uni3c_data["offload"]:
                        self.uni3c_controlnet.to(self.offload_device)

            # Asynchronous block offloading with CUDA streams and events
            if torch.cuda.is_available():
                cuda_stream = None #torch.cuda.Stream(device=device, priority=0) # todo causes issues on some systems
                events = [torch.cuda.Event() for _ in self.blocks]
                swap_start_idx = len(self.blocks) - self.blocks_to_swap if self.blocks_to_swap > 0 else len(self.blocks)
            else:
                cuda_stream = None
                events = None
                swap_start_idx = len(self.blocks)

            # lynx ref
            if lynx_ref_buffer is None and lynx_ref_feature_extractor:
                lynx_ref_buffer = {}

            attn_override_blocks = attention_mode = None
            attention_mode_override_active = False
            attention_mode_override = transformer_options.get("attention_mode_override", None)
            if attention_mode_override is not None:
                attn_override_blocks = attention_mode_override.get("blocks", range(len(self.blocks)))
                if attention_mode_override["start_step"] <= current_step < attention_mode_override["end_step"]:
                    attention_mode_override_active = True
                    if attention_mode_override["verbose"]:
                        tqdm.write(f"Applying attention mode override: {attention_mode_override['mode']} at step {current_step} on blocks: {attn_override_blocks if attn_override_blocks is not None else 'all'}")

            for b, block in enumerate(self.blocks):
                mm.throw_exception_if_processing_interrupted()
                if attention_mode_override_active and b in attn_override_blocks:
                    attention_mode = attention_mode_override['mode']
                else:
                    attention_mode = None
                block_idx = f"{b:02d}"
                if lynx_ref_buffer is not None and not lynx_ref_feature_extractor:
                    lynx_ref_feature = lynx_ref_buffer.get(block_idx, None)
                else:
                    lynx_ref_feature = None
                # FlashVSR
                if flashvsr_LQ_latent is not None and b < len(flashvsr_LQ_latent):
                    x += flashvsr_LQ_latent[b].to(x) * flashvsr_strength
                # Prefetch blocks if enabled
                if self.prefetch_blocks > 0:
                    for prefetch_offset in range(1, self.prefetch_blocks + 1):
                        prefetch_idx = b + prefetch_offset
                        if prefetch_idx < len(self.blocks) and self.blocks_to_swap > 0 and prefetch_idx >= swap_start_idx:
                            context_mgr = torch.cuda.stream(cuda_stream) if torch.cuda.is_available() else nullcontext()
                            with context_mgr:
                                self.blocks[prefetch_idx].to(self.main_device, non_blocking=self.use_non_blocking)
                                if events is not None:
                                    events[prefetch_idx].record(cuda_stream)
                if self.block_swap_debug:
                    transfer_start = time.perf_counter()
                # Wait for block to be ready
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    if self.prefetch_blocks > 0 and events is not None:
                        if not events[b].query():
                            events[b].synchronize()
                    block.to(self.main_device)
                if self.block_swap_debug:
                    transfer_end = time.perf_counter()
                    transfer_time = transfer_end - transfer_start
                    compute_start = time.perf_counter()
                #skip layer guidance
                if self.slg_blocks is not None:
                    if b in self.slg_blocks and is_uncond:
                        if self.slg_start_percent <= current_step_percentage <= self.slg_end_percent:
                            continue

                x_onetoall_ref = None
                if onetoall_ref_block_samples is not None:
                    interval_ref = len(self.blocks) / len(onetoall_ref_block_samples)
                    interval_ref = int(np.ceil(interval_ref))
                    x_onetoall_ref = onetoall_ref_block_samples[b // interval_ref]

                # ---run block----#
                x, x_ip, lynx_ref_feature, x_ovi = block(x, x_ip=x_ip, lynx_ref_feature=lynx_ref_feature, x_ovi=x_ovi, x_onetoall_ref=x_onetoall_ref, onetoall_freqs=onetoall_freqs, attention_mode_override=attention_mode, **kwargs)
                # ---post block----#

                # dual controlnet
                if dual_control_input is not None and (hasattr(block, "control_blocks_dense") or hasattr(block, "control_blocks_sparse")):
                    if dense_latent is not None and hasattr(block, "control_blocks_dense"):
                        dense = block.control_blocks_dense(dense, control_context, control_t_mod, control_freqs, clip_fea=clip_fea_control)
                    if sparse_latent is not None and hasattr(block, "control_blocks_sparse"):
                        sparse = block.control_blocks_sparse(sparse, control_context, control_t_mod, control_freqs, clip_fea=clip_fea_control)

                    if prev_latent is not None:
                        x[:, -self.original_seq_len:] += block.control_combine_linears(dense + sparse) * dual_control_input["strength"]
                    else:
                        x += block.control_combine_linears(dense + sparse) * dual_control_input["strength"]

                if self.audio_injector is not None and s2v_audio_input is not None:
                    x = self.audio_injector_forward(b, x, merged_audio_emb, scale=s2v_audio_scale) #s2v
                if block.has_face_fuser_block and motion_vec is not None:
                    x = self.wananimate_forward(block, x, motion_vec, strength=wananim_face_strength)
                if self.block_swap_debug:
                    compute_end = time.perf_counter()
                    compute_time = compute_end - compute_start
                    to_cpu_transfer_start = time.perf_counter()
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
                if self.block_swap_debug:
                    to_cpu_transfer_end = time.perf_counter()
                    to_cpu_transfer_time = to_cpu_transfer_end - to_cpu_transfer_start
                    log.info(f"Block {b}: transfer_time={transfer_time:.4f}s, compute_time={compute_time:.4f}s, to_cpu_transfer_time={to_cpu_transfer_time:.4f}s")
                # lynx ref
                if lynx_ref_feature_extractor:
                    if b in lynx_ref_blocks_to_use:
                        log.info(f"storing to lynx ref buffer for block {block_idx}")
                        lynx_ref_buffer[block_idx] = lynx_ref_feature
                #uni3c controlnet
                if uni3c_controlnet_states is not None and b < len(uni3c_controlnet_states):
                    x[:, :self.original_seq_len] += uni3c_controlnet_states[b].to(x) * uni3c_data["controlnet_weight"]
                #controlnet
                if (controlnet is not None) and (b % controlnet["controlnet_stride"] == 0) and (b // controlnet["controlnet_stride"] < len(controlnet["controlnet_states"])):
                    x[:, :self.original_seq_len] += controlnet["controlnet_states"][b // controlnet["controlnet_stride"]].to(x) * controlnet["controlnet_weight"]
                # One-to-All-Animation controlnet
                if onetoall_control_enabled:
                    if prev_x is not None and (b - 1) < len(self.controlnet.blocks):
                        #tqdm.write(f"Applying One-to-All ControlNet at block {b}")
                        if b == 1:
                            ctrl_in = prev_x + controlnet_tokens
                        elif prev_control is not None:
                            ctrl_in = prev_control

                        self.controlnet.blocks[b - 1].to(self.main_device)
                        control_out = self.controlnet.blocks[b - 1](ctrl_in, e0, seq_lens, freqs, e_tr=e0_token_replace, tr_num=replace_token_num,tr_start=token_replace_start, split_rope=False)
                        self.controlnet.blocks[b - 1].to(self.offload_device, non_blocking=self.use_non_blocking)
                        prev_control = control_out

                        control_out_proj = self.controlnet_zero[b - 1](control_out)
                        x = x + control_out_proj * one_to_all_controlnet_strength
                    if b < len(self.controlnet.blocks): # Store prev_x only while controlnet is active
                        prev_x = x
                    elif b == len(self.controlnet.blocks): # Controlnet done, free memory
                        prev_x = None
                        prev_control = None
                        if controlnet_tokens is not None:
                            del controlnet_tokens
                            controlnet_tokens = None
                        mm.soft_empty_cache()

            if lynx_ref_feature_extractor:
                return lynx_ref_buffer

            if self.enable_teacache and (self.teacache_start_step <= current_step <= self.teacache_end_step) and pred_id is not None:
                self.teacache_state.update(
                    pred_id,
                    previous_residual=(x.to(original_x.device) - original_x),
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                    previous_modulated_input=previous_modulated_input
                )
            elif self.enable_magcache and (self.magcache_start_step <= current_step <= self.magcache_end_step) and pred_id is not None:
                self.magcache_state.update(
                    pred_id,
                    residual_cache=(x.to(original_x.device) - original_x)
                )
            elif self.enable_easycache and (self.easycache_start_step <= current_step <= self.easycache_end_step) and pred_id is not None:
                x_out = x.clone().to(original_x.device)
                output_change = (x_out - original_x).abs().mean()
                input_change = (original_x - x_out).abs().mean()
                self.easycache_state.update(
                    pred_id,
                    previous_raw_input=original_x,
                    previous_raw_output=x_out,
                    cache=x.to(original_x.device) - original_x,
                    k = output_change / input_change,
                    accumulated_error = 0.0,
                    cache_ovi = x_ovi.clone().to(original_x.device) - original_x_ovi if x_ovi is not None else None
                )



        if self.enable_easycache and (self.easycache_start_step <= current_step <= self.easycache_end_step) and pred_id is not None:
            self.easycache_state.update(
                pred_id,
                previous_raw_output=x.clone(),
            )

        if self.ref_conv is not None and fun_ref is not None:
            fun_ref_length = fun_ref.size(1)
            x = x[:, fun_ref_length:]
            #grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if end_ref_latent is not None:
            end_ref_latent_length = end_ref_latent.size(1)
            x = x[:, :-end_ref_latent_length]
            #grid_sizes = torch.stack([torch.tensor([u[0] - end_ref_latent_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        #if attn_cond is not None:
        #    x = x[:, :self.original_seq_len]
            #grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if prev_latent is not None:
            x = x[:, -self.original_seq_len:]
        else:
            x = x[:, :self.original_seq_len]

        x = self.head(x, e.to(x.device), temp_length=F,
                      e_tr=e_token_replace.to(x.device) if use_token_replace else None, tr_start=token_replace_start, tr_num=replace_token_num)

        if x_ovi is not None:
            x_ovi = self.audio_model.head(x_ovi, e_ovi.to(x_ovi.device))
            grid_sizes_ovi = [gs[0] for gs in grid_sizes_ovi]
            assert len(x) == len(grid_sizes_ovi)
            x_ovi = [u[:gs] for u, gs in zip(x_ovi, grid_sizes_ovi)]
            x_ovi = [u.float() for u in x_ovi]

        x = self.unpatchify(x, original_grid_sizes)
        x = [u[:, prefix_frames:suffix_frames, ...].float() for u in x]
        return (x, x_ovi, pred_id) if pred_id is not None else (x, x_ovi, None)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
