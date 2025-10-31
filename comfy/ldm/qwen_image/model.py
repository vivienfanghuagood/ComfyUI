# https://github.com/QwenLM/Qwen-Image (Apache 2.0)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import repeat
import os

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.layers import EmbedND
import comfy.ldm.common_dit
import comfy.patcher_extension

USE_TRITON_FUSION = os.environ.get("QWEN_USE_TRITON", "0") == "1"
USE_PROFILER = os.environ.get("QWEN_USE_PROFILER", "0") == "1"

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    USE_TRITON_FUSION = False

if TRITON_AVAILABLE and USE_TRITON_FUSION:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def fused_linear_gelu_kernel(
        x_ptr, weight_ptr, bias_ptr, output_ptr,
        M, K, N,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        USE_TANH_APPROX: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        mask_m = offs_m < M
        mask_n = offs_n < N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            
            w_ptrs = weight_ptr + offs_n[:, None] * stride_wn + (k + offs_k[None, :]) * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            
            acc += tl.dot(x_tile, tl.trans(w_tile))
        
        bias_ptrs = bias_ptr + offs_n
        bias_tile = tl.load(bias_ptrs, mask=mask_n, other=0.0)
        acc = acc + bias_tile[None, :]
        
        if USE_TANH_APPROX:
            sqrt_2_over_pi = 0.7978845608028654
            acc_cubed = acc * acc * acc
            inner = sqrt_2_over_pi * (acc + 0.044715 * acc_cubed)
            tanh_inner = tl.libdevice.tanh(inner)
            gelu_out = 0.5 * acc * (1.0 + tanh_inner)
        else:
            gelu_out = 0.5 * acc * (1.0 + tl.libdevice.erf(acc * 0.7071067811865476))
        
        out_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, gelu_out.to(output_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        ],
        key=['hidden_dim'],
    )
    @triton.jit
    def modulate_kernel(
        x_ptr, mod_params_ptr, output_ptr, gate_ptr,
        total_rows, hidden_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1)
        
        if row_idx >= total_rows:
            return
        
        row_start = row_idx * hidden_dim
        h_idx = col_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        mask = h_idx < hidden_dim
        base_offset = row_start + h_idx
        
        x = tl.load(x_ptr + base_offset, mask=mask, other=0.0)
        shift = tl.load(mod_params_ptr + h_idx, mask=mask, other=0.0)
        scale = tl.load(mod_params_ptr + hidden_dim + h_idx, mask=mask, other=0.0)
        gate = tl.load(mod_params_ptr + 2 * hidden_dim + h_idx, mask=mask, other=0.0)
        
        scale_plus_1 = scale + 1.0
        output = tl.fma(x, scale_plus_1, shift)
        
        tl.store(output_ptr + base_offset, output, mask=mask)
        tl.store(gate_ptr + base_offset, gate, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        ],
        key=['hidden_dim'],
    )
    @triton.jit
    def fused_norm_modulate_kernel_optimized(
        x_ptr, mod_params_ptr, output_ptr, gate_ptr,
        total_rows, hidden_dim,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        USE_BF16_ACCUM: tl.constexpr,
    ):
        """
        Single-pass fused LayerNorm + Modulate kernel.
        Each thread block processes one complete row (token).
        Optimized for hidden_dim=3072 on AMD MI300X.
        """
        row_idx = tl.program_id(0)
        
        if row_idx >= total_rows:
            return
        
        row_start = row_idx * hidden_dim
        hidden_dim_float = tl.cast(hidden_dim, tl.float32)
        
        # Phase 1: Compute statistics with optimized reduction
        # Process in chunks to improve cache locality
        num_blocks = hidden_dim // BLOCK_SIZE
        
        if USE_BF16_ACCUM:
            # Use BF16 for intermediate accumulation (faster but less precise)
            mean_acc = tl.zeros([1], dtype=tl.bfloat16)
            m2_acc = tl.zeros([1], dtype=tl.bfloat16)
        else:
            # Use FP32 for accumulation (more precise)
            mean_acc = 0.0
            m2_acc = 0.0
        
        # Unrolled reduction loop for better performance
        for block_idx in range(num_blocks):
            h_idx = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x_vals = tl.load(x_ptr + row_start + h_idx)
            
            if USE_BF16_ACCUM:
                x_bf16 = x_vals.to(tl.bfloat16)
                mean_acc += tl.sum(x_bf16)
                m2_acc += tl.sum(x_bf16 * x_bf16)
            else:
                x_f32 = x_vals.to(tl.float32)
                mean_acc += tl.sum(x_f32)
                m2_acc += tl.sum(x_f32 * x_f32)
        
        # Convert to FP32 for final statistics
        if USE_BF16_ACCUM:
            mean_acc_f32 = mean_acc.to(tl.float32)
            m2_acc_f32 = m2_acc.to(tl.float32)
        else:
            mean_acc_f32 = mean_acc
            m2_acc_f32 = m2_acc
        
        mean = mean_acc_f32 / hidden_dim_float
        variance = m2_acc_f32 / hidden_dim_float - mean * mean
        rstd = 1.0 / tl.sqrt(variance + eps)
        
        # Phase 2: Apply normalization and modulation
        for block_idx in range(num_blocks):
            h_idx = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            
            x_vals = tl.load(x_ptr + row_start + h_idx).to(tl.float32)
            shift = tl.load(mod_params_ptr + h_idx).to(tl.float32)
            scale = tl.load(mod_params_ptr + hidden_dim + h_idx).to(tl.float32)
            gate = tl.load(mod_params_ptr + 2 * hidden_dim + h_idx).to(tl.float32)
            
            # Fused normalization and modulation
            x_normalized = (x_vals - mean) * rstd
            output = tl.fma(x_normalized, scale + 1.0, shift)
            
            tl.store(output_ptr + row_start + h_idx, output.to(x_ptr.dtype.element_ty))
            tl.store(gate_ptr + row_start + h_idx, gate.to(x_ptr.dtype.element_ty))

    def triton_fused_norm_modulate(x, mod_params, norm_fn, eps=1e-6):
        batch, seq_len, hidden_dim = x.shape
        
        x_2d = x.reshape(-1, hidden_dim).contiguous()
        mod_params_2d = mod_params.reshape(-1, hidden_dim * 3).contiguous()
        
        output = torch.empty_like(x_2d)
        gate = torch.empty_like(x_2d)
        
        total_rows = x_2d.shape[0]
        USE_BF16_ACCUM = os.environ.get("QWEN_USE_BF16_ACCUM", "0") == "1"
        
        grid = (total_rows,)
        
        fused_norm_modulate_kernel_optimized[grid](
            x_2d, mod_params_2d, output, gate,
            total_rows, hidden_dim,
            eps=eps,
            USE_BF16_ACCUM=USE_BF16_ACCUM,
        )
        
        return output.view(batch, seq_len, hidden_dim), gate.view(batch, seq_len, hidden_dim)

    def triton_modulate(x, mod_params):
        batch, seq_len, hidden_dim = x.shape
        
        x_2d = x.reshape(-1, hidden_dim).contiguous()
        mod_params_2d = mod_params.reshape(-1, hidden_dim * 3).contiguous()
        
        output = torch.empty_like(x_2d)
        gate = torch.empty_like(x_2d)
        
        total_rows = x_2d.shape[0]
        num_col_blocks = triton.cdiv(hidden_dim, 256)
        
        grid = (total_rows, num_col_blocks)
        
        modulate_kernel[grid](
            x_2d, mod_params_2d, output, gate,
            total_rows, hidden_dim,
        )
        
        return output.view(batch, seq_len, hidden_dim), gate.view(batch, seq_len, hidden_dim)

    def triton_fused_linear_gelu(x, weight, bias, use_tanh_approx=True):
        batch, seq_len, in_features = x.shape
        out_features = weight.shape[0]
        
        x_2d = x.reshape(-1, in_features).contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        
        M = x_2d.shape[0]
        K = in_features
        N = out_features
        
        output = torch.empty((M, N), dtype=x.dtype, device=x.device)
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        
        fused_linear_gelu_kernel[grid](
            x_2d, weight, bias, output,
            M, K, N,
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(0), weight.stride(1),
            USE_TANH_APPROX=use_tanh_approx,
        )
        
        return output.view(batch, seq_len, out_features)

class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=bias, dtype=dtype, device=device)
        self.approximate = approximate

    def forward(self, hidden_states):
        if USE_TRITON_FUSION and TRITON_AVAILABLE and hidden_states.is_cuda:
            try:
                use_tanh = self.approximate == "tanh"
                return triton_fused_linear_gelu(
                    hidden_states, 
                    self.proj.weight, 
                    self.proj.bias,
                    use_tanh_approx=use_tanh
                )
            except:
                pass
        
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(GELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, operations=operations))
        self.net.append(nn.Dropout(dropout))
        self.net.append(operations.Linear(inner_dim, dim_out, bias=bias, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb(x, freqs_cis):
    if x.shape[1] == 0:
        return x

    t_ = x.reshape(*x.shape[:-1], -1, 1, 2)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.reshape(*x.shape)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return timesteps_emb


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections
        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Text stream projections
        self.add_q_proj = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Output projections
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype, device=device),
            nn.Dropout(dropout)
        ])
        self.to_add_out = operations.Linear(self.inner_dim, self.out_context_dim, bias=out_bias, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_txt = encoder_hidden_states.shape[1]

        img_query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        img_key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        img_value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        txt_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        joint_hidden_states = optimized_attention_masked(joint_query, joint_key, joint_value, self.heads, attention_mask, transformer_options=transformer_options)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if USE_TRITON_FUSION and TRITON_AVAILABLE and x.is_cuda:
            try:
                return triton_modulate(x, mod_params)
            except Exception as e:
                print(f"[TRITON] modulate failed: {e}, falling back to PyTorch")
        
        shift, scale, gate = torch.chunk(mod_params, 3, dim=-1)
        return torch.addcmul(shift.unsqueeze(1), x, 1 + scale.unsqueeze(1)), gate.unsqueeze(1)

    def _fused_norm_modulate(self, x: torch.Tensor, mod_params: torch.Tensor, norm_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        if USE_TRITON_FUSION and TRITON_AVAILABLE and x.is_cuda:
            try:
                return triton_fused_norm_modulate(x, mod_params, norm_fn, eps=1e-6)
            except Exception as e:
                print(f"[TRITON] fused_norm_modulate failed: {e}, falling back to PyTorch")
        
        x_normed = norm_fn(x)
        return self._modulate(x_normed, mod_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        if USE_TRITON_FUSION and TRITON_AVAILABLE and hidden_states.is_cuda:
            try:
                img_modulated, img_gate1 = self._fused_norm_modulate(hidden_states, img_mod1, self.img_norm1)
                txt_modulated, txt_gate1 = self._fused_norm_modulate(encoder_hidden_states, txt_mod1, self.txt_norm1)
            except:
                img_normed = self.img_norm1(hidden_states)
                img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
                txt_normed = self.txt_norm1(encoder_hidden_states)
                txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        else:
            img_normed = self.img_norm1(hidden_states)
            img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
            txt_normed = self.txt_norm1(encoder_hidden_states)
            txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            transformer_options=transformer_options,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        if USE_TRITON_FUSION and TRITON_AVAILABLE and hidden_states.is_cuda:
            try:
                img_modulated2, img_gate2 = self._fused_norm_modulate(hidden_states, img_mod2, self.img_norm2)
                hidden_states = torch.addcmul(hidden_states, img_gate2, self.img_mlp(img_modulated2))
                txt_modulated2, txt_gate2 = self._fused_norm_modulate(encoder_hidden_states, txt_mod2, self.txt_norm2)
                encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, self.txt_mlp(txt_modulated2))
            except:
                img_normed2 = self.img_norm2(hidden_states)
                img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
                hidden_states = torch.addcmul(hidden_states, img_gate2, self.img_mlp(img_modulated2))
                txt_normed2 = self.txt_norm2(encoder_hidden_states)
                txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
                encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, self.txt_mlp(txt_modulated2))
        else:
            img_normed2 = self.img_norm2(hidden_states)
            img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
            hidden_states = torch.addcmul(hidden_states, img_gate2, self.img_mlp(img_modulated2))
            txt_normed2 = self.txt_norm2(encoder_hidden_states)
            txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
            encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, self.txt_mlp(txt_modulated2))

        return encoder_hidden_states, hidden_states


class LastLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=False,
        eps=1e-6,
        bias=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = torch.addcmul(shift[:, None, :], self.norm(x), (1 + scale)[:, None, :])
        return x


class QwenImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        image_model=None,
        final_layer=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        
        self._forward_count = 0
        self._profiler = None
        self._profiler_enabled = USE_PROFILER

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                device=device,
                operations=operations
            )
            for _ in range(num_layers)
        ])

        if final_layer:
            self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
            self.proj_out = operations.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True, dtype=dtype, device=device)

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1) - (h_len // 2)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0) - (w_len // 2)
        return hidden_states, repeat(img_ids, "h w c -> b (h w) c", b=bs), orig_shape

    def forward(self, x, timestep, context, attention_mask=None, guidance=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, attention_mask, guidance, ref_latents, transformer_options, **kwargs)

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        control=None,
        **kwargs
    ):
        self._forward_count += 1
        
        if self._profiler_enabled:
            if self._forward_count == 10:
                print("[PROFILER] Starting profiler at forward #10")
                self._profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                self._profiler.__enter__()
            elif self._forward_count == 20:
                if self._profiler is not None:
                    self._profiler.__exit__(None, None, None)
                    print("[PROFILER] Stopped profiler at forward #20")
                    print("\n" + "="*80)
                    print("Profiler Summary (by CUDA time)")
                    print("="*80)
                    print(self._profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                    
                    trace_file = "qwen_image_trace.json"
                    self._profiler.export_chrome_trace(trace_file)
                    print(f"\n[PROFILER] Trace saved to {trace_file}")
                    print("="*80 + "\n")
                    self._profiler = None
        
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]
