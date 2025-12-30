# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast

from einops import rearrange, repeat

# 检查是否强制使用纯PyTorch实现
SELECTIVE_SCAN_FORCE_FALLBACK = os.environ.get("SELECTIVE_SCAN_FORCE_FALLBACK", "FALSE").upper() == "TRUE"
CAUSAL_CONV1D_FORCE_FALLBACK = os.environ.get("CAUSAL_CONV1D_FORCE_FALLBACK", "FALSE").upper() == "TRUE"

# 总是需要导入selective_scan_fn（包含Python参考实现）
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref, selective_fused_scan_fn

# 尝试导入全C++ VisionMamba实现
try:
    import vision_mamba_cpp
    HAS_VISION_MAMBA_CPP = True
except ImportError:
    vision_mamba_cpp = None
    HAS_VISION_MAMBA_CPP = False

# 只有在非回退模式下才尝试导入CUDA优化版本
if not SELECTIVE_SCAN_FORCE_FALLBACK:
    try:
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_fn_no_out_proj, bimamba_inner_fn
    except ImportError:
        mamba_inner_fn, mamba_inner_fn_no_out_proj, bimamba_inner_fn = None, None, None
else:
    mamba_inner_fn, mamba_inner_fn_no_out_proj, bimamba_inner_fn = None, None, None

if not CAUSAL_CONV1D_FORCE_FALLBACK:
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_fn, causal_conv1d_update = None, None
else:
    causal_conv1d_fn, causal_conv1d_update = None, None

# 只有在CUDA版本可用时才使用mamba_inner_fn系列函数
USE_CUDA_MAMBA = (not SELECTIVE_SCAN_FORCE_FALLBACK) and mamba_inner_fn is not None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_divide_out=False,
        init_layer_scale=None,
        use_cpp_scan=False,  # 使用C++优化实现
        use_fixlen_scan=False,  # 使用两阶段优化算法
        use_fused_bidirectional=False,  # 使用融合双向扫描
        use_full_cpp=False,  # 使用完整C++ Mamba实现（包括卷积、投影等）
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out
        self.use_cpp_scan = use_cpp_scan
        self.use_fixlen_scan = use_fixlen_scan
        self.use_fused_bidirectional = use_fused_bidirectional
        self.use_full_cpp = use_full_cpp and HAS_VISION_MAMBA_CPP

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # 全C++实现路径
        if self.use_full_cpp and HAS_VISION_MAMBA_CPP:
            return vision_mamba_cpp.mamba_forward(
                hidden_states,
                self.in_proj.weight,
                self.in_proj.bias if self.in_proj.bias is not None else torch.tensor([]),
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj.weight,
                self.x_proj_b.weight,
                self.dt_proj.weight,
                self.dt_proj.bias,
                self.dt_proj_b.weight,
                self.dt_proj_b.bias,
                self.A_log,
                self.A_b_log,
                self.D,
                self.D_b,
                self.out_proj.weight,
                self.out_proj.bias if self.out_proj.bias is not None else torch.tensor([]),
                self.d_conv,
                self.dt_rank,
                self.d_state,
                self.use_fused_bidirectional,
                self.use_fixlen_scan
            )

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 在纯PyTorch回退模式下使用参考实现
        if not USE_CUDA_MAMBA or SELECTIVE_SCAN_FORCE_FALLBACK:
            # 使用纯PyTorch实现，BiMamba v2双向扫描
            A_b = -torch.exp(self.A_b_log.float())
            
            # 选择使用融合双向扫描还是分离的两次扫描
            if self.use_fused_bidirectional:
                # 使用融合双向扫描优化
                #print('Using fused bidirectional selective_scan (data stacking optimization)')
                out = self._forward_fuse_reference(
                    xz, A, A_b, seqlen, batch, dim, conv_state, ssm_state, inference_params
                )
                # 应用输出投影
                out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                # 使用原始的分离双向扫描
                #print('bidirectional Mamba using pure PyTorch selective_scan reference implementation.')
                # 前向扫描：使用正向参数
                out = self._forward_reference(
                    xz, A, seqlen, batch, dim, conv_state, ssm_state, inference_params,
                    conv1d=self.conv1d,
                    x_proj=self.x_proj,
                    dt_proj=self.dt_proj,
                    D=self.D,
                    dt_bias=self.dt_proj.bias
                )
                
                # 反向扫描：使用反向参数和翻转的输入
                out_b = self._forward_reference(
                    xz.flip([-1]), A_b, seqlen, batch, dim, conv_state, ssm_state, inference_params,
                    conv1d=self.conv1d_b,
                    x_proj=self.x_proj_b,
                    dt_proj=self.dt_proj_b,
                    D=self.D_b,
                    dt_bias=self.dt_proj_b.bias
                )
                
                # 合并前向和反向结果，应用输出投影
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        # CUDA优化版本
        elif self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )    
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                # print("F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)")
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def _forward_reference(self, xz, A, seqlen, batch, dim, conv_state, ssm_state, inference_params,
                          conv1d, x_proj, dt_proj, D, dt_bias):
        """纯PyTorch参考实现
        
        Args:
            xz: 输入张量 (b, 2*d_inner, l)
            A: 状态转移矩阵 (d_inner, d_state)
            seqlen: 序列长度
            batch: batch大小
            dim: 原始维度
            conv_state: 卷积状态（用于增量推理）
            ssm_state: SSM状态（用于增量推理）
            inference_params: 推理参数
            conv1d: 卷积层（前向或反向）
            x_proj: x投影层（前向或反向）
            dt_proj: dt投影层（前向或反向）
            D: D参数（前向或反向）
            dt_bias: dt偏置（前向或反向）
        """
        x, z = xz.chunk(2, dim=1)
        
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            
        # 使用传入的卷积层替代causal_conv1d
        x = self.act(conv1d(x)[..., :seqlen])

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        
        # 使用参考实现的selective_scan
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
            use_cpp=self.use_cpp_scan,
            use_fixlen=self.use_fixlen_scan,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        # 返回 (b, d, l) 格式，与GPU版本一致
        return y
    
    def _forward_fuse_reference(self, xz, A_fwd, A_bwd, seqlen, batch, dim, conv_state, ssm_state, inference_params):
        """融合双向Selective Scan - 在N维度concat，两阶段批量计算
        
        核心策略：
        1. 在状态空间维度N上concat正向和反向参数，形成2N的融合状态空间
        2. 阶段1：一次递推循环同时计算正向和反向的所有隐藏状态x（共2N个）
        3. 阶段2：一次einsum批量计算所有输出y
        
        数学原理：
        正向：x_fwd[i] = deltaA_fwd[i] * x_fwd[i-1] + deltaB_u_fwd[i]  (N个状态)
        反向：x_bwd[i] = deltaA_bwd[i] * x_bwd[i-1] + deltaB_u_bwd[i]  (N个状态)
        融合：x_bi[i] = deltaA_bi[i] * x_bi[i-1] + deltaB_u_bi[i]      (2N个状态，包含前N和后N)
        
        预期加速：2-3x
        
        Returns:
            out: 融合后的输出 (b, d, l)
        """
        x, z = xz.chunk(2, dim=1)  # x: (b, d_inner, l), z: (b, d_inner, l)
        
        # ===== 步骤1：反转反向输入 =====
        x_bwd_flip = x.flip(dims=[2])  # (b, d_inner, l)
        z_bwd_flip = z.flip(dims=[2])
        
        # ===== 步骤2：卷积处理 =====
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        
        x_fwd_conv = self.act(self.conv1d(x)[..., :seqlen])  # (b, d_inner, l)
        x_bwd_conv_flip = self.act(self.conv1d_b(x_bwd_flip)[..., :seqlen])  # (b, d_inner, l)
        
        # ===== 步骤3：参数投影 =====
        # 正向
        x_dbl_fwd = self.x_proj(rearrange(x_fwd_conv, "b d l -> (b l) d"))
        dt_fwd, B_fwd, C_fwd = torch.split(x_dbl_fwd, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_fwd = self.dt_proj.weight @ dt_fwd.t()
        dt_fwd = rearrange(dt_fwd, "d (b l) -> b d l", l=seqlen)
        B_fwd = rearrange(B_fwd, "(b l) n -> b n l", l=seqlen).contiguous()
        C_fwd = rearrange(C_fwd, "(b l) n -> b n l", l=seqlen).contiguous()
        
        # 反向
        x_dbl_bwd = self.x_proj_b(rearrange(x_bwd_conv_flip, "b d l -> (b l) d"))
        dt_bwd, B_bwd, C_bwd = torch.split(x_dbl_bwd, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_bwd = self.dt_proj_b.weight @ dt_bwd.t()
        dt_bwd = rearrange(dt_bwd, "d (b l) -> b d l", l=seqlen)
        B_bwd = rearrange(B_bwd, "(b l) n -> b n l", l=seqlen).contiguous()
        C_bwd = rearrange(C_bwd, "(b l) n -> b n l", l=seqlen).contiguous()
        
        # 应用delta_bias和softplus
        dt_fwd = dt_fwd + self.dt_proj.bias[..., None].float()
        dt_bwd = dt_bwd + self.dt_proj_b.bias[..., None].float()
        dt_fwd = F.softplus(dt_fwd)
        dt_bwd = F.softplus(dt_bwd)
        
        # ===== 步骤4：在N维度concat - 融合两个方向的状态空间！=====
        # 关键思想：正向有N个状态，反向有N个状态，concat成2N个状态
        A_bi = torch.cat([A_fwd, A_bwd], dim=1)  # (d, 2n)
        B_bi = torch.cat([B_fwd, B_bwd], dim=1)  # (b, 2n, l)
        C_bi = torch.cat([C_fwd, C_bwd], dim=1)  # (b, 2n, l)
        
        # 卷积输出concat在d维度（正向d个通道+反向d个通道）
        x_conv_bi = torch.cat([x_fwd_conv, x_bwd_conv_flip], dim=1)  # (b, 2d, l)
        dt_bi = torch.cat([dt_fwd, dt_bwd], dim=1)  # (b, 2d, l)
        D_bi = torch.cat([self.D, self.D_b], dim=0)  # (2d,)
        z_bi = torch.cat([z, z_bwd_flip], dim=1)  # (b, 2d, l)
        
        # ===== 步骤5-8：调用selective_fused_scan_fn完成融合双向扫描 =====
        # 从步骤5开始，包括deltaA和deltaB_u的计算
        # 根据use_cpp_scan和use_fixlen_scan参数选择实现
        out = selective_fused_scan_fn(
            dt_fwd, dt_bwd,
            A_fwd, A_bwd,
            B_fwd, B_bwd,
            x_fwd_conv, x_bwd_conv_flip,
            C_fwd, C_bwd,
            self.D, self.D_b,
            z_fwd=z if z is not None else None,
            z_bwd_flip=z_bwd_flip if z is not None else None,
            use_cpp=self.use_cpp_scan,
            use_fixlen=self.use_fixlen_scan
        )
        
        return out


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
