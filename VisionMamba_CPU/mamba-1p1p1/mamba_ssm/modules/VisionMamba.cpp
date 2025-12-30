//Presented by KeJi
//Date: 2025-12-29

#include <torch/extension.h>
#include <vector>
#include <cmath>

/*
 * Vision Mamba C++ 完整实现
 *
 * 实现BiMamba v2的完整前向传播，调用selective_scan.cpp中的优化函数
 *
 * 注意：此文件需要与selective_scan.cpp一起编译
 */

// 引入selective_scan函数声明（定义在selective_scan.cpp中）
extern torch::Tensor Selective_Scan_Ref_Cpu(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D,
    const torch::Tensor& z,
    const torch::Tensor& delta_bias,
    bool delta_softplus,
    bool return_last_state
);

extern torch::Tensor Selective_Scan_Ref_Fixlen_Cpu(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D,
    const torch::Tensor& z,
    const torch::Tensor& delta_bias,
    bool delta_softplus,
    bool return_last_state
);

extern torch::Tensor Selective_Fused_Scan_Cpu(
    const torch::Tensor& dt_fwd,
    const torch::Tensor& dt_bwd,
    const torch::Tensor& A_fwd,
    const torch::Tensor& A_bwd,
    const torch::Tensor& B_fwd,
    const torch::Tensor& B_bwd,
    const torch::Tensor& x_fwd_conv,
    const torch::Tensor& x_bwd_conv_flip,
    const torch::Tensor& C_fwd,
    const torch::Tensor& C_bwd,
    const torch::Tensor& D_fwd,
    const torch::Tensor& D_bwd,
    const torch::Tensor& z_fwd,
    const torch::Tensor& z_bwd_flip
);

extern torch::Tensor Selective_Fused_Scan_Fixlen_Cpu(
    const torch::Tensor& dt_fwd,
    const torch::Tensor& dt_bwd,
    const torch::Tensor& A_fwd,
    const torch::Tensor& A_bwd,
    const torch::Tensor& B_fwd,
    const torch::Tensor& B_bwd,
    const torch::Tensor& x_fwd_conv,
    const torch::Tensor& x_bwd_conv_flip,
    const torch::Tensor& C_fwd,
    const torch::Tensor& C_bwd,
    const torch::Tensor& D_fwd,
    const torch::Tensor& D_bwd,
    const torch::Tensor& z_fwd,
    const torch::Tensor& z_bwd_flip
);

// 辅助函数
inline bool is_none(const torch::Tensor& t) {
    return !t.defined() || t.numel() == 0;
}

// 分离双向前向传播实现（Original/Fixlen）
torch::Tensor Mamba_Forward_Bidirectional_Cpu(
    const torch::Tensor& xz,                    // (B, 2*d_inner, L)
    const torch::Tensor& conv1d_fwd_weight,     // (d_inner, 1, d_conv)
    const torch::Tensor& conv1d_fwd_bias,       // (d_inner,)
    const torch::Tensor& conv1d_bwd_weight,     // (d_inner, 1, d_conv)
    const torch::Tensor& conv1d_bwd_bias,       // (d_inner,)
    const torch::Tensor& x_proj_fwd_weight,     // (dt_rank+2*d_state, d_inner)
    const torch::Tensor& x_proj_bwd_weight,     // (dt_rank+2*d_state, d_inner)
    const torch::Tensor& dt_proj_fwd_weight,    // (d_inner, dt_rank)
    const torch::Tensor& dt_proj_fwd_bias,      // (d_inner,)
    const torch::Tensor& dt_proj_bwd_weight,    // (d_inner, dt_rank)
    const torch::Tensor& dt_proj_bwd_bias,      // (d_inner,)
    const torch::Tensor& A_log_fwd,             // (d_inner, d_state)
    const torch::Tensor& A_log_bwd,             // (d_inner, d_state)
    const torch::Tensor& D_fwd,                 // (d_inner,)
    const torch::Tensor& D_bwd,                 // (d_inner,)
    const torch::Tensor& out_proj_weight,       // (d_model, d_inner)
    const torch::Tensor& out_proj_bias,         // (d_model,) 可选
    int64_t d_conv,
    int64_t dt_rank,
    int64_t d_state,
    bool use_fixlen = false
) {
    const int64_t batch = xz.size(0);
    const int64_t d_inner = xz.size(1) / 2;
    const int64_t seqlen = xz.size(2);
    
    // 分离x和z
    auto chunks = xz.chunk(2, 1);
    auto x = chunks[0];  // (B, d_inner, L)
    auto z = chunks[1];  // (B, d_inner, L)
    
    // 反向输入翻转
    auto x_bwd_flip = x.flip({2});
    auto z_bwd_flip = z.flip({2});
    
    // ===== 正向卷积 =====
    // 重塑卷积权重为2D: (d_inner, 1, d_conv) -> (d_inner, d_conv)
    auto conv_w_fwd = conv1d_fwd_weight.squeeze(1);  // (d_inner, d_conv)
    auto conv_w_bwd = conv1d_bwd_weight.squeeze(1);  // (d_inner, d_conv)
    
    // 手动实现分组卷积
    // F.conv1d with groups=d_inner
    auto x_fwd_padded = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({d_conv - 1, 0}));
    auto x_bwd_padded = torch::nn::functional::pad(x_bwd_flip, torch::nn::functional::PadFuncOptions({d_conv - 1, 0}));
    
    // 使用分组卷积
    auto x_fwd_conv = torch::nn::functional::conv1d(
        x_fwd_padded, 
        conv1d_fwd_weight,
        torch::nn::functional::Conv1dFuncOptions().groups(d_inner).bias(conv1d_fwd_bias)
    );
    x_fwd_conv = x_fwd_conv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, seqlen)});
    x_fwd_conv = torch::silu(x_fwd_conv);
    
    auto x_bwd_conv_flip = torch::nn::functional::conv1d(
        x_bwd_padded,
        conv1d_bwd_weight,
        torch::nn::functional::Conv1dFuncOptions().groups(d_inner).bias(conv1d_bwd_bias)
    );
    x_bwd_conv_flip = x_bwd_conv_flip.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, seqlen)});
    x_bwd_conv_flip = torch::silu(x_bwd_conv_flip);
    
    // ===== 参数投影 =====
    // x_proj: (d_inner, L) -> (dt_rank+2*d_state, L)
    // rearrange "b d l -> (b l) d" then linear
    auto x_fwd_flat = x_fwd_conv.permute({0, 2, 1}).reshape({batch * seqlen, d_inner});  // (BL, d_inner)
    auto x_bwd_flat = x_bwd_conv_flip.permute({0, 2, 1}).reshape({batch * seqlen, d_inner});
    
    auto x_dbl_fwd = torch::mm(x_fwd_flat, x_proj_fwd_weight.t());  // (BL, dt_rank+2*d_state)
    auto x_dbl_bwd = torch::mm(x_bwd_flat, x_proj_bwd_weight.t());
    
    // 分离dt, B, C
    auto splits_fwd = x_dbl_fwd.split({dt_rank, d_state, d_state}, 1);
    auto dt_fwd_flat = splits_fwd[0];  // (BL, dt_rank)
    auto B_fwd_flat = splits_fwd[1];   // (BL, d_state)
    auto C_fwd_flat = splits_fwd[2];   // (BL, d_state)
    
    auto splits_bwd = x_dbl_bwd.split({dt_rank, d_state, d_state}, 1);
    auto dt_bwd_flat = splits_bwd[0];
    auto B_bwd_flat = splits_bwd[1];
    auto C_bwd_flat = splits_bwd[2];
    
    // dt投影: dt_proj_weight @ dt.t() -> (d_inner, BL)
    auto dt_fwd = torch::mm(dt_proj_fwd_weight, dt_fwd_flat.t());  // (d_inner, BL)
    dt_fwd = dt_fwd.reshape({d_inner, batch, seqlen}).permute({1, 0, 2});  // (B, d_inner, L)
    
    auto dt_bwd = torch::mm(dt_proj_bwd_weight, dt_bwd_flat.t());
    dt_bwd = dt_bwd.reshape({d_inner, batch, seqlen}).permute({1, 0, 2});
    
    // B, C reshape: (BL, d_state) -> (B, d_state, L)
    auto B_fwd = B_fwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    auto C_fwd = C_fwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    
    auto B_bwd = B_bwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    auto C_bwd = C_bwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    
    // A = -exp(A_log)
    auto A_fwd = -torch::exp(A_log_fwd.to(torch::kFloat32));
    auto A_bwd = -torch::exp(A_log_bwd.to(torch::kFloat32));
    
    // ===== Selective Scan =====
    torch::Tensor y_fwd, y_bwd;
    
    if (use_fixlen) {
        y_fwd = Selective_Scan_Ref_Fixlen_Cpu(
            x_fwd_conv, dt_fwd, A_fwd, B_fwd, C_fwd,
            D_fwd.to(torch::kFloat32), z, dt_proj_fwd_bias.to(torch::kFloat32),
            true, false
        );
        y_bwd = Selective_Scan_Ref_Fixlen_Cpu(
            x_bwd_conv_flip, dt_bwd, A_bwd, B_bwd, C_bwd,
            D_bwd.to(torch::kFloat32), z_bwd_flip, dt_proj_bwd_bias.to(torch::kFloat32),
            true, false
        );
    } else {
        y_fwd = Selective_Scan_Ref_Cpu(
            x_fwd_conv, dt_fwd, A_fwd, B_fwd, C_fwd,
            D_fwd.to(torch::kFloat32), z, dt_proj_fwd_bias.to(torch::kFloat32),
            true, false
        );
        y_bwd = Selective_Scan_Ref_Cpu(
            x_bwd_conv_flip, dt_bwd, A_bwd, B_bwd, C_bwd,
            D_bwd.to(torch::kFloat32), z_bwd_flip, dt_proj_bwd_bias.to(torch::kFloat32),
            true, false
        );
    }
    
    // 合并正向和反向结果
    auto y = y_fwd + y_bwd.flip({2});  // (B, d_inner, L)
    
    // 输出投影: (B, L, d_inner) @ (d_inner, d_model) -> (B, L, d_model)
    y = y.permute({0, 2, 1});  // (B, L, d_inner)
    torch::Tensor out;
    if (is_none(out_proj_bias)) {
        out = torch::mm(y.reshape({batch * seqlen, d_inner}), out_proj_weight.t());
    } else {
        out = torch::addmm(out_proj_bias, y.reshape({batch * seqlen, d_inner}), out_proj_weight.t());
    }
    out = out.reshape({batch, seqlen, -1});  // (B, L, d_model)
    
    return out;
}

// 融合双向前向传播实现（Fused/Fused-Fixlen）
torch::Tensor Mamba_Forward_Fused_Cpu(
    const torch::Tensor& xz,                    // (B, 2*d_inner, L)
    const torch::Tensor& conv1d_fwd_weight,     // (d_inner, 1, d_conv)
    const torch::Tensor& conv1d_fwd_bias,       // (d_inner,)
    const torch::Tensor& conv1d_bwd_weight,     // (d_inner, 1, d_conv)
    const torch::Tensor& conv1d_bwd_bias,       // (d_inner,)
    const torch::Tensor& x_proj_fwd_weight,     // (dt_rank+2*d_state, d_inner)
    const torch::Tensor& x_proj_bwd_weight,     // (dt_rank+2*d_state, d_inner)
    const torch::Tensor& dt_proj_fwd_weight,    // (d_inner, dt_rank)
    const torch::Tensor& dt_proj_fwd_bias,      // (d_inner,)
    const torch::Tensor& dt_proj_bwd_weight,    // (d_inner, dt_rank)
    const torch::Tensor& dt_proj_bwd_bias,      // (d_inner,)
    const torch::Tensor& A_log_fwd,             // (d_inner, d_state)
    const torch::Tensor& A_log_bwd,             // (d_inner, d_state)
    const torch::Tensor& D_fwd,                 // (d_inner,)
    const torch::Tensor& D_bwd,                 // (d_inner,)
    const torch::Tensor& out_proj_weight,       // (d_model, d_inner)
    const torch::Tensor& out_proj_bias,         // (d_model,) 可选
    int64_t d_conv,
    int64_t dt_rank,
    int64_t d_state,
    bool use_fixlen = false
) {
    const int64_t batch = xz.size(0);
    const int64_t d_inner = xz.size(1) / 2;
    const int64_t seqlen = xz.size(2);
    
    // 分离x和z
    auto chunks = xz.chunk(2, 1);
    auto x = chunks[0];  // (B, d_inner, L)
    auto z = chunks[1];  // (B, d_inner, L)
    
    // 反向输入翻转
    auto x_bwd_flip = x.flip({2});
    auto z_bwd_flip = z.flip({2});
    
    // ===== 卷积 =====
    auto x_fwd_padded = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({d_conv - 1, 0}));
    auto x_bwd_padded = torch::nn::functional::pad(x_bwd_flip, torch::nn::functional::PadFuncOptions({d_conv - 1, 0}));
    
    auto x_fwd_conv = torch::nn::functional::conv1d(
        x_fwd_padded, 
        conv1d_fwd_weight,
        torch::nn::functional::Conv1dFuncOptions().groups(d_inner).bias(conv1d_fwd_bias)
    );
    x_fwd_conv = x_fwd_conv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, seqlen)});
    x_fwd_conv = torch::silu(x_fwd_conv);
    
    auto x_bwd_conv_flip = torch::nn::functional::conv1d(
        x_bwd_padded,
        conv1d_bwd_weight,
        torch::nn::functional::Conv1dFuncOptions().groups(d_inner).bias(conv1d_bwd_bias)
    );
    x_bwd_conv_flip = x_bwd_conv_flip.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, seqlen)});
    x_bwd_conv_flip = torch::silu(x_bwd_conv_flip);
    
    // ===== 参数投影 =====
    auto x_fwd_flat = x_fwd_conv.permute({0, 2, 1}).reshape({batch * seqlen, d_inner});
    auto x_bwd_flat = x_bwd_conv_flip.permute({0, 2, 1}).reshape({batch * seqlen, d_inner});
    
    auto x_dbl_fwd = torch::mm(x_fwd_flat, x_proj_fwd_weight.t());
    auto x_dbl_bwd = torch::mm(x_bwd_flat, x_proj_bwd_weight.t());
    
    auto splits_fwd = x_dbl_fwd.split({dt_rank, d_state, d_state}, 1);
    auto dt_fwd_flat = splits_fwd[0];
    auto B_fwd_flat = splits_fwd[1];
    auto C_fwd_flat = splits_fwd[2];
    
    auto splits_bwd = x_dbl_bwd.split({dt_rank, d_state, d_state}, 1);
    auto dt_bwd_flat = splits_bwd[0];
    auto B_bwd_flat = splits_bwd[1];
    auto C_bwd_flat = splits_bwd[2];
    
    auto dt_fwd = torch::mm(dt_proj_fwd_weight, dt_fwd_flat.t());
    dt_fwd = dt_fwd.reshape({d_inner, batch, seqlen}).permute({1, 0, 2});
    
    auto dt_bwd = torch::mm(dt_proj_bwd_weight, dt_bwd_flat.t());
    dt_bwd = dt_bwd.reshape({d_inner, batch, seqlen}).permute({1, 0, 2});
    
    auto B_fwd = B_fwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    auto C_fwd = C_fwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    auto B_bwd = B_bwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    auto C_bwd = C_bwd_flat.reshape({batch, seqlen, d_state}).permute({0, 2, 1}).contiguous();
    
    // 应用delta_bias和softplus
    dt_fwd = dt_fwd + dt_proj_fwd_bias.unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
    dt_bwd = dt_bwd + dt_proj_bwd_bias.unsqueeze(0).unsqueeze(-1).to(torch::kFloat32);
    dt_fwd = torch::nn::functional::softplus(dt_fwd);
    dt_bwd = torch::nn::functional::softplus(dt_bwd);
    
    // A = -exp(A_log)
    auto A_fwd = -torch::exp(A_log_fwd.to(torch::kFloat32));
    auto A_bwd = -torch::exp(A_log_bwd.to(torch::kFloat32));
    
    // ===== 融合Selective Scan =====
    torch::Tensor y;
    
    if (use_fixlen) {
        y = Selective_Fused_Scan_Fixlen_Cpu(
            dt_fwd, dt_bwd,
            A_fwd, A_bwd,
            B_fwd, B_bwd,
            x_fwd_conv, x_bwd_conv_flip,
            C_fwd, C_bwd,
            D_fwd.to(torch::kFloat32), D_bwd.to(torch::kFloat32),
            z, z_bwd_flip
        );
    } else {
        y = Selective_Fused_Scan_Cpu(
            dt_fwd, dt_bwd,
            A_fwd, A_bwd,
            B_fwd, B_bwd,
            x_fwd_conv, x_bwd_conv_flip,
            C_fwd, C_bwd,
            D_fwd.to(torch::kFloat32), D_bwd.to(torch::kFloat32),
            z, z_bwd_flip
        );
    }
    
    // 输出投影
    y = y.permute({0, 2, 1});  // (B, L, d_inner)
    torch::Tensor out;
    if (is_none(out_proj_bias)) {
        out = torch::mm(y.reshape({batch * seqlen, d_inner}), out_proj_weight.t());
    } else {
        out = torch::addmm(out_proj_bias, y.reshape({batch * seqlen, d_inner}), out_proj_weight.t());
    }
    out = out.reshape({batch, seqlen, -1});  // (B, L, d_model)
    
    return out;
}

// 完整的Mamba前向传播入口函数
torch::Tensor Mamba_Forward_Cpu(
    const torch::Tensor& hidden_states,         // (B, L, d_model)
    const torch::Tensor& in_proj_weight,        // (2*d_inner, d_model)
    const torch::Tensor& in_proj_bias,          // (2*d_inner,) 可选
    const torch::Tensor& conv1d_fwd_weight,     // (d_inner, 1, d_conv)
    const torch::Tensor& conv1d_fwd_bias,       // (d_inner,)
    const torch::Tensor& conv1d_bwd_weight,     // (d_inner, 1, d_conv)
    const torch::Tensor& conv1d_bwd_bias,       // (d_inner,)
    const torch::Tensor& x_proj_fwd_weight,     // (dt_rank+2*d_state, d_inner)
    const torch::Tensor& x_proj_bwd_weight,     // (dt_rank+2*d_state, d_inner)
    const torch::Tensor& dt_proj_fwd_weight,    // (d_inner, dt_rank)
    const torch::Tensor& dt_proj_fwd_bias,      // (d_inner,)
    const torch::Tensor& dt_proj_bwd_weight,    // (d_inner, dt_rank)
    const torch::Tensor& dt_proj_bwd_bias,      // (d_inner,)
    const torch::Tensor& A_log_fwd,             // (d_inner, d_state)
    const torch::Tensor& A_log_bwd,             // (d_inner, d_state)
    const torch::Tensor& D_fwd,                 // (d_inner,)
    const torch::Tensor& D_bwd,                 // (d_inner,)
    const torch::Tensor& out_proj_weight,       // (d_model, d_inner)
    const torch::Tensor& out_proj_bias,         // (d_model,) 可选
    int64_t d_conv,
    int64_t dt_rank,
    int64_t d_state,
    bool use_fused = false,
    bool use_fixlen = false
) {
    const int64_t batch = hidden_states.size(0);
    const int64_t seqlen = hidden_states.size(1);
    const int64_t d_model = hidden_states.size(2);
    
    // 输入投影: (B, L, d_model) -> (B, 2*d_inner, L)
    // in_proj_weight @ hidden_states.t() -> (2*d_inner, BL)
    auto h_flat = hidden_states.reshape({batch * seqlen, d_model});  // (BL, d_model)
    auto xz = torch::mm(h_flat, in_proj_weight.t());  // (BL, 2*d_inner)
    
    if (!is_none(in_proj_bias)) {
        xz = xz + in_proj_bias;
    }
    
    // reshape to (B, 2*d_inner, L)
    xz = xz.reshape({batch, seqlen, -1}).permute({0, 2, 1}).contiguous();
    
    // 选择实现方式
    if (use_fused) {
        return Mamba_Forward_Fused_Cpu(
            xz,
            conv1d_fwd_weight, conv1d_fwd_bias,
            conv1d_bwd_weight, conv1d_bwd_bias,
            x_proj_fwd_weight, x_proj_bwd_weight,
            dt_proj_fwd_weight, dt_proj_fwd_bias,
            dt_proj_bwd_weight, dt_proj_bwd_bias,
            A_log_fwd, A_log_bwd,
            D_fwd, D_bwd,
            out_proj_weight, out_proj_bias,
            d_conv, dt_rank, d_state, use_fixlen
        );
    } else {
        return Mamba_Forward_Bidirectional_Cpu(
            xz,
            conv1d_fwd_weight, conv1d_fwd_bias,
            conv1d_bwd_weight, conv1d_bwd_bias,
            x_proj_fwd_weight, x_proj_bwd_weight,
            dt_proj_fwd_weight, dt_proj_fwd_bias,
            dt_proj_bwd_weight, dt_proj_bwd_bias,
            A_log_fwd, A_log_bwd,
            D_fwd, D_bwd,
            out_proj_weight, out_proj_bias,
            d_conv, dt_rank, d_state, use_fixlen
        );
    }
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mamba_forward", &Mamba_Forward_Cpu,
          "Mamba Forward CPU - complete BiMamba v2 implementation",
          py::arg("hidden_states"),
          py::arg("in_proj_weight"),
          py::arg("in_proj_bias"),
          py::arg("conv1d_fwd_weight"),
          py::arg("conv1d_fwd_bias"),
          py::arg("conv1d_bwd_weight"),
          py::arg("conv1d_bwd_bias"),
          py::arg("x_proj_fwd_weight"),
          py::arg("x_proj_bwd_weight"),
          py::arg("dt_proj_fwd_weight"),
          py::arg("dt_proj_fwd_bias"),
          py::arg("dt_proj_bwd_weight"),
          py::arg("dt_proj_bwd_bias"),
          py::arg("A_log_fwd"),
          py::arg("A_log_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("out_proj_weight"),
          py::arg("out_proj_bias"),
          py::arg("d_conv"),
          py::arg("dt_rank"),
          py::arg("d_state"),
          py::arg("use_fused") = false,
          py::arg("use_fixlen") = false);
    
    m.def("mamba_forward_fused", &Mamba_Forward_Fused_Cpu,
          "Mamba Forward Fused CPU - fused bidirectional implementation",
          py::arg("xz"),
          py::arg("conv1d_fwd_weight"),
          py::arg("conv1d_fwd_bias"),
          py::arg("conv1d_bwd_weight"),
          py::arg("conv1d_bwd_bias"),
          py::arg("x_proj_fwd_weight"),
          py::arg("x_proj_bwd_weight"),
          py::arg("dt_proj_fwd_weight"),
          py::arg("dt_proj_fwd_bias"),
          py::arg("dt_proj_bwd_weight"),
          py::arg("dt_proj_bwd_bias"),
          py::arg("A_log_fwd"),
          py::arg("A_log_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("out_proj_weight"),
          py::arg("out_proj_bias"),
          py::arg("d_conv"),
          py::arg("dt_rank"),
          py::arg("d_state"),
          py::arg("use_fixlen") = false);
    
    m.def("mamba_forward_bidirectional", &Mamba_Forward_Bidirectional_Cpu,
          "Mamba Forward Bidirectional CPU - separated bidirectional implementation",
          py::arg("xz"),
          py::arg("conv1d_fwd_weight"),
          py::arg("conv1d_fwd_bias"),
          py::arg("conv1d_bwd_weight"),
          py::arg("conv1d_bwd_bias"),
          py::arg("x_proj_fwd_weight"),
          py::arg("x_proj_bwd_weight"),
          py::arg("dt_proj_fwd_weight"),
          py::arg("dt_proj_fwd_bias"),
          py::arg("dt_proj_bwd_weight"),
          py::arg("dt_proj_bwd_bias"),
          py::arg("A_log_fwd"),
          py::arg("A_log_bwd"),
          py::arg("D_fwd"),
          py::arg("D_bwd"),
          py::arg("out_proj_weight"),
          py::arg("out_proj_bias"),
          py::arg("d_conv"),
          py::arg("dt_rank"),
          py::arg("d_state"),
          py::arg("use_fixlen") = false);
}
