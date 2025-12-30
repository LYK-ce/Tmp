//Presented by KeJi
//Date: 2025-12-22

#include <torch/extension.h>
#include <vector>

/*
 * Vision Mamba Selective Scan C++ 完整实现
 * 
 * 完全复刻selective_scan_interface.py:235的selective_scan_ref函数
 */

// 辅助函数：检查tensor是否为None
inline bool is_none(const torch::Tensor& t) {
    return !t.defined() || t.numel() == 0;
}

// 完整复刻selective_scan_ref
torch::Tensor Selective_Scan_Ref_Cpu(
    const torch::Tensor& u,              // (B, D, L)
    const torch::Tensor& delta,          // (B, D, L)
    const torch::Tensor& A,              // (D, N)
    const torch::Tensor& B,              // (D, N) or (B, N, L)
    const torch::Tensor& C,              // (D, N) or (B, N, L)
    const torch::Tensor& D = torch::Tensor(),       // (D,) - 可选
    const torch::Tensor& z = torch::Tensor(),       // (B, D, L) - 可选
    const torch::Tensor& delta_bias = torch::Tensor(),  // (D,) - 可选
    bool delta_softplus = false,         // 是否对delta应用softplus
    bool return_last_state = false       // 是否返回last_state（简化：忽略）
) {
    // Line 250-252: 数据类型转换
    auto dtype_in = u.dtype();
    auto u_f = u.to(torch::kFloat32);
    auto delta_f = delta.to(torch::kFloat32);
    
    // Line 253-256: 处理delta_bias和delta_softplus
    if (!is_none(delta_bias)) {
        delta_f = delta_f + delta_bias.unsqueeze(-1).to(torch::kFloat32);
    }
    if (delta_softplus) {
        delta_f = torch::nn::functional::softplus(delta_f);
    }
    
    // Line 257: 获取维度
    const int64_t batch = u_f.size(0);
    const int64_t dim = A.size(0);
    const int64_t dstate = A.size(1);
    const int64_t seq_len = u_f.size(2);
    
    // Line 258-259: 判断B和C是否是variable
    bool is_variable_B = B.dim() >= 3;
    bool is_variable_C = C.dim() >= 3;
    
    // Line 265-267: 转换为float
    auto B_f = B.to(torch::kFloat32);
    auto C_f = C.to(torch::kFloat32);
    
    // Line 268: 初始化隐藏状态 x
    auto x = torch::zeros({batch, dim, dstate}, u_f.options());
    
    // Line 269: 初始化输出列表
    std::vector<torch::Tensor> ys;
    ys.reserve(seq_len);
    
    // Line 270: deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    // delta: (B,D,L), A: (D,N) -> (B,D,L,N)
    auto deltaA = torch::exp(
        delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    );
    
    // Line 271-278: 计算deltaB_u
    torch::Tensor deltaB_u;
    if (!is_variable_B) {
        // Line 272: deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        // delta: (B,D,L), B: (D,N), u: (B,D,L) -> (B,D,L,N)
        deltaB_u = delta_f.unsqueeze(-1) * B_f.unsqueeze(0).unsqueeze(2) * u_f.unsqueeze(-1);
    } else {
        if (B_f.dim() == 3) {
            // Line 275: deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            // delta: (B,D,L), B: (B,N,L), u: (B,D,L) -> (B,D,L,N)
            // B: (B,N,L) -> (B,1,N,L) -> transpose to (B,1,L,N)
            deltaB_u = delta_f.unsqueeze(-1) * 
                      B_f.unsqueeze(1).transpose(2, 3) * 
                      u_f.unsqueeze(-1);
        } else {
            throw std::runtime_error("Grouped B (dim==4) not supported");
        }
    }
    
    // Line 282-295: 主循环
    for (int64_t i = 0; i < seq_len; i++) {
        // Line 283: x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        x = deltaA.index({torch::indexing::Slice(), torch::indexing::Slice(), i}) * x
            + deltaB_u.index({torch::indexing::Slice(), torch::indexing::Slice(), i});
        
        // Line 284-290: 计算y
        torch::Tensor y;
        if (!is_variable_C) {
            // Line 285: y = torch.einsum('bdn,dn->bd', x, C)
            // x: (B,D,N), C: (D,N) -> (B,D)
            y = torch::sum(x * C_f.unsqueeze(0), -1);
        } else {
            if (C_f.dim() == 3) {
                // Line 287-288: y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                // x: (B,D,N), C[:,:,i]: (B,N) -> (B,D)
                auto C_i = C_f.index({torch::indexing::Slice(), torch::indexing::Slice(), i});  // (B,N)
                y = torch::sum(x * C_i.unsqueeze(1), -1);  // (B,D)
            } else {
                throw std::runtime_error("Grouped C (dim==4) not supported");
            }
        }
        
        // Line 295: ys.append(y)
        ys.push_back(y);
    }
    
    // Line 296: y = torch.stack(ys, dim=2)  # (batch, dim, L)
    auto y = torch::stack(ys, 2);
    
    // Line 297: out = y if D is None else y + u * rearrange(D, "d -> d 1")
    torch::Tensor out;
    if (is_none(D)) {
        out = y;
    } else {
        // u: (B,D,L), D: (D,) -> (1,D,1) -> (B,D,L)
        out = y + u_f * D.to(torch::kFloat32).unsqueeze(0).unsqueeze(-1);
    }
    
    // Line 298-299: 应用z门控
    if (!is_none(z)) {
        out = out * torch::nn::functional::silu(z.to(torch::kFloat32));
    }
    
    // Line 300: 转换回原始数据类型
    out = out.to(dtype_in);
    
    return out;
}

// 优化的两阶段算法
torch::Tensor Selective_Scan_Ref_Fixlen_Cpu(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D = torch::Tensor(),
    const torch::Tensor& z = torch::Tensor(),
    const torch::Tensor& delta_bias = torch::Tensor(),
    bool delta_softplus = false,
    bool return_last_state = false
) {
    // 前处理同原始版本
    auto dtype_in = u.dtype();
    auto u_f = u.to(torch::kFloat32);
    auto delta_f = delta.to(torch::kFloat32);
    
    if (!is_none(delta_bias)) {
        delta_f = delta_f + delta_bias.unsqueeze(-1).to(torch::kFloat32);
    }
    if (delta_softplus) {
        delta_f = torch::nn::functional::softplus(delta_f);
    }
    
    const int64_t batch = u_f.size(0);
    const int64_t dim = A.size(0);
    const int64_t dstate = A.size(1);
    const int64_t seq_len = u_f.size(2);
    
    bool is_variable_B = B.dim() >= 3;
    bool is_variable_C = C.dim() >= 3;
    
    auto B_f = B.to(torch::kFloat32);
    auto C_f = C.to(torch::kFloat32);
    
    // 计算deltaA
    auto deltaA = torch::exp(
        delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
    );
    
    // 计算deltaB_u
    torch::Tensor deltaB_u;
    if (!is_variable_B) {
        deltaB_u = delta_f.unsqueeze(-1) * B_f.unsqueeze(0).unsqueeze(2) * u_f.unsqueeze(-1);
    } else {
        if (B_f.dim() == 3) {
            deltaB_u = delta_f.unsqueeze(-1) * 
                      B_f.unsqueeze(1).transpose(2, 3) * 
                      u_f.unsqueeze(-1);
        } else {
            throw std::runtime_error("Grouped B not supported");
        }
    }
    
    // 阶段1：递推计算所有隐藏状态（使用select优化，避免clone）
    // 公式：deltaB_u[i] = deltaA[i] * deltaB_u[i-1] + deltaB_u[i]
    // 改写为：deltaB_u[i] += deltaA[i] * deltaB_u[i-1]
    for (int64_t i = 1; i < seq_len; i++) {
        auto deltaA_i = deltaA.select(2, i);       // (B, D, N) - 视图
        auto prev = deltaB_u.select(2, i - 1);     // (B, D, N) - 视图
        // 原地加法：curr += deltaA_i * prev
        deltaB_u.select(2, i).add_(deltaA_i * prev);
    }
    
    // 阶段2：批量计算输出
    torch::Tensor y;
    if (!is_variable_C) {
        y = torch::sum(deltaB_u * C_f.unsqueeze(0).unsqueeze(2), -1);
    } else {
        if (C_f.dim() == 3) {
            y = torch::sum(deltaB_u * C_f.unsqueeze(1).transpose(2, 3), -1);
        } else {
            throw std::runtime_error("Grouped C not supported");
        }
    }
    
    // 添加D和z
    torch::Tensor out;
    if (is_none(D)) {
        out = y;
    } else {
        out = y + u_f * D.to(torch::kFloat32).unsqueeze(0).unsqueeze(-1);
    }
    
    if (!is_none(z)) {
        out = out * torch::nn::functional::silu(z.to(torch::kFloat32));
    }
    
    out = out.to(dtype_in);
    
    return out;
}

// 融合双向Selective Scan（标准版本）
// 使用(B,D,L,2N)布局保持与广播操作兼容，避免permute开销
torch::Tensor Selective_Fused_Scan_Cpu(
    const torch::Tensor& dt_fwd,            // (B, D, L)
    const torch::Tensor& dt_bwd,            // (B, D, L)
    const torch::Tensor& A_fwd,             // (D, N)
    const torch::Tensor& A_bwd,             // (D, N)
    const torch::Tensor& B_fwd,             // (B, N, L)
    const torch::Tensor& B_bwd,             // (B, N, L)
    const torch::Tensor& x_fwd_conv,        // (B, D, L)
    const torch::Tensor& x_bwd_conv_flip,   // (B, D, L)
    const torch::Tensor& C_fwd,             // (B, N, L)
    const torch::Tensor& C_bwd,             // (B, N, L)
    const torch::Tensor& D_fwd,             // (D,)
    const torch::Tensor& D_bwd,             // (D,)
    const torch::Tensor& z_fwd = torch::Tensor(),       // (B, D, L) - 可选
    const torch::Tensor& z_bwd_flip = torch::Tensor()   // (B, D, L) - 可选
) {
    auto dtype_in = dt_fwd.dtype();
    
    const int64_t batch = dt_fwd.size(0);
    const int64_t dim = dt_fwd.size(1);
    const int64_t seq_len = dt_fwd.size(2);
    const int64_t dstate = A_fwd.size(1);
    
    // 使用(B,D,L,2N)布局，与广播操作兼容，避免permute
    auto deltaA_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    auto deltaB_u_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    
    // 计算并写入（无需permute）
    // dt: (B,D,L) -> (B,D,L,1), A: (D,N) -> (1,D,1,N)
    // 正向部分 [:,:,:,:n] -> narrow(3, 0, dstate)
    deltaA_bi.narrow(3, 0, dstate).copy_(
        torch::exp(dt_fwd.unsqueeze(-1) * A_fwd.unsqueeze(0).unsqueeze(2))
    );
    // B: (B,N,L) -> (B,1,L,N)
    deltaB_u_bi.narrow(3, 0, dstate).copy_(
        dt_fwd.unsqueeze(-1) * B_fwd.unsqueeze(1).transpose(2, 3) * x_fwd_conv.unsqueeze(-1)
    );
    
    // 反向部分 [:,:,:,n:] -> narrow(3, dstate, dstate)
    deltaA_bi.narrow(3, dstate, dstate).copy_(
        torch::exp(dt_bwd.unsqueeze(-1) * A_bwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, dstate, dstate).copy_(
        dt_bwd.unsqueeze(-1) * B_bwd.unsqueeze(1).transpose(2, 3) * x_bwd_conv_flip.unsqueeze(-1)
    );
    
    // 阶段1：递推 - (B,D,L,2N)布局
    for (int64_t i = 1; i < seq_len; i++) {
        auto deltaA_i = deltaA_bi.select(2, i);       // (B, D, 2N) - 视图
        auto prev = deltaB_u_bi.select(2, i - 1);     // (B, D, 2N) - 视图
        auto curr = deltaB_u_bi.select(2, i);         // (B, D, 2N) - 视图
        deltaB_u_bi.select(2, i).copy_(deltaA_i * prev + curr);
    }
    
    // 阶段2：分离并计算输出（无需转置）
    auto deltaB_u_fwd_out = deltaB_u_bi.narrow(3, 0, dstate);       // (b, d, l, n)
    auto deltaB_u_bwd_out = deltaB_u_bi.narrow(3, dstate, dstate);  // (b, d, l, n)
    
    // 批量计算输出：y = einsum('bdln,bnl->bdl')
    auto y_fwd = torch::sum(deltaB_u_fwd_out * C_fwd.unsqueeze(1).transpose(2, 3), -1);  // (b, d, l)
    auto y_bwd = torch::sum(deltaB_u_bwd_out * C_bwd.unsqueeze(1).transpose(2, 3), -1);  // (b, d, l)
    
    // 添加D项
    y_fwd = y_fwd + x_fwd_conv * D_fwd.unsqueeze(0).unsqueeze(-1);
    y_bwd = y_bwd + x_bwd_conv_flip * D_bwd.unsqueeze(0).unsqueeze(-1);
    
    // 门控
    if (!is_none(z_fwd)) {
        y_fwd = y_fwd * torch::nn::functional::silu(z_fwd);
    }
    if (!is_none(z_bwd_flip)) {
        y_bwd = y_bwd * torch::nn::functional::silu(z_bwd_flip);
    }
    
    // 反转并合并输出
    y_bwd = y_bwd.flip({2});  // 反转回原序列顺序
    auto out = y_fwd + y_bwd;  // (b, d, l)
    
    return out.to(dtype_in);
}

// 融合双向Selective Scan（优化版本，使用select避免clone + 预分配消除cat + (B,D,L,2N)布局避免permute）
torch::Tensor Selective_Fused_Scan_Fixlen_Cpu(
    const torch::Tensor& dt_fwd,            // (B, D, L)
    const torch::Tensor& dt_bwd,            // (B, D, L)
    const torch::Tensor& A_fwd,             // (D, N)
    const torch::Tensor& A_bwd,             // (D, N)
    const torch::Tensor& B_fwd,             // (B, N, L)
    const torch::Tensor& B_bwd,             // (B, N, L)
    const torch::Tensor& x_fwd_conv,        // (B, D, L)
    const torch::Tensor& x_bwd_conv_flip,   // (B, D, L)
    const torch::Tensor& C_fwd,             // (B, N, L)
    const torch::Tensor& C_bwd,             // (B, N, L)
    const torch::Tensor& D_fwd,             // (D,)
    const torch::Tensor& D_bwd,             // (D,)
    const torch::Tensor& z_fwd = torch::Tensor(),       // (B, D, L) - 可选
    const torch::Tensor& z_bwd_flip = torch::Tensor()   // (B, D, L) - 可选
) {
    auto dtype_in = dt_fwd.dtype();
    
    const int64_t batch = dt_fwd.size(0);
    const int64_t dim = dt_fwd.size(1);
    const int64_t seq_len = dt_fwd.size(2);
    const int64_t dstate = A_fwd.size(1);
    
    // 使用(B,D,L,2N)布局，与广播操作兼容，避免permute
    auto deltaA_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    auto deltaB_u_bi = torch::empty({batch, dim, seq_len, 2 * dstate}, dt_fwd.options().dtype(torch::kFloat32));
    
    // 计算并写入（无需permute）
    // 正向部分 [:,:,:,:n]
    deltaA_bi.narrow(3, 0, dstate).copy_(
        torch::exp(dt_fwd.unsqueeze(-1) * A_fwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, 0, dstate).copy_(
        dt_fwd.unsqueeze(-1) * B_fwd.unsqueeze(1).transpose(2, 3) * x_fwd_conv.unsqueeze(-1)
    );
    
    // 反向部分 [:,:,:,n:]
    deltaA_bi.narrow(3, dstate, dstate).copy_(
        torch::exp(dt_bwd.unsqueeze(-1) * A_bwd.unsqueeze(0).unsqueeze(2))
    );
    deltaB_u_bi.narrow(3, dstate, dstate).copy_(
        dt_bwd.unsqueeze(-1) * B_bwd.unsqueeze(1).transpose(2, 3) * x_bwd_conv_flip.unsqueeze(-1)
    );
    
    // 阶段1：递推（原地修改，(B,D,L,2N)布局）
    for (int64_t i = 1; i < seq_len; i++) {
        auto deltaA_i = deltaA_bi.select(2, i);       // (B, D, 2N) - 视图
        auto prev = deltaB_u_bi.select(2, i - 1);     // (B, D, 2N) - 视图
        // 原地加法：curr += deltaA_i * prev
        deltaB_u_bi.select(2, i).add_(deltaA_i * prev);
    }
    
    // 阶段2：分离并计算输出（无需转置）
    auto deltaB_u_fwd_out = deltaB_u_bi.narrow(3, 0, dstate);       // (b, d, l, n)
    auto deltaB_u_bwd_out = deltaB_u_bi.narrow(3, dstate, dstate);  // (b, d, l, n)
    
    // 批量计算输出
    auto y_fwd = torch::sum(deltaB_u_fwd_out * C_fwd.unsqueeze(1).transpose(2, 3), -1);
    auto y_bwd = torch::sum(deltaB_u_bwd_out * C_bwd.unsqueeze(1).transpose(2, 3), -1);
    
    // 添加D项
    y_fwd = y_fwd + x_fwd_conv * D_fwd.unsqueeze(0).unsqueeze(-1);
    y_bwd = y_bwd + x_bwd_conv_flip * D_bwd.unsqueeze(0).unsqueeze(-1);
    
    // 门控
    if (!is_none(z_fwd)) {
        y_fwd = y_fwd * torch::nn::functional::silu(z_fwd);
    }
    if (!is_none(z_bwd_flip)) {
        y_bwd = y_bwd * torch::nn::functional::silu(z_bwd_flip);
    }
    
    // 反转并合并输出
    y_bwd = y_bwd.flip({2});
    auto out = y_fwd + y_bwd;
    
    return out.to(dtype_in);
}


// PYBIND11_MODULE removed for linking with VisionMamba.cpp
