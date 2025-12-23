//Presented by KeJi
//Date: 2025-12-22

#include <torch/extension.h>

// 优化版selective scan实现 - 使用PyTorch C++ API而不是手写循环
// 这样可以调用Intel MKL/OpenBLAS等优化库
torch::Tensor Selective_Scan_Cpu(
    torch::Tensor u,       // (b, l, d_in)
    torch::Tensor delta,   // (b, l, d_in)
    torch::Tensor A,       // (d_in, n)
    torch::Tensor B,       // (b, l, n)
    torch::Tensor C,       // (b, l, n)
    torch::Tensor D        // (d_in,)
) {
    // 确保所有张量都是连续的
    u = u.contiguous();
    delta = delta.contiguous();
    A = A.contiguous();
    B = B.contiguous();
    C = C.contiguous();
    D = D.contiguous();
    
    const int64_t batch = u.size(0);
    const int64_t seq_len = u.size(1);
    const int64_t d_in = u.size(2);
    const int64_t n = A.size(1);
    
    // 使用PyTorch优化的张量运算，而不是手写循环
    // deltaA = exp(delta[:,:,:,None] * A[None,None,:,:])
    // 形状: (b, l, d_in, n)
    auto deltaA = torch::exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    );
    
    // deltaB_u = delta[:,:,:,None] * B[:,:,None,:] * u[:,:,:,None]
    // 形状: (b, l, d_in, n)
    auto deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1);
    
    // 初始化隐藏状态 h: (b, d_in, n)
    auto h = torch::zeros({batch, d_in, n}, u.options());
    
    // 预分配输出张量
    auto y = torch::zeros({batch, seq_len, d_in}, u.options());
    
    // 原始selective scan: 逐个时间步处理（这部分必须串行）
    for (int64_t t = 0; t < seq_len; t++) {
        // 更新隐藏状态: h = deltaA[:, t] * h + deltaB_u[:, t]
        // 使用张量运算而不是嵌套循环
        h = deltaA.index({torch::indexing::Slice(), t}) * h 
            + deltaB_u.index({torch::indexing::Slice(), t});
        
        // 计算输出: y[:, t] = sum(h * C[:, t, :].unsqueeze(1), dim=-1) + D * u[:, t]
        // 使用torch::sum而不是手写循环
        y.index({torch::indexing::Slice(), t}) = 
            torch::sum(h * C.index({torch::indexing::Slice(), t}).unsqueeze(1), -1);
    }
    
    // 添加跳跃连接: y = y + u * D
    y = y + u * D.unsqueeze(0).unsqueeze(0);
    
    return y;
}

// 优化的两阶段计算版本 - 使用PyTorch C++ API
torch::Tensor Selective_Scan_Fixlen_Cpu(
    torch::Tensor u,       // (b, l, d_in)
    torch::Tensor delta,   // (b, l, d_in)
    torch::Tensor A,       // (d_in, n)
    torch::Tensor B,       // (b, l, n)
    torch::Tensor C,       // (b, l, n)
    torch::Tensor D        // (d_in,)
) {
    // 确保所有张量都是连续的
    u = u.contiguous();
    delta = delta.contiguous();
    A = A.contiguous();
    B = B.contiguous();
    C = C.contiguous();
    D = D.contiguous();
    
    const int64_t batch = u.size(0);
    const int64_t seq_len = u.size(1);
    const int64_t d_in = u.size(2);
    const int64_t n = A.size(1);
    
    // 使用PyTorch优化的张量运算
    // deltaA = exp(delta[:,:,:,None] * A[None,None,:,:])
    auto deltaA = torch::exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    );
    
    // deltaB_u = delta[:,:,:,None] * B[:,:,None,:] * u[:,:,:,None]
    // 这个张量会被原地修改为存储所有隐藏状态
    auto deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1);
    
    // 阶段1: 递推计算所有隐藏状态，原地存储在deltaB_u中
    // for t in range(1, l):
    //     deltaB_u[:, t] = deltaA[:, t] * deltaB_u[:, t-1] + deltaB_u[:, t]
    for (int64_t t = 1; t < seq_len; t++) {
        // 使用张量运算而不是4层嵌套循环
        deltaB_u.index({torch::indexing::Slice(), t}) = 
            deltaA.index({torch::indexing::Slice(), t}) * 
            deltaB_u.index({torch::indexing::Slice(), t - 1}) +
            deltaB_u.index({torch::indexing::Slice(), t});
    }
    
    // 阶段2: 批量计算所有输出
    // y = sum(deltaB_u * C.unsqueeze(2), dim=-1)
    // 使用torch::sum而不是einsum或手写循环，性能相当且更清晰
    auto y = torch::sum(deltaB_u * C.unsqueeze(2), -1);
    
    // 添加跳跃连接
    y = y + u * D.unsqueeze(0).unsqueeze(0);
    
    return y;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan", &Selective_Scan_Cpu,
          "Selective Scan CPU (Original algorithm with PyTorch API)");
    m.def("selective_scan_fixlen", &Selective_Scan_Fixlen_Cpu,
          "Selective Scan CPU (Optimized two-stage algorithm with PyTorch API)");
}
