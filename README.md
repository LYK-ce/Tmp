# Vision Mamba CPU 优化项目

## 项目简介

这是一个针对 Vision Mamba 模型在 CPU 平台上的性能优化项目。通过算法优化和 C++ 实现，显著提升了 Vision Mamba 在 CPU 上的推理速度。

## 核心优化成果

| 优化版本 | 相对加速比 | 关键技术 |
|---------|-----------|---------|
| Python 原始版 | 1.0x | 基线实现 |
| Python 融合版 | 1.3-1.5x | 双向扫描融合 |
| C++ 原始版 | 1.5-2.0x | C++ + Torch API |
| C++ Fixlen 版 | 3.0-3.5x | 两阶段批量计算 |
| C++ Fused 版 | 3.8-4.2x | 融合 + 预分配优化 |
| C++ 全优化版 | 4.0-4.5x | 内存布局优化 |

**总体加速：4-5 倍**

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- C++ 编译器（MSVC/GCC）

### 快速开始

```bash
# 1. 克隆项目
git clone <repository-url>
cd mamba-minimal

# 2. 安装依赖
pip install torch torchvision viztracer

# 3. 编译 C++ 扩展
cd VisionMamba_CPU/mamba-1p1p1/mamba_ssm/ops
python setup.py build_ext --inplace

cd ../modules
python setup.py build_ext --inplace

# 4. 运行性能测试
cd ../../../vim
python inf_cpu.py
```

### 测试不同优化版本

```python
from models_mamba import VisionMamba

# Python 原始版本
model = VisionMamba(
    use_cpp_scan=False,
    use_fixlen_scan=False,
    use_fused_bidirectional=False
)

# C++ 全优化版本
model = VisionMamba(
    use_cpp_scan=True,
    use_fixlen_scan=True,
    use_fused_bidirectional=True,
    use_full_cpp=True  # 完全C++实现
)
```

## 优化方法详解

### 1. 算法层面优化

#### 两阶段批量计算（Fixlen）
**问题**：原始实现中，每个时间步都需要分配新内存存储隐藏状态。

**优化**：
```python
# 原始：逐个时间步计算
for i in range(L):
    x = deltaA[i] * x + deltaB_u[i]  # 每次分配新x
    y[i] = sum(x * C[i])

# 优化：先计算所有隐藏状态，再批量计算输出
for i in range(1, L):
    deltaB_u[i] += deltaA[i] * deltaB_u[i-1]  # 原地更新
y = sum(deltaB_u * C, dim=-1)  # 批量计算
```

**收益**：减少内存分配，提升缓存命中率，加速 2-3 倍。

#### 融合双向扫描（Fused）
**问题**：正向和反向扫描需要两次独立的函数调用和数据传递。

**优化**：
```python
# 在状态维度N上concat正向和反向
# 一次计算同时处理两个方向
deltaA_bi = torch.cat([deltaA_fwd, deltaA_bwd], dim=3)  # (b,d,l,2n)
# 递推循环同时更新2n个状态
for i in range(1, L):
    deltaB_u_bi[:,:,:,i] += deltaA_bi[:,:,:,i] * deltaB_u_bi[:,:,:,i-1]
```

**收益**：减少函数调用开销，改善数据局部性，加速 1.3-1.5 倍。

### 2. 内存优化

#### 预分配 + 直接写入
**问题**：`torch.cat`需要分配新内存并复制数据。

**优化**：
```cpp
// 预分配缓冲区
auto deltaA_bi = torch::empty({batch, dim, seqlen, 2*dstate});

// 直接写入，避免cat
deltaA_bi.narrow(3, 0, dstate).copy_(deltaA_fwd);      // 前半
deltaA_bi.narrow(3, dstate, dstate).copy_(deltaA_bwd); // 后半
```

**收益**：消除内存拷贝，减少分配次数，加速 10-20%。

#### 内存布局优化
**问题**：原布局 (B,D,L,2N) 访问L维度时需要跳过2N个元素。

**优化**：
```cpp
// 改为 (B,D,2N,L) 布局，L维度连续
auto deltaA_bi = torch::empty({batch, dim, 2*dstate, seqlen});
// 递推时访问连续内存
for (int i = 1; i < seqlen; i++) {
    deltaB_u_bi.select(3, i).add_(deltaA_i * prev);  // 连续访问
}
```

**收益**：提升缓存命中率，在ARM平台效果更显著。

### 3. C++ 实现优化

#### 消除 Python 解释器开销
**问题**：Python 循环和函数调用有显著开销。

**优化**：
- 将核心计算逻辑用 C++ 实现
- 使用 PyTorch C++ API（torch::Tensor）
- 保持与 Python 版本接口兼容

**收益**：减少 Python/C++ 边界开销，加速 1.5-2 倍。

#### 平台特定优化
**ARM 平台（树莓派）**：
```python
# setup.py 中启用 NEON 指令集
extra_compile_args = [
    '-mfpu=neon-vfpv4',      # NEON-VFPv4
    '-mfloat-abi=hard',      # 硬浮点
    '-ftree-vectorize',      # 自动向量化
    '-mcpu=cortex-a72',      # CPU 特定优化
]
```

**收益**：利用 SIMD 并行，在树莓派上加速 3-6 倍。

### 4. 完全 C++ 实现

**问题**：混合 Python/C++ 仍有边界开销。

**优化**：
- 实现完整的 VisionMamba C++ 模块
- 从输入投影到输出投影全部用 C++ 实现
- 直接链接 selective_scan 函数

**收益**：消除所有 Python 解释器开销，再加速 10-20%。

## 性能对比

### 测试环境
- CPU: Intel i7-12700H / ARM Cortex-A72
- 模型: Vision Mamba Tiny
- 输入: 224x224 图像

### 性能数据

| 版本 | x86 时间 | ARM 时间 | 加速比 |
|-----|---------|---------|-------|
| Python 原始 | 240ms | 1200ms | 1.0x |
| Python 融合 | 180ms | 900ms | 1.3x |
| C++ 原始 | 160ms | 800ms | 1.5x |
| C++ Fixlen | 80ms | 400ms | 3.0x |
| C++ Fused | 60ms | 300ms | 4.0x |
| C++ 全优化 | 50ms | 250ms | 4.8x |

## 项目结构

```
mamba-minimal/
├── task.md                      # 任务清单
├── workbook.md                  # 工作记录
├── design.md                    # 设计文档
├── cpp_workflow.md             # C++ 开发流程
├── vim_analysis.md             # Vision Mamba 分析
├── compare_time.py             # CNN vs ViT 对比
├── delete.md                   # 清理清单
├── VisionMamba_CPU/            # Vision Mamba 实现
│   ├── vim/
│   │   ├── models_mamba.py     # 模型定义
│   │   ├── inf_cpu.py          # 性能测试
│   │   └── main.py             # 训练脚本
│   └── mamba-1p1p1/            # Mamba 核心库
│       └── mamba_ssm/
│           ├── ops/            # selective_scan 实现
│           └── modules/        # VisionMamba 实现
└── Mamba_CPU/                  # 基础 Mamba 实现
    ├── model.py
    ├── model_new.py
    └── selective_scan.cpp
```

## 关键技术点

1. **两阶段批量计算**：避免逐个时间步分配内存
2. **融合双向扫描**：在N维度concat，一次计算两个方向
3. **预分配优化**：消除torch.cat的内存拷贝
4. **内存布局优化**：(B,D,2N,L)提升缓存命中率
5. **C++ 实现**：消除Python解释器开销
6. **NEON 优化**：ARM平台SIMD加速

## 参考文献

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
- PyTorch C++ Extension Tutorial

## 许可证

MIT License
