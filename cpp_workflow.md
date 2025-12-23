#Presented by KeJi
#Date: 2025-12-20

# PyTorch C++ Extension 工作流程

## 核心原则
**同一份C++代码，两种使用方式：开发用JIT，发布用setuptools**

---

## 方法1：JIT即时编译（开发阶段）

### 优势
- ✅ 修改代码自动重编译
- ✅ 无需setup.py
- ✅ 快速迭代

### 使用方式
```python
from torch.utils.cpp_extension import load

# 直接加载编译
selective_scan = load(
    name='selective_scan_cpp',
    sources=['selective_scan.cpp'],
    extra_cflags=['-O3'],
    verbose=True
)

# 调用
y = selective_scan.selective_scan(u, delta, A, B, C, D)
```

---

## 方法2：setuptools编译（发布阶段）

### 优势
- ✅ 可打包发布（wheel）
- ✅ 性能最优
- ✅ 生产稳定

### setup.py
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='selective_scan_cpp',
    ext_modules=[
        CppExtension(
            name='selective_scan_cpp',
            sources=['selective_scan.cpp'],
            extra_compile_args=['-O3']
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 编译安装
```bash
python setup.py install
# 或打包: python setup.py bdist_wheel
```

### Python调用
```python
import selective_scan_cpp
y = selective_scan_cpp.selective_scan(u, delta, A, B, C, D)
```

---

## C++代码模板（通用）

### selective_scan.cpp
```cpp
#include <torch/extension.h>

torch::Tensor selective_scan_cpu(
    torch::Tensor u, torch::Tensor delta, torch::Tensor A,
    torch::Tensor B, torch::Tensor C, torch::Tensor D
) {
    // 获取维度
    const int64_t batch = u.size(0);
    const int64_t seq_len = u.size(1);
    const int64_t d_in = u.size(2);
    const int64_t n = A.size(1);
    
    // 预分配输出
    auto y = torch::zeros_like(u);
    auto h = torch::zeros({batch, d_in, n}, u.options());
    
    // 核心计算循环
    auto u_a = u.accessor<float, 3>();
    auto delta_a = delta.accessor<float, 3>();
    auto A_a = A.accessor<float, 2>();
    auto B_a = B.accessor<float, 3>();
    auto C_a = C.accessor<float, 3>();
    auto D_a = D.accessor<float, 1>();
    auto y_a = y.accessor<float, 3>();
    auto h_a = h.accessor<float, 3>();
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t t = 0; t < seq_len; t++) {
            // 更新隐藏状态
            for (int64_t d = 0; d < d_in; d++) {
                for (int64_t i = 0; i < n; i++) {
                    float deltaA = expf(delta_a[b][t][d] * A_a[d][i]);
                    float deltaB_u = delta_a[b][t][d] * B_a[b][t][i] * u_a[b][t][d];
                    h_a[b][d][i] = deltaA * h_a[b][d][i] + deltaB_u;
                }
            }
            
            // 计算输出
            for (int64_t d = 0; d < d_in; d++) {
                float dot = 0.0f;
                for (int64_t i = 0; i < n; i++) {
                    dot += h_a[b][d][i] * C_a[b][t][i];
                }
                y_a[b][t][d] = dot + D_a[d] * u_a[b][t][d];
            }
        }
    }
    
    return y;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan", &selective_scan_cpu, "Selective Scan CPU");
}
```

---

## 优化技巧

### 编译优化（gcc/MinGW）
```python
extra_compile_args=[
    '-O3',              # 最高优化
    '-march=native',    # CPU特化
    '-fopenmp',         # OpenMP并行
    '-ffast-math',      # 快速数学
]
extra_link_args=['-fopenmp']  # 链接OpenMP库
```

### 编译优化（MSVC）
```python
extra_compile_args=[
    '/O2',              # 优化
    '/openmp',          # OpenMP
    '/fp:fast'          # 快速浮点
]
```

### OpenMP并行
```cpp
#include <omp.h>

#pragma omp parallel for collapse(2)
for (int64_t b = 0; b < batch; b++) {
    for (int64_t d = 0; d < d_in; d++) {
        // ... 计算代码 ...
    }
}
```

---

## 推荐工作流

### 开发流程
```bash
# 1. 快速测试（JIT）
python test_jit.py

# 2. 修改C++代码
vim selective_scan.cpp

# 3. 再次测试（自动重编译）
python test_jit.py
```

### 发布流程
```bash
# 1. 最终验证
python test_setuptools.py

# 2. 正式编译
python setup.py install

# 3. 打包分发
python setup.py bdist_wheel
```

---

## 环境要求

### Windows - 方案1：MSVC（官方推荐）
```bash
# 安装Visual Studio Build Tools
# 下载：https://visualstudio.microsoft.com/downloads/
# 包含MSVC编译器
```

### Windows - 方案2：MinGW（gcc替代）
```bash
# 1. 安装MinGW-w64
# 下载：https://www.mingw-w64.org/

# 2. 设置环境变量
set CC=gcc
set CXX=g++
set DISTUTILS_USE_SDK=1

# 3. 验证安装
gcc --version
g++ --version
```

**MinGW配置setup.py**：
```python
import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='selective_scan_cpp',
    ext_modules=[
        CppExtension(
            name='selective_scan_cpp',
            sources=['selective_scan.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fopenmp'],
            extra_link_args=['-fopenmp']
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### Linux（默认gcc）
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"

# 验证
gcc --version
g++ --version
```

### Python包
```bash
pip install torch pybind11
```

---

## 编译器对比

| 编译器 | 平台 | 优势 | 劣势 |
|-------|------|------|------|
| **gcc** | Linux | ✅ 默认首选 | - |
| **MinGW** | Windows | ✅ 编译快<br>✅ 熟悉语法 | ⚠️ ABI兼容性 |
| **MSVC** | Windows | ✅ 官方支持<br>✅ 兼容性好 | ⚠️ 编译慢 |
| **clang** | macOS | ✅ 系统默认 | - |

**推荐**：
- Linux → gcc
- Windows开发 → MinGW（快速）
- Windows发布 → MSVC（稳定）

---

## 总结

| 特性 | JIT | setuptools |
|------|-----|-----------|
| 使用场景 | 开发调试 | 生产发布 |
| 编译时机 | 首次导入 | 手动执行 |
| 修改后 | 自动重编译 | 需重新安装 |
| 性能 | 相同 | 相同 |
| 代码 | **完全相同** | **完全相同** |

**核心**：C++代码100%通用，选择合适的编译方式即可。
