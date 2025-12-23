# Vision Mamba 树莓派部署指南

## 概述

本指南详细说明如何在树莓派上部署和运行 Vision Mamba (Vim) 模型。由于树莓派硬件限制（无 CUDA 支持、有限的计算能力），我们需要对原始代码进行一些修改，以确保模型可以在树莓派上正常运行。

## 已实施的修改项

以下是我们已经完成的修改：

1. **causal-conv1d 库修改**
   - 添加了 `CAUSAL_CONV1D_FORCE_FALLBACK` 环境变量支持
   - 实现了纯 PyTorch 版本的 causal_conv1d 操作
   - 当环境变量设置为 "TRUE" 时，自动切换到纯 PyTorch 实现
   - 修改了 setup.py 文件，当设置环境变量时跳过 CUDA 构建

2. **mamba-ssm 库修改**
   - 添加了 `SELECTIVE_SCAN_FORCE_FALLBACK` 环境变量支持
   - 实现了纯 PyTorch 版本的选择性扫描操作
   - 当环境变量设置为 "TRUE" 时，自动切换到纯 PyTorch 实现
   - 修改了 setup.py 文件，当设置环境变量时跳过 CUDA 构建
   - 移除了 triton 的硬依赖，改为可选依赖

3. **创建树莓派专用推理脚本**
   - 创建了 `vim/infer_rpi.py` 脚本，用于在树莓派上进行图像推理
   - 该脚本默认使用 CPU 运行，适合树莓派环境

4. **主程序修改**
   - 在 `vim/main.py` 中添加了对环境变量的检查和提示

## 部署步骤

### 1. 硬件要求

- 树莓派 4B 或更高版本（推荐 8GB 内存版本）
- 至少 16GB 的 microSD 卡

### 2. 系统环境配置

#### 2.1 安装操作系统

在树莓派上安装 64 位的 Raspberry Pi OS。

#### 2.2 安装 Python 环境

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 依赖
sudo apt install python3-pip python3-venv python3-dev -y
```

#### 2.3 创建虚拟环境

```bash
# 创建并激活虚拟环境
python3 -m venv vimberrypi
source vimberrypi/bin/activate
```

#### 2.4 安装 PyTorch

```bash
# 安装 PyTorch（适用于 ARM64 架构）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2.5 安装其他依赖

```bash
# 安装基础依赖
pip install numpy pillow einops timm requests

# 安装 causal_conv1d（仅作为接口，实际使用纯 PyTorch 实现）
# 首先设置环境变量以跳过 CUDA 构建
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
cd causal-conv1d
pip install -e .
cd ..

# 安装 mamba_ssm（仅作为接口，实际使用纯 PyTorch 实现）
# 首先设置环境变量以跳过 CUDA 构建
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE
cd mamba-1p1p1
pip install -e .
cd ..
```

### 3. 环境变量设置

在运行任何 Vim 模型之前，需要设置以下环境变量以强制使用纯 PyTorch 实现：

```bash
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE
```

### 4. 下载预训练模型（可选）

从 Hugging Face 下载 Vim-Tiny 模型：

```bash
# 创建模型目录
mkdir -p models

# 下载 Vim-Tiny 模型权重
wget -O models/vim_tiny.pth https://huggingface.co/hustvl/Vim-tiny-midclstok/resolve/main/pytorch_model.bin
```

### 5. 运行推理

#### 5.1 使用专用推理脚本（推荐）

```bash
# 设置环境变量
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE

# 运行推理
python vim/infer_rpi.py --image path/to/your/image.jpg --checkpoint models/vim_tiny.pth
```

#### 5.2 使用原始评估脚本

```bash
# 设置环境变量
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE

# 运行评估
python vim/main.py --eval \
--resume models/vim_tiny.pth \
--model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--device cpu \
--batch-size 1
```

#### 5.3 测试模型FLOPs和推理时间（不使用预训练模型）

如果您只想测试模型的计算复杂度和推理时间，而不需要预训练模型，可以使用以下脚本：

```bash
# 设置环境变量
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE

# 运行测试脚本
python vim/test_flops.py
```

这个脚本会创建一个随机初始化的 Vim-Tiny 模型，并测试其推理时间。

## 性能优化建议

1. **使用较小的图像尺寸**：如果可能，使用较小的图像输入尺寸以减少计算量。

2. **限制线程数**：可以通过设置环境变量来限制 PyTorch 使用的线程数：
   ```bash
   export OMP_NUM_THREADS=4
   ```

3. **使用 swap 空间**：如果内存不足，可以增加 swap 空间大小。

## 故障排除

1. **内存不足**：降低 batch size 到 1，并确保有足够的 swap 空间。

2. **依赖安装失败**：
   - 确保在安装前设置了正确的环境变量
   - 如果仍有问题，可以尝试添加 `--no-build-isolation` 参数
   - Triton 是一个 GPU 专用库，在树莓派上无法安装，我们的修改已经将其设为可选依赖

3. **推理速度慢**：这是正常的，因为使用的是纯 PyTorch 实现而非优化的 CUDA 内核。

4. **模型创建错误**：
   - 确保在运行前设置了正确的环境变量
   - 检查是否正确安装了所有依赖项
   - 如果出现 `TypeError: the first argument must be callable` 错误，通常是由于缺少依赖项或环境变量设置不正确
   - 如果出现 `'NoneType' object is not callable` 错误，通常是由于在纯PyTorch回退模式下fused_add_norm功能不可用

5. **fused_add_norm问题**：
   - 在纯PyTorch回退模式下，我们自动禁用了fused_add_norm功能
   - 这可能会略微影响性能，但确保了模型可以正常运行
   - 现在我们在前向传播中也添加了回退处理，确保即使在启用fused_add_norm的情况下也能正常工作

6. **Mamba CUDA功能回退**：
   - 我们已经实现了Mamba模块的纯PyTorch参考实现
   - 当设置环境变量时，会自动使用纯PyTorch实现替代CUDA优化版本
   - 这会显著降低推理速度，但确保了在无CUDA设备上的兼容性

## 注意事项

1. 在树莓派上运行 Vim 模型会比在 GPU 上慢很多，这是由于使用了纯 PyTorch 实现而非优化的 CUDA 内核。

2. 建议使用 Vim-Tiny 模型，它是为移动/边缘设备设计的较小模型。

3. 确保已设置正确的环境变量以使用纯 PyTorch 实现。

4. 如果遇到内存不足的问题，请使用较小的 batch size（如 1）。

## 测试

可以使用以下脚本测试环境配置是否正确：

```bash
# 设置环境变量
export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE
export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE

# 运行简单的测试
python -c "import torch; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('Environment setup correctly')"
```

如果输出 "Environment setup correctly"，则说明环境配置正确。