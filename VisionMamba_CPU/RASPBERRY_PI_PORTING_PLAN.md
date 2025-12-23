# Vision Mamba 移植到树莓派计划

## 项目概述

Vision Mamba (Vim) 是一种基于双向状态空间模型（SSM）的新型视觉骨干网络。本项目旨在将 Vim 移植到树莓派上运行，以便在边缘设备上进行视觉识别任务。

## 树莓派环境限制

1. **硬件限制**
   - ARM架构处理器（通常是Cortex-A系列）
   - 内存有限（1GB-8GB，典型为4GB）
   - 无专用GPU，依赖CPU计算
   - 存储空间有限（通常使用SD卡）

2. **软件限制**
   - 需要使用PyTorch CPU版本
   - 无法使用CUDA加速
   - 无法编译C++扩展（缺少编译工具链）

## 移植步骤

### 第一步：环境配置

1. **安装Python环境**
   - 使用系统自带Python或通过apt安装Python 3.8-3.10
   - 使用venv创建虚拟环境

2. **安装PyTorch**
   - 安装适用于ARM的PyTorch CPU版本
   - 安装对应的torchvision

3. **安装基础依赖**
   - 安装numpy, pillow等基础库
   - 安装timm等模型相关库

### 第二步：修改causal-conv1d依赖

1. **添加环境变量检查**
   - 添加`CAUSAL_CONV1D_FORCE_FALLBACK`环境变量支持
   - 当该环境变量为"TRUE"时，使用纯PyTorch实现替代CUDA实现

2. **实现纯PyTorch版本**
   - 创建纯PyTorch实现的causal conv1d操作
   - 替换原有的CUDA实现

### 第三步：修改mamba-ssm依赖

1. **添加环境变量检查**
   - 添加`SELECTIVE_SCAN_FORCE_FALLBACK`环境变量支持
   - 当该环境变量为"TRUE"时，使用纯PyTorch实现替代CUDA实现

2. **实现纯PyTorch版本**
   - 创建纯PyTorch实现的选择性扫描操作
   - 替换原有的CUDA实现

### 第四步：调整模型代码

1. **修改模型加载和推理代码**
   - 默认强制使用CPU设备
   - 调整批处理大小为1
   - 添加内存优化选项

2. **创建树莓派专用推理脚本**
   - 创建简化版推理脚本，减少依赖
   - 支持单图像推理
   - 添加模型量化选项（可选）

### 第五步：测试和验证

1. **基础功能测试**
   - 模型加载测试
   - 单图像推理测试

2. **性能测试**
   - 内存使用测试
   - 推理时间测试

## 预期问题和解决方案

### 1. 性能问题
**问题**：推理速度可能过慢
**解决方案**：
- 使用模型量化
- 提供更小的模型变体
- 优化数据预处理流程

### 2. 内存不足
**问题**：模型可能超出树莓派内存限制
**解决方案**：
- 减小批处理大小
- 添加模型量化支持
- 使用模型简化版本

## 实施细节

### causal-conv1d修改方案

1. 在`causal_conv1d/causal_conv1d.py`中添加环境变量检查
2. 创建纯PyTorch实现的`causal_conv1d`函数
3. 当检测到`CAUSAL_CONV1D_FORCE_FALLBACK=TRUE`时，使用纯PyTorch实现

### mamba-ssm修改方案

1. 在`mamba_ssm/ops/selective_scan_interface.py`中添加环境变量检查
2. 创建纯PyTorch实现的`selective_scan`函数
3. 当检测到`SELECTIVE_SCAN_FORCE_FALLBACK=TRUE`时，使用纯PyTorch实现

### Vim模型修改方案

1. 在`vim/main.py`中添加CPU设备强制设置
2. 创建简化版推理脚本`vim/infer_rpi.py`
3. 添加模型量化支持（可选）

## 测试计划

1. **环境变量测试**
   - 测试`CAUSAL_CONV1D_FORCE_FALLBACK`环境变量是否生效
   - 测试`SELECTIVE_SCAN_FORCE_FALLBACK`环境变量是否生效

2. **功能测试**
   - 测试模型在CPU上加载是否正常
   - 测试单图像推理是否正常

3. **性能测试**
   - 测试推理时间
   - 测试内存占用

## 验收标准

1. 能够在树莓派上成功运行Vim-Tiny模型
2. 单张图像推理时间在可接受范围内（<30秒）
3. 内存占用不超过树莓派总内存的80%
4. 推理精度与原模型保持一致