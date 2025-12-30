#Presented by KeJi
#Date: 2025-12-29

"""
VisionMamba C++ 扩展编译脚本

功能：
1. 编译VisionMamba.cpp和selective_scan_core.cpp为一个共享库
2. 支持Windows(MSVC)和Linux(GCC)平台
3. 针对ARM平台(RK3588)启用NEON优化

使用方法：
    python setup.py build_ext --inplace
    
注意：
    需要先运行 prepare_sources.py 来生成 selective_scan_core.cpp
    或者手动从 selective_scan.cpp 中移除 PYBIND11_MODULE 部分
"""

import os
import platform
import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 获取当前目录和ops目录
current_dir = os.path.dirname(os.path.abspath(__file__))
ops_dir = os.path.join(os.path.dirname(current_dir), 'ops')

# 检查是否存在selective_scan_core.cpp（无PYBIND11的版本）
selective_scan_core = os.path.join(ops_dir, 'selective_scan_core.cpp')
selective_scan_orig = os.path.join(ops_dir, 'selective_scan.cpp')

# 如果不存在core版本，从原始文件生成
if not os.path.exists(selective_scan_core):
    print(f"Generating {selective_scan_core} from {selective_scan_orig}...")
    with open(selective_scan_orig, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除PYBIND11_MODULE部分
    pybind_start = content.find('// Python绑定')
    if pybind_start == -1:
        pybind_start = content.find('PYBIND11_MODULE')
    
    if pybind_start != -1:
        content = content[:pybind_start]
        content += '\n// PYBIND11_MODULE removed for linking with VisionMamba.cpp\n'
    
    with open(selective_scan_core, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Generated {selective_scan_core}")

# 源文件列表
sources = [
    os.path.join(current_dir, 'VisionMamba.cpp'),
    selective_scan_core,
]

# 编译参数
extra_compile_args = []
extra_link_args = []

system = platform.system()
machine = platform.machine().lower()

if system == 'Windows':
    # Windows MSVC
    extra_compile_args = ['/O2', '/fp:fast']
elif system == 'Linux':
    # Linux GCC
    extra_compile_args = ['-O3', '-ffast-math', '-fopenmp']
    extra_link_args = ['-fopenmp']
    
    # ARM平台优化
    if machine in ['aarch64', 'arm64']:
        extra_compile_args.extend([
            '-march=armv8-a+simd',
            '-ftree-vectorize',
            '-fno-math-errno',
        ])
        # 针对RK3588的Cortex-A76/A55核心优化
        if os.path.exists('/proc/device-tree/model'):
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read()
                    if 'rk3588' in model.lower() or 'rock' in model.lower():
                        extra_compile_args.extend([
                            '-mcpu=cortex-a76',
                            '-mtune=cortex-a76',
                        ])
            except:
                pass

# 扩展模块定义
ext_modules = [
    CppExtension(
        name='vision_mamba_cpp',
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[ops_dir],
    ),
]

setup(
    name='vision_mamba_cpp',
    version='1.0.0',
    description='Vision Mamba C++ Extension',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
