#Presented by KeJi
#Date: 2025-12-22

"""
Setup script for Vision Mamba Selective Scan C++ extension
跨平台编译：支持 Windows (x86_64), Linux (x86_64), ARM (树莓派)
编译适配官方Mamba格式(b,d,l)的C++优化实现
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sys
import platform

def get_compile_args():
    """
    根据平台和CPU架构返回最优的编译参数
    
    支持的平台：
    - Windows (MSVC) - x86_64
    - Linux (GCC) - x86_64
    - Linux (GCC) - ARM64 (树莓派)
    """
    machine = platform.machine().lower()
    
    if sys.platform == 'win32':
        # ===== Windows (MSVC) =====
        print(f"[INFO] 检测到平台: Windows ({machine})")
        return {
            'extra_compile_args': [
                '/O2',              # 优化速度
                '/Oi',              # 启用内联函数
                '/Ot',              # 优先速度
                '/GL',              # 全程序优化
                '/std:c++17',       # C++17标准
                '/openmp',          # OpenMP多线程
                '/fp:fast',         # 快速浮点运算
            ],
            'extra_link_args': [
                '/LTCG',            # 链接时代码生成
            ]
        }
    
    elif 'arm' in machine or 'aarch64' in machine:
        # ===== ARM/树莓派 (GCC) =====
        print(f"[INFO] 检测到平台: ARM/树莓派 ({machine})")
        
        # 区分64位ARMv8和32位ARMv7架构
        is_aarch64 = 'aarch64' in machine or machine == 'arm64'
        
        if is_aarch64:
            # ===== ARMv8 / aarch64（64位ARM - 树莓派3/4/5 64位系统） =====
            print("[INFO] 架构: ARMv8 (aarch64) - 64位ARM，NEON是标准特性")
            arm_compile_args = [
                '-O3',                          # 最高优化等级
                '-std=c++17',                   # C++17标准
                '-march=native',                # 自动检测CPU特性（NEON默认启用）
                '-mtune=native',                # 针对当前CPU微架构调优
                # 注意：ARMv8不需要-mfpu和-mfloat-abi参数
                '-ffast-math',                  # 快速数学运算
                '-funsafe-math-optimizations',  # 激进数学优化
                '-funroll-loops',               # 循环展开
                '-ftree-vectorize',             # 自动向量化
                '-fvect-cost-model=unlimited',  # 无限制向量化
                '-fprefetch-loop-arrays',       # 数组预取
                '-fomit-frame-pointer',         # 省略帧指针
                '-fno-signed-zeros',            # 无符号零优化
                '-fno-trapping-math',           # 非陷阱数学
                '-frename-registers',           # 寄存器重命名
                '-fopenmp',                     # OpenMP多线程
                '-flto',                        # 链接时优化
                '-fuse-linker-plugin',          # 链接器插件
            ]
        else:
            # ===== ARMv7（32位ARM - 树莓派1/2/3 32位系统） =====
            print("[INFO] 架构: ARMv7 (armv7l) - 32位ARM，需要明确启用NEON")
            arm_compile_args = [
                '-O3',
                '-std=c++17',
                '-march=native',
                '-mtune=native',
                '-mfpu=neon-vfpv4',            # 32位ARM需要明确指定FPU
                '-mfloat-abi=hard',            # 32位ARM需要指定硬浮点ABI
                '-ffast-math',
                '-funsafe-math-optimizations',
                '-funroll-loops',
                '-ftree-vectorize',
                '-fvect-cost-model=unlimited',
                '-fprefetch-loop-arrays',
                '-fomit-frame-pointer',
                '-fno-signed-zeros',
                '-fno-trapping-math',
                '-frename-registers',
                '-fopenmp',
                '-flto',
                '-fuse-linker-plugin',
            ]
        
        # CPU型号检测和特定优化（适用于32位和64位）
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'cortex-a72' in cpuinfo:
                    print("[INFO] 检测到Cortex-A72（树莓派4）")
                    if is_aarch64:
                        arm_compile_args.extend(['-mcpu=cortex-a72'])
                elif 'cortex-a76' in cpuinfo:
                    print("[INFO] 检测到Cortex-A76（树莓派5）")
                    if is_aarch64:
                        arm_compile_args.extend(['-mcpu=cortex-a76'])
                elif 'cortex-a53' in cpuinfo:
                    print("[INFO] 检测到Cortex-A53（树莓派3）")
                    if is_aarch64:
                        arm_compile_args.extend(['-mcpu=cortex-a53'])
        except:
            print("[WARN] 无法读取/proc/cpuinfo，使用-march=native自动检测")
        
        print("[INFO] NEON向量化优化已启用（128位SIMD）")
        
        return {
            'extra_compile_args': arm_compile_args,
            'extra_link_args': [
                '-flto',                    # LTO链接
                '-fuse-linker-plugin',      # 链接器插件
                '-Wl,-O3',                  # 链接器优化
            ]
        }
    
    else:
        # ===== Linux/Mac x86_64 (GCC/Clang) =====
        print(f"[INFO] 检测到平台: Linux/Mac ({machine})")
        return {
            'extra_compile_args': [
                '-O3',              # 最高优化
                '-std=c++17',       # C++17标准
                '-march=native',    # 针对当前CPU优化（自动检测AVX/AVX2）
                '-mtune=native',    # CPU调优
                '-ffast-math',      # 快速数学运算
                '-funroll-loops',   # 循环展开
                '-ftree-vectorize', # 树向量化（启用SIMD）
                '-fopenmp',         # OpenMP多线程
                '-flto',            # 链接时优化
            ],
            'extra_link_args': [
                '-flto',            # LTO链接
            ]
        }

# 获取编译参数
compile_config = get_compile_args()

print("[INFO] 编译参数:")
print(f"  - extra_compile_args: {compile_config['extra_compile_args']}")
print(f"  - extra_link_args: {compile_config.get('extra_link_args', [])}")

setup(
    name='selective_scan_cpp',
    version='1.0.0',
    ext_modules=[
        CppExtension(
            name='selective_scan_cpp',
            sources=['selective_scan.cpp'],
            extra_compile_args=compile_config['extra_compile_args'],
            extra_link_args=compile_config.get('extra_link_args', []),
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    description='Vision Mamba Selective Scan CPU optimization - Cross-platform (Windows/Linux/ARM)',
    author='KeJi',
)
