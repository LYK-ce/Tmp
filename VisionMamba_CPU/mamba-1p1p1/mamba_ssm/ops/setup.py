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
        
        # 树莓派特定优化参数
        arm_compile_args = [
            '-O3',                          # 最高优化等级
            '-std=c++17',                   # C++17标准
            '-march=native',                # 自动检测CPU架构（NEON, ARMv7/v8）
            '-mtune=native',                # 针对当前CPU微架构调优
            '-mfpu=neon-vfpv4',            # 明确启用NEON-VFPv4（树莓派3/4）
            '-mfloat-abi=hard',            # 硬浮点ABI（性能更好）
            '-ffast-math',                  # 快速数学运算（放宽IEEE 754）
            '-funsafe-math-optimizations',  # 更激进的数学优化
            '-funroll-loops',               # 循环展开
            '-ftree-vectorize',             # 自动向量化（启用NEON）
            '-fvect-cost-model=unlimited',  # 无限制向量化成本模型
            '-fprefetch-loop-arrays',       # 循环数组预取
            '-fomit-frame-pointer',         # 省略帧指针（释放寄存器）
            '-fno-signed-zeros',            # 无符号零优化
            '-fno-trapping-math',           # 非陷阱数学
            '-frename-registers',           # 寄存器重命名
            '-fopenmp',                     # OpenMP多线程
            '-flto',                        # 链接时优化（LTO）
            '-fuse-linker-plugin',          # 使用链接器插件
        ]
        
        # 树莓派4/5特有优化（ARMv8-A Cortex-A72/A76）
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'cortex-a72' in cpuinfo or 'cortex-a76' in cpuinfo:
                    print("[INFO] 检测到树莓派4/5，启用Cortex-A7x优化")
                    arm_compile_args.extend([
                        '-mcpu=cortex-a72',     # 或cortex-a76（树莓派5）
                        '-mcache-line-size=64', # 64字节缓存行
                    ])
                elif 'cortex-a53' in cpuinfo:
                    print("[INFO] 检测到树莓派3，启用Cortex-A53优化")
                    arm_compile_args.extend([
                        '-mcpu=cortex-a53',
                        '-mcache-line-size=64',
                    ])
        except:
            print("[WARN] 无法读取/proc/cpuinfo，使用通用ARM优化")
        
        print("[INFO] NEON向量化已启用（128位SIMD指令集）")
        
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
