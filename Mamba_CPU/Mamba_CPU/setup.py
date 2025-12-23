#Presented by KeJi
#Date: 2025-12-22

"""
Setup script for selective_scan C++ extension
使用MSVC编译（Windows）或gcc（Linux）
"""

import os
import sys
import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 检测平台并设置编译参数
if sys.platform == 'win32':
    # Windows MSVC
    extra_compile_args = ['/O2', '/std:c++17', '/openmp']
    extra_link_args = []
else:
    # Linux/Mac gcc
    extra_compile_args = ['-O3', '-std=c++17', '-fopenmp', '-march=native']
    extra_link_args = ['-fopenmp']

setup(
    name='selective_scan_cpp',
    ext_modules=[
        CppExtension(
            name='selective_scan_cpp',
            sources=['selective_scan.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
