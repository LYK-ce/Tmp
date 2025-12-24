#!/usr/bin/env python3
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import time
import os
import traceback
from timm.models import create_model
import models_mamba
import sys

# 设置环境变量以强制使用纯PyTorch实现
os.environ["CAUSAL_CONV1D_FORCE_FALLBACK"] = "TRUE"
os.environ["SELECTIVE_SCAN_FORCE_FALLBACK"] = "TRUE"

def main():
    print("=" * 60)
    print("Vision Mamba 模型性能测试报告")
    print("模型名称: Vim-Tiny")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # 系统信息
    print("系统信息:")
    print("-" * 30)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: CPU (纯PyTorch回退模式)")
    print()
    
    try:
        # 测试标准224x224模型的推理时间
        print("模型信息:")
        print("-" * 30)
        model = create_model(
            'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None,
        )
        
        # 将模型设置为评估模式
        model.eval()
        
        # 创建一个随机输入张量（模拟224x224的RGB图像）
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估计模型大小（参数量 * 4字节/参数）
        model_size_mb = params * 4 / (1024 * 1024)
        
        print(f"总参数量: {params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {model_size_mb:.2f} MB")
        print()
        
        # 测试推理时间
        print("推理性能 (输入尺寸: 224×224):")
        print("-" * 30)
        
        # 预热运行
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        
        # 单次推理时间测试
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        end_time = time.time()
        
        single_inference_time = end_time - start_time
        single_fps = 1.0 / single_inference_time
        
        print(f"单次推理时间: {single_inference_time:.6f} 秒")
        print(f"单次推理FPS: {single_fps:.2f}")
        
        # 批量推理时间测试（批大小=1，运行10次取平均）
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        end_time = time.time()
        
        batch_inference_time = (end_time - start_time) / 10
        batch_fps = 1.0 / batch_inference_time
        
        print(f"批量推理时间 (批大小=1): {batch_inference_time:.6f} 秒")
        print(f"批量处理FPS: {batch_fps:.2f}")
        
        # 处理1000张图片的测试
        start_time = time.time()
        with torch.no_grad():
            for i in range(1000):
                _ = model(input_tensor)
        end_time = time.time()
        
        total_time_1000 = end_time - start_time
        fps_1000 = 1000.0 / total_time_1000
        
        print(f"处理 1000 张图片总时间: {total_time_1000:.4f} 秒")
        print(f"处理 1000 张图片FPS: {fps_1000:.2f}")
        
        print("\n测试完成!")

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        print("\n请确保您已设置环境变量:")
        print("export CAUSAL_CONV1D_FORCE_FALLBACK=TRUE")
        print("export SELECTIVE_SCAN_FORCE_FALLBACK=TRUE")

if __name__ == '__main__':
    main()