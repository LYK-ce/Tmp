#Presented by KeJi
#Date: 2025-12-22

"""
Vision Mamba CPU推理性能测试脚本

功能：
1. 创建Vim-tiny模型（随机初始化或加载预训练）
2. 执行单次inference
3. 使用VizTracer分析性能瓶颈

使用方式：
    python inf_cpu.py
    # 生成vim_cpu_profile.html性能分析报告
"""

import os
import sys
import torch
import time

# 强制使用纯PyTorch实现（CPU环境）
os.environ['SELECTIVE_SCAN_FORCE_FALLBACK'] = 'TRUE'
os.environ['CAUSAL_CONV1D_FORCE_FALLBACK'] = 'TRUE'

# 添加必要目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # vim目录

# 添加mamba-1p1p1目录以导入mamba_ssm模块
mamba_dir = os.path.join(os.path.dirname(current_dir), 'mamba-1p1p1')
if os.path.exists(mamba_dir):
    sys.path.insert(0, mamba_dir)
else:
    print(f"警告: 未找到mamba-1p1p1目录: {mamba_dir}")

from timm.models import create_model
import models_mamba
from viztracer import VizTracer


def create_vim_tiny_model(pretrained=False, checkpoint_path=None,
                         use_cpp_scan=False, use_fixlen_scan=False,
                         use_fused_bidirectional=False):
    """
    创建Vim-tiny模型
    
    参数:
        pretrained: 是否使用预训练权重（如果可用）
        checkpoint_path: 预训练模型检查点路径
        use_cpp_scan: 使用C++优化实现（来自Mamba_CPU）
        use_fixlen_scan: 使用两阶段优化算法（仅当use_cpp_scan=True时有效）
        use_fused_bidirectional: 使用融合双向扫描（合并正向和反向计算）
    
    配置：
    - img_size: 224
    - patch_size: 16
    - embed_dim: 192
    - depth: 24
    - d_state: 16
    - num_classes: 1000
    """
    scan_type = "Python-Ref"
    if use_fused_bidirectional:
        scan_type = "Python-Fused-BiDir"
    elif use_cpp_scan:
        scan_type = "C++-Fixlen" if use_fixlen_scan else "C++-Original"
    print(f"创建Vim-tiny模型 (Scan类型: {scan_type})...")
    
    # 使用timm的create_model接口（参考infer_rpi.py）
    model = create_model(
        'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        use_cpp_scan=use_cpp_scan,
        use_fixlen_scan=use_fixlen_scan,
        use_fused_bidirectional=use_fused_bidirectional,
    )
    
    # 如果提供了检查点路径，加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return model


def warmup_inference(model, input_tensor, num_warmup=3):
    """模型预热"""
    print(f"\n预热推理 ({num_warmup}次)...")
    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(input_tensor)
            print(f"  预热 {i+1}/{num_warmup} 完成")


def benchmark_inference(model, input_tensor, num_runs=10):
    """性能基准测试"""
    print(f"\n基准性能测试 ({num_runs}次)...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()
            
            elapsed = (end - start) * 1000  # 转换为毫秒
            times.append(elapsed)
            print(f"  Run {i+1}/{num_runs}: {elapsed:.2f} ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n性能统计:")
    print(f"  平均时间: {avg_time:.2f} ms")
    print(f"  最小时间: {min_time:.2f} ms")
    print(f"  最大时间: {max_time:.2f} ms")
    print(f"  输出形状: {output.shape}")
    
    return avg_time, output


def profile_with_viztracer(model, input_tensor, output_file='vim_cpu_profile.html'):
    """使用VizTracer进行性能分析"""
    print(f"\n使用VizTracer进行性能分析...")
    print(f"输出文件: {output_file}")
    
    tracer = VizTracer(
        max_stack_depth=20,
        min_duration=100,  # 只记录>100us的函数调用
        ignore_frozen=True,
        log_sparse=True,
        verbose=0
    )
    
    tracer.start()
    
    # 执行推理
    with torch.no_grad():
        output = model(input_tensor)
    
    tracer.stop()
    tracer.save(output_file)
    
    print(f"VizTracer分析完成，报告已保存到: {output_file}")
    print(f"在浏览器中打开该文件查看详细性能分析")
    
    return output


def main():
    """主函数"""
    print("=" * 80)
    print("Vision Mamba CPU推理性能测试 - 对比三种Selective Scan实现")
    print("=" * 80)
    
    # 检查环境
    print("\n环境配置:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"  CPU线程数: {torch.get_num_threads()}")
    print(f"  SELECTIVE_SCAN_FORCE_FALLBACK: {os.environ.get('SELECTIVE_SCAN_FORCE_FALLBACK', 'FALSE')}")
    
    # 检查C++扩展是否可用
    try:
        import selective_scan_cpp
        print(f"  C++扩展: ✓ 可用")
        cpp_available = True
    except ImportError:
        print(f"  C++扩展: ✗ 不可用（将使用Python fallback）")
        cpp_available = False
    
    # 创建随机输入（batch_size=1, channels=3, height=224, width=224）
    print("\n创建输入张量...")
    input_tensor = torch.randn(1, 3, 224, 224)
    print(f"输入形状: {input_tensor.shape}")
    
    # 定义测试配置：Python和C++版本的原始、两阶段、融合实验
    test_configs = [
        # === Python实现 ===
        {
            'name': 'Python-Original',
            'use_cpp_scan': False,
            'use_fixlen_scan': False,
            'use_fused_bidirectional': False,
            'desc': 'Python原始实现（分离双向扫描）'
        },
        {
            'name': 'Python-Fused',
            'use_cpp_scan': False,
            'use_fixlen_scan': False,
            'use_fused_bidirectional': True,
            'desc': 'Python融合实现（N维度concat批量计算）'
        },
        
        # === C++实现 ===
        {
            'name': 'CPP-Original',
            'use_cpp_scan': True,
            'use_fixlen_scan': False,
            'use_fused_bidirectional': False,
            'desc': 'C++原始实现（分离双向扫描）'
        },
        {
            'name': 'CPP-Fixlen',
            'use_cpp_scan': True,
            'use_fixlen_scan': True,
            'use_fused_bidirectional': False,
            'desc': 'C++两阶段实现（优化循环）'
        },
        {
            'name': 'CPP-Fused',
            'use_cpp_scan': True,
            'use_fixlen_scan': False,
            'use_fused_bidirectional': True,
            'desc': 'C++融合实现（N维度concat批量计算）'
        },
        {
            'name': 'CPP-Fused-Fixlen',
            'use_cpp_scan': True,
            'use_fixlen_scan': True,
            'use_fused_bidirectional': True,
            'desc': 'C++融合+两阶段实现（双重优化）'
        },
    ]
    
    models = {}
    outputs = {}
    timings = {}
    
    # ===== 关键修改：先创建基础模型并保存参数 =====
    print("\n" + "=" * 80)
    print("步骤1: 创建基础模型并保存参数（确保所有测试使用相同参数）")
    print("=" * 80)
    
    base_model = create_vim_tiny_model(
        use_cpp_scan=False,  # 创建Python版本作为基础
        use_fixlen_scan=False
    )
    # 保存参数
    base_state_dict = base_model.state_dict()
    print(f"✓ 基础模型参数已保存（参数量: {sum(p.numel() for p in base_model.parameters())/1e6:.2f}M）")
    
    # 删除基础模型释放内存
    del base_model
    
    print("\n" + "=" * 80)
    print("步骤2: 测试各种配置（使用相同的模型参数）")
    print("=" * 80)
    
    # 测试每种配置
    for config in test_configs:
        name = config['name']
        print("\n" + "=" * 80)
        print(f"测试配置: {name} - {config['desc']}")
        print("=" * 80)
        
        # 如果C++不可用但请求使用C++，跳过
        if config['use_cpp_scan'] and not cpp_available:
            print(f"⚠ 跳过 {name}（C++扩展不可用）")
            continue
        
        # 创建模型
        model = create_vim_tiny_model(
            use_cpp_scan=config['use_cpp_scan'],
            use_fixlen_scan=config['use_fixlen_scan'],
            use_fused_bidirectional=config['use_fused_bidirectional']
        )
        
        # ===== 关键：加载相同的参数 =====
        model.load_state_dict(base_state_dict)
        print(f"✓ 已加载基础模型参数")
        
        models[name] = model
        
        # 预热
        warmup_inference(model, input_tensor, num_warmup=2)
        
        # 基准测试
        avg_time, output = benchmark_inference(model, input_tensor, num_runs=10)
        outputs[name] = output
        timings[name] = avg_time
        
        # VizTracer性能分析
        profile_file = f'vim_{name.lower().replace("+", "p")}_profile.html'
        _ = profile_with_viztracer(model, input_tensor, output_file=profile_file)
    
    # 对比输出一致性
    if len(outputs) > 1:
        print("\n" + "=" * 80)
        print("输出一致性验证")
        print("=" * 80)
        
        ref_name = list(outputs.keys())[0]
        ref_output = outputs[ref_name]
        
        for name, output in outputs.items():
            if name == ref_name:
                continue
            diff = torch.abs(output - ref_output).max().item()
            print(f"{name} vs {ref_name}: 最大差异 = {diff:.2e}", end="")
            if diff < 1e-4:
                print(" ✓ 一致")
            else:
                print(" ✗ 不一致!")
    
    # 性能对比
    if len(timings) > 1:
        print("\n" + "=" * 80)
        print("性能对比")
        print("=" * 80)
        
        ref_name = 'Python-Ref' if 'Python-Ref' in timings else list(timings.keys())[0]
        ref_time = timings[ref_name]
        
        print(f"{'配置':<20} {'时间(ms)':<12} {'相对加速':<12} {'说明'}")
        print("-" * 80)
        
        for name, avg_time in timings.items():
            speedup = ref_time / avg_time
            marker = "⭐" if speedup > 2.0 else ("✓" if speedup > 1.0 else "")
            print(f"{name:<20} {avg_time:>8.2f} ms   {speedup:>6.2f}x      {marker}")
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"模型: Vim-tiny")
    if models:
        first_model = list(models.values())[0]
        print(f"参数量: {sum(p.numel() for p in first_model.parameters())/1e6:.2f}M")
    print(f"输入尺寸: {input_tensor.shape}")
    if outputs:
        first_output = list(outputs.values())[0]
        print(f"输出尺寸: {first_output.shape}")
    print(f"测试配置数: {len(timings)}")
    print("\n生成的性能分析文件:")
    for name in outputs.keys():
        profile_file = f'vim_{name.lower().replace("+", "p")}_profile.html'
        print(f"  - {profile_file}")
    print("=" * 80)
    
    print("\n提示:")
    print("  1. 在浏览器中打开HTML文件查看详细性能分析")
    print("  2. 对比不同实现的selective_scan调用耗时")
    print("  3. 查看两阶段优化算法的性能提升")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
