#Presented by KeJi
#Date : 2025-12-20

"""
Mamba模型CPU性能分析脚本
使用viztracer追踪和分析模型推理的时间瓶颈
对比原始版本和优化版本的性能
"""

import torch
import time
from viztracer import VizTracer

# 检查依赖
try:
    from model import Mamba as MambaOld, ModelArgs
    print("[OK] 成功导入model模块（Python原始版本）")
except ImportError as e:
    print(f"[ERROR] 导入model失败: {e}")
    exit(1)

try:
    from model_new import Mamba as MambaNew
    print("[OK] 成功导入model_new模块（Python优化版本）")
except ImportError as e:
    print(f"[ERROR] 导入model_new失败: {e}")
    exit(1)

try:
    from model_cpp import Mamba as MambaCpp, CPP_EXTENSION_AVAILABLE
    if CPP_EXTENSION_AVAILABLE:
        print("[OK] 成功导入model_cpp模块（C++版本可用）")
    else:
        print("[WARNING] model_cpp模块已导入，但C++扩展不可用（将使用Python实现）")
except ImportError as e:
    print(f"[ERROR] 导入model_cpp失败: {e}")
    exit(1)

try:
    import viztracer
    print("[OK] viztracer已安装")
except ImportError:
    print("[ERROR] 请安装viztracer: pip install viztracer")
    exit(1)


def create_test_model(model_type='python_old'):
    """创建测试用的小型Mamba模型
    
    Args:
        model_type: 模型类型，可选：
            - 'python_old': Python原始版本
            - 'python_new': Python优化版本
            - 'cpp_old': C++原始版本
            - 'cpp_new': C++优化版本
    """
    version_map = {
        'python_old': 'Python原始版本',
        'python_new': 'Python优化版本',
        'cpp_old': 'C++原始版本（逐个处理）',
        'cpp_new': 'C++优化版本（两阶段计算）'
    }
    version_name = version_map.get(model_type, model_type)
    print(f"\n=== 创建测试模型 ({version_name}) ===")
    
    # 使用自定义配置创建小型模型以便快速测试
    args = ModelArgs(
        d_model=256,      # 较小的隐藏维度
        n_layer=4,        # 较少的层数
        vocab_size=1000,  # 较小的词表
        d_state=16,
        expand=2
    )
    
    # 根据类型创建模型
    if model_type == 'python_old':
        model = MambaOld(args)
    elif model_type == 'python_new':
        model = MambaNew(args)
    elif model_type == 'cpp_old':
        model = MambaCpp(args, use_cpp=True, use_fixlen=False)
    elif model_type == 'cpp_new':
        model = MambaCpp(args, use_cpp=True, use_fixlen=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.eval()
    
    print(f"模型参数: d_model={args.d_model}, n_layer={args.n_layer}, d_inner={args.d_inner}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    return model, args


def prepare_input_data(batch_size=1, seq_length=128, vocab_size=1000):
    """准备测试输入数据"""
    print(f"\n=== 准备输入数据 ===")
    print(f"Batch size: {batch_size}, 序列长度: {seq_length}")
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    return input_ids


def warmup_model(model, input_ids, num_warmup=3):
    """预热模型，避免首次推理的初始化开销"""
    print(f"\n=== 预热模型 ({num_warmup}次) ===")
    
    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(input_ids)
            print(f"预热 {i+1}/{num_warmup} 完成")


def benchmark_without_profiler(model, input_ids, num_runs=10):
    """不使用profiler的基准测试，获得真实性能"""
    print(f"\n=== 基准性能测试 ({num_runs}次) ===")
    
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.perf_counter()
            output = model(input_ids)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"运行 {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n平均时间: {avg_time*1000:.2f} ms")
    print(f"最小时间: {min_time*1000:.2f} ms")
    print(f"最大时间: {max_time*1000:.2f} ms")
    
    return avg_time


def profile_with_viztracer(model, input_ids, output_file='mamba_profile.html'):
    """使用viztracer进行详细的性能分析"""
    print(f"\n=== 使用VizTracer进行性能分析 ===")
    
    tracer = VizTracer(
        max_stack_depth=20,        # 追踪调用栈深度
        min_duration=0.0001,       # 只显示>=0.1ms的调用
        verbose=0,                 # 静默模式
        ignore_frozen=True,        # 忽略Python标准库
        log_sparse=True            # 启用自定义标记
    )
    
    print("开始追踪...")
    tracer.start()
    
    # 执行模型推理
    with torch.no_grad():
        output = model(input_ids)
    
    tracer.stop()
    print("追踪完成")
    
    # 保存结果
    tracer.save(output_file)
    print(f"\n性能分析结果已保存到: {output_file}")
    print(f"请在浏览器中打开该文件查看详细的时间线分析")
    
    return output


def run_multiple_sequence_lengths(model, vocab_size=1000):
    """测试不同序列长度的性能"""
    print("\n" + "="*60)
    print("=== 不同序列长度性能对比 ===")
    print("="*60)
    
    seq_lengths = [64, 128, 256, 512]
    results = []
    
    for seq_len in seq_lengths:
        print(f"\n--- 测试序列长度: {seq_len} ---")
        input_ids = prepare_input_data(batch_size=1, seq_length=seq_len, vocab_size=vocab_size)
        
        # 预热
        with torch.no_grad():
            _ = model(input_ids)
        
        # 测试
        times = []
        with torch.no_grad():
            for _ in range(5):
                start = time.perf_counter()
                _ = model(input_ids)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        results.append((seq_len, avg_time))
        print(f"平均时间: {avg_time*1000:.2f} ms ({avg_time*1000/seq_len:.3f} ms/token)")
    
    # 显示汇总
    print("\n" + "="*60)
    print("序列长度 | 平均时间  | 每token时间")
    print("-"*60)
    for seq_len, avg_time in results:
        print(f"{seq_len:8d} | {avg_time*1000:7.2f} ms | {avg_time*1000/seq_len:7.3f} ms")
    print("="*60)


def compare_models(model_old, model_new, input_ids, num_runs=10):
    """对比两个模型的性能"""
    print("\n" + "="*60)
    print("=== 性能对比: 原始版本 vs 优化版本 ===")
    print("="*60)
    
    # 测试原始版本
    print("\n--- 原始版本 ---")
    times_old = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.perf_counter()
            _ = model_old(input_ids)
            elapsed = time.perf_counter() - start
            times_old.append(elapsed)
    avg_old = sum(times_old) / len(times_old)
    print(f"平均时间: {avg_old*1000:.2f} ms")
    
    # 测试优化版本
    print("\n--- 优化版本 ---")
    times_new = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.perf_counter()
            _ = model_new(input_ids)
            elapsed = time.perf_counter() - start
            times_new.append(elapsed)
    avg_new = sum(times_new) / len(times_new)
    print(f"平均时间: {avg_new*1000:.2f} ms")
    
    # 计算加速比
    speedup = avg_old / avg_new
    improvement = (avg_old - avg_new) / avg_old * 100
    
    print("\n" + "="*60)
    print("对比结果:")
    print(f"原始版本: {avg_old*1000:.2f} ms")
    print(f"优化版本: {avg_new*1000:.2f} ms")
    print(f"加速比: {speedup:.2f}x")
    print(f"性能提升: {improvement:.1f}%")
    print("="*60)
    
    return avg_old, avg_new, speedup


def verify_outputs(models, input_ids):
    """验证所有模型的输出是否相同"""
    print("\n" + "="*60)
    print("=== 输出结果验证 ===")
    print("="*60)
    
    # 收集所有模型的输出
    outputs = {}
    with torch.no_grad():
        for model_type, model in models.items():
            output = model(input_ids)
            outputs[model_type] = output
            print(f"{model_type}: shape={output.shape}, mean={output.mean():.6f}, std={output.std():.6f}")
    
    # 以python_old为基准进行对比
    baseline = outputs['python_old']
    all_match = True
    
    print("\n相对误差对比（相对于python_old）:")
    print("-"*60)
    print(f"{'版本':<25} | {'最大绝对误差':>15} | {'平均相对误差':>15} | {'是否一致':>10}")
    print("-"*60)
    
    for model_type in models.keys():
        if model_type == 'python_old':
            print(f"{model_type:<25} | {'0.000000':>15} | {'0.00%':>15} | {'基准':>10}")
            continue
        
        output = outputs[model_type]
        
        # 计算绝对误差
        abs_diff = torch.abs(baseline - output)
        max_abs_error = abs_diff.max().item()
        
        # 计算相对误差
        rel_diff = abs_diff / (torch.abs(baseline) + 1e-8)
        mean_rel_error = rel_diff.mean().item() * 100
        
        # 判断是否一致（使用较宽松的阈值，因为浮点运算）
        is_close = torch.allclose(baseline, output, rtol=1e-4, atol=1e-5)
        match_str = "[OK]" if is_close else "[DIFF]"
        
        if not is_close:
            all_match = False
        
        print(f"{model_type:<25} | {max_abs_error:>15.6e} | {mean_rel_error:>14.4f}% | {match_str:>10}")
    
    print("="*60)
    
    if all_match:
        print("\n[OK] 所有模型输出结果一致！")
        print("验证通过：不同实现方式产生相同的数值结果")
    else:
        print("\n[WARNING] 部分模型输出存在差异")
        print("可能原因：浮点运算顺序不同导致的数值误差")
        print("如果相对误差<0.01%，通常可以接受")
    
    return all_match


def main():
    """主函数 - 测试四种版本的Mamba实现"""
    print("="*60)
    print("Mamba模型CPU性能分析与对比")
    print("包含Python和C++两种实现，每种各有两个版本")
    print("="*60)
    
    # 1. 创建四个版本的模型
    model_types = ['python_old', 'python_new', 'cpp_old', 'cpp_new']
    models = {}
    
    print("\n" + "="*60)
    print("第一部分: 创建所有模型版本")
    print("="*60)
    
    # 先创建第一个模型，保存其权重
    first_model, args = create_test_model(model_types[0])
    models[model_types[0]] = first_model
    baseline_state_dict = first_model.state_dict()
    
    # 创建其他模型并加载相同的权重
    for model_type in model_types[1:]:
        model, _ = create_test_model(model_type)
        model.load_state_dict(baseline_state_dict)  # 加载相同权重
        models[model_type] = model
        print(f"[OK] {model_type} 已加载相同权重")
    
    # 2. 准备输入
    input_ids = prepare_input_data(batch_size=1, seq_length=128, vocab_size=args.vocab_size)
    
    # 3. 验证输出结果一致性
    print("\n" + "="*60)
    print("第二部分: 验证输出结果一致性")
    print("="*60)
    verify_outputs(models, input_ids)
    
    # 4. 预热所有模型
    print("\n" + "="*60)
    print("第三部分: 预热所有模型")
    print("="*60)
    for model_type, model in models.items():
        print(f"\n--- 预热 {model_type} ---")
        warmup_model(model, input_ids, num_warmup=2)
    
    # 5. 基准测试所有版本
    print("\n" + "="*60)
    print("第四部分: 基准性能测试")
    print("="*60)
    
    benchmark_results = {}
    for model_type, model in models.items():
        print(f"\n--- {model_type} 基准测试 ---")
        avg_time = benchmark_without_profiler(model, input_ids, num_runs=10)
        benchmark_results[model_type] = avg_time
    
    # 6. VizTracer性能分析
    print("\n" + "="*60)
    print("第五部分: VizTracer性能分析")
    print("="*60)
    
    profile_files = {
        'python_old': 'mamba_python_old_profile.html',
        'python_new': 'mamba_python_new_profile.html',
        'cpp_old': 'mamba_cpp_old_profile.html',
        'cpp_new': 'mamba_cpp_new_profile.html'
    }
    
    for model_type, model in models.items():
        print(f"\n--- {model_type} VizTracer分析 ---")
        profile_with_viztracer(model, input_ids, output_file=profile_files[model_type])
    
    # 7. 性能对比汇总
    print("\n" + "="*60)
    print("第六部分: 性能对比汇总")
    print("="*60)
    
    print("\n所有版本性能对比:")
    print("-"*60)
    print(f"{'版本':<25} | {'平均时间':>12} | {'相对加速':>10}")
    print("-"*60)
    
    baseline_time = benchmark_results['python_old']
    for model_type in model_types:
        avg_time = benchmark_results[model_type]
        speedup = baseline_time / avg_time
        print(f"{model_type:<25} | {avg_time*1000:>10.2f} ms | {speedup:>9.2f}x")
    print("="*60)
    
    # 7. 详细对比
    print("\n" + "="*60)
    print("详细版本对比")
    print("="*60)
    
    # Python版本对比
    print("\n【Python实现对比】")
    py_old = benchmark_results['python_old']
    py_new = benchmark_results['python_new']
    py_speedup = py_old / py_new
    print(f"原始版本: {py_old*1000:.2f} ms")
    print(f"优化版本: {py_new*1000:.2f} ms")
    print(f"加速比: {py_speedup:.2f}x ({(py_speedup-1)*100:.1f}% 提升)")
    
    # C++版本对比
    print("\n【C++实现对比】")
    cpp_old = benchmark_results['cpp_old']
    cpp_new = benchmark_results['cpp_new']
    cpp_speedup = cpp_old / cpp_new
    print(f"原始算法: {cpp_old*1000:.2f} ms")
    print(f"优化算法: {cpp_new*1000:.2f} ms")
    print(f"加速比: {cpp_speedup:.2f}x ({(cpp_speedup-1)*100:.1f}% 提升)")
    
    # Python vs C++对比
    print("\n【Python vs C++对比（原始算法）】")
    py_cpp_speedup_old = py_old / cpp_old
    print(f"Python: {py_old*1000:.2f} ms")
    print(f"C++: {cpp_old*1000:.2f} ms")
    print(f"C++加速比: {py_cpp_speedup_old:.2f}x")
    
    print("\n【Python vs C++对比（优化算法）】")
    py_cpp_speedup_new = py_new / cpp_new
    print(f"Python: {py_new*1000:.2f} ms")
    print(f"C++: {cpp_new*1000:.2f} ms")
    print(f"C++加速比: {py_cpp_speedup_new:.2f}x")
    
    # 总体最佳提升
    print("\n【总体优化效果】")
    total_speedup = py_old / cpp_new
    print(f"初始版本（Python原始）: {py_old*1000:.2f} ms")
    print(f"最终版本（C++优化）: {cpp_new*1000:.2f} ms")
    print(f"总加速比: {total_speedup:.2f}x ({(total_speedup-1)*100:.1f}% 性能提升)")
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    print("\n生成的文件:")
    for model_type, file in profile_files.items():
        print(f"{model_type:>15}: {file}")
    print("\n下一步:")
    print("1. 在浏览器中打开HTML文件查看详细的时间线分析")
    print("2. 重点关注 selective_scan() 函数的性能变化")
    print("3. 对比Python和C++实现的差异")
    print("4. 分析两阶段计算优化的效果")


if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    
    # 运行主程序
    main()
