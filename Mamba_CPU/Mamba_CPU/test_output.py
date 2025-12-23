#Presented by KeJi
#Date: 2025-12-20

"""
验证原始版本和优化版本的输出是否完全相同
"""

import torch
import numpy as np
from model import Mamba as MambaOld, ModelArgs
from model_new import Mamba as MambaNew


def create_identical_models():
    """创建两个参数完全相同的模型"""
    print("=== 创建测试模型 ===")
    
    # 设置随机种子确保参数初始化一致
    torch.manual_seed(42)
    args = ModelArgs(
        d_model=256,
        n_layer=4,
        vocab_size=1000,
        d_state=16,
        expand=2
    )
    model_old = MambaOld(args)
    model_old.eval()
    
    # 重置随机种子，确保第二个模型参数相同
    torch.manual_seed(42)
    model_new = MambaNew(args)
    model_new.eval()
    
    # 验证参数是否相同
    print("\n检查模型参数是否相同...")
    params_match = True
    for (name_old, param_old), (name_new, param_new) in zip(
        model_old.named_parameters(), 
        model_new.named_parameters()
    ):
        if not torch.allclose(param_old, param_new, rtol=1e-5, atol=1e-8):
            print(f"  ✗ 参数不匹配: {name_old}")
            params_match = False
    
    if params_match:
        print("  ✓ 所有参数完全相同")
    
    return model_old, model_new, args


def compare_outputs(model_old, model_new, input_ids, test_name="测试"):
    """比较两个模型的输出"""
    print(f"\n=== {test_name} ===")
    print(f"输入形状: {input_ids.shape}")
    
    with torch.no_grad():
        output_old = model_old(input_ids)
        output_new = model_new(input_ids)
    
    print(f"原始版本输出形状: {output_old.shape}")
    print(f"优化版本输出形状: {output_new.shape}")
    
    # 检查形状
    if output_old.shape != output_new.shape:
        print("✗ 输出形状不同！")
        return False
    
    # 计算差异
    abs_diff = torch.abs(output_old - output_new)
    rel_diff = abs_diff / (torch.abs(output_old) + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"\n数值差异分析:")
    print(f"  最大绝对差异: {max_abs_diff:.2e}")
    print(f"  平均绝对差异: {mean_abs_diff:.2e}")
    print(f"  最大相对差异: {max_rel_diff:.2e}")
    print(f"  平均相对差异: {mean_rel_diff:.2e}")
    
    # 使用不同的容差进行检查
    tolerances = [
        (1e-5, 1e-8, "严格"),
        (1e-4, 1e-6, "标准"),
        (1e-3, 1e-5, "宽松"),
    ]
    
    print(f"\n容差检查:")
    all_match = False
    for rtol, atol, name in tolerances:
        is_close = torch.allclose(output_old, output_new, rtol=rtol, atol=atol)
        status = "✓ 通过" if is_close else "✗ 未通过"
        print(f"  {name} (rtol={rtol}, atol={atol}): {status}")
        if is_close and not all_match:
            all_match = True
    
    # 统计相同元素的比例
    exact_match = (output_old == output_new).float().mean().item() * 100
    print(f"\n完全相同的元素比例: {exact_match:.2f}%")
    
    # 显示一些样本值对比
    print(f"\n样本值对比（前5个元素）:")
    flat_old = output_old.flatten()[:5]
    flat_new = output_new.flatten()[:5]
    for i, (v_old, v_new) in enumerate(zip(flat_old, flat_new)):
        diff = abs(v_old.item() - v_new.item())
        print(f"  [{i}] 原始={v_old.item():.6f}, 优化={v_new.item():.6f}, 差异={diff:.2e}")
    
    return all_match


def test_single_pass():
    """测试单次前向传播"""
    model_old, model_new, args = create_identical_models()
    
    # 准备输入
    torch.manual_seed(123)
    input_ids = torch.randint(0, args.vocab_size, (1, 128))
    
    compare_outputs(model_old, model_new, input_ids, "单次前向传播测试")


def test_multiple_sequences():
    """测试多个不同的序列"""
    model_old, model_new, args = create_identical_models()
    
    print("\n" + "="*60)
    print("测试多个不同序列")
    print("="*60)
    
    test_cases = [
        (1, 32, "短序列"),
        (1, 128, "中等序列"),
        (1, 256, "长序列"),
        (2, 64, "批量输入"),
    ]
    
    all_passed = True
    for batch_size, seq_len, name in test_cases:
        torch.manual_seed(seq_len)  # 使用序列长度作为种子
        input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        
        passed = compare_outputs(model_old, model_new, input_ids, f"{name} (batch={batch_size}, seq_len={seq_len})")
        all_passed = all_passed and passed
    
    return all_passed


def test_intermediate_values():
    """测试中间层的输出"""
    print("\n" + "="*60)
    print("测试中间层selective_scan的输出")
    print("="*60)
    
    torch.manual_seed(42)
    args = ModelArgs(
        d_model=256,
        n_layer=1,  # 只用一层便于测试
        vocab_size=1000,
        d_state=16,
        expand=2
    )
    
    model_old = MambaOld(args)
    model_old.eval()
    
    torch.manual_seed(42)
    model_new = MambaNew(args)
    model_new.eval()
    
    # 准备测试输入
    torch.manual_seed(123)
    input_ids = torch.randint(0, args.vocab_size, (1, 64))
    
    # 获取embedding输出
    with torch.no_grad():
        x_old = model_old.embedding(input_ids)
        x_new = model_new.embedding(input_ids)
        
        # 通过norm
        x_old = model_old.layers[0].norm(x_old)
        x_new = model_new.layers[0].norm(x_new)
        
        # 通过in_proj
        x_and_res_old = model_old.layers[0].mixer.in_proj(x_old)
        x_and_res_new = model_new.layers[0].mixer.in_proj(x_new)
        
        (u_old, res_old) = x_and_res_old.split(split_size=[args.d_inner, args.d_inner], dim=-1)
        (u_new, res_new) = x_and_res_new.split(split_size=[args.d_inner, args.d_inner], dim=-1)
        
        print("\n检查in_proj输出:")
        print(f"  u最大差异: {torch.abs(u_old - u_new).max().item():.2e}")
        print(f"  res最大差异: {torch.abs(res_old - res_new).max().item():.2e}")
        
        # 通过conv1d
        from einops import rearrange
        u_old = rearrange(u_old, 'b l d_in -> b d_in l')
        u_old = model_old.layers[0].mixer.conv1d(u_old)[:, :, :64]
        u_old = rearrange(u_old, 'b d_in l -> b l d_in')
        
        u_new = rearrange(u_new, 'b l d_in -> b d_in l')
        u_new = model_new.layers[0].mixer.conv1d(u_new)[:, :, :64]
        u_new = rearrange(u_new, 'b d_in l -> b l d_in')
        
        print("\n检查conv1d输出:")
        print(f"  最大差异: {torch.abs(u_old - u_new).max().item():.2e}")
        
        # 通过silu
        import torch.nn.functional as F
        u_old = F.silu(u_old)
        u_new = F.silu(u_new)
        
        print("\n检查silu输出:")
        print(f"  最大差异: {torch.abs(u_old - u_new).max().item():.2e}")
        
        # 直接调用selective_scan（这是核心测试）
        y_old = model_old.layers[0].mixer.ssm(u_old)
        y_new = model_new.layers[0].mixer.ssm(u_new)
        
        print("\n检查selective_scan (ssm)输出:")
        print(f"  形状: {y_old.shape} vs {y_new.shape}")
        print(f"  最大绝对差异: {torch.abs(y_old - y_new).max().item():.2e}")
        print(f"  平均绝对差异: {torch.abs(y_old - y_new).mean().item():.2e}")
        
        is_close = torch.allclose(y_old, y_new, rtol=1e-5, atol=1e-7)
        print(f"  数值是否接近(rtol=1e-5, atol=1e-7): {'✓ 是' if is_close else '✗ 否'}")
        
        if not is_close:
            # 显示差异分布
            diff = torch.abs(y_old - y_new)
            print(f"\n  差异分布:")
            print(f"    最小: {diff.min().item():.2e}")
            print(f"    25%: {diff.kthvalue(int(diff.numel()*0.25)).values.item():.2e}")
            print(f"    中位数: {diff.median().item():.2e}")
            print(f"    75%: {diff.kthvalue(int(diff.numel()*0.75)).values.item():.2e}")
            print(f"    最大: {diff.max().item():.2e}")


def main():
    """主函数"""
    print("="*60)
    print("Mamba模型输出一致性验证")
    print("="*60)
    
    # 测试1: 单次前向传播
    print("\n【测试1】单次前向传播")
    test_single_pass()
    
    # 测试2: 多个不同序列
    print("\n【测试2】多个不同序列")
    all_passed = test_multiple_sequences()
    
    # 测试3: 中间值验证
    print("\n【测试3】中间层验证")
    test_intermediate_values()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("\n如果所有测试的数值差异都在合理范围内（通常<1e-5），")
    print("则说明两个版本在数学上是等价的。")
    print("\n注意: 由于浮点运算顺序不同，可能存在微小的数值差异，")
    print("这是正常的，只要相对误差很小即可。")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
