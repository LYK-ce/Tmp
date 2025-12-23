#Presented by KeJi
#Date: 2025-12-22

# Vision Mamba C++扩展编译说明

## 快速编译（推荐）

### Windows（MSVC）

**在Developer Command Prompt for VS中运行**：

```bash
cd VisionMamba_CPU\mamba-1p1p1\mamba_ssm\ops
set DISTUTILS_USE_SDK=1
python setup.py build_ext --inplace
```

或直接双击运行：
```bash
compile_and_test.bat
```

### Linux/Mac（gcc）

```bash
cd VisionMamba_CPU/mamba-1p1p1/mamba_ssm/ops  
python setup.py build_ext --inplace
```

## 编译产物

- **Windows**: `selective_scan_cpp.cp313-win_amd64.pyd`
- **Linux**: `selective_scan_cpp.cpython-313-x86_64-linux-gnu.so`

## 测试

编译完成后：

```bash
cd ../../vim
python inf_cpu.py
```

将生成3个VizTracer报告：
- `vim_python-ref_profile.html`
- `vim_c-original_profile.html`
- `vim_c-fixlen_profile.html`

## 期待结果

- ✓ 输出一致性：差异<1e-4
- ✓ C++-Fixlen性能：2.5-3x加速

## 故障排除

**问题1**: cl.exe未找到
- **解决**: 使用"Developer Command Prompt for VS"

**问题2**: ninja构建失败  
- **解决**: 确保已安装ninja（`pip install ninja`）

**问题3**: Python版本不匹配
- **解决**: .pyd文件是cp313，需要Python 3.13
