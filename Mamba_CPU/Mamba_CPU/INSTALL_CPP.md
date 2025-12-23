#Presented by KeJi
#Date: 2025-12-20

# C++ 扩展安装指南

## 问题说明

运行 `test.py` 时遇到错误：
```
subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
```

**原因**: Windows系统未找到C++编译器（MSVC的`cl.exe`）

## 解决方案

### 方案1: 安装MSVC（推荐，官方支持）

#### 步骤：
1. 下载 Visual Studio Build Tools:
   https://visualstudio.microsoft.com/downloads/
   
2. 运行安装程序，选择 **"Desktop development with C++"**

3. 确保勾选以下组件：
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows 10 SDK

4. 安装完成后，**重启命令行**

5. 验证安装：
   ```cmd
   where cl
   ```
   应该输出类似：`C:\Program Files\Microsoft Visual Studio\...\cl.exe`

#### 优点
- ✅ PyTorch官方推荐
- ✅ 最佳兼容性
- ✅ 稳定性好

#### 缺点
- ⚠️ 安装包较大（~7GB）
- ⚠️ 编译速度较慢

---

### 方案2: 安装MinGW-w64（轻量级）

#### 步骤：
1. 下载MinGW-w64（选择x86_64版本）:
   https://github.com/niXman/mingw-builds-binaries/releases
   
2. 解压到无空格路径，如：`C:\mingw64`

3. 添加到系统PATH:
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在系统变量PATH中添加：`C:\mingw64\bin`

4. **重启命令行**

5. 验证安装：
   ```cmd
   gcc --version
   g++ --version
   ```

6. 设置环境变量（每次运行前执行）：
   ```cmd
   set CC=gcc
   set CXX=g++
   set DISTUTILS_USE_SDK=1
   ```

#### 优点
- ✅ 体积小（~500MB）
- ✅ 编译速度快
- ✅ Linux风格编译器

#### 缺点
- ⚠️ 需要手动设置环境变量
- ⚠️ 可能存在ABI兼容性问题

---

### 方案3: 仅使用Python版本（无需编译器）

#### 说明
如果暂时无法安装编译器，代码已自动降级到Python实现。

#### 运行方式
```bash
python test.py
```

程序会显示：
```
⚠ model_cpp模块已导入，但C++扩展不可用（将使用Python实现）
警告: 请求使用C++实现但扩展不可用，降级到Python实现
```

#### 影响
- `cpp_old` 和 `cpp_new` 模型会自动使用Python实现
- 可以正常运行测试，但无法对比C++性能提升

---

## 推荐工作流

### 开发阶段（现在）
1. 先使用**方案3**测试Python版本功能正确性
2. 确认代码逻辑无误

### 性能测试阶段（稍后）
1. 安装编译器（推荐**方案1**）
2. 重新运行 `test.py` 对比C++性能
3. 使用VizTracer分析性能提升

---

## 验证安装

安装编译器后，运行以下测试：

```bash
python -c "from model_cpp import CPP_EXTENSION_AVAILABLE; print('C++ OK' if CPP_EXTENSION_AVAILABLE else 'Failed')"
```

预期输出：
```
✓ C++扩展加载成功
C++ OK
```

---

## 常见问题

### Q1: MSVC安装后仍然报错
**A**: 确保使用"Developer Command Prompt for VS"或重启系统

### Q2: MinGW找不到gcc
**A**: 检查PATH环境变量是否正确设置，确保包含`mingw64\bin`

### Q3: 能否混用MSVC和MinGW？
**A**: 不推荐，选择其一即可

### Q4: 编译很慢怎么办？
**A**: 首次编译较慢（1-2分钟），后续修改代码会自动增量编译（快很多）

---

## 参考资料

- PyTorch C++ Extension官方文档：
  https://pytorch.org/tutorials/advanced/cpp_extension.html
  
- 本项目的C++工作流程：
  参见 [`cpp_workflow.md`](cpp_workflow.md)
