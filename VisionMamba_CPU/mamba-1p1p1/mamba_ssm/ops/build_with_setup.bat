@echo off
REM Presented by KeJi
REM Date: 2025-12-22
REM 
REM 使用setup.py编译Vision Mamba C++扩展

echo ========================================
echo 设置MSVC环境
echo ========================================

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo.
echo ========================================
echo 编译C++扩展
echo ========================================

REM 设置必需的环境变量
set DISTUTILS_USE_SDK=1

REM 编译
python setup.py build_ext --inplace

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 编译失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 编译成功！
echo ========================================
echo.
echo 生成的文件：selective_scan_cpp.cp313-win_amd64.pyd
echo.
echo 现在可以运行测试：
echo   cd ..\..\..\..\vim
echo   python inf_cpu.py
echo.
pause
