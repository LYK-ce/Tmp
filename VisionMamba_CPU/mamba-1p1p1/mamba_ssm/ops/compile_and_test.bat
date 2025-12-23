@echo off
REM Presented by KeJi
REM Date: 2025-12-22
REM
REM 使用与Mamba_CPU相同的方法编译和测试

echo ================================================================================
echo Vision Mamba C++扩展编译和测试
echo ================================================================================
echo.

REM 查找vcvarsall.bat（支持多个VS版本）
set "VCVARS_PATH="

REM 尝试VS 2026
if exist "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS_PATH=C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvarsall.bat"
    echo [找到] Visual Studio 2026 Community
    goto :setup_env
)

REM 尝试VS 2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    echo [找到] Visual Studio 2022 Community
    goto :setup_env
)

REM 尝试VS 2019
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
    echo [找到] Visual Studio 2019 Community
    goto :setup_env
)

REM 未找到
echo [错误] 未找到Visual Studio
echo.
echo 请安装Visual Studio Build Tools或Community版本
echo 下载地址: https://visualstudio.microsoft.com/downloads/
echo.
pause
exit /b 1

:setup_env
echo.
echo [步骤1] 设置MSVC环境...
call "%VCVARS_PATH%" x64
if %ERRORLEVEL% NEQ 0 (
    echo [错误] MSVC环境设置失败
    pause
    exit /b 1
)

echo [OK] MSVC环境已设置
echo.

REM 验证cl.exe可用
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] cl.exe不在PATH中，即使运行了vcvarsall.bat
    pause
    exit /b 1
)
echo [OK] 找到cl.exe

echo.
echo [步骤2] 编译C++扩展...
set DISTUTILS_USE_SDK=1
python setup.py build_ext --inplace

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo [成功] 编译完成!
    echo ========================================
    echo.
    echo [生成文件] selective_scan_cpp.cp313-win_amd64.pyd
    echo.
) else (
    echo.
    echo ========================================
    echo [失败] 编译失败，错误代码: %ERRORLEVEL%
    echo ========================================
    pause
    exit /b %ERRORLEVEL%
)

echo [步骤3] 运行inf_cpu.py测试...
echo.
cd ..\..\..\..\vim
python inf_cpu.py

echo.
echo ========================================
echo 完成！
echo ========================================
pause
