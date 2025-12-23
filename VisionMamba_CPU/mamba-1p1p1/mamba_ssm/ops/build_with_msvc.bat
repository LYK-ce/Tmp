@echo off
REM Presented by KeJi
REM Date: 2025-12-22
REM 
REM 编译Vision Mamba Selective Scan C++扩展
REM 需要在Developer Command Prompt for VS中运行

echo ========================================
echo 编译Vision Mamba C++扩展
echo ========================================
echo.

REM 查找vcvarsall.bat
set "VCVARS_PATH="
for %%V in (2026 2022 2019) do (
    if exist "C:\Program Files\Microsoft Visual Studio\%%V\Community\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VCVARS_PATH=C:\Program Files\Microsoft Visual Studio\%%V\Community\VC\Auxiliary\Build\vcvarsall.bat"
        echo 找到Visual Studio %%V
        goto :found_vs
    )
)

:found_vs
if not defined VCVARS_PATH (
    echo 错误: 未找到Visual Studio
    pause
    exit /b 1
)

echo 设置MSVC环境...
call "%VCVARS_PATH%" x64

REM 设置环境变量
set DISTUTILS_USE_SDK=1

echo.
echo 开始编译...
python setup.py build_ext --inplace

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 编译成功!
    echo 生成文件: selective_scan_cpp.cp313-win_amd64.pyd
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 编译失败! 错误代码: %ERRORLEVEL%
    echo ========================================
)

echo.
pause
