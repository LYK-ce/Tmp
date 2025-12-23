@echo off
REM Presented by KeJi
REM Date: 2025-12-22
REM 
REM 此脚本自动设置MSVC环境并运行test.py

echo ========================================
echo 设置MSVC编译环境
echo ========================================

REM 查找并调用vcvarsall.bat
set VCVARSALL_FOUND=0

if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo 找到 Visual Studio 18 Community
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    set VCVARSALL_FOUND=1
)

if %VCVARSALL_FOUND%==0 if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo 找到 Visual Studio 2022 Community
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    set VCVARSALL_FOUND=1
)

if %VCVARSALL_FOUND%==0 if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    echo 找到 Visual Studio 2022 Professional
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
    set VCVARSALL_FOUND=1
)

if %VCVARSALL_FOUND%==0 if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    echo 找到 Visual Studio 2022 Enterprise
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
    set VCVARSALL_FOUND=1
)

if %VCVARSALL_FOUND%==0 (
    echo [ERROR] 未找到MSVC编译器环境
    echo 请安装Visual Studio或Visual Studio Build Tools
    pause
    exit /b 1
)

echo.
echo ========================================
echo MSVC环境设置完成
echo ========================================
echo.

REM 验证cl.exe是否可用
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] cl.exe仍然无法找到
    pause
    exit /b 1
)

echo [OK] cl.exe已就绪
cl /? 2>&1 | findstr /C:"C/C++"
echo.

echo ========================================
echo 运行test.py
echo ========================================
echo.

python test.py

echo.
echo ========================================
echo 测试完成
echo ========================================
pause
