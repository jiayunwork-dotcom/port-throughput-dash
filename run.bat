@echo off
chcp 65001 >nul
title 港口集装箱吞吐量预测分析平台

setlocal
cd /d "%~dp0"

echo ============================================
echo   港口集装箱吞吐量预测与堆场优化分析平台
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo Python版本: %PYVER%
echo.

set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%" (
    echo 首次运行，正在创建虚拟环境...
    python -m venv "%VENV_DIR%"
    echo 虚拟环境创建完成
)

call "%VENV_DIR%\Scripts\activate.bat"

echo 检查并安装依赖...
python -m pip install --upgrade pip >nul
if exist "%~dp0requirements.txt" (
    pip install -r "%~dp0requirements.txt"
)

echo.
echo 检查数据文件...
if not exist "%~dp0data\船舶靠泊记录.csv" (
    echo 正在生成模拟数据...
    if not exist "%~dp0data" mkdir "%~dp0data"
    if not exist "%~dp0reports" mkdir "%~dp0reports"
    python -c "import sys; sys.path.insert(0, r'%~dp0'); from src.data_generator import PortDataGenerator; gen = PortDataGenerator(); gen.generate_all(r'%~dp0data')"
)

echo.
echo 启动Dash应用...
echo 访问地址: http://localhost:8050
echo 按 Ctrl+C 停止服务
echo.

python "%~dp0app.py"

pause
