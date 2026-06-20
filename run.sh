#!/bin/bash
# 港口集装箱吞吐量预测分析平台 - 启动脚本

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "  港口集装箱吞吐量预测与堆场优化分析平台"
echo "============================================"
echo ""

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "错误: 未找到Python解释器，请先安装Python 3.8+"
    exit 1
fi

echo "检测Python版本..."
PYTHON_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python版本: $PYTHON_VER"

VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "首次运行，正在创建虚拟环境..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "虚拟环境创建完成: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo ""
echo "检查依赖安装..."
pip install --upgrade pip > /dev/null 2>&1

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
fi

echo ""
echo "检查数据文件..."
if [ ! -d "$PROJECT_DIR/data" ] || [ ! -f "$PROJECT_DIR/data/船舶靠泊记录.csv" ]; then
    echo "正在生成模拟数据..."
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/reports"
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from src.data_generator import PortDataGenerator
gen = PortDataGenerator()
gen.generate_all('$PROJECT_DIR/data')
"
fi

echo ""
echo "启动Dash应用..."
echo "访问地址: http://localhost:8050"
echo "按 Ctrl+C 停止服务"
echo ""

$PYTHON_CMD "$PROJECT_DIR/app.py"
