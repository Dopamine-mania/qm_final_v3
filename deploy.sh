#!/bin/bash
# qm_final_v3 一键部署脚本

set -e  # 遇到错误立即退出

echo "🚀 开始部署 qm_final_v3 系统..."

# 检查Python版本
echo "📋 检查Python版本..."
python3 --version || {
    echo "❌ Python 3 未安装，请先安装 Python 3.8+"
    exit 1
}

# 检查系统资源
echo "📋 检查系统资源..."
echo "内存信息:"
free -h || echo "无法获取内存信息"
echo "磁盘空间:"
df -h . || echo "无法获取磁盘信息"

# 创建虚拟环境
echo "🔧 创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
else
    echo "ℹ️  虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "🔧 升级pip..."
pip install --upgrade pip

# 安装核心依赖
echo "📦 安装核心依赖..."
pip install numpy>=1.21.0 matplotlib>=3.4.0 scipy>=1.7.0 pyyaml>=6.0 || {
    echo "❌ 核心依赖安装失败"
    exit 1
}

# 安装PyTorch
echo "📦 安装PyTorch..."
pip install torch>=2.0.0 torchaudio>=2.0.0 torchvision>=0.15.0 || {
    echo "⚠️  GPU版本安装失败，尝试CPU版本..."
    pip install torch==2.0.0+cpu torchaudio==2.0.0+cpu torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html || {
        echo "❌ PyTorch安装失败"
        exit 1
    }
}

# 安装其他必要依赖
echo "📦 安装其他依赖..."
pip install pandas>=1.3.0 psutil>=5.8.0 tqdm>=4.62.0 pillow>=9.0.0 click>=8.0.0 pydantic>=2.0.0

# 安装可选依赖 (不强制)
echo "📦 安装可选依赖..."
pip install opencv-python>=4.5.0 || echo "⚠️  OpenCV安装失败，视频功能将受限"
pip install librosa>=0.9.0 soundfile>=0.10.0 || echo "⚠️  音频处理库安装失败"
pip install pyaudio>=0.2.11 || echo "⚠️  PyAudio安装失败，录音功能将被禁用"

# 创建必要目录
echo "🔧 创建目录结构..."
mkdir -p logs
mkdir -p data/therapy_sessions
mkdir -p outputs

# 设置权限
echo "🔧 设置文件权限..."
chmod +x main.py
chmod +x system_check.py

# 运行系统检查
echo "🔍 运行系统健康检查..."
python system_check.py

# 运行测试
echo "🧪 运行系统测试..."
python main.py --test

echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 快速命令："
echo "  测试系统: python main.py --test"
echo "  启动系统: python main.py"
echo "  性能检查: python performance_optimizer.py"
echo "  系统状态: python system_check.py"
echo ""
echo "📝 注意事项："
echo "  - 确保虚拟环境已激活: source venv/bin/activate"
echo "  - 查看日志文件: logs/ 目录"
echo "  - 配置文件: configs/six_layer_architecture.yaml"
echo ""
echo "✅ 系统已准备就绪，可以开始测试！"