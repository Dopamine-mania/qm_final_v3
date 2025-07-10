# QM_Final3 部署指南

## 系统要求

### 最低配置
- **操作系统**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+
- **Python**: 3.8+
- **内存**: 8GB RAM
- **存储**: 2GB 可用空间
- **网络**: 稳定的互联网连接

### 推荐配置
- **操作系统**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.9+
- **内存**: 16GB RAM
- **GPU**: NVIDIA GPU (可选，用于GPU加速)
- **存储**: 5GB 可用空间

## 安装步骤

### 1. 克隆仓库
```bash
git clone <your-github-repo-url>
cd qm_final3
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境（可选）
```bash
# 创建日志目录
mkdir -p logs

# 创建数据目录
mkdir -p data/therapy_sessions

# 创建输出目录
mkdir -p outputs
```

## 运行系统

### 测试模式
```bash
python main.py --test
```

### 演示模式
```bash
python main.py --demo
```

### 完整运行
```bash
python main.py
```

## 可选依赖

### 音视频处理 (推荐)
```bash
# 音频处理
pip install pyaudio librosa soundfile

# 视频处理
pip install opencv-python

# 图形界面
pip install pygame
```

### GPU加速 (可选)
```bash
# NVIDIA GPU支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'xyz'**
   - 确保已激活虚拟环境
   - 重新安装依赖: `pip install -r requirements.txt`

2. **pygame/OpenCV 不可用警告**
   - 这是正常的，系统会使用替代方案
   - 如需完整功能，请安装对应依赖

3. **权限错误**
   - 确保对项目目录有写权限
   - 检查日志和数据目录的权限

### 性能优化

1. **启用GPU加速**
   ```bash
   # 检查GPU可用性
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **调整配置**
   - 修改 `configs/six_layer_architecture.yaml`
   - 根据硬件调整批处理大小和超时时间

## 日志和监控

### 日志文件
- 主日志: `logs/qm_final3.log`
- 系统会自动创建带时间戳的日志文件

### 监控指标
- 处理时间
- 内存使用
- 错误率
- 系统状态

## 服务器部署特别说明

### 1. 无图形界面环境
系统设计为支持无GUI环境，会自动禁用图形相关功能。

### 2. 网络配置
如果需要远程访问，确保防火墙设置正确。

### 3. 自动启动
可以使用systemd或cron设置自动启动。

### 4. 资源限制
在资源受限的环境中，可以调整配置以降低资源消耗。

## 技术支持

如有问题，请检查：
1. 日志文件中的错误信息
2. 系统健康检查输出
3. 依赖安装是否完整

系统状态检查：
```bash
python main.py --test
```

## 版本信息
- 系统版本: 3.0.0
- 支持的Python版本: 3.8+
- 最后更新: 2025-07-10