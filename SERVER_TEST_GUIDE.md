# 学校服务器测试指南

## 🚀 快速部署步骤

### 1. 克隆仓库
```bash
git clone https://github.com/Dopamine-mania/qm_final_v3.git
cd qm_final_v3
```

### 2. 环境设置
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 系统测试
```bash
# 健康检查
python system_check.py

# 功能测试
python main.py --test

# 演示模式（推荐）
python main.py --demo
```

## 📊 预期测试结果

### 系统健康检查
应该看到：
```
整体状态: PASS
总检查项目: 7
通过: 7
成功率: 100.0%
```

### 演示模式输出
应该生成：
- 音频文件：`outputs/demo_*.wav`
- 视频帧：`outputs/demo_*.png` 
- 分析报告：`outputs/demo_*_report.json`

### 日志文件
- 主日志：`logs/qm_final3.log`
- 系统报告：`system_check_report.json`

## 🔍 关键测试指标

### 1. 六层架构初始化
所有6层应该成功初始化：
- InputLayer ✅
- FusionLayer ✅ (27维情绪识别)
- MappingLayer ✅ (KG-MLP混合)
- GenerationLayer ✅ (音视频生成)
- RenderingLayer ✅ (同步渲染)
- TherapyLayer ✅ (FSM治疗流程)

### 2. 性能指标
- 系统延迟：<500ms（目标）
- 内存使用：正常范围
- CPU使用：稳定

### 3. 功能验证
- 情绪识别：输出27维情绪分类
- 音频生成：生成治疗音乐文件
- 视频生成：生成视觉内容帧
- 数据保存：自动保存到outputs/目录

## ⚠️ 常见问题处理

### 警告信息（正常）
```
WARNING: pygame不可用，本地播放功能将受限
WARNING: OpenCV不可用，视频渲染功能将受限
WARNING: PyAudio不可用，跳过音频录制
```
这些警告是正常的，系统会使用备用方案。

### 依赖问题
如果遇到依赖问题：
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### 权限问题
确保对以下目录有写权限：
- `logs/`
- `outputs/` 
- `data/`

## 📝 测试反馈格式

请提供以下信息：

### 1. 环境信息
- 操作系统版本
- Python版本
- 可用内存/CPU

### 2. 测试结果
- `python system_check.py` 的完整输出
- `python main.py --test` 的执行结果
- `python main.py --demo` 的运行情况

### 3. 错误信息
- 任何错误或异常的完整堆栈跟踪
- 日志文件中的错误信息

### 4. 性能数据
- 各层处理时间
- 总体响应时间
- 资源使用情况

### 5. 生成文件
- 是否成功生成音频文件
- 是否成功生成视频帧
- 分析报告是否完整

## 🎯 关键验证点

### 必须通过的测试
1. ✅ 系统健康检查100%通过
2. ✅ 六层架构全部初始化成功  
3. ✅ 演示模式生成音视频文件
4. ✅ 27维情绪分类正常工作
5. ✅ FSM治疗流程状态正确

### 性能基准
- 启动时间：<30秒
- 单次处理：<2秒
- 内存峰值：<2GB

根据测试结果，我会进行相应的优化和修正！