# 🧹 项目清理计划

## 📋 当前状态分析

项目确实变得臃肿，有很多重复和过时的文件。需要立即清理！

### 🎯 保留的核心文件（精简版）

#### 🚀 **用户界面**
- `gradio_demo_optimized.py` - 最新推荐的双模式演示
- `quick_test.py` - 系统测试工具
- `install_dependencies.py` - 依赖安装脚本

#### 🏗️ **核心系统**
- `main.py` - 主程序入口
- `layers/` - 六层架构核心
- `core/` - 核心组件
- `configs/` - 配置文件

#### 📚 **文档**
- `README.md` - 项目主文档
- `OPTIMIZED_DEMO_GUIDE.md` - 优化演示指南
- `API_CONFIG_TEMPLATE.md` - API配置指南

#### 🔧 **工具**
- `requirements.txt` - 依赖列表

### ❌ 需要删除的冗余文件

#### 过时的Gradio版本
- `app.py` - 旧版Streamlit界面
- `gradio_app.py` - 旧版Gradio界面
- `gradio_debug.py` - 调试版本
- `gradio_demo_enhanced.py` - 被优化版本替代
- `gradio_demo_mode.py` - 被优化版本替代
- `gradio_demo_simple.py` - 被优化版本替代
- `gradio_fixed.py` - 被优化版本替代

#### 过时的工具脚本
- `create_public_link.py` - 旧版隧道工具
- `create_tunnel.py` - 旧版隧道工具
- `simple_tunnel.py` - 旧版隧道工具
- `deploy.sh` - 旧版部署脚本
- `gpu_test.py` - 测试脚本
- `performance_optimizer.py` - 旧版优化工具
- `system_check.py` - 被quick_test.py替代

#### 过时的文档
- `DEPLOYMENT.md` - 旧版部署文档
- `ENHANCED_DEMO_GUIDE.md` - 被优化版本替代
- `SYNC_VIDEO_FIX.md` - 已集成到优化版本
- `SERVER_TEST_GUIDE.md` - 信息已整合
- `PROJECT_SUMMARY.md` - 信息已整合

#### 临时和测试文件
- `commit_msg.txt` - 临时文件
- `extracted_text.txt` - 临时文件
- `system_check_report.json` - 测试输出
- `example_input_layer_usage.py` - 示例文件
- `test_*.py` - 所有测试文件
- `therapy_evaluator.py` - 测试文件

### 📊 清理后的项目结构

```
qm_final3/
├── 📁 core/                    # 核心组件
├── 📁 layers/                  # 六层架构
├── 📁 configs/                 # 配置文件
├── 📁 data/                    # 数据目录
├── 📁 logs/                    # 日志文件
├── 📁 outputs/                 # 输出结果
├── 📄 main.py                  # 主程序
├── 📄 gradio_demo_optimized.py # 用户界面
├── 📄 quick_test.py            # 系统测试
├── 📄 install_dependencies.py  # 依赖安装
├── 📄 requirements.txt         # 依赖列表
├── 📄 README.md               # 项目文档
├── 📄 OPTIMIZED_DEMO_GUIDE.md # 使用指南
└── 📄 API_CONFIG_TEMPLATE.md  # API配置
```

## 🎯 清理原则

1. **保留最新最优** - 只保留最新的优化版本
2. **去除重复** - 删除功能重复的文件
3. **清理临时** - 删除所有临时和测试文件
4. **整合文档** - 合并重复的文档内容
5. **专注核心** - 只保留核心功能文件

## ⚡ 立即执行清理

准备删除约25个冗余文件，将项目从臃肿状态恢复到简洁高效的状态。