# 🌙 qm_final3 - 六层架构心境流转系统

**基于情绪驱动的细粒度三阶段音乐治疗叙事系统**

## 🎯 项目概述

qm_final3 是基于qm_final2升级的六层架构系统，实现了Draft文档中描述的先进技术方案：

- **27维细粒度情绪分类**：专门针对睡眠前状态的情绪识别
- **多模态融合**：文本+语音+面部表情的综合分析
- **KG-MLP混合映射**：知识图谱与深度学习的融合架构
- **亚500ms实时性能**：极低延迟的实时音视频生成
- **FSM驱动的治疗流程**：基于ISO原则的三阶段治疗

## 🏗️ 六层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    六层架构 (Six-Layer Architecture)           │
├─────────────────────────────────────────────────────────────┤
│ Layer 6: Therapy Layer    │ FSM治疗流程 + ISO三阶段          │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Rendering Layer  │ 实时渲染 + 音视频同步            │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Generation Layer │ 音乐生成 + 视频生成              │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Mapping Layer    │ KG-MLP混合映射                  │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Fusion Layer     │ 27维情绪融合                    │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Input Layer      │ 多模态数据采集                  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
qm_final3/
├── 📁 layers/              # 六层架构核心
│   ├── base_layer.py       # 基础层接口和管道
│   ├── input_layer.py      # 输入层 - 多模态数据采集
│   ├── fusion_layer.py     # 融合层 - 27维情绪识别
│   ├── mapping_layer.py    # 映射层 - KG-MLP混合映射
│   ├── generation_layer.py # 生成层 - 音视频生成
│   ├── rendering_layer.py  # 渲染层 - 实时渲染
│   └── therapy_layer.py    # 治疗层 - FSM治疗流程
├── 📁 core/                # 核心组件
│   ├── theory/             # 理论基础 (从qm_final2迁移)
│   ├── models/             # 模型管理
│   └── utils.py            # 工具函数
├── 📁 configs/             # 配置文件
│   ├── six_layer_architecture.yaml  # 六层架构配置
│   ├── emotion_27d.yaml    # 27维情绪配置
│   ├── theory_params.yaml  # 理论参数 (从qm_final2)
│   └── models.yaml         # 模型配置 (从qm_final2)
├── 📁 api/                 # API接口
├── 📁 evaluation/          # 评估和测试
├── 📁 data/                # 数据管理
├── 📁 outputs/             # 输出结果
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv qm_final3_env
source qm_final3_env/bin/activate  # Windows: qm_final3_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行系统

```bash
# 测试模式 - 验证系统配置
python main.py --test

# 演示模式 - 基础功能演示
python main.py --demo

# 完整模式 - 启动完整系统
python main.py

# 详细日志模式
python main.py --verbose
```

### 3. 使用自定义配置

```bash
# 使用自定义配置文件
python main.py --config configs/custom_config.yaml
```

## 🔧 核心技术特性

### 1. 27维情绪分类

基于Draft文档的创新设计，扩展了传统的9种基础情绪：

**基础情绪 (9维)**：
- 愤怒、恐惧/焦虑、厌恶、悲伤、娱乐、喜悦、灵感、温柔、中性

**睡眠专用情绪 (18维)**：
- 反刍思考、睡眠焦虑、身体疲惫、精神疲惫、过度觉醒
- 就寝担忧、睡眠恐惧、思维奔逸、躯体紧张、情感麻木
- 不安能量、睡眠挫败、就寝孤独、预期焦虑、睡眠完美主义
- 卧室不适、睡眠监控焦虑、晨起恐惧

### 2. 多模态融合

- **文本分析**：情绪关键词提取、语义特征分析
- **语音分析**：韵律特征、音调变化、语速分析
- **面部表情**：基于MediaPipe的实时面部情绪识别
- **融合策略**：confidence-weighted hybrid fusion

### 3. KG-MLP混合映射

- **知识图谱**：编码睡眠治疗专家知识
- **MLP网络**：学习个性化情绪-音乐映射
- **混合融合**：专家知识与数据驱动的平衡

### 4. 实时性能优化

- **目标延迟**：<500ms 端到端处理
- **层级优化**：每层独立的性能监控
- **异步处理**：并行多模态数据处理
- **硬件加速**：GPU优化的模型推理

## 🔄 升级自qm_final2

### 主要改进

1. **架构升级**：从3层架构升级为6层架构
2. **情绪扩展**：从9维情绪扩展为27维情绪空间
3. **多模态完善**：增加面部表情识别
4. **映射升级**：从简单映射升级为KG-MLP架构
5. **性能优化**：实现亚500ms实时响应

### 迁移的组件

- ✅ **配置系统**：theory_params.yaml, models.yaml等
- ✅ **理论基础**：ISO原则、音乐心理学、睡眠生理学
- ✅ **API框架**：基础API接口结构
- 🔄 **情绪识别**：扩展为27维情绪分类
- 🔄 **音乐生成**：集成MusicGen和质量评估
- 🔄 **视频生成**：治疗性视觉内容生成

## 📊 性能指标

### 目标性能

- **总延迟**：<500ms (端到端)
- **情绪识别准确率**：>85%
- **治疗效果**：>75% 情绪转换成功率
- **系统稳定性**：>99% 可用性

### 各层性能分配

- **Input Layer**: 50ms
- **Fusion Layer**: 150ms  
- **Mapping Layer**: 100ms
- **Generation Layer**: 150ms
- **Rendering Layer**: 80ms
- **Therapy Layer**: 50ms

## 🧪 开发状态

### 已完成 ✅

- [x] 项目结构创建
- [x] 六层架构接口定义
- [x] 输入层完整实现
- [x] 配置系统迁移和扩展
- [x] 核心工具模块
- [x] 主程序框架

### 开发中 🔄

- [ ] 融合层（27维情绪识别）
- [ ] 映射层（KG-MLP架构）
- [ ] 生成层（音视频生成）
- [ ] 渲染层（实时渲染）
- [ ] 治疗层（FSM治疗流程）

### 待开发 📋

- [ ] Web界面集成
- [ ] API服务完善
- [ ] 性能优化
- [ ] 临床验证准备
- [ ] 文档完善

## 🛠️ 开发指南

### 添加新层

1. 继承`BaseLayer`类
2. 实现`_process_impl`方法
3. 在`layers/__init__.py`中注册
4. 在`main.py`中初始化

### 配置管理

- 主配置：`configs/six_layer_architecture.yaml`
- 情绪配置：`configs/emotion_27d.yaml`
- 理论参数：`configs/theory_params.yaml`

### 性能监控

系统内置性能监控，可通过以下方式获取：

```python
# 获取系统状态
status = system.get_system_status()

# 获取层状态
layer_status = layer.get_status()

# 获取管道状态
pipeline_status = pipeline.get_pipeline_status()
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交代码
4. 创建Pull Request

## 📝 许可证

MIT License

## 👨‍💻 作者

陈万新 - 毕业设计项目
版本：3.0.0
日期：2025年1月

---

**基于qm_final2升级，实现Draft文档中的先进架构设计**