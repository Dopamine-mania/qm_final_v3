#!/usr/bin/env python3
"""
qm_final3 - 基于情绪驱动的细粒度三阶段音乐治疗叙事系统

六层架构实现：
1. Input Layer - 多模态数据采集
2. Fusion Layer - 多模态情绪识别融合
3. Mapping Layer - KG-MLP混合映射
4. Generation Layer - 音视频生成
5. Rendering Layer - 实时渲染优化
6. Therapy Layer - FSM治疗流程

基于qm_final2升级，实现draft文档中的先进架构设计。

作者：陈万新
版本：3.0.0
日期：2025年1月
"""

__version__ = "3.0.0"
__author__ = "陈万新"
__description__ = "基于情绪驱动的细粒度三阶段音乐治疗叙事系统"

# 导入核心组件
from .layers import *
from .core import *

# 系统配置
SYSTEM_NAME = "心境流转 qm_final3"
ARCHITECTURE_TYPE = "六层模块化架构"
TARGET_LATENCY = 500  # ms
EMOTION_DIMENSIONS = 27