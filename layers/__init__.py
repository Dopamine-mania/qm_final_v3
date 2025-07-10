#!/usr/bin/env python3
"""
六层架构核心模块

层级架构设计：
1. Input Layer (输入层) - 多模态数据采集和预处理
2. Fusion Layer (融合层) - 27维情绪识别与多模态融合
3. Mapping Layer (映射层) - KG-MLP混合情绪到音乐参数映射
4. Generation Layer (生成层) - 实时音视频内容生成
5. Rendering Layer (渲染层) - 低延迟同步渲染
6. Therapy Layer (治疗层) - FSM驱动的三阶段治疗流程

每层都实现标准化的接口，支持层间数据流和状态管理。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# 导入各层模块
from .base_layer import BaseLayer, LayerInterface, LayerData, LayerConfig, LayerPipeline
from .input_layer import InputLayer, InputLayerConfig
from .fusion_layer import FusionLayer, FusionLayerConfig
from .mapping_layer import MappingLayer, MappingLayerConfig

# 导入已实现的层
from .generation_layer import GenerationLayer, GenerationLayerConfig
from .rendering_layer import RenderingLayer, RenderingLayerConfig
from .therapy_layer import TherapyLayer, TherapyLayerConfig

__all__ = [
    # 基础组件
    'BaseLayer', 'LayerInterface', 'LayerData', 'LayerConfig', 'LayerPipeline',
    
    # 已实现的层
    'InputLayer', 'InputLayerConfig',
    'FusionLayer', 'FusionLayerConfig',
    'MappingLayer', 'MappingLayerConfig',
    'GenerationLayer', 'GenerationLayerConfig',
    'RenderingLayer', 'RenderingLayerConfig',
    'TherapyLayer', 'TherapyLayerConfig'
]