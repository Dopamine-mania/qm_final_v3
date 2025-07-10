#!/usr/bin/env python3
"""
核心组件模块

包含系统的核心理论基础、模型管理、工具函数等基础组件。
这些组件为六层架构提供支撑，确保系统的理论正确性和技术可靠性。
"""

# 导入理论基础模块
from .theory import *
from .models import *
from .utils import *

__all__ = [
    # 理论基础
    'ISOPrinciple', 'MusicPsychology', 'SleepPhysiology', 'ValenceArousal',
    
    # 模型管理
    'ModelFactory', 'ModelRegistry', 'ModelAdapter',
    
    # 工具函数
    'ConfigLoader', 'DataValidator', 'PerformanceMonitor'
]