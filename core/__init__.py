#!/usr/bin/env python3
"""
核心组件模块

包含系统的核心理论基础、模型管理、工具函数等基础组件。
这些组件为六层架构提供支撑，确保系统的理论正确性和技术可靠性。
"""

# 导入理论基础模块
from .theory import *
from .utils import *

# 导入模型模块
try:
    from .models import *
except ImportError as e:
    import logging
    logging.warning(f"模型模块导入失败: {e}")

__all__ = [
    # 理论基础
    'ISOPrinciple', 'MusicPsychology', 'SleepPhysiology', 'ValenceArousal',
    
    # 模型类
    'ModelFactory', 'EmotionVector', 'MusicParameters', 'ProcessingResult',
    'SystemConfig', 'TherapySession', 'UserProfile', 'PerformanceMetrics',
    
    # 工具函数
    'ConfigLoader', 'setup_logging', 'get_project_root'
]