#!/usr/bin/env python3
"""
理论基础模块

包含系统的核心理论实现，包括：
- ISO原则：音乐治疗的同步-引导-巩固三阶段理论
- 音乐心理学：音乐参数与情绪状态的映射理论
- 睡眠生理学：睡眠过程的生理和心理机制
- 效价唤醒模型：情绪的二维空间表示理论

这些理论为整个系统提供科学基础，确保治疗的有效性和安全性。
"""

# 导入各理论模块
from .iso_principle import ISOPrinciple, ISOStage
from .music_psychology import MusicPsychologyModel, MusicalCharacteristics, PsychoacousticProfile
from .sleep_physiology import SleepPhysiologyModel, SleepStage, PhysiologicalState
from .valence_arousal import ValenceArousalModel, EmotionMapping, MultimodalEmotion

__all__ = [
    # ISO原则
    'ISOPrinciple', 'ISOStage',
    
    # 音乐心理学
    'MusicPsychologyModel', 'MusicalCharacteristics', 'PsychoacousticProfile',
    
    # 睡眠生理学
    'SleepPhysiologyModel', 'SleepStage', 'PhysiologicalState',
    
    # 效价唤醒模型
    'ValenceArousalModel', 'EmotionMapping', 'MultimodalEmotion'
]