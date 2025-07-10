#!/usr/bin/env python3
"""
核心数据模型和类型定义

提供系统中使用的核心数据结构、模型类和类型定义。
包括配置模型、数据传输对象、状态管理等。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# =====================================================
# 基础枚举类型
# =====================================================

class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    SENSOR = "sensor"

class EmotionCategory(Enum):
    """情绪类别枚举"""
    # 基础情绪 (9种)
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"
    
    # 睡眠特化情绪 (18种)
    RESTLESSNESS = "restlessness"
    DROWSINESS = "drowsiness"
    INSOMNIA_ANXIETY = "insomnia_anxiety"
    SLEEP_FRUSTRATION = "sleep_frustration"
    BEDTIME_WORRY = "bedtime_worry"
    SLEEP_DEPRESSION = "sleep_depression"
    NIGHTMARE_FEAR = "nightmare_fear"
    SLEEP_ANTICIPATION = "sleep_anticipation"
    MORNING_ANXIETY = "morning_anxiety"
    SLEEP_GUILT = "sleep_guilt"
    FATIGUE_IRRITATION = "fatigue_irritation"
    SLEEP_LONELINESS = "sleep_loneliness"
    COMFORT_SEEKING = "comfort_seeking"
    SLEEP_CONTENTMENT = "sleep_contentment"
    DEEP_RELAXATION = "deep_relaxation"
    SLEEP_GRATITUDE = "sleep_gratitude"
    PEACEFUL_DROWSINESS = "peaceful_drowsiness"
    SLEEP_SECURITY = "sleep_security"

class TherapyStage(Enum):
    """治疗阶段枚举"""
    ASSESSMENT = "assessment"
    SYNCHRONIZATION = "synchronization"
    GUIDANCE = "guidance"
    CONSOLIDATION = "consolidation"
    MONITORING = "monitoring"

# =====================================================
# 配置数据模型
# =====================================================

@dataclass
class SystemConfig:
    """系统配置模型"""
    name: str = "qm_final3"
    version: str = "3.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    max_concurrent_tasks: int = 4
    enable_performance_monitoring: bool = True

@dataclass
class LayerConfigBase:
    """层配置基类"""
    layer_name: str
    enabled: bool = True
    debug_mode: bool = False
    max_processing_time: float = 100.0  # 毫秒
    use_gpu: bool = False
    batch_size: int = 1

@dataclass
class ModalityConfig:
    """模态配置"""
    enabled: bool = True
    priority: float = 1.0
    confidence_threshold: float = 0.5
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionConfig:
    """情绪识别配置"""
    total_dimensions: int = 27
    base_emotions: int = 9
    sleep_emotions: int = 18
    confidence_threshold: float = 0.7
    enable_relationships: bool = True
    fusion_strategy: str = "confidence_weighted"

@dataclass
class TherapyConfig:
    """治疗配置"""
    session_duration: float = 1800.0  # 30分钟
    sync_duration_ratio: float = 0.25
    guidance_duration_ratio: float = 0.50
    consolidation_duration_ratio: float = 0.25
    effectiveness_threshold: float = 0.7
    enable_adaptive_adjustment: bool = True

# =====================================================
# 数据传输对象
# =====================================================

@dataclass
class ProcessingResult:
    """处理结果数据模型"""
    status: ProcessingStatus
    data: Any
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModalityData:
    """模态数据模型"""
    modality_type: ModalityType
    raw_data: Any
    features: Optional[np.ndarray] = None
    confidence: float = 0.0
    preprocessing_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EmotionVector:
    """情绪向量模型"""
    emotions: np.ndarray  # 27维情绪向量
    confidence: float
    dominant_emotion: EmotionCategory
    arousal: float = 0.0  # -1到1
    valence: float = 0.0  # -1到1
    sleep_specific_score: float = 0.0  # 睡眠特化程度
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MusicParameters:
    """音乐参数模型"""
    tempo: float = 60.0  # BPM
    key: str = "C"
    scale: str = "major"
    dynamics: float = 0.5  # 0-1
    timbre: Dict[str, float] = field(default_factory=dict)
    binaural_frequency: Optional[float] = None
    spatial_params: Dict[str, Any] = field(default_factory=dict)
    therapy_stage: TherapyStage = TherapyStage.SYNCHRONIZATION

@dataclass
class ContentGenerationRequest:
    """内容生成请求模型"""
    emotion_vector: EmotionVector
    music_parameters: MusicParameters
    duration: float = 30.0  # 秒
    content_type: str = "both"  # audio, video, both
    quality_level: str = "medium"
    therapy_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeneratedContent:
    """生成内容模型"""
    audio_data: Optional[np.ndarray] = None
    video_data: Optional[np.ndarray] = None
    audio_metadata: Dict[str, Any] = field(default_factory=dict)
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0
    quality_score: float = 0.0
    therapy_alignment: float = 0.0

@dataclass
class RenderingState:
    """渲染状态模型"""
    is_rendering: bool = False
    audio_position: float = 0.0
    video_position: float = 0.0
    sync_drift_ms: float = 0.0
    buffer_health: float = 1.0
    quality_level: str = "medium"
    frame_rate: float = 30.0
    sample_rate: int = 44100

@dataclass
class TherapySession:
    """治疗会话模型"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    current_stage: TherapyStage = TherapyStage.ASSESSMENT
    emotion_history: List[EmotionVector] = field(default_factory=list)
    effectiveness_scores: List[float] = field(default_factory=list)
    therapy_notes: Dict[str, Any] = field(default_factory=dict)
    session_config: TherapyConfig = field(default_factory=TherapyConfig)

@dataclass
class UserProfile:
    """用户档案模型"""
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    sleep_patterns: Dict[str, Any] = field(default_factory=dict)
    emotion_preferences: Dict[str, float] = field(default_factory=dict)
    therapy_history: List[str] = field(default_factory=list)  # session_ids
    effectiveness_trends: Dict[str, List[float]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

# =====================================================
# 性能监控模型
# =====================================================

@dataclass
class PerformanceMetrics:
    """性能指标模型"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: Optional[float] = None
    processing_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LayerPerformance:
    """层性能模型"""
    layer_name: str
    total_processed: int = 0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    error_count: int = 0
    success_rate: float = 1.0
    recent_processing_times: List[float] = field(default_factory=list)

# =====================================================
# 状态管理模型
# =====================================================

@dataclass
class SystemState:
    """系统状态模型"""
    is_running: bool = False
    current_sessions: int = 0
    total_processed: int = 0
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    layer_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# =====================================================
# 工厂函数和工具类
# =====================================================

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_emotion_vector(emotions: Union[List[float], np.ndarray], 
                            confidence: float = 0.0) -> EmotionVector:
        """创建情绪向量"""
        if isinstance(emotions, list):
            emotions = np.array(emotions)
        
        if len(emotions) != 27:
            raise ValueError(f"情绪向量必须是27维，当前维度: {len(emotions)}")
        
        # 找到主导情绪
        dominant_idx = np.argmax(emotions)
        emotion_categories = list(EmotionCategory)
        dominant_emotion = emotion_categories[dominant_idx] if dominant_idx < len(emotion_categories) else EmotionCategory.NEUTRAL
        
        # 计算效价和唤醒度 (简化计算)
        valence = float(np.mean(emotions[:4]) - np.mean(emotions[4:8]))  # 正面情绪 - 负面情绪
        arousal = float(np.std(emotions))  # 使用标准差作为唤醒度的近似
        
        # 计算睡眠特化分数
        sleep_specific_score = float(np.mean(emotions[9:27]))  # 睡眠特化情绪的平均值
        
        return EmotionVector(
            emotions=emotions,
            confidence=confidence,
            dominant_emotion=dominant_emotion,
            arousal=arousal,
            valence=valence,
            sleep_specific_score=sleep_specific_score
        )
    
    @staticmethod
    def create_music_parameters(emotion_vector: EmotionVector, 
                              therapy_stage: TherapyStage = TherapyStage.SYNCHRONIZATION) -> MusicParameters:
        """根据情绪向量创建音乐参数"""
        # 根据效价和唤醒度调整参数
        base_tempo = 60.0
        tempo_adjustment = emotion_vector.arousal * 30.0  # 唤醒度影响节拍
        tempo = max(40.0, min(120.0, base_tempo + tempo_adjustment))
        
        # 根据效价选择调式
        scale = "major" if emotion_vector.valence > 0 else "minor"
        
        # 根据睡眠特化程度调整动态
        dynamics = max(0.1, min(0.8, 0.5 - emotion_vector.sleep_specific_score * 0.3))
        
        # 双耳节拍频率 (用于睡眠诱导)
        binaural_freq = None
        if therapy_stage in [TherapyStage.SYNCHRONIZATION, TherapyStage.GUIDANCE]:
            if emotion_vector.sleep_specific_score > 0.5:
                binaural_freq = 6.0  # Theta波段，有助于睡眠
        
        return MusicParameters(
            tempo=tempo,
            scale=scale,
            dynamics=dynamics,
            binaural_frequency=binaural_freq,
            therapy_stage=therapy_stage
        )

# =====================================================
# 验证函数
# =====================================================

def validate_emotion_vector(emotions: np.ndarray) -> bool:
    """验证情绪向量的有效性"""
    if not isinstance(emotions, np.ndarray):
        return False
    if emotions.shape != (27,):
        return False
    if not np.all(emotions >= 0) or not np.all(emotions <= 1):
        return False
    return True

def validate_music_parameters(params: MusicParameters) -> bool:
    """验证音乐参数的有效性"""
    if not (20.0 <= params.tempo <= 200.0):
        return False
    if not (0.0 <= params.dynamics <= 1.0):
        return False
    if params.binaural_frequency and not (0.5 <= params.binaural_frequency <= 30.0):
        return False
    return True

# =====================================================
# 常量定义
# =====================================================

# 27维情绪标签
EMOTION_LABELS = [
    # 基础情绪 (0-8)
    "joy", "sadness", "anger", "fear", "disgust", 
    "surprise", "trust", "anticipation", "neutral",
    # 睡眠特化情绪 (9-26)
    "restlessness", "drowsiness", "insomnia_anxiety", "sleep_frustration",
    "bedtime_worry", "sleep_depression", "nightmare_fear", "sleep_anticipation",
    "morning_anxiety", "sleep_guilt", "fatigue_irritation", "sleep_loneliness",
    "comfort_seeking", "sleep_contentment", "deep_relaxation", "sleep_gratitude",
    "peaceful_drowsiness", "sleep_security"
]

# 默认配置
DEFAULT_SYSTEM_CONFIG = SystemConfig()
DEFAULT_EMOTION_CONFIG = EmotionConfig()
DEFAULT_THERAPY_CONFIG = TherapyConfig()

logger.info("核心模型模块初始化完成")