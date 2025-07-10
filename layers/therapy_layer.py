#!/usr/bin/env python3
"""
治疗层 (Therapy Layer) - Layer 6

FSM驱动的三阶段治疗流程，核心功能包括：
1. 有限状态机驱动的治疗会话管理
2. ISO原则的三阶段治疗实施
3. 实时治疗效果监测和调整
4. 个性化治疗方案适配
5. 治疗数据记录和分析

处理流程：
Rendering Layer → Therapy Layer → 用户反馈 → 治疗调整
"""

import numpy as np
import asyncio
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import uuid

from .base_layer import BaseLayer, LayerData, LayerConfig
from core.utils import (
    ConfigLoader, DataValidator, PerformanceMonitor, 
    get_project_root
)
from core.theory.iso_principle import ISOPrinciple, ISOStage, EmotionState

logger = logging.getLogger(__name__)

class TherapyState(Enum):
    """治疗状态枚举"""
    IDLE = "idle"                           # 空闲状态
    ASSESSMENT = "assessment"               # 初始评估
    PREPARATION = "preparation"             # 治疗准备
    SYNCHRONIZATION = "synchronization"     # 同频阶段
    GUIDANCE = "guidance"                   # 引导阶段
    CONSOLIDATION = "consolidation"         # 巩固阶段
    MONITORING = "monitoring"               # 效果监测
    ADJUSTMENT = "adjustment"               # 治疗调整
    COMPLETION = "completion"               # 治疗完成
    ERROR = "error"                         # 错误状态

class TherapyEvent(Enum):
    """治疗事件枚举"""
    START_SESSION = "start_session"
    ASSESSMENT_COMPLETE = "assessment_complete"
    PREPARATION_READY = "preparation_ready"
    STAGE_TRANSITION = "stage_transition"
    STAGE_COMPLETE = "stage_complete"
    EFFECTIVENESS_CHECK = "effectiveness_check"
    ADJUSTMENT_NEEDED = "adjustment_needed"
    SESSION_COMPLETE = "session_complete"
    ERROR_OCCURRED = "error_occurred"
    USER_FEEDBACK = "user_feedback"
    EMERGENCY_STOP = "emergency_stop"

class InterventionType(Enum):
    """干预类型枚举"""
    MUSIC_THERAPY = "music_therapy"
    VISUAL_THERAPY = "visual_therapy"
    BREATHING_GUIDANCE = "breathing_guidance"
    PROGRESSIVE_RELAXATION = "progressive_relaxation"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    MINDFULNESS = "mindfulness"

@dataclass
class TherapyLayerConfig(LayerConfig):
    """治疗层配置"""
    # 会话配置
    default_session_duration: float = 1200.0  # 20分钟
    max_session_duration: float = 3600.0      # 最大1小时
    min_session_duration: float = 300.0       # 最小5分钟
    
    # ISO阶段配置
    synchronization_duration_ratio: float = 0.25
    guidance_duration_ratio: float = 0.50
    consolidation_duration_ratio: float = 0.25
    
    # 监测配置
    effectiveness_check_interval: float = 60.0  # 每分钟检查
    feedback_collection_interval: float = 120.0  # 每2分钟收集反馈
    
    # 自适应配置
    enable_adaptive_adjustment: bool = True
    adjustment_sensitivity: float = 0.7
    max_adjustments_per_session: int = 3
    
    # 安全配置
    enable_emergency_protocols: bool = True
    stress_threshold: float = 0.8
    effectiveness_threshold: float = 0.4
    
    # 数据记录
    enable_session_recording: bool = True
    enable_effectiveness_tracking: bool = True
    
    # 性能配置
    max_processing_time: float = 50.0  # ms

@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    sleep_issues: List[str] = field(default_factory=list)
    therapy_history: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    baseline_emotions: Dict[str, float] = field(default_factory=dict)
    effectiveness_scores: List[float] = field(default_factory=list)

@dataclass
class TherapySession:
    """治疗会话"""
    session_id: str
    user_id: str
    start_time: datetime
    planned_duration: float
    current_state: TherapyState
    current_stage: Optional[ISOStage] = None
    
    # 情绪状态跟踪
    initial_emotion: Optional[EmotionState] = None
    current_emotion: Optional[EmotionState] = None
    target_emotion: Optional[EmotionState] = None
    
    # 治疗进度
    progress_percentage: float = 0.0
    stage_start_time: Optional[datetime] = None
    
    # 效果指标
    effectiveness_score: float = 0.0
    user_satisfaction: float = 0.0
    stress_level: float = 0.0
    
    # 治疗记录
    interventions_applied: List[str] = field(default_factory=list)
    adjustments_made: List[str] = field(default_factory=list)
    feedback_collected: List[Dict] = field(default_factory=list)
    
    # 会话元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

class TherapyFSM:
    """治疗有限状态机"""
    
    def __init__(self, config: TherapyLayerConfig):
        self.config = config
        self.current_state = TherapyState.IDLE
        self.session: Optional[TherapySession] = None
        
        # 状态转换表
        self.state_transitions = {
            TherapyState.IDLE: {
                TherapyEvent.START_SESSION: TherapyState.ASSESSMENT
            },
            TherapyState.ASSESSMENT: {
                TherapyEvent.ASSESSMENT_COMPLETE: TherapyState.PREPARATION,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.PREPARATION: {
                TherapyEvent.PREPARATION_READY: TherapyState.SYNCHRONIZATION,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.SYNCHRONIZATION: {
                TherapyEvent.STAGE_COMPLETE: TherapyState.GUIDANCE,
                TherapyEvent.ADJUSTMENT_NEEDED: TherapyState.ADJUSTMENT,
                TherapyEvent.EMERGENCY_STOP: TherapyState.COMPLETION,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.GUIDANCE: {
                TherapyEvent.STAGE_COMPLETE: TherapyState.CONSOLIDATION,
                TherapyEvent.ADJUSTMENT_NEEDED: TherapyState.ADJUSTMENT,
                TherapyEvent.EMERGENCY_STOP: TherapyState.COMPLETION,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.CONSOLIDATION: {
                TherapyEvent.STAGE_COMPLETE: TherapyState.MONITORING,
                TherapyEvent.ADJUSTMENT_NEEDED: TherapyState.ADJUSTMENT,
                TherapyEvent.EMERGENCY_STOP: TherapyState.COMPLETION,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.MONITORING: {
                TherapyEvent.EFFECTIVENESS_CHECK: TherapyState.MONITORING,
                TherapyEvent.SESSION_COMPLETE: TherapyState.COMPLETION,
                TherapyEvent.ADJUSTMENT_NEEDED: TherapyState.ADJUSTMENT,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.ADJUSTMENT: {
                TherapyEvent.STAGE_TRANSITION: TherapyState.SYNCHRONIZATION,
                TherapyEvent.SESSION_COMPLETE: TherapyState.COMPLETION,
                TherapyEvent.ERROR_OCCURRED: TherapyState.ERROR
            },
            TherapyState.COMPLETION: {
                TherapyEvent.START_SESSION: TherapyState.ASSESSMENT
            },
            TherapyState.ERROR: {
                TherapyEvent.START_SESSION: TherapyState.ASSESSMENT
            }
        }
        
        # 状态处理函数
        self.state_handlers = {
            TherapyState.ASSESSMENT: self._handle_assessment,
            TherapyState.PREPARATION: self._handle_preparation,
            TherapyState.SYNCHRONIZATION: self._handle_synchronization,
            TherapyState.GUIDANCE: self._handle_guidance,
            TherapyState.CONSOLIDATION: self._handle_consolidation,
            TherapyState.MONITORING: self._handle_monitoring,
            TherapyState.ADJUSTMENT: self._handle_adjustment,
            TherapyState.COMPLETION: self._handle_completion
        }
        
        logger.info("治疗FSM初始化完成")
    
    def trigger_event(self, event: TherapyEvent, **kwargs) -> bool:
        """触发事件，执行状态转换"""
        current_state = self.current_state
        
        # 检查是否允许该事件
        if event not in self.state_transitions.get(current_state, {}):
            logger.warning(f"事件 {event.value} 在状态 {current_state.value} 下不被允许")
            return False
        
        # 执行状态转换
        new_state = self.state_transitions[current_state][event]
        
        logger.info(f"状态转换: {current_state.value} -> {new_state.value} (事件: {event.value})")
        
        self.current_state = new_state
        
        # 更新会话状态
        if self.session:
            self.session.current_state = new_state
        
        # 调用状态处理函数
        if new_state in self.state_handlers:
            return self.state_handlers[new_state](**kwargs)
        
        return True
    
    def _handle_assessment(self, **kwargs) -> bool:
        """处理初始评估状态"""
        if not self.session:
            return False
        
        # 分析用户当前情绪状态
        current_emotion_data = kwargs.get('current_emotion', {})
        self.session.initial_emotion = EmotionState(
            valence=current_emotion_data.get('valence', 0.0),
            arousal=current_emotion_data.get('arousal', 0.0),
            confidence=current_emotion_data.get('confidence', 1.0)
        )
        self.session.current_emotion = self.session.initial_emotion
        
        # 设置目标情绪状态（适合睡眠的状态）
        self.session.target_emotion = EmotionState(
            valence=0.2,   # 轻微积极
            arousal=-0.6,  # 低唤醒
            confidence=0.9
        )
        
        # 评估完成，准备进入下一阶段
        logger.info(f"评估完成 - 初始情绪: V={self.session.initial_emotion.valence:.2f}, "
                   f"A={self.session.initial_emotion.arousal:.2f}")
        
        return True
    
    def _handle_preparation(self, **kwargs) -> bool:
        """处理治疗准备状态"""
        if not self.session:
            return False
        
        # 计算各阶段持续时间
        total_duration = self.session.planned_duration
        self.session.metadata['stage_durations'] = {
            'synchronization': total_duration * self.config.synchronization_duration_ratio,
            'guidance': total_duration * self.config.guidance_duration_ratio,
            'consolidation': total_duration * self.config.consolidation_duration_ratio
        }
        
        # 选择治疗干预方案
        interventions = self._select_interventions()
        self.session.interventions_applied = interventions
        
        logger.info(f"治疗准备完成 - 总时长: {total_duration}s, 干预方案: {interventions}")
        
        return True
    
    def _handle_synchronization(self, **kwargs) -> bool:
        """处理同频阶段"""
        if not self.session:
            return False
        
        self.session.current_stage = ISOStage.SYNCHRONIZATION
        self.session.stage_start_time = datetime.now()
        
        # 更新进度
        self.session.progress_percentage = 10.0  # 开始阶段
        
        logger.info("同频阶段开始 - 匹配用户当前情绪状态")
        
        return True
    
    def _handle_guidance(self, **kwargs) -> bool:
        """处理引导阶段"""
        if not self.session:
            return False
        
        self.session.current_stage = ISOStage.GUIDANCE
        self.session.stage_start_time = datetime.now()
        
        # 更新进度
        self.session.progress_percentage = 40.0  # 主要治疗阶段
        
        logger.info("引导阶段开始 - 逐步引导情绪过渡")
        
        return True
    
    def _handle_consolidation(self, **kwargs) -> bool:
        """处理巩固阶段"""
        if not self.session:
            return False
        
        self.session.current_stage = ISOStage.CONSOLIDATION
        self.session.stage_start_time = datetime.now()
        
        # 更新进度
        self.session.progress_percentage = 70.0  # 接近完成
        
        logger.info("巩固阶段开始 - 稳定在目标情绪状态")
        
        return True
    
    def _handle_monitoring(self, **kwargs) -> bool:
        """处理效果监测状态"""
        if not self.session:
            return False
        
        # 更新进度
        self.session.progress_percentage = 90.0  # 即将完成
        
        # 计算治疗效果
        effectiveness = self._calculate_effectiveness()
        self.session.effectiveness_score = effectiveness
        
        logger.info(f"效果监测 - 治疗效果: {effectiveness:.2f}")
        
        return True
    
    def _handle_adjustment(self, **kwargs) -> bool:
        """处理治疗调整状态"""
        if not self.session:
            return False
        
        adjustment_reason = kwargs.get('reason', '未知原因')
        
        # 记录调整
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'reason': adjustment_reason,
            'stage': self.session.current_stage.value if self.session.current_stage else None,
            'effectiveness_before': self.session.effectiveness_score
        }
        
        self.session.adjustments_made.append(adjustment_record)
        
        logger.info(f"治疗调整 - 原因: {adjustment_reason}")
        
        return True
    
    def _handle_completion(self, **kwargs) -> bool:
        """处理治疗完成状态"""
        if not self.session:
            return False
        
        # 更新进度
        self.session.progress_percentage = 100.0
        
        # 计算最终效果评分
        final_effectiveness = self._calculate_effectiveness()
        self.session.effectiveness_score = final_effectiveness
        
        # 记录会话结束时间
        self.session.metadata['end_time'] = datetime.now().isoformat()
        
        logger.info(f"治疗会话完成 - 最终效果: {final_effectiveness:.2f}")
        
        return True
    
    def _select_interventions(self) -> List[str]:
        """选择适合的治疗干预方案"""
        interventions = [InterventionType.MUSIC_THERAPY.value]
        
        if self.session and self.session.initial_emotion:
            # 根据初始情绪状态选择干预
            if self.session.initial_emotion.arousal > 0.5:
                # 高唤醒，添加放松技术
                interventions.append(InterventionType.PROGRESSIVE_RELAXATION.value)
                interventions.append(InterventionType.BREATHING_GUIDANCE.value)
            
            if self.session.initial_emotion.valence < -0.3:
                # 负面情绪，添加认知重构
                interventions.append(InterventionType.COGNITIVE_RESTRUCTURING.value)
            
            # 总是包含视觉治疗和正念
            interventions.append(InterventionType.VISUAL_THERAPY.value)
            interventions.append(InterventionType.MINDFULNESS.value)
        
        return interventions
    
    def _calculate_effectiveness(self) -> float:
        """计算治疗效果"""
        if not self.session or not self.session.initial_emotion or not self.session.current_emotion:
            return 0.0
        
        # 计算情绪改善程度
        initial_distance = self.session.initial_emotion.distance_to(self.session.target_emotion)
        current_distance = self.session.current_emotion.distance_to(self.session.target_emotion)
        
        # 效果 = (初始距离 - 当前距离) / 初始距离
        if initial_distance > 0:
            improvement = (initial_distance - current_distance) / initial_distance
            effectiveness = max(0.0, min(1.0, improvement))
        else:
            effectiveness = 1.0  # 已经在目标状态
        
        return effectiveness

class TherapyMonitor:
    """治疗监测器"""
    
    def __init__(self, config: TherapyLayerConfig):
        self.config = config
        self.monitoring_active = False
        self.last_check_time = None
        self.feedback_history = []
        
        logger.info("治疗监测器初始化完成")
    
    def start_monitoring(self, session: TherapySession):
        """开始监测"""
        self.monitoring_active = True
        self.last_check_time = time.time()
        logger.info(f"开始监测会话: {session.session_id}")
    
    def stop_monitoring(self):
        """停止监测"""
        self.monitoring_active = False
        logger.info("停止监测")
    
    def check_effectiveness(self, session: TherapySession) -> Dict[str, Any]:
        """检查治疗效果"""
        current_time = time.time()
        
        # 检查是否需要效果评估
        if (self.last_check_time and 
            current_time - self.last_check_time < self.config.effectiveness_check_interval):
            return {'needs_check': False}
        
        self.last_check_time = current_time
        
        # 评估当前效果
        effectiveness = session.effectiveness_score
        stress_level = session.stress_level
        
        # 判断是否需要调整
        needs_adjustment = (
            effectiveness < self.config.effectiveness_threshold or
            stress_level > self.config.stress_threshold
        )
        
        return {
            'needs_check': True,
            'effectiveness': effectiveness,
            'stress_level': stress_level,
            'needs_adjustment': needs_adjustment,
            'recommendation': self._get_adjustment_recommendation(session)
        }
    
    def collect_feedback(self, session: TherapySession, feedback_data: Dict[str, Any]):
        """收集用户反馈"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session.session_id,
            'feedback': feedback_data,
            'stage': session.current_stage.value if session.current_stage else None
        }
        
        self.feedback_history.append(feedback_entry)
        session.feedback_collected.append(feedback_entry)
        
        # 更新用户满意度
        if 'satisfaction' in feedback_data:
            session.user_satisfaction = feedback_data['satisfaction']
        
        logger.info(f"收集用户反馈: {feedback_data}")
    
    def _get_adjustment_recommendation(self, session: TherapySession) -> str:
        """获取调整建议"""
        if session.effectiveness_score < 0.3:
            return "效果不佳，建议调整治疗策略"
        elif session.stress_level > 0.8:
            return "压力水平过高，建议放缓治疗节奏"
        elif session.effectiveness_score > 0.8:
            return "效果良好，继续当前治疗方案"
        else:
            return "效果一般，可考虑微调参数"

class TherapyDataRecorder:
    """治疗数据记录器"""
    
    def __init__(self, config: TherapyLayerConfig):
        self.config = config
        self.records = []
        self.session_data = {}
        
        # 创建数据目录
        self.data_dir = get_project_root() / "data" / "therapy_sessions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"治疗数据记录器初始化完成 - 数据目录: {self.data_dir}")
    
    def start_recording(self, session: TherapySession):
        """开始记录会话"""
        if not self.config.enable_session_recording:
            return
        
        self.session_data[session.session_id] = {
            'session_info': session,
            'timeline': [],
            'metrics': []
        }
        
        logger.info(f"开始记录会话: {session.session_id}")
    
    def record_event(self, session_id: str, event_type: str, data: Dict[str, Any]):
        """记录事件"""
        if session_id not in self.session_data:
            return
        
        event_record = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        self.session_data[session_id]['timeline'].append(event_record)
    
    def record_metrics(self, session_id: str, metrics: Dict[str, float]):
        """记录指标"""
        if session_id not in self.session_data:
            return
        
        metric_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        self.session_data[session_id]['metrics'].append(metric_record)
    
    def save_session(self, session: TherapySession):
        """保存会话数据"""
        if not self.config.enable_session_recording:
            return
        
        try:
            # 转换会话数据为可序列化格式
            session_dict = {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'start_time': session.start_time.isoformat(),
                'planned_duration': session.planned_duration,
                'final_state': session.current_state.value,
                'progress_percentage': session.progress_percentage,
                'effectiveness_score': session.effectiveness_score,
                'user_satisfaction': session.user_satisfaction,
                'stress_level': session.stress_level,
                'interventions_applied': session.interventions_applied,
                'adjustments_made': session.adjustments_made,
                'feedback_collected': session.feedback_collected,
                'metadata': session.metadata
            }
            
            # 添加时间线和指标
            if session.session_id in self.session_data:
                session_dict['timeline'] = self.session_data[session.session_id]['timeline']
                session_dict['metrics'] = self.session_data[session.session_id]['metrics']
            
            # 保存到文件
            filename = f"session_{session.session_id}_{session.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"会话数据已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存会话数据失败: {e}")

class TherapyLayer(BaseLayer):
    """治疗层 - FSM驱动的三阶段治疗流程"""
    
    def __init__(self, config: TherapyLayerConfig):
        super().__init__(config)
        self.config = config
        self.layer_name = "therapy_layer"
        
        # 初始化组件
        self.fsm = TherapyFSM(config)
        self.monitor = TherapyMonitor(config)
        self.recorder = TherapyDataRecorder(config)
        self.iso_principle = ISOPrinciple()
        
        # 用户档案管理
        self.user_profiles: Dict[str, UserProfile] = {}
        self.active_sessions: Dict[str, TherapySession] = {}
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 治疗统计
        self.therapy_stats = {
            'total_sessions': 0,
            'completed_sessions': 0,
            'avg_effectiveness': 0.0,
            'avg_satisfaction': 0.0,
            'total_therapy_time': 0.0
        }
        
        logger.info("治疗层初始化完成")
    
    def start_therapy_session(self, user_id: str, duration: float = None) -> str:
        """开始治疗会话"""
        session_id = str(uuid.uuid4())
        
        # 创建会话
        session = TherapySession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            planned_duration=duration or self.config.default_session_duration,
            current_state=TherapyState.IDLE
        )
        
        # 设置FSM会话
        self.fsm.session = session
        self.active_sessions[session_id] = session
        
        # 开始记录
        self.recorder.start_recording(session)
        
        # 触发开始事件
        success = self.fsm.trigger_event(TherapyEvent.START_SESSION)
        
        if success:
            self.therapy_stats['total_sessions'] += 1
            logger.info(f"治疗会话开始: {session_id}, 用户: {user_id}")
            return session_id
        else:
            # 清理失败的会话
            del self.active_sessions[session_id]
            raise RuntimeError("治疗会话启动失败")
    
    def update_emotion_state(self, session_id: str, emotion_data: Dict[str, float]):
        """更新情绪状态"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # 更新当前情绪
        session.current_emotion = EmotionState(
            valence=emotion_data.get('valence', 0.0),
            arousal=emotion_data.get('arousal', 0.0),
            confidence=emotion_data.get('confidence', 1.0)
        )
        
        # 记录指标
        self.recorder.record_metrics(session_id, {
            'valence': session.current_emotion.valence,
            'arousal': session.current_emotion.arousal,
            'confidence': session.current_emotion.confidence,
            'effectiveness': session.effectiveness_score
        })
        
        logger.debug(f"情绪状态更新: {session_id}, V={session.current_emotion.valence:.2f}")
    
    def check_session_progress(self, session_id: str) -> Dict[str, Any]:
        """检查会话进度"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        # 检查时间进度
        elapsed_time = (datetime.now() - session.start_time).total_seconds()
        time_progress = elapsed_time / session.planned_duration
        
        # 检查是否需要阶段转换
        stage_duration = session.metadata.get('stage_durations', {})
        current_stage = session.current_stage
        
        stage_transition_needed = False
        if current_stage and session.stage_start_time:
            stage_elapsed = (datetime.now() - session.stage_start_time).total_seconds()
            stage_planned = stage_duration.get(current_stage.value, 0)
            
            if stage_elapsed >= stage_planned:
                stage_transition_needed = True
        
        # 监测治疗效果
        monitoring_result = self.monitor.check_effectiveness(session)
        
        return {
            'session_id': session_id,
            'current_state': session.current_state.value,
            'current_stage': current_stage.value if current_stage else None,
            'progress_percentage': session.progress_percentage,
            'time_progress': min(time_progress, 1.0),
            'elapsed_time': elapsed_time,
            'stage_transition_needed': stage_transition_needed,
            'monitoring_result': monitoring_result,
            'effectiveness_score': session.effectiveness_score
        }
    
    def handle_user_feedback(self, session_id: str, feedback: Dict[str, Any]):
        """处理用户反馈"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # 收集反馈
        self.monitor.collect_feedback(session, feedback)
        
        # 记录事件
        self.recorder.record_event(session_id, 'user_feedback', feedback)
        
        # 检查是否需要调整
        if feedback.get('discomfort', 0) > 7 or feedback.get('effectiveness', 10) < 3:
            self.fsm.trigger_event(TherapyEvent.ADJUSTMENT_NEEDED, 
                                 reason=f"用户反馈: 不适度{feedback.get('discomfort', 0)}")
    
    def complete_session(self, session_id: str) -> Dict[str, Any]:
        """完成治疗会话"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        # 触发完成事件
        self.fsm.trigger_event(TherapyEvent.SESSION_COMPLETE)
        
        # 停止监测
        self.monitor.stop_monitoring()
        
        # 保存会话数据
        self.recorder.save_session(session)
        
        # 更新统计
        self.therapy_stats['completed_sessions'] += 1
        self.therapy_stats['total_therapy_time'] += (datetime.now() - session.start_time).total_seconds()
        
        # 更新平均效果
        total_completed = self.therapy_stats['completed_sessions']
        current_avg_eff = self.therapy_stats['avg_effectiveness']
        self.therapy_stats['avg_effectiveness'] = (
            (current_avg_eff * (total_completed - 1) + session.effectiveness_score) / total_completed
        )
        
        # 更新平均满意度
        current_avg_sat = self.therapy_stats['avg_satisfaction']
        self.therapy_stats['avg_satisfaction'] = (
            (current_avg_sat * (total_completed - 1) + session.user_satisfaction) / total_completed
        )
        
        # 创建会话摘要
        summary = {
            'session_id': session_id,
            'duration': (datetime.now() - session.start_time).total_seconds(),
            'effectiveness_score': session.effectiveness_score,
            'user_satisfaction': session.user_satisfaction,
            'interventions_applied': session.interventions_applied,
            'adjustments_made': len(session.adjustments_made),
            'final_state': session.current_state.value
        }
        
        # 从活跃会话中移除
        del self.active_sessions[session_id]
        
        logger.info(f"治疗会话完成: {session_id}, 效果: {session.effectiveness_score:.2f}")
        
        return summary
    
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """治疗层处理实现"""
        self.performance_monitor.start_timer("therapy_layer_processing")
        
        try:
            # 验证输入数据
            if not input_data.data:
                raise ValueError("输入数据为空")
            
            # 提取渲染结果
            rendering_result = input_data.data.get('rendering_result', {})
            
            # 获取或创建会话ID
            session_id = input_data.metadata.get('session_id')
            if not session_id and self.active_sessions:
                # 使用第一个活跃会话
                session_id = list(self.active_sessions.keys())[0]
            
            therapy_response = {'message': '治疗层就绪，等待会话开始'}
            
            # 如果有活跃会话，处理治疗逻辑
            if session_id and session_id in self.active_sessions:
                # 检查会话进度
                progress = self.check_session_progress(session_id)
                
                # 处理阶段转换
                if progress.get('stage_transition_needed'):
                    current_stage = self.active_sessions[session_id].current_stage
                    if current_stage == ISOStage.SYNCHRONIZATION:
                        self.fsm.trigger_event(TherapyEvent.STAGE_COMPLETE)
                    elif current_stage == ISOStage.GUIDANCE:
                        self.fsm.trigger_event(TherapyEvent.STAGE_COMPLETE)
                    elif current_stage == ISOStage.CONSOLIDATION:
                        self.fsm.trigger_event(TherapyEvent.STAGE_COMPLETE)
                
                # 处理效果监测
                monitoring = progress.get('monitoring_result', {})
                if monitoring.get('needs_adjustment'):
                    self.fsm.trigger_event(TherapyEvent.ADJUSTMENT_NEEDED,
                                         reason=monitoring.get('recommendation'))
                
                therapy_response = {
                    'session_progress': progress,
                    'therapy_guidance': self._generate_therapy_guidance(session_id),
                    'intervention_active': rendering_result.get('audio_rendered', False) or 
                                         rendering_result.get('video_rendered', False)
                }
            
            # 创建输出数据
            output_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'therapy_response': therapy_response,
                    'therapy_stats': self.therapy_stats.copy(),
                    'active_sessions': list(self.active_sessions.keys()),
                    'fsm_state': self.fsm.current_state.value
                },
                metadata={
                    'source_layer': input_data.layer_name,
                    'session_id': session_id,
                    'therapy_active': len(self.active_sessions) > 0
                },
                confidence=input_data.confidence
            )
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer("therapy_layer_processing")
            output_data.processing_time = processing_time
            
            # 更新统计信息
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            logger.info(f"治疗层处理完成 - 活跃会话: {len(self.active_sessions)}, "
                       f"FSM状态: {self.fsm.current_state.value}, 耗时: {processing_time*1000:.1f}ms")
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"治疗层处理失败: {e}")
            
            # 创建错误输出
            error_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'error': str(e),
                    'therapy_response': {'message': '治疗层暂时不可用'}
                },
                metadata={'error': True, 'source_layer': input_data.layer_name},
                confidence=0.0
            )
            
            processing_time = self.performance_monitor.end_timer("therapy_layer_processing")
            error_data.processing_time = processing_time
            
            return error_data
    
    def _generate_therapy_guidance(self, session_id: str) -> Dict[str, Any]:
        """生成治疗指导"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {}
        
        current_stage = session.current_stage
        guidance = {
            'stage': current_stage.value if current_stage else 'preparation',
            'instructions': '',
            'focus_areas': [],
            'expected_duration': 0
        }
        
        if current_stage == ISOStage.SYNCHRONIZATION:
            guidance.update({
                'instructions': '专注于当前的感受，让音乐与您的情绪状态同步',
                'focus_areas': ['情绪识别', '当下觉察', '音乐感受'],
                'expected_duration': session.metadata.get('stage_durations', {}).get('synchronization', 300)
            })
        elif current_stage == ISOStage.GUIDANCE:
            guidance.update({
                'instructions': '跟随音乐的引导，允许您的情绪自然地转变',
                'focus_areas': ['情绪转换', '深度放松', '内心平静'],
                'expected_duration': session.metadata.get('stage_durations', {}).get('guidance', 600)
            })
        elif current_stage == ISOStage.CONSOLIDATION:
            guidance.update({
                'instructions': '保持这种平静的状态，为良好的睡眠做准备',
                'focus_areas': ['状态维持', '睡前准备', '身心放松'],
                'expected_duration': session.metadata.get('stage_durations', {}).get('consolidation', 300)
            })
        else:
            guidance.update({
                'instructions': '准备开始治疗会话，请找一个舒适的位置',
                'focus_areas': ['环境准备', '身心调整'],
                'expected_duration': 60
            })
        
        return guidance
    
    def get_therapy_stats(self) -> Dict[str, Any]:
        """获取治疗统计信息"""
        return {
            'therapy_stats': self.therapy_stats.copy(),
            'active_sessions_count': len(self.active_sessions),
            'fsm_state': self.fsm.current_state.value,
            'user_profiles_count': len(self.user_profiles)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取治疗层状态"""
        base_status = super().get_status()
        
        # 添加治疗层特有的状态信息
        therapy_status = {
            'fsm_state': self.fsm.current_state.value,
            'active_sessions': len(self.active_sessions),
            'therapy_stats': self.get_therapy_stats(),
            'iso_principle_enabled': True,
            'monitoring_active': self.monitor.monitoring_active,
            'recording_enabled': self.config.enable_session_recording,
            'performance_stats': self.performance_monitor.get_all_stats()
        }
        
        base_status.update(therapy_status)
        return base_status
    
    def shutdown(self):
        """关闭治疗层"""
        # 完成所有活跃会话
        for session_id in list(self.active_sessions.keys()):
            try:
                self.complete_session(session_id)
            except Exception as e:
                logger.error(f"关闭会话失败 {session_id}: {e}")
        
        # 停止监测
        self.monitor.stop_monitoring()
        
        logger.info("治疗层已关闭")