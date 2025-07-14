#!/usr/bin/env python3
"""
治疗效果评估和反馈机制
实现睡眠治疗的效果评估、用户反馈收集和自适应优化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TherapyStage(Enum):
    """治疗阶段"""
    MATCH = "match_stage"
    GUIDE = "guide_stage" 
    TARGET = "target_stage"

class SleepQuality(Enum):
    """睡眠质量等级"""
    VERY_POOR = 1
    POOR = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5

@dataclass
class TherapySession:
    """治疗会话数据"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    initial_emotion: str
    target_emotion: str
    
    # ISO三阶段数据
    stages: Dict[str, Dict[str, Any]]
    
    # 生理指标（如果可用）
    heart_rate_data: Optional[List[float]] = None
    breathing_rate_data: Optional[List[float]] = None
    movement_data: Optional[List[float]] = None
    
    # 用户反馈
    subjective_rating: Optional[int] = None  # 1-10主观评分
    sleep_quality: Optional[SleepQuality] = None
    sleep_latency: Optional[float] = None  # 入睡时间（分钟）
    total_sleep_time: Optional[float] = None  # 总睡眠时间（小时）
    
    # 治疗效果指标
    therapy_effectiveness: Optional[float] = None
    stage_effectiveness: Optional[Dict[str, float]] = None

@dataclass
class UserFeedback:
    """用户反馈数据"""
    session_id: str
    user_id: str
    timestamp: datetime
    
    # 治疗体验评分
    music_preference: int  # 1-10 音乐喜好度
    video_preference: int  # 1-10 视频喜好度
    overall_experience: int  # 1-10 总体体验
    
    # 治疗效果评估
    relaxation_level: int  # 1-10 放松程度
    stress_reduction: int  # 1-10 压力缓解
    sleep_readiness: int  # 1-10 睡眠准备度
    
    # 定性反馈
    liked_aspects: List[str]  # 喜欢的方面
    disliked_aspects: List[str]  # 不喜欢的方面
    suggestions: str  # 改进建议
    
    # 生理感受
    physical_comfort: int  # 1-10 身体舒适度
    mental_calm: int  # 1-10 心理平静度

class TherapyEvaluator:
    """治疗效果评估器"""
    
    def __init__(self, data_dir: str = "therapy_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 评估模型参数
        self.effectiveness_weights = {
            'physiological': 0.4,  # 生理指标权重
            'subjective': 0.3,     # 主观评分权重
            'behavioral': 0.2,     # 行为指标权重  
            'temporal': 0.1        # 时间指标权重
        }
        
        # 阶段效果权重
        self.stage_weights = {
            'match_stage': 0.3,   # 匹配阶段
            'guide_stage': 0.4,   # 引导阶段（最重要）
            'target_stage': 0.3   # 目标阶段
        }
        
        logger.info("治疗效果评估器初始化完成")
    
    def evaluate_session(self, session: TherapySession) -> Dict[str, float]:
        """评估单次治疗会话效果"""
        logger.info(f"开始评估治疗会话: {session.session_id}")
        
        evaluation_results = {
            'overall_effectiveness': 0.0,
            'stage_effectiveness': {},
            'improvement_areas': [],
            'recommendations': []
        }
        
        # 1. 生理指标评估
        physiological_score = self._evaluate_physiological_indicators(session)
        
        # 2. 主观反馈评估
        subjective_score = self._evaluate_subjective_feedback(session)
        
        # 3. 行为指标评估
        behavioral_score = self._evaluate_behavioral_indicators(session)
        
        # 4. 时间效率评估
        temporal_score = self._evaluate_temporal_efficiency(session)
        
        # 计算综合效果评分
        overall_effectiveness = (
            physiological_score * self.effectiveness_weights['physiological'] +
            subjective_score * self.effectiveness_weights['subjective'] +
            behavioral_score * self.effectiveness_weights['behavioral'] +
            temporal_score * self.effectiveness_weights['temporal']
        )
        
        evaluation_results['overall_effectiveness'] = overall_effectiveness
        
        # 5. 分阶段效果评估
        stage_effectiveness = self._evaluate_stage_effectiveness(session)
        evaluation_results['stage_effectiveness'] = stage_effectiveness
        
        # 6. 生成改进建议
        recommendations = self._generate_recommendations(session, evaluation_results)
        evaluation_results['recommendations'] = recommendations
        
        # 7. 识别改进区域
        improvement_areas = self._identify_improvement_areas(evaluation_results)
        evaluation_results['improvement_areas'] = improvement_areas
        
        logger.info(f"治疗会话评估完成 - 总体效果: {overall_effectiveness:.2f}")
        
        return evaluation_results
    
    def _evaluate_physiological_indicators(self, session: TherapySession) -> float:
        """评估生理指标"""
        if not any([session.heart_rate_data, session.breathing_rate_data, session.movement_data]):
            logger.warning("无生理指标数据，使用默认评分")
            return 0.7  # 默认中等评分
        
        physiological_scores = []
        
        # 心率变化评估
        if session.heart_rate_data:
            hr_score = self._analyze_heart_rate_trend(session.heart_rate_data)
            physiological_scores.append(hr_score)
        
        # 呼吸频率评估
        if session.breathing_rate_data:
            br_score = self._analyze_breathing_pattern(session.breathing_rate_data)
            physiological_scores.append(br_score)
        
        # 肢体活动评估
        if session.movement_data:
            movement_score = self._analyze_movement_reduction(session.movement_data)
            physiological_scores.append(movement_score)
        
        return np.mean(physiological_scores) if physiological_scores else 0.7
    
    def _analyze_heart_rate_trend(self, heart_rate_data: List[float]) -> float:
        """分析心率变化趋势"""
        if len(heart_rate_data) < 2:
            return 0.5
        
        # 计算心率下降趋势
        initial_hr = np.mean(heart_rate_data[:len(heart_rate_data)//4])  # 前25%
        final_hr = np.mean(heart_rate_data[-len(heart_rate_data)//4:])   # 后25%
        
        hr_reduction = (initial_hr - final_hr) / initial_hr
        
        # 期望心率下降5-15%
        if 0.05 <= hr_reduction <= 0.15:
            return 0.9  # 理想下降
        elif 0.0 <= hr_reduction < 0.05:
            return 0.7  # 轻微下降
        elif hr_reduction > 0.15:
            return 0.6  # 下降过多
        else:
            return 0.3  # 心率上升（不理想）
    
    def _analyze_breathing_pattern(self, breathing_data: List[float]) -> float:
        """分析呼吸模式"""
        if len(breathing_data) < 10:
            return 0.5
        
        # 计算呼吸频率变异性
        breathing_variability = np.std(breathing_data)
        mean_breathing_rate = np.mean(breathing_data)
        
        # 分析呼吸频率下降
        initial_br = np.mean(breathing_data[:len(breathing_data)//4])
        final_br = np.mean(breathing_data[-len(breathing_data)//4:])
        br_reduction = (initial_br - final_br) / initial_br
        
        # 综合评分
        variability_score = 0.8 if breathing_variability < 2.0 else 0.5  # 低变异性好
        trend_score = 0.9 if br_reduction > 0.1 else 0.6  # 呼吸频率下降好
        
        return (variability_score + trend_score) / 2
    
    def _analyze_movement_reduction(self, movement_data: List[float]) -> float:
        """分析肢体活动减少"""
        if len(movement_data) < 5:
            return 0.5
        
        # 计算活动减少趋势
        initial_movement = np.mean(movement_data[:len(movement_data)//3])
        final_movement = np.mean(movement_data[-len(movement_data)//3:])
        
        movement_reduction = (initial_movement - final_movement) / (initial_movement + 1e-6)
        
        # 活动减少60%以上为优秀
        if movement_reduction >= 0.6:
            return 0.95
        elif movement_reduction >= 0.4:
            return 0.8
        elif movement_reduction >= 0.2:
            return 0.6
        else:
            return 0.4
    
    def _evaluate_subjective_feedback(self, session: TherapySession) -> float:
        """评估主观反馈"""
        if session.subjective_rating is None:
            return 0.7  # 默认评分
        
        # 将1-10评分转换为0-1分数
        subjective_score = session.subjective_rating / 10.0
        
        # 如果有睡眠质量评估，纳入考虑
        if session.sleep_quality:
            quality_score = session.sleep_quality.value / 5.0
            subjective_score = (subjective_score + quality_score) / 2
        
        return subjective_score
    
    def _evaluate_behavioral_indicators(self, session: TherapySession) -> float:
        """评估行为指标"""
        behavioral_scores = []
        
        # 入睡时间评估
        if session.sleep_latency is not None:
            # 理想入睡时间: 15-30分钟
            if 15 <= session.sleep_latency <= 30:
                latency_score = 1.0
            elif 10 <= session.sleep_latency < 15:
                latency_score = 0.9
            elif 30 < session.sleep_latency <= 45:
                latency_score = 0.7
            elif session.sleep_latency > 60:
                latency_score = 0.3
            else:
                latency_score = 0.6
            behavioral_scores.append(latency_score)
        
        # 总睡眠时间评估
        if session.total_sleep_time is not None:
            # 理想睡眠时间: 7-9小时
            if 7 <= session.total_sleep_time <= 9:
                sleep_score = 1.0
            elif 6 <= session.total_sleep_time < 7:
                sleep_score = 0.8
            elif 5 <= session.total_sleep_time < 6:
                sleep_score = 0.6
            else:
                sleep_score = 0.4
            behavioral_scores.append(sleep_score)
        
        return np.mean(behavioral_scores) if behavioral_scores else 0.7
    
    def _evaluate_temporal_efficiency(self, session: TherapySession) -> float:
        """评估时间效率"""
        if not session.end_time:
            return 0.7
        
        # 计算治疗总时长
        treatment_duration = (session.end_time - session.start_time).total_seconds() / 60  # 分钟
        
        # 理想治疗时长: 15-20分钟
        if 15 <= treatment_duration <= 20:
            return 1.0
        elif 10 <= treatment_duration < 15:
            return 0.8
        elif 20 < treatment_duration <= 30:
            return 0.7
        else:
            return 0.5
    
    def _evaluate_stage_effectiveness(self, session: TherapySession) -> Dict[str, float]:
        """评估各阶段效果"""
        stage_effectiveness = {}
        
        for stage_name in ['match_stage', 'guide_stage', 'target_stage']:
            if stage_name in session.stages:
                stage_data = session.stages[stage_name]
                
                # 基于阶段特定指标评估
                if stage_name == 'match_stage':
                    # 匹配阶段：关注情绪识别准确性
                    effectiveness = self._evaluate_match_stage(stage_data, session)
                elif stage_name == 'guide_stage':
                    # 引导阶段：关注过渡平滑性
                    effectiveness = self._evaluate_guide_stage(stage_data, session)
                else:  # target_stage
                    # 目标阶段：关注最终达成效果
                    effectiveness = self._evaluate_target_stage(stage_data, session)
                
                stage_effectiveness[stage_name] = effectiveness
        
        return stage_effectiveness
    
    def _evaluate_match_stage(self, stage_data: Dict[str, Any], session: TherapySession) -> float:
        """评估匹配阶段效果"""
        # 匹配阶段主要看情绪识别是否准确
        base_score = 0.8  # 基础评分
        
        # 如果初始情绪匹配准确，加分
        if session.initial_emotion in ['sleep_anxiety', 'hyperarousal', 'mental_fatigue']:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _evaluate_guide_stage(self, stage_data: Dict[str, Any], session: TherapySession) -> float:
        """评估引导阶段效果"""
        # 引导阶段是最关键的，基于多个指标评估
        base_score = 0.7
        
        # 如果有生理数据，检查引导阶段的改善
        if session.heart_rate_data and len(session.heart_rate_data) >= 3:
            # 检查中间阶段的心率变化
            mid_start = len(session.heart_rate_data) // 3
            mid_end = 2 * len(session.heart_rate_data) // 3
            
            pre_guide = np.mean(session.heart_rate_data[:mid_start])
            during_guide = np.mean(session.heart_rate_data[mid_start:mid_end])
            
            hr_improvement = (pre_guide - during_guide) / pre_guide
            if hr_improvement > 0.05:
                base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _evaluate_target_stage(self, stage_data: Dict[str, Any], session: TherapySession) -> float:
        """评估目标阶段效果"""
        # 目标阶段主要看最终效果
        base_score = 0.75
        
        # 基于主观评分调整
        if session.subjective_rating:
            if session.subjective_rating >= 8:
                base_score += 0.2
            elif session.subjective_rating >= 6:
                base_score += 0.1
            elif session.subjective_rating < 4:
                base_score -= 0.2
        
        return max(0.0, min(base_score, 1.0))
    
    def _generate_recommendations(self, session: TherapySession, evaluation: Dict[str, Any]) -> List[str]:
        """生成个性化建议"""
        recommendations = []
        
        overall_score = evaluation['overall_effectiveness']
        
        if overall_score < 0.6:
            recommendations.append("建议调整音乐节拍，尝试更慢的tempo以增强放松效果")
            recommendations.append("考虑延长引导阶段时间，给身心更多时间适应")
        
        if overall_score >= 0.8:
            recommendations.append("治疗效果良好，建议保持当前设置")
        
        # 基于阶段效果给出具体建议
        stage_scores = evaluation.get('stage_effectiveness', {})
        
        if stage_scores.get('match_stage', 0) < 0.7:
            recommendations.append("匹配阶段效果欠佳，建议优化情绪识别算法")
        
        if stage_scores.get('guide_stage', 0) < 0.7:
            recommendations.append("引导阶段需要改进，建议增加渐进式放松技术")
        
        if stage_scores.get('target_stage', 0) < 0.7:
            recommendations.append("目标阶段效果不足，建议调整最终放松深度")
        
        # 基于生理指标给建议
        if session.heart_rate_data:
            hr_trend = self._analyze_heart_rate_trend(session.heart_rate_data)
            if hr_trend < 0.6:
                recommendations.append("心率下降不明显，建议增加深呼吸引导")
        
        return recommendations
    
    def _identify_improvement_areas(self, evaluation: Dict[str, Any]) -> List[str]:
        """识别需要改进的区域"""
        improvement_areas = []
        
        overall_score = evaluation['overall_effectiveness']
        stage_scores = evaluation.get('stage_effectiveness', {})
        
        if overall_score < 0.7:
            improvement_areas.append("整体治疗效果")
        
        for stage, score in stage_scores.items():
            if score < 0.7:
                improvement_areas.append(f"{stage}阶段效果")
        
        return improvement_areas
    
    def collect_user_feedback(self, session_id: str, user_id: str, 
                            feedback_data: Dict[str, Any]) -> UserFeedback:
        """收集用户反馈"""
        feedback = UserFeedback(
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            music_preference=feedback_data.get('music_preference', 5),
            video_preference=feedback_data.get('video_preference', 5),
            overall_experience=feedback_data.get('overall_experience', 5),
            relaxation_level=feedback_data.get('relaxation_level', 5),
            stress_reduction=feedback_data.get('stress_reduction', 5),
            sleep_readiness=feedback_data.get('sleep_readiness', 5),
            liked_aspects=feedback_data.get('liked_aspects', []),
            disliked_aspects=feedback_data.get('disliked_aspects', []),
            suggestions=feedback_data.get('suggestions', ''),
            physical_comfort=feedback_data.get('physical_comfort', 5),
            mental_calm=feedback_data.get('mental_calm', 5)
        )
        
        # 保存反馈数据
        self._save_feedback(feedback)
        
        logger.info(f"用户反馈收集完成 - 会话: {session_id}")
        return feedback
    
    def _save_feedback(self, feedback: UserFeedback):
        """保存用户反馈"""
        feedback_file = self.data_dir / f"feedback_{feedback.user_id}_{feedback.session_id}.json"
        
        feedback_dict = {
            'session_id': feedback.session_id,
            'user_id': feedback.user_id,
            'timestamp': feedback.timestamp.isoformat(),
            'music_preference': feedback.music_preference,
            'video_preference': feedback.video_preference,
            'overall_experience': feedback.overall_experience,
            'relaxation_level': feedback.relaxation_level,
            'stress_reduction': feedback.stress_reduction,
            'sleep_readiness': feedback.sleep_readiness,
            'liked_aspects': feedback.liked_aspects,
            'disliked_aspects': feedback.disliked_aspects,
            'suggestions': feedback.suggestions,
            'physical_comfort': feedback.physical_comfort,
            'mental_calm': feedback.mental_calm
        }
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_dict, f, ensure_ascii=False, indent=2)
    
    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """分析用户使用模式"""
        user_feedback_files = list(self.data_dir.glob(f"feedback_{user_id}_*.json"))
        
        if not user_feedback_files:
            return {'error': '无用户反馈数据'}
        
        feedback_history = []
        for file in user_feedback_files:
            with open(file, 'r', encoding='utf-8') as f:
                feedback_history.append(json.load(f))
        
        # 分析趋势
        analysis = {
            'total_sessions': len(feedback_history),
            'average_satisfaction': np.mean([f['overall_experience'] for f in feedback_history]),
            'improvement_trend': self._calculate_improvement_trend(feedback_history),
            'preferred_features': self._identify_preferred_features(feedback_history),
            'common_complaints': self._identify_common_complaints(feedback_history),
            'personalized_recommendations': self._generate_personalized_recommendations(feedback_history)
        }
        
        return analysis
    
    def _calculate_improvement_trend(self, feedback_history: List[Dict[str, Any]]) -> str:
        """计算改进趋势"""
        if len(feedback_history) < 2:
            return "数据不足"
        
        # 按时间排序
        sorted_feedback = sorted(feedback_history, key=lambda x: x['timestamp'])
        
        # 计算体验评分趋势
        scores = [f['overall_experience'] for f in sorted_feedback]
        trend = np.polyfit(range(len(scores)), scores, 1)[0]  # 线性趋势
        
        if trend > 0.1:
            return "显著改善"
        elif trend > 0:
            return "轻微改善"
        elif trend > -0.1:
            return "基本稳定"
        else:
            return "需要关注"
    
    def _identify_preferred_features(self, feedback_history: List[Dict[str, Any]]) -> List[str]:
        """识别偏好特征"""
        music_avg = np.mean([f['music_preference'] for f in feedback_history])
        video_avg = np.mean([f['video_preference'] for f in feedback_history])
        
        preferences = []
        if music_avg >= 7:
            preferences.append("音乐治疗")
        if video_avg >= 7:
            preferences.append("视觉治疗")
        
        # 分析常被提及的喜欢方面
        all_liked = []
        for feedback in feedback_history:
            all_liked.extend(feedback.get('liked_aspects', []))
        
        from collections import Counter
        common_liked = Counter(all_liked).most_common(3)
        preferences.extend([item[0] for item in common_liked])
        
        return preferences
    
    def _identify_common_complaints(self, feedback_history: List[Dict[str, Any]]) -> List[str]:
        """识别常见抱怨"""
        all_disliked = []
        for feedback in feedback_history:
            all_disliked.extend(feedback.get('disliked_aspects', []))
        
        from collections import Counter
        common_complaints = Counter(all_disliked).most_common(3)
        return [item[0] for item in common_complaints]
    
    def _generate_personalized_recommendations(self, feedback_history: List[Dict[str, Any]]) -> List[str]:
        """生成个性化建议"""
        recommendations = []
        
        avg_relaxation = np.mean([f['relaxation_level'] for f in feedback_history])
        avg_sleep_readiness = np.mean([f['sleep_readiness'] for f in feedback_history])
        
        if avg_relaxation < 6:
            recommendations.append("建议增加深度放松技术")
        
        if avg_sleep_readiness < 6:
            recommendations.append("建议延长目标阶段时间")
        
        # 基于用户建议生成改进方案
        all_suggestions = [f.get('suggestions', '') for f in feedback_history]
        if any('节拍' in s or '音乐' in s for s in all_suggestions):
            recommendations.append("考虑调整音乐参数以匹配个人偏好")
        
        return recommendations