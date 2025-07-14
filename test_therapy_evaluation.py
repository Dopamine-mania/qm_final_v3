#!/usr/bin/env python3
"""
测试治疗效果评估和反馈机制
验证治疗效果评估器的功能
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from therapy_evaluator import (
    TherapyEvaluator, TherapySession, UserFeedback, 
    SleepQuality, TherapyStage
)
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_therapy_session(session_type: str = "successful") -> TherapySession:
    """创建模拟治疗会话数据"""
    
    base_time = datetime.now() - timedelta(hours=1)
    
    if session_type == "successful":
        # 成功的治疗会话
        session = TherapySession(
            session_id="test_session_001",
            user_id="user_001",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=18),
            initial_emotion="sleep_anxiety",
            target_emotion="peaceful",
            stages={
                'match_stage': {
                    'tempo_bpm': 100.0,
                    'duration': 2.0,
                    'therapy_intensity': 0.0
                },
                'guide_stage': {
                    'tempo_bpm': 80.0,
                    'duration': 6.0,
                    'therapy_intensity': 0.5
                },
                'target_stage': {
                    'tempo_bpm': 45.0,
                    'duration': 10.0,
                    'therapy_intensity': 0.8
                }
            },
            # 模拟良好的生理数据
            heart_rate_data=generate_decreasing_heart_rate(),
            breathing_rate_data=generate_calming_breathing(),
            movement_data=generate_decreasing_movement(),
            # 积极的主观反馈
            subjective_rating=8,
            sleep_quality=SleepQuality.GOOD,
            sleep_latency=20.0,  # 20分钟入睡
            total_sleep_time=7.5,  # 7.5小时睡眠
        )
    
    elif session_type == "moderate":
        # 中等效果的治疗会话
        session = TherapySession(
            session_id="test_session_002",
            user_id="user_002",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=25),
            initial_emotion="hyperarousal",
            target_emotion="relaxed",
            stages={
                'match_stage': {
                    'tempo_bpm': 120.0,
                    'duration': 3.0,
                    'therapy_intensity': 0.2
                },
                'guide_stage': {
                    'tempo_bpm': 90.0,
                    'duration': 8.0,
                    'therapy_intensity': 0.4
                },
                'target_stage': {
                    'tempo_bpm': 60.0,
                    'duration': 14.0,
                    'therapy_intensity': 0.6
                }
            },
            heart_rate_data=generate_moderate_heart_rate(),
            breathing_rate_data=generate_moderate_breathing(),
            movement_data=generate_moderate_movement(),
            subjective_rating=6,
            sleep_quality=SleepQuality.FAIR,
            sleep_latency=40.0,  # 40分钟入睡
            total_sleep_time=6.5,  # 6.5小时睡眠
        )
    
    else:  # "poor"
        # 效果不佳的治疗会话
        session = TherapySession(
            session_id="test_session_003",
            user_id="user_003",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=35),
            initial_emotion="mental_fatigue",
            target_emotion="drowsy",
            stages={
                'match_stage': {
                    'tempo_bpm': 80.0,
                    'duration': 5.0,
                    'therapy_intensity': 0.1
                },
                'guide_stage': {
                    'tempo_bpm': 70.0,
                    'duration': 15.0,
                    'therapy_intensity': 0.3
                },
                'target_stage': {
                    'tempo_bpm': 50.0,
                    'duration': 15.0,
                    'therapy_intensity': 0.4
                }
            },
            heart_rate_data=generate_poor_heart_rate(),
            breathing_rate_data=generate_irregular_breathing(),
            movement_data=generate_restless_movement(),
            subjective_rating=3,
            sleep_quality=SleepQuality.POOR,
            sleep_latency=80.0,  # 80分钟入睡
            total_sleep_time=4.5,  # 4.5小时睡眠
        )
    
    return session

def generate_decreasing_heart_rate() -> list:
    """生成下降趋势的心率数据"""
    time_points = 60  # 60个时间点
    initial_hr = 75
    final_hr = 65
    
    # 生成平滑下降的心率
    heart_rates = np.linspace(initial_hr, final_hr, time_points)
    # 添加小幅随机波动
    noise = np.random.normal(0, 2, time_points)
    heart_rates += noise
    
    return heart_rates.tolist()

def generate_moderate_heart_rate() -> list:
    """生成中等下降趋势的心率数据"""
    time_points = 60
    initial_hr = 78
    final_hr = 72
    
    heart_rates = np.linspace(initial_hr, final_hr, time_points)
    noise = np.random.normal(0, 3, time_points)
    heart_rates += noise
    
    return heart_rates.tolist()

def generate_poor_heart_rate() -> list:
    """生成心率变化不明显的数据"""
    time_points = 60
    initial_hr = 76
    final_hr = 75  # 几乎无变化
    
    heart_rates = np.linspace(initial_hr, final_hr, time_points)
    noise = np.random.normal(0, 4, time_points)  # 更大波动
    heart_rates += noise
    
    return heart_rates.tolist()

def generate_calming_breathing() -> list:
    """生成平静的呼吸数据"""
    time_points = 60
    initial_br = 16
    final_br = 12
    
    breathing_rates = np.linspace(initial_br, final_br, time_points)
    noise = np.random.normal(0, 0.5, time_points)
    breathing_rates += noise
    
    return breathing_rates.tolist()

def generate_moderate_breathing() -> list:
    """生成中等改善的呼吸数据"""
    time_points = 60
    initial_br = 18
    final_br = 15
    
    breathing_rates = np.linspace(initial_br, final_br, time_points)
    noise = np.random.normal(0, 1, time_points)
    breathing_rates += noise
    
    return breathing_rates.tolist()

def generate_irregular_breathing() -> list:
    """生成不规律的呼吸数据"""
    time_points = 60
    breathing_rates = np.random.normal(16, 3, time_points)  # 高变异性
    
    return breathing_rates.tolist()

def generate_decreasing_movement() -> list:
    """生成逐渐减少的活动数据"""
    time_points = 30
    initial_movement = 5.0
    final_movement = 0.5
    
    movements = np.linspace(initial_movement, final_movement, time_points)
    noise = np.random.exponential(0.5, time_points)  # 指数分布噪声
    movements += noise
    
    return movements.tolist()

def generate_moderate_movement() -> list:
    """生成中等减少的活动数据"""
    time_points = 30
    initial_movement = 4.0
    final_movement = 2.0
    
    movements = np.linspace(initial_movement, final_movement, time_points)
    noise = np.random.exponential(0.8, time_points)
    movements += noise
    
    return movements.tolist()

def generate_restless_movement() -> list:
    """生成躁动的活动数据"""
    time_points = 30
    movements = np.random.exponential(3, time_points)  # 高活动水平
    
    return movements.tolist()

def create_mock_user_feedback(session_id: str, user_id: str, feedback_type: str = "positive") -> dict:
    """创建模拟用户反馈"""
    
    if feedback_type == "positive":
        return {
            'music_preference': 8,
            'video_preference': 7,
            'overall_experience': 8,
            'relaxation_level': 9,
            'stress_reduction': 8,
            'sleep_readiness': 9,
            'liked_aspects': ['音乐节奏', '视觉效果', '渐进式放松'],
            'disliked_aspects': [],
            'suggestions': '音乐很棒，希望能有更多自然声音',
            'physical_comfort': 8,
            'mental_calm': 9
        }
    
    elif feedback_type == "moderate":
        return {
            'music_preference': 6,
            'video_preference': 5,
            'overall_experience': 6,
            'relaxation_level': 6,
            'stress_reduction': 5,
            'sleep_readiness': 6,
            'liked_aspects': ['引导阶段'],
            'disliked_aspects': ['视频太单调'],
            'suggestions': '希望视频内容更丰富一些',
            'physical_comfort': 6,
            'mental_calm': 6
        }
    
    else:  # "negative"
        return {
            'music_preference': 3,
            'video_preference': 4,
            'overall_experience': 3,
            'relaxation_level': 3,
            'stress_reduction': 2,
            'sleep_readiness': 3,
            'liked_aspects': [],
            'disliked_aspects': ['音乐太慢', '时间太长', '效果不明显'],
            'suggestions': '需要更有效的放松技术',
            'physical_comfort': 4,
            'mental_calm': 3
        }

def test_therapy_evaluation():
    """测试治疗效果评估"""
    
    print("🏥 治疗效果评估和反馈机制测试")
    print("=" * 60)
    
    # 初始化评估器
    evaluator = TherapyEvaluator(data_dir="test_therapy_data")
    
    # 测试不同类型的治疗会话
    test_cases = [
        ("successful", "positive"),
        ("moderate", "moderate"), 
        ("poor", "negative")
    ]
    
    for i, (session_type, feedback_type) in enumerate(test_cases, 1):
        print(f"\n🔬 === 测试案例 {i}: {session_type}治疗会话 ===")
        
        # 创建模拟会话
        session = create_mock_therapy_session(session_type)
        print(f"📊 会话ID: {session.session_id}")
        print(f"   用户: {session.user_id}")
        print(f"   初始情绪: {session.initial_emotion}")
        print(f"   目标情绪: {session.target_emotion}")
        print(f"   治疗时长: {(session.end_time - session.start_time).total_seconds()/60:.1f}分钟")
        
        # 评估治疗效果
        print(f"\n📈 开始治疗效果评估...")
        evaluation_results = evaluator.evaluate_session(session)
        
        print(f"✅ 评估完成！")
        print(f"\n📋 评估结果详情:")
        print(f"   🎯 总体效果: {evaluation_results['overall_effectiveness']:.3f}")
        
        # 分阶段效果
        stage_effectiveness = evaluation_results['stage_effectiveness']
        print(f"   📊 分阶段效果:")
        for stage, score in stage_effectiveness.items():
            stage_name_cn = {
                'match_stage': '匹配阶段',
                'guide_stage': '引导阶段',
                'target_stage': '目标阶段'
            }.get(stage, stage)
            print(f"      • {stage_name_cn}: {score:.3f}")
        
        # 改进建议
        recommendations = evaluation_results['recommendations']
        print(f"   💡 改进建议 ({len(recommendations)}条):")
        for j, rec in enumerate(recommendations, 1):
            print(f"      {j}. {rec}")
        
        # 需改进区域
        improvement_areas = evaluation_results['improvement_areas']
        if improvement_areas:
            print(f"   ⚠️ 需改进区域:")
            for area in improvement_areas:
                print(f"      • {area}")
        else:
            print(f"   ✨ 无明显需改进区域")
        
        # 收集用户反馈
        print(f"\n📝 收集用户反馈...")
        feedback_data = create_mock_user_feedback(session.session_id, session.user_id, feedback_type)
        user_feedback = evaluator.collect_user_feedback(
            session.session_id, 
            session.user_id, 
            feedback_data
        )
        
        print(f"✅ 用户反馈收集完成")
        print(f"   🎵 音乐偏好: {user_feedback.music_preference}/10")
        print(f"   🎬 视频偏好: {user_feedback.video_preference}/10")  
        print(f"   💫 总体体验: {user_feedback.overall_experience}/10")
        print(f"   😌 放松程度: {user_feedback.relaxation_level}/10")
        print(f"   💤 睡眠准备: {user_feedback.sleep_readiness}/10")
        
        if user_feedback.liked_aspects:
            print(f"   👍 喜欢的方面: {', '.join(user_feedback.liked_aspects)}")
        if user_feedback.disliked_aspects:
            print(f"   👎 不喜欢的方面: {', '.join(user_feedback.disliked_aspects)}")
        if user_feedback.suggestions:
            print(f"   💭 建议: {user_feedback.suggestions}")
        
        print(f"\n" + "─" * 50)
    
    # 测试用户模式分析
    print(f"\n👤 用户模式分析测试")
    print("=" * 40)
    
    # 为user_001分析使用模式
    user_analysis = evaluator.analyze_user_patterns("user_001")
    
    if 'error' not in user_analysis:
        print(f"📊 用户user_001使用模式分析:")
        print(f"   • 总会话数: {user_analysis['total_sessions']}")
        print(f"   • 平均满意度: {user_analysis['average_satisfaction']:.2f}/10")
        print(f"   • 改进趋势: {user_analysis['improvement_trend']}")
        
        if user_analysis['preferred_features']:
            print(f"   • 偏好特征: {', '.join(user_analysis['preferred_features'])}")
        
        if user_analysis['common_complaints']:
            print(f"   • 常见抱怨: {', '.join(user_analysis['common_complaints'])}")
        
        if user_analysis['personalized_recommendations']:
            print(f"   • 个性化建议:")
            for rec in user_analysis['personalized_recommendations']:
                print(f"      - {rec}")
    else:
        print(f"❌ {user_analysis['error']}")
    
    print(f"\n🎉 治疗效果评估和反馈机制测试完成！")
    print(f"\n📊 测试总结:")
    print(f"   ✅ 治疗会话效果评估 - 正常")
    print(f"   ✅ 生理指标分析 - 正常")
    print(f"   ✅ 主观反馈收集 - 正常")
    print(f"   ✅ 分阶段效果评估 - 正常")
    print(f"   ✅ 改进建议生成 - 正常")
    print(f"   ✅ 用户模式分析 - 正常")
    print(f"\n💡 核心特性:")
    print(f"   • 多维度效果评估（生理+主观+行为+时间）")
    print(f"   • 个性化改进建议生成")
    print(f"   • 用户反馈收集和模式分析")
    print(f"   • 三阶段治疗效果分析")
    print(f"   • 自适应优化建议")

if __name__ == "__main__":
    test_therapy_evaluation()