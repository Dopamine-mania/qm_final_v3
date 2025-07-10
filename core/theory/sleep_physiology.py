"""
睡眠生理学模块
基于睡眠研究的生理学原理实现

核心原理：
1. 脑波优化：促进α波，减少β波
2. 心率同步：音乐BPM与静息心率同步
3. 副交感神经激活：降低皮质唤醒
4. 内分泌调节：促进褐黑素分泌

参考文献：
- Music improves sleep quality by modulating brain activity (2024)
- The effects of music on the sleep quality (2024) 
- Music therapy for sleep disorders (2024)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import yaml
import json
import math

# 导入相关模块
from .valence_arousal import EmotionState

# 配置加载
try:
    with open(Path(__file__).parent.parent.parent / "configs" / "theory_params.yaml", "r", encoding="utf-8") as f:
        THEORY_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    THEORY_CONFIG = {
        "sleep_physiology": {
            "brainwave_targets": {
                "alpha_range": [8, 12],
                "theta_range": [4, 8],
                "delta_range": [0.5, 4]
            },
            "heart_rate_sync": {
                "resting_bpm_range": [50, 60],
                "transition_bpm": [70, 90],
                "sync_tolerance": 5
            },
            "parasympathetic_activation": {
                "enable": True,
                "low_frequency_emphasis": True,
                "harmonic_resonance": True
            }
        }
    }

logger = logging.getLogger(__name__)

class SleepStage(Enum):
    """
    睡眠阶段分类
    """
    AWAKE = "awake"                    # 清醒状态
    DROWSY = "drowsy"                  # 困倦状态  
    LIGHT_SLEEP = "light_sleep"        # 浅睡眠
    DEEP_SLEEP = "deep_sleep"          # 深睡眠
    REM_SLEEP = "rem_sleep"            # REM睡眠

class BrainwaveType(Enum):
    """
    脑波类型
    """
    GAMMA = "gamma"      # 30-100 Hz - 高度集中
    BETA = "beta"        # 13-30 Hz - 清醒、焦虑
    ALPHA = "alpha"      # 8-13 Hz - 放松、冥想
    THETA = "theta"      # 4-8 Hz - 深度放松、初睡
    DELTA = "delta"      # 0.5-4 Hz - 深睡眠

@dataclass
class BrainwaveProfile:
    """
    脑波概谱
    """
    gamma_power: float = 0.0
    beta_power: float = 0.0
    alpha_power: float = 0.0
    theta_power: float = 0.0
    delta_power: float = 0.0
    
    def normalize(self) -> 'BrainwaveProfile':
        """归一化脑波功率"""
        total = self.gamma_power + self.beta_power + self.alpha_power + self.theta_power + self.delta_power
        if total == 0:
            return self
        
        return BrainwaveProfile(
            gamma_power=self.gamma_power / total,
            beta_power=self.beta_power / total,
            alpha_power=self.alpha_power / total,
            theta_power=self.theta_power / total,
            delta_power=self.delta_power / total
        )
    
    def get_dominant_wave(self) -> BrainwaveType:
        """获取主导脑波类型"""
        powers = {
            BrainwaveType.GAMMA: self.gamma_power,
            BrainwaveType.BETA: self.beta_power,
            BrainwaveType.ALPHA: self.alpha_power,
            BrainwaveType.THETA: self.theta_power,
            BrainwaveType.DELTA: self.delta_power
        }
        return max(powers, key=powers.get)

@dataclass
class PhysiologicalState:
    """
    生理状态
    """
    heart_rate: float          # 心率 (BPM)
    brainwave_profile: BrainwaveProfile
    stress_level: float        # 应激水平 (0-1)
    drowsiness_level: float    # 困倦度 (0-1)
    sleep_readiness: float     # 睡眠准备度 (0-1)
    timestamp: Optional[float] = None
    
    def get_sleep_stage(self) -> SleepStage:
        """根据生理指标判断睡眠阶段"""
        dominant_wave = self.brainwave_profile.get_dominant_wave()
        
        if self.drowsiness_level < 0.3:
            return SleepStage.AWAKE
        elif self.drowsiness_level < 0.6:
            return SleepStage.DROWSY
        elif dominant_wave in [BrainwaveType.ALPHA, BrainwaveType.THETA]:
            return SleepStage.LIGHT_SLEEP
        elif dominant_wave == BrainwaveType.DELTA:
            return SleepStage.DEEP_SLEEP
        else:
            return SleepStage.DROWSY

@dataclass
class MusicTherapyParameters:
    """
    音乐疗法参数
    """
    target_bpm: float          # 目标BPM
    dominant_frequency: float  # 主频率 (Hz)
    harmonic_content: List[float]  # 谐波内容
    volume_level: float        # 音量级别 (0-1)
    stereo_width: float        # 立体声宽度 (0-1)
    frequency_emphasis: Dict[str, float]  # 频率强调
    
class SleepPhysiologyModel:
    """
    睡眠生理学模型
    
    结合生理学、神经科学和音乐疗法的原理，
    为睡前情绪疗愈提供科学依据。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or THEORY_CONFIG["sleep_physiology"]
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 脑波目标
        self.brainwave_targets = self.config["brainwave_targets"]
        
        # 心率同步参数
        self.heart_rate_sync = self.config["heart_rate_sync"]
        
        # 副交感神经激活参数
        self.parasympathetic_config = self.config["parasympathetic_activation"]
        
        # 预定义的生理模式
        self.physiological_patterns = self._create_physiological_patterns()
        
        self.logger.info("睡眠生理学模型初始化完成")
    
    def _create_physiological_patterns(self) -> Dict[str, Dict]:
        """
        创建预定义的生理模式
        """
        patterns = {
            "awake_alert": {
                "heart_rate": 80,
                "brainwave": BrainwaveProfile(beta_power=0.6, alpha_power=0.3, theta_power=0.1),
                "stress_level": 0.7,
                "drowsiness_level": 0.1
            },
            "awake_relaxed": {
                "heart_rate": 70,
                "brainwave": BrainwaveProfile(alpha_power=0.5, beta_power=0.3, theta_power=0.2),
                "stress_level": 0.3,
                "drowsiness_level": 0.3
            },
            "drowsy": {
                "heart_rate": 65,
                "brainwave": BrainwaveProfile(alpha_power=0.4, theta_power=0.4, beta_power=0.2),
                "stress_level": 0.2,
                "drowsiness_level": 0.6
            },
            "sleep_ready": {
                "heart_rate": 55,
                "brainwave": BrainwaveProfile(theta_power=0.5, alpha_power=0.3, delta_power=0.2),
                "stress_level": 0.1,
                "drowsiness_level": 0.8
            },
            "light_sleep": {
                "heart_rate": 50,
                "brainwave": BrainwaveProfile(theta_power=0.4, delta_power=0.4, alpha_power=0.2),
                "stress_level": 0.05,
                "drowsiness_level": 0.9
            }
        }
        
        return patterns
    
    def analyze_current_state(self, emotion_state: EmotionState) -> PhysiologicalState:
        """
        根据情绪状态分析当前生理状态
        
        Args:
            emotion_state: 当前情绪状态
            
        Returns:
            预估的生理状态
        """
        # 根据情绪的valence和arousal映射到生理指标
        
        # 心率映射：高arousal -> 高心率
        base_hr = 70
        arousal_effect = emotion_state.arousal * 20  # arousal对心率的影响
        estimated_hr = base_hr + arousal_effect
        estimated_hr = max(50, min(100, estimated_hr))  # 限制在合理范围
        
        # 脑波模式映射
        brainwave = self._estimate_brainwave_from_emotion(emotion_state)
        
        # 应激水平：低 valence + 高 arousal -> 高应激
        stress = max(0, (0.5 - emotion_state.valence) * 0.5 + emotion_state.arousal * 0.3)
        stress = min(1.0, stress)
        
        # 困倦度：低 arousal -> 高困倦
        drowsiness = max(0, (0.5 - emotion_state.arousal) * 0.8)
        drowsiness = min(1.0, drowsiness)
        
        # 睡眠准备度：综合指标
        sleep_readiness = self._calculate_sleep_readiness(
            estimated_hr, brainwave, stress, drowsiness
        )
        
        state = PhysiologicalState(
            heart_rate=estimated_hr,
            brainwave_profile=brainwave,
            stress_level=stress,
            drowsiness_level=drowsiness,
            sleep_readiness=sleep_readiness
        )
        
        self.logger.info(f"生理状态分析: HR={estimated_hr:.1f}, 应激={stress:.2f}, 困倦={drowsiness:.2f}")
        
        return state
    
    def _estimate_brainwave_from_emotion(self, emotion_state: EmotionState) -> BrainwaveProfile:
        """
        根据情绪状态估计脑波模式
        """
        # 基础脑波分布
        base_profile = BrainwaveProfile(
            gamma_power=0.05,
            beta_power=0.3,
            alpha_power=0.4,
            theta_power=0.2,
            delta_power=0.05
        )
        
        # 根据arousal调整
        if emotion_state.arousal > 0.5:  # 高唤醒
            base_profile.beta_power += 0.2
            base_profile.alpha_power -= 0.1
            base_profile.theta_power -= 0.1
        elif emotion_state.arousal < -0.3:  # 低判醒
            base_profile.beta_power -= 0.2
            base_profile.alpha_power += 0.1
            base_profile.theta_power += 0.1
        
        # 根据valence调整
        if emotion_state.valence < -0.3:  # 消极情绪
            base_profile.beta_power += 0.1  # 应激反应
            base_profile.alpha_power -= 0.1
        
        return base_profile.normalize()
    
    def _calculate_sleep_readiness(self, 
                                 heart_rate: float,
                                 brainwave: BrainwaveProfile,
                                 stress: float,
                                 drowsiness: float) -> float:
        """
        计算睡眠准备度
        """
        # 心率因子：越接近静息心率越好
        optimal_hr_range = self.heart_rate_sync["resting_bpm_range"]
        hr_score = 1.0 - min(1.0, abs(heart_rate - np.mean(optimal_hr_range)) / 20)
        
        # 脑波因子：α和θ波越多越好
        brainwave_score = (brainwave.alpha_power + brainwave.theta_power) - brainwave.beta_power * 0.5
        brainwave_score = max(0, min(1, brainwave_score))
        
        # 应激因子：应激越低越好
        stress_score = 1.0 - stress
        
        # 困倦因子：适度困倦最好
        drowsiness_score = min(1.0, drowsiness * 1.2)  # 轻微增强困倦的权重
        
        # 加权平均
        weights = [0.25, 0.3, 0.25, 0.2]  # hr, brainwave, stress, drowsiness
        scores = [hr_score, brainwave_score, stress_score, drowsiness_score]
        
        sleep_readiness = sum(w * s for w, s in zip(weights, scores))
        
        return max(0, min(1, sleep_readiness))
    
    def generate_therapy_progression(self, 
                                   initial_state: PhysiologicalState,
                                   target_sleep_stage: SleepStage = SleepStage.DROWSY,
                                   duration_minutes: float = 20.0) -> List[PhysiologicalState]:
        """
        生成生理疗愈进程
        
        Args:
            initial_state: 初始生理状态
            target_sleep_stage: 目标睡眠阶段
            duration_minutes: 持续时间（分钟）
            
        Returns:
            生理状态变化序列
        """
        # 获取目标模式
        target_patterns = {
            SleepStage.AWAKE: "awake_relaxed",
            SleepStage.DROWSY: "drowsy",
            SleepStage.LIGHT_SLEEP: "sleep_ready",
            SleepStage.DEEP_SLEEP: "light_sleep"
        }
        
        target_pattern_name = target_patterns.get(target_sleep_stage, "sleep_ready")
        target_pattern = self.physiological_patterns[target_pattern_name]
        
        # 创建目标状态
        target_state = PhysiologicalState(
            heart_rate=target_pattern["heart_rate"],
            brainwave_profile=target_pattern["brainwave"],
            stress_level=target_pattern["stress_level"],
            drowsiness_level=target_pattern["drowsiness_level"],
            sleep_readiness=0.8  # 目标睡眠准备度
        )
        
        # 生成时间序列（每分钟一个点）
        num_points = int(duration_minutes) + 1
        progression = []
        
        for i in range(num_points):
            t = i / (num_points - 1)  # 0 到 1
            
            # 使用非线性插值（S型曲线）
            smooth_t = self._smooth_transition(t)
            
            # 插值生成中间状态
            interpolated_state = self._interpolate_physiological_states(
                initial_state, target_state, smooth_t
            )
            
            interpolated_state.timestamp = i / 60.0  # 转换为小时
            progression.append(interpolated_state)
        
        self.logger.info(f"生成{duration_minutes}分钟的疗愈进程，共{len(progression)}个点")
        
        return progression
    
    def _smooth_transition(self, t: float, steepness: float = 3.0) -> float:
        """
        平滑过渡函数（S型曲线）
        """
        # Sigmoid函数变种
        return 1 / (1 + math.exp(-steepness * (t - 0.5)))
    
    def _interpolate_physiological_states(self, 
                                        start: PhysiologicalState,
                                        end: PhysiologicalState,
                                        t: float) -> PhysiologicalState:
        """
        插值生理状态
        """
        # 线性插值各个指标
        heart_rate = start.heart_rate * (1 - t) + end.heart_rate * t
        stress_level = start.stress_level * (1 - t) + end.stress_level * t
        drowsiness_level = start.drowsiness_level * (1 - t) + end.drowsiness_level * t
        sleep_readiness = start.sleep_readiness * (1 - t) + end.sleep_readiness * t
        
        # 脑波插值
        brainwave = BrainwaveProfile(
            gamma_power=start.brainwave_profile.gamma_power * (1 - t) + end.brainwave_profile.gamma_power * t,
            beta_power=start.brainwave_profile.beta_power * (1 - t) + end.brainwave_profile.beta_power * t,
            alpha_power=start.brainwave_profile.alpha_power * (1 - t) + end.brainwave_profile.alpha_power * t,
            theta_power=start.brainwave_profile.theta_power * (1 - t) + end.brainwave_profile.theta_power * t,
            delta_power=start.brainwave_profile.delta_power * (1 - t) + end.brainwave_profile.delta_power * t
        ).normalize()
        
        return PhysiologicalState(
            heart_rate=heart_rate,
            brainwave_profile=brainwave,
            stress_level=stress_level,
            drowsiness_level=drowsiness_level,
            sleep_readiness=sleep_readiness
        )
    
    def calculate_music_parameters(self, 
                                 target_state: PhysiologicalState,
                                 current_bpm: Optional[float] = None) -> MusicTherapyParameters:
        """
        根据目标生理状态计算音乐疗法参数
        
        Args:
            target_state: 目标生理状态
            current_bpm: 当前BPM（用于渐进调整）
            
        Returns:
            音乐疗法参数
        """
        # 目标BPM：基于目标心率
        target_bpm = target_state.heart_rate * 0.8  # 略低于心率
        target_bpm = max(40, min(80, target_bpm))  # 限制在疗愈范围
        
        # 如果有当前BPM，则逐渐调整
        if current_bpm is not None:
            max_change = 5  # 每次最大变化5 BPM
            if abs(target_bpm - current_bpm) > max_change:
                if target_bpm > current_bpm:
                    target_bpm = current_bpm + max_change
                else:
                    target_bpm = current_bpm - max_change
        
        # 主频率：基于目标脑波
        dominant_wave = target_state.brainwave_profile.get_dominant_wave()
        frequency_ranges = {
            BrainwaveType.BETA: [15, 25],
            BrainwaveType.ALPHA: [8, 12],
            BrainwaveType.THETA: [4, 8],
            BrainwaveType.DELTA: [1, 4]
        }
        
        freq_range = frequency_ranges.get(dominant_wave, [8, 12])
        dominant_frequency = np.mean(freq_range)
        
        # 谐波内容：基于副交感神经激活
        harmonics = self._calculate_therapeutic_harmonics(dominant_frequency)
        
        # 音量级别：根据应激水平调整
        volume = 0.7 - target_state.stress_level * 0.3  # 应激越高音量越低
        volume = max(0.2, min(0.8, volume))
        
        # 立体声宽度：睡眠时窄窄一些
        stereo_width = 0.8 - target_state.drowsiness_level * 0.3
        stereo_width = max(0.3, min(1.0, stereo_width))
        
        # 频率强调
        frequency_emphasis = self._calculate_frequency_emphasis(target_state)
        
        params = MusicTherapyParameters(
            target_bpm=target_bpm,
            dominant_frequency=dominant_frequency,
            harmonic_content=harmonics,
            volume_level=volume,
            stereo_width=stereo_width,
            frequency_emphasis=frequency_emphasis
        )
        
        self.logger.info(f"计算音乐参数: BPM={target_bpm:.1f}, 主频={dominant_frequency:.1f}Hz, 音量={volume:.2f}")
        
        return params
    
    def _calculate_therapeutic_harmonics(self, fundamental_freq: float) -> List[float]:
        """
        计算疗愈谐波
        
        基于音乐疗法理论，某些谐波比例具有特殊的放松效果
        """
        harmonics = []
        
        # 基频
        harmonics.append(fundamental_freq)
        
        # 五度谐波 (3:2 比例)
        harmonics.append(fundamental_freq * 1.5)
        
        # 四度谐波 (4:3 比例)
        harmonics.append(fundamental_freq * 1.333)
        
        # 八度谐波 (2:1 比例)
        harmonics.append(fundamental_freq * 2.0)
        
        # 黄金比例谐波 (特殊的放松效果)
        golden_ratio = 1.618
        harmonics.append(fundamental_freq * golden_ratio)
        
        # 过滤超出人耳听觉范围的频率
        harmonics = [h for h in harmonics if 20 <= h <= 20000]
        
        return harmonics
    
    def _calculate_frequency_emphasis(self, target_state: PhysiologicalState) -> Dict[str, float]:
        """
        计算频率强调
        
        根据目标状态调整不同频段的强调
        """
        emphasis = {
            "sub_bass": 0.3,      # 20-60 Hz
            "bass": 0.5,          # 60-250 Hz
            "low_mid": 0.7,       # 250-500 Hz
            "mid": 0.8,           # 500-2000 Hz
            "high_mid": 0.6,      # 2000-4000 Hz
            "presence": 0.4,      # 4000-6000 Hz
            "brilliance": 0.2     # 6000-20000 Hz
        }
        
        # 根据目标状态调整
        if target_state.drowsiness_level > 0.6:  # 高困倦状态
            # 强调低频，减弱高频
            emphasis["sub_bass"] += 0.2
            emphasis["bass"] += 0.2
            emphasis["presence"] -= 0.2
            emphasis["brilliance"] -= 0.3
        
        if target_state.stress_level < 0.3:  # 低应激状态
            # 增强中频的温暖感
            emphasis["low_mid"] += 0.1
            emphasis["mid"] += 0.1
        
        # 确保所有值在合理范围内
        for key in emphasis:
            emphasis[key] = max(0.0, min(1.0, emphasis[key]))
        
        return emphasis
    
    def validate_therapy_effectiveness(self, 
                                     initial_state: PhysiologicalState,
                                     final_state: PhysiologicalState) -> Dict[str, float]:
        """
        验证疗愈有效性
        
        Returns:
            有效性指标
        """
        # 心率降低效果
        hr_improvement = (initial_state.heart_rate - final_state.heart_rate) / initial_state.heart_rate
        hr_improvement = max(0, hr_improvement)  # 只考虑改善
        
        # 应激水平降低
        stress_improvement = initial_state.stress_level - final_state.stress_level
        stress_improvement = max(0, stress_improvement)
        
        # 困倦度提升
        drowsiness_improvement = final_state.drowsiness_level - initial_state.drowsiness_level
        drowsiness_improvement = max(0, drowsiness_improvement)
        
        # 睡眠准备度提升
        sleep_readiness_improvement = final_state.sleep_readiness - initial_state.sleep_readiness
        sleep_readiness_improvement = max(0, sleep_readiness_improvement)
        
        # 脑波改善（α和θ波增加，β波减少）
        beneficial_waves_initial = (initial_state.brainwave_profile.alpha_power + 
                                   initial_state.brainwave_profile.theta_power)
        beneficial_waves_final = (final_state.brainwave_profile.alpha_power + 
                                 final_state.brainwave_profile.theta_power)
        brainwave_improvement = beneficial_waves_final - beneficial_waves_initial
        brainwave_improvement = max(0, brainwave_improvement)
        
        # 综合有效性分数
        weights = [0.25, 0.25, 0.2, 0.2, 0.1]  # 各指标权重
        improvements = [
            hr_improvement,
            stress_improvement,
            drowsiness_improvement,
            sleep_readiness_improvement,
            brainwave_improvement
        ]
        
        overall_effectiveness = sum(w * i for w, i in zip(weights, improvements))
        
        return {
            "heart_rate_improvement": hr_improvement,
            "stress_reduction": stress_improvement,
            "drowsiness_increase": drowsiness_improvement,
            "sleep_readiness_increase": sleep_readiness_improvement,
            "brainwave_improvement": brainwave_improvement,
            "overall_effectiveness": overall_effectiveness
        }
    
    def export_physiology_data(self, 
                             progression: List[PhysiologicalState],
                             filepath: str) -> None:
        """
        导出生理数据
        """
        export_data = {
            "sleep_physiology_model_version": "1.0",
            "timestamp": np.datetime64('now').astype(str),
            "model_config": self.config,
            "progression_data": [
                {
                    "timestamp": state.timestamp,
                    "heart_rate": state.heart_rate,
                    "brainwave_profile": {
                        "gamma": state.brainwave_profile.gamma_power,
                        "beta": state.brainwave_profile.beta_power,
                        "alpha": state.brainwave_profile.alpha_power,
                        "theta": state.brainwave_profile.theta_power,
                        "delta": state.brainwave_profile.delta_power
                    },
                    "stress_level": state.stress_level,
                    "drowsiness_level": state.drowsiness_level,
                    "sleep_readiness": state.sleep_readiness,
                    "sleep_stage": state.get_sleep_stage().value
                } for state in progression
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"生理数据已导出到: {filepath}")

# 工具函数
def create_sleep_model(config_path: Optional[str] = None) -> SleepPhysiologyModel:
    """
    创建睡眠生理学模型的便捷函数
    """
    config = None
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get("sleep_physiology")
    
    return SleepPhysiologyModel(config)

# 示例使用
if __name__ == "__main__":
    # 创建睡眠生理学模型
    sleep_model = SleepPhysiologyModel()
    
    # 模拟初始情绪状态（焦虑、高判醒）
    initial_emotion = EmotionState(valence=-0.2, arousal=0.6, confidence=0.9)
    
    # 分析当前生理状态
    initial_physiology = sleep_model.analyze_current_state(initial_emotion)
    
    # 生成疗愈进程
    therapy_progression = sleep_model.generate_therapy_progression(
        initial_physiology,
        target_sleep_stage=SleepStage.DROWSY,
        duration_minutes=20.0
    )
    
    # 计算音乐参数
    target_state = therapy_progression[-1]
    music_params = sleep_model.calculate_music_parameters(target_state)
    
    # 验证有效性
    effectiveness = sleep_model.validate_therapy_effectiveness(
        initial_physiology, target_state
    )
    
    print("睡眠生理学模型演示:")
    print(f"初始状态: HR={initial_physiology.heart_rate:.1f}, 应激={initial_physiology.stress_level:.2f}")
    print(f"目标状态: HR={target_state.heart_rate:.1f}, 睡眠准备度={target_state.sleep_readiness:.2f}")
    print(f"音乐参数: BPM={music_params.target_bpm:.1f}, 主频={music_params.dominant_frequency:.1f}Hz")
    print(f"疗愈有效性: {effectiveness['overall_effectiveness']:.3f}")
