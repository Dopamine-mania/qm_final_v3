"""
ISO原则理论实现
基于2024年最新学术研究的ISO原则数字化实现

ISO原则 (Iso-Principle):
从与当前情绪状态匹配的音乐开始，逐渐过渡到期望的情绪状态。

参考文献:
- Starcke & von Georgi (2024). Music listening according to the iso principle modulates affective state.
- Generated Therapeutic Music Based on the ISO Principle (2024).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import yaml

# 配置加载
try:
    with open(Path(__file__).parent.parent.parent / "configs" / "theory_params.yaml", "r", encoding="utf-8") as f:
        THEORY_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # 如果配置文件不存在，使用默认值
    THEORY_CONFIG = {
        "iso_principle": {
            "stage_durations": {"synchronization": 0.25, "guidance": 0.50, "consolidation": 0.25},
            "transition_params": {"smoothness_factor": 0.8, "step_size": 0.05, "adaptation_rate": 0.7},
            "target_sleep_state": {"valence": 0.2, "arousal": -0.6, "confidence": 0.9}
        }
    }

logger = logging.getLogger(__name__)

class ISOStage(Enum):
    """
    ISO原则的三个阶段
    """
    SYNCHRONIZATION = "synchronization"  # 同频阶段：匹配当前情绪
    GUIDANCE = "guidance"                # 引导阶段：逐渐过渡
    CONSOLIDATION = "consolidation"      # 巩固阶段：稳定目标状态

@dataclass
class EmotionState:
    """
    情绪状态表示（Valence-Arousal模型）
    """
    valence: float    # 效价（-1到1，消极到积极）
    arousal: float    # 唤醒度（-1到1，低判醒到高唤醒）
    confidence: float = 1.0  # 置信度（0到1）
    timestamp: Optional[float] = None  # 时间戳
    
    def __post_init__(self):
        """"验证数值范围"""
        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, -1.0, 1.0)
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
    
    def distance_to(self, other: 'EmotionState') -> float:
        """
        计算与另一个情绪状态的欧式距离
        """
        return np.sqrt((self.valence - other.valence)**2 + (self.arousal - other.arousal)**2)
    
    def weighted_distance_to(self, other: 'EmotionState', 
                           valence_weight: float = 1.0, 
                           arousal_weight: float = 1.0) -> float:
        """
        计算加权距离（可以强调某个维度）
        """
        valence_diff = (self.valence - other.valence) * valence_weight
        arousal_diff = (self.arousal - other.arousal) * arousal_weight
        return np.sqrt(valence_diff**2 + arousal_diff**2)

@dataclass 
class ISOStageConfig:
    """
    ISO阶段配置
    """
    stage: ISOStage
    duration_ratio: float  # 在整个过程中的时间占比
    target_state: EmotionState
    transition_params: Dict[str, float]
    music_requirements: Dict[str, any] = None
    
class ISOPrinciple:
    """
    ISO原则核心实现类
    
    基于2024年最新研究的ISO原则数字化实现：
    1. 同频阶段：匹配用户当前情绪状态
    2. 引导阶段：渐进式情绪过渡
    3. 巩固阶段：稳定在睡前理想状态
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or THEORY_CONFIG["iso_principle"]
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 从配置中提取参数
        self.stage_durations = self.config["stage_durations"]
        self.transition_params = self.config["transition_params"]
        
        # 目标睡前状态
        target_config = self.config["target_sleep_state"]
        self.target_sleep_state = EmotionState(
            valence=target_config["valence"],
            arousal=target_config["arousal"],
            confidence=target_config["confidence"]
        )
        
        self.logger.info(f"ISO原则初始化完成，目标状态: V={self.target_sleep_state.valence}, A={self.target_sleep_state.arousal}")
    
    def create_iso_stages(self, 
                         current_state: EmotionState,
                         total_duration: float = 1200.0) -> List[ISOStageConfig]:
        """
        创建ISO三阶段配置
        
        Args:
            current_state: 用户当前情绪状态
            total_duration: 总时长（秒），默认20分钟
            
        Returns:
            三个阶段的配置列表
        """
        stages = []
        
        # 阶段1：同频 - 匹配当前状态
        sync_duration = total_duration * self.stage_durations["synchronization"]
        sync_stage = ISOStageConfig(
            stage=ISOStage.SYNCHRONIZATION,
            duration_ratio=self.stage_durations["synchronization"],
            target_state=current_state,  # 保持当前状态
            transition_params={
                "smoothness_factor": 1.0,  # 完全匹配
                "adaptation_rate": 0.9,
                "stability_emphasis": True
            }
        )
        
        # 阶段2：引导 - 渐进过渡
        guidance_duration = total_duration * self.stage_durations["guidance"]
        # 计算中间过渡状态
        intermediate_state = self._calculate_intermediate_state(current_state, self.target_sleep_state)
        guidance_stage = ISOStageConfig(
            stage=ISOStage.GUIDANCE,
            duration_ratio=self.stage_durations["guidance"],
            target_state=intermediate_state,
            transition_params=self.transition_params.copy()
        )
        
        # 阶段3：巩固 - 稳定在目标状态
        consol_duration = total_duration * self.stage_durations["consolidation"]
        consol_stage = ISOStageConfig(
            stage=ISOStage.CONSOLIDATION,
            duration_ratio=self.stage_durations["consolidation"],
            target_state=self.target_sleep_state,
            transition_params={
                "smoothness_factor": 0.9,
                "adaptation_rate": 0.6,
                "sleep_emphasis": True
            }
        )
        
        stages = [sync_stage, guidance_stage, consol_stage]
        
        self.logger.info(f"创建ISO三阶段，总时长: {total_duration}秒")
        self.logger.info(f"同频: {sync_duration:.1f}s, 引导: {guidance_duration:.1f}s, 巩固: {consol_duration:.1f}s")
        
        return stages
    
    def generate_emotion_trajectory(self, 
                                  stages: List[ISOStageConfig],
                                  num_points: int = 100) -> List[EmotionState]:
        """
        生成平滑的情绪轨迹
        
        Args:
            stages: ISO阶段配置列表
            num_points: 轨迹点数量
            
        Returns:
            情绪轨迹点列表
        """
        trajectory = []
        
        for i, stage in enumerate(stages):
            stage_points = int(num_points * stage.duration_ratio)
            
            if i == 0:
                # 第一阶段：保持稳定
                stage_trajectory = self._generate_stable_trajectory(
                    stage.target_state, stage_points
                )
            else:
                # 后续阶段：平滑过渡
                prev_state = trajectory[-1] if trajectory else stages[0].target_state
                stage_trajectory = self._generate_transition_trajectory(
                    prev_state, stage.target_state, stage_points, 
                    stage.transition_params["smoothness_factor"]
                )
            
            trajectory.extend(stage_trajectory)
        
        self.logger.info(f"生成情绪轨迹，共{len(trajectory)}个点")
        return trajectory
    
    def _calculate_intermediate_state(self, 
                                    start: EmotionState, 
                                    end: EmotionState,
                                    weight: float = 0.6) -> EmotionState:
        """
        计算中间过渡状态
        
        Args:
            start: 起始状态
            end: 目标状态  
            weight: 偏向目标状态的权重
            
        Returns:
            中间状态
        """
        intermediate_valence = start.valence * (1 - weight) + end.valence * weight
        intermediate_arousal = start.arousal * (1 - weight) + end.arousal * weight
        
        return EmotionState(
            valence=intermediate_valence,
            arousal=intermediate_arousal,
            confidence=min(start.confidence, end.confidence)
        )
    
    def _generate_stable_trajectory(self, 
                                  state: EmotionState, 
                                  num_points: int) -> List[EmotionState]:
        """
        生成稳定状态轨迹（微小随机波动）
        """
        trajectory = []
        
        # 添加微小的自然波动
        noise_scale = 0.02  # 很小的噪声
        
        for i in range(num_points):
            # 添加微小随机波动，模拟自然情绪变化
            valence_noise = np.random.normal(0, noise_scale)
            arousal_noise = np.random.normal(0, noise_scale)
            
            new_state = EmotionState(
                valence=state.valence + valence_noise,
                arousal=state.arousal + arousal_noise,
                confidence=state.confidence,
                timestamp=i / num_points
            )
            
            trajectory.append(new_state)
        
        return trajectory
    
    def _generate_transition_trajectory(self, 
                                      start: EmotionState,
                                      end: EmotionState,
                                      num_points: int,
                                      smoothness: float = 0.8) -> List[EmotionState]:
        """
        生成平滑过渡轨迹
        
        Args:
            start: 起始状态
            end: 目标状态
            num_points: 轨迹点数
            smoothness: 平滑度（0-1）
        """
        trajectory = []
        
        # 使用平滑插值函数
        t_values = np.linspace(0, 1, num_points)
        
        # 应用平滑函数（sigmoid变换）
        if smoothness > 0:
            # S形曲线，更自然的过渡
            smooth_t = self._smooth_interpolation(t_values, smoothness)
        else:
            smooth_t = t_values
        
        for i, t in enumerate(smooth_t):
            # 线性插值
            valence = start.valence * (1 - t) + end.valence * t
            arousal = start.arousal * (1 - t) + end.arousal * t
            confidence = start.confidence * (1 - t) + end.confidence * t
            
            # 添加微小自然波动
            noise_scale = 0.01
            valence += np.random.normal(0, noise_scale)
            arousal += np.random.normal(0, noise_scale)
            
            new_state = EmotionState(
                valence=valence,
                arousal=arousal,
                confidence=confidence,
                timestamp=t
            )
            
            trajectory.append(new_state)
        
        return trajectory
    
    def _smooth_interpolation(self, t_values: np.ndarray, smoothness: float) -> np.ndarray:
        """
        平滑插值函数
        
        使用sigmoid函数创建S形过渡，更符合自然情绪变化
        """
        # 调整sigmoid的陡峭程度
        k = 6 * smoothness  # 控制曲线陡峭程度
        
        # Sigmoid函数：平滑S形过渡
        smooth_t = 1 / (1 + np.exp(-k * (t_values - 0.5)))
        
        # 归一化到[0,1]
        smooth_t = (smooth_t - smooth_t.min()) / (smooth_t.max() - smooth_t.min())
        
        return smooth_t
    
    def validate_trajectory(self, trajectory: List[EmotionState]) -> Dict[str, float]:
        """
        验证轨迹质量
        
        Returns:
            验证指标字典
        """
        if len(trajectory) < 2:
            return {"error": "轨迹点数不足"}
        
        # 计算轨迹平滑度
        smoothness = self._calculate_trajectory_smoothness(trajectory)
        
        # 计算目标达成度
        final_state = trajectory[-1]
        target_achievement = 1.0 - final_state.distance_to(self.target_sleep_state) / 2.0  # 最大距离为2
        
        # 计算过渡自然度（变化率方差）
        naturalness = self._calculate_transition_naturalness(trajectory)
        
        return {
            "smoothness": smoothness,
            "target_achievement": target_achievement,
            "naturalness": naturalness,
            "overall_quality": (smoothness + target_achievement + naturalness) / 3.0
        }
    
    def _calculate_trajectory_smoothness(self, trajectory: List[EmotionState]) -> float:
        """
        计算轨迹平滑度
        """
        if len(trajectory) < 3:
            return 1.0
        
        # 计算二阶导数的方差（曲率变化）
        valence_values = [state.valence for state in trajectory]
        arousal_values = [state.arousal for state in trajectory]
        
        valence_smoothness = self._calculate_smoothness_metric(valence_values)
        arousal_smoothness = self._calculate_smoothness_metric(arousal_values)
        
        return (valence_smoothness + arousal_smoothness) / 2.0
    
    def _calculate_smoothness_metric(self, values: List[float]) -> float:
        """
        计算单个维度的平滑度
        """
        if len(values) < 3:
            return 1.0
        
        # 计算一阶导数
        first_diff = np.diff(values)
        # 计算二阶导数
        second_diff = np.diff(first_diff)
        
        # 平滑度 = 1 / (1 + 二阶导数方差)
        smoothness = 1.0 / (1.0 + np.var(second_diff))
        
        return smoothness
    
    def _calculate_transition_naturalness(self, trajectory: List[EmotionState]) -> float:
        """
        计算过渡自然度
        """
        if len(trajectory) < 2:
            return 1.0
        
        # 计算相邻点之间的距离
        distances = []
        for i in range(1, len(trajectory)):
            dist = trajectory[i].distance_to(trajectory[i-1])
            distances.append(dist)
        
        # 自然度 = 1 / (1 + 距离方差)，距离变化越小越自然
        naturalness = 1.0 / (1.0 + np.var(distances))
        
        return naturalness
    
    def get_stage_requirements(self, stage: ISOStage) -> Dict[str, any]:
        """
        获取特定阶段的音乐要求
        
        Args:
            stage: ISO阶段
            
        Returns:
            音乐生成要求字典
        """
        base_requirements = {
            "genre": "ambient",
            "avoid_sudden_changes": True,
            "therapeutic_focus": True
        }
        
        if stage == ISOStage.SYNCHRONIZATION:
            return {
                **base_requirements,
                "emotional_matching": "exact",
                "stability_emphasis": True,
                "variation_level": "minimal"
            }
        
        elif stage == ISOStage.GUIDANCE:
            return {
                **base_requirements,
                "emotional_matching": "transitional",
                "transition_smoothness": "high",
                "variation_level": "moderate",
                "progression_type": "gradual"
            }
        
        elif stage == ISOStage.CONSOLIDATION:
            return {
                **base_requirements,
                "emotional_matching": "sleep_optimized",
                "stability_emphasis": True,
                "sleep_induction": True,
                "variation_level": "minimal"
            }
        
        return base_requirements
    
    def export_trajectory_data(self, 
                             trajectory: List[EmotionState], 
                             filepath: str) -> None:
        """
        导出轨迹数据用于分析
        """
        import json
        
        data = {
            "iso_principle_version": "1.0",
            "timestamp": np.datetime64('now').astype(str),
            "config": self.config,
            "trajectory": [
                {
                    "valence": state.valence,
                    "arousal": state.arousal,
                    "confidence": state.confidence,
                    "timestamp": state.timestamp
                } for state in trajectory
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"轨迹数据已导出到: {filepath}")

# 工具函数
def create_iso_planner(config_path: Optional[str] = None) -> ISOPrinciple:
    """
    创建ISO原则规划器的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ISOPrinciple实例
    """
    config = None
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get("iso_principle")
    
    return ISOPrinciple(config)

def emotion_state_from_dict(data: Dict[str, float]) -> EmotionState:
    """
    从字典创建EmotionState
    """
    return EmotionState(
        valence=data.get("valence", 0.0),
        arousal=data.get("arousal", 0.0),
        confidence=data.get("confidence", 1.0)
    )

# 示例使用
if __name__ == "__main__":
    # 创建ISO原则实例
    iso = ISOPrinciple()
    
    # 模拟用户当前情绪状态（焦虑、高唤醒）
    current_emotion = EmotionState(valence=-0.3, arousal=0.7, confidence=0.9)
    
    # 创建ISO三阶段
    stages = iso.create_iso_stages(current_emotion, total_duration=1200)  # 20分钟
    
    # 生成情绪轨迹
    trajectory = iso.generate_emotion_trajectory(stages, num_points=120)  # 每10秒一个点
    
    # 验证轨迹质量
    quality_metrics = iso.validate_trajectory(trajectory)
    
    print("ISO原则演示:")
    print(f"当前情绪: V={current_emotion.valence:.2f}, A={current_emotion.arousal:.2f}")
    print(f"目标情绪: V={iso.target_sleep_state.valence:.2f}, A={iso.target_sleep_state.arousal:.2f}")
    print(f"轨迹质量: {quality_metrics['overall_quality']:.3f}")
    print(f"目标达成度: {quality_metrics['target_achievement']:.3f}")
