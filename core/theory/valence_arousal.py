"""
Valence-Arousal情绪模型
基于学术研究的二维情绪空间模型实现

Valence-Arousal模型是情绪心理学中广泛使用的二维情绪表示模型：
- Valence (效价): 情绪的积极性/消极性 (-1 到 +1)
- Arousal (唤醒度): 情绪的激活程度 (-1 到 +1)

参考文献:
- Russell, J. A. (1980). A circumplex model of affect.
- Music Emotion Maps in Arousal-Valence Space (2024)
- Audio features dedicated to the detection and tracking of arousal and valence (2024)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import yaml
import json
from collections import defaultdict

# 导入ISO原则中的EmotionState
from .iso_principle import EmotionState

# 配置加载
try:
    with open(Path(__file__).parent.parent.parent / "configs" / "theory_params.yaml", "r", encoding="utf-8") as f:
        THEORY_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    THEORY_CONFIG = {
        "valence_arousal": {
            "emotion_regions": {
                "high_valence_high_arousal": [0.5, 1.0, 0.5, 1.0],
                "high_valence_low_arousal": [0.5, 1.0, -1.0, 0.0],
                "low_valence_high_arousal": [-1.0, 0.0, 0.5, 1.0],
                "low_valence_low_arousal": [-1.0, 0.0, -1.0, 0.0]
            },
            "fusion_weights": {
                "text_emotion": 0.6,
                "speech_emotion": 0.4,
                "confidence_threshold": 0.7
            }
        }
    }

logger = logging.getLogger(__name__)

class EmotionQuadrant(Enum):
    """
    情绪四象限分类
    """
    HIGH_VALENCE_HIGH_AROUSAL = "high_valence_high_arousal"    # 兴奋、快乐
    HIGH_VALENCE_LOW_AROUSAL = "high_valence_low_arousal"      # 放松、宁静
    LOW_VALENCE_HIGH_AROUSAL = "low_valence_high_arousal"      # 焦虑、愤怒
    LOW_VALENCE_LOW_AROUSAL = "low_valence_low_arousal"        # 悲伤、抑郁

class BasicEmotion(Enum):
    """
    基本情绪类型（映射到V-A空间）
    """
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CALM = "calm"
    EXCITED = "excited"
    BORED = "bored"
    CONTENT = "content"

@dataclass
class EmotionMapping:
    """
    情绪映射定义
    """
    basic_emotion: BasicEmotion
    valence: float
    arousal: float
    confidence: float = 1.0
    synonyms: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_emotion_state(self) -> EmotionState:
        """转换为EmotionState"""
        return EmotionState(
            valence=self.valence,
            arousal=self.arousal,
            confidence=self.confidence
        )

@dataclass
class MultimodalEmotion:
    """
    多模态情绪表示
    """
    text_emotion: Optional[EmotionState] = None
    speech_emotion: Optional[EmotionState] = None
    fused_emotion: Optional[EmotionState] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ValenceArousalModel:
    """
    Valence-Arousal情绪模型核心实现
    
    功能包括：
    1. 情绪分类和映射
    2. 多模态情绪融合
    3. 情绪距离和相似度计算
    4. 情绪区域分析
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or THEORY_CONFIG["valence_arousal"]
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化情绪映射
        self.emotion_mappings = self._create_emotion_mappings()
        
        # 可能从配置中加载区域定义
        self.emotion_regions = self.config["emotion_regions"]
        
        # 融合权重
        self.fusion_weights = self.config["fusion_weights"]
        
        self.logger.info("Valence-Arousal模型初始化完成")
    
    def _create_emotion_mappings(self) -> Dict[BasicEmotion, EmotionMapping]:
        """
        创建基本情绪到V-A空间的映射
        
        基于学术研究的标准映射值
        """
        mappings = {
            BasicEmotion.HAPPY: EmotionMapping(
                basic_emotion=BasicEmotion.HAPPY,
                valence=0.8, arousal=0.4,
                synonyms=["joyful", "cheerful", "delighted", "pleased"],
                description="快乐、愛快的状态"
            ),
            BasicEmotion.SAD: EmotionMapping(
                basic_emotion=BasicEmotion.SAD,
                valence=-0.7, arousal=-0.3,
                synonyms=["depressed", "melancholy", "gloomy", "sorrowful"],
                description="悲伤、伤心的状态"
            ),
            BasicEmotion.ANGRY: EmotionMapping(
                basic_emotion=BasicEmotion.ANGRY,
                valence=-0.6, arousal=0.7,
                synonyms=["furious", "irritated", "mad", "enraged"],
                description="愤怒、烦躁的状态"
            ),
            BasicEmotion.FEAR: EmotionMapping(
                basic_emotion=BasicEmotion.FEAR,
                valence=-0.5, arousal=0.6,
                synonyms=["afraid", "anxious", "worried", "scared"],
                description="恐惧、焦虑的状态"
            ),
            BasicEmotion.SURPRISED: EmotionMapping(
                basic_emotion=BasicEmotion.SURPRISED,
                valence=0.1, arousal=0.8,
                synonyms=["astonished", "amazed", "shocked"],
                description="惊讶、意外的状态"
            ),
            BasicEmotion.DISGUSTED: EmotionMapping(
                basic_emotion=BasicEmotion.DISGUSTED,
                valence=-0.8, arousal=0.2,
                synonyms=["repulsed", "revolted", "sickened"],
                description="反感、厌恶的状态"
            ),
            BasicEmotion.CALM: EmotionMapping(
                basic_emotion=BasicEmotion.CALM,
                valence=0.3, arousal=-0.6,
                synonyms=["peaceful", "serene", "relaxed", "tranquil"],
                description="平静、放松的状态"
            ),
            BasicEmotion.EXCITED: EmotionMapping(
                basic_emotion=BasicEmotion.EXCITED,
                valence=0.7, arousal=0.8,
                synonyms=["thrilled", "energetic", "enthusiastic"],
                description="兴奋、激动的状态"
            ),
            BasicEmotion.BORED: EmotionMapping(
                basic_emotion=BasicEmotion.BORED,
                valence=-0.2, arousal=-0.7,
                synonyms=["uninterested", "listless", "weary"],
                description="无聊、厦倦的状态"
            ),
            BasicEmotion.CONTENT: EmotionMapping(
                basic_emotion=BasicEmotion.CONTENT,
                valence=0.5, arousal=-0.2,
                synonyms=["satisfied", "pleased", "comfortable"],
                description="满足、安逸的状态"
            )
        }
        
        return mappings
    
    def classify_emotion_by_va(self, 
                              valence: float, 
                              arousal: float) -> Tuple[EmotionQuadrant, BasicEmotion]:
        """
        根据V-A值分类情绪
        
        Args:
            valence: 效价值 (-1 到 1)
            arousal: 唤醒度 (-1 到 1)
            
        Returns:
            情绪四象限和最相似的基本情绪
        """
        # 确定四象限
        if valence >= 0 and arousal >= 0:
            quadrant = EmotionQuadrant.HIGH_VALENCE_HIGH_AROUSAL
        elif valence >= 0 and arousal < 0:
            quadrant = EmotionQuadrant.HIGH_VALENCE_LOW_AROUSAL
        elif valence < 0 and arousal >= 0:
            quadrant = EmotionQuadrant.LOW_VALENCE_HIGH_AROUSAL
        else:
            quadrant = EmotionQuadrant.LOW_VALENCE_LOW_AROUSAL
        
        # 找到最相似的基本情绪
        target_state = EmotionState(valence=valence, arousal=arousal)
        closest_emotion = self.find_closest_basic_emotion(target_state)
        
        return quadrant, closest_emotion
    
    def find_closest_basic_emotion(self, emotion_state: EmotionState) -> BasicEmotion:
        """
        找到最相似的基本情绪
        """
        min_distance = float('inf')
        closest_emotion = BasicEmotion.CALM  # 默认值
        
        for emotion, mapping in self.emotion_mappings.items():
            mapped_state = mapping.to_emotion_state()
            distance = emotion_state.distance_to(mapped_state)
            
            if distance < min_distance:
                min_distance = distance
                closest_emotion = emotion
        
        return closest_emotion
    
    def get_emotion_region(self, emotion_state: EmotionState) -> str:
        """
        获取情绪所在区域
        """
        v, a = emotion_state.valence, emotion_state.arousal
        
        for region_name, bounds in self.emotion_regions.items():
            v_min, v_max, a_min, a_max = bounds
            if v_min <= v <= v_max and a_min <= a <= a_max:
                return region_name
        
        return "unknown_region"
    
    def fuse_multimodal_emotions(self, 
                                text_emotion: Optional[EmotionState],
                                speech_emotion: Optional[EmotionState],
                                custom_weights: Optional[Dict[str, float]] = None) -> MultimodalEmotion:
        """
        融合多模态情绪
        
        Args:
            text_emotion: 文本情绪
            speech_emotion: 语音情绪
            custom_weights: 自定义融合权重
            
        Returns:
            融合后的多模态情绪
        """
        weights = custom_weights or self.fusion_weights
        result = MultimodalEmotion(
            text_emotion=text_emotion,
            speech_emotion=speech_emotion
        )
        
        # 记录置信度
        if text_emotion:
            result.confidence_scores["text"] = text_emotion.confidence
        if speech_emotion:
            result.confidence_scores["speech"] = speech_emotion.confidence
        
        # 检查是否有足够的数据进行融合
        if not text_emotion and not speech_emotion:
            self.logger.warning("没有可用的情绪数据进行融合")
            return result
        
        # 单模态情况
        if text_emotion and not speech_emotion:
            result.fused_emotion = text_emotion
            result.confidence_scores["fusion"] = text_emotion.confidence
            return result
        
        if speech_emotion and not text_emotion:
            result.fused_emotion = speech_emotion
            result.confidence_scores["fusion"] = speech_emotion.confidence
            return result
        
        # 多模态融合
        result.fused_emotion = self._weighted_emotion_fusion(
            text_emotion, speech_emotion, weights
        )
        
        # 计算融合置信度
        fusion_confidence = self._calculate_fusion_confidence(
            text_emotion, speech_emotion, weights
        )
        result.confidence_scores["fusion"] = fusion_confidence
        
        # 记录融合元数据
        result.metadata.update({
            "fusion_method": "weighted_average",
            "weights_used": weights,
            "modalities_count": 2,
            "quality_score": self._assess_fusion_quality(result)
        })
        
        self.logger.info(f"多模态情绪融合完成，置信度: {fusion_confidence:.3f}")
        
        return result
    
    def _weighted_emotion_fusion(self, 
                               text_emotion: EmotionState,
                               speech_emotion: EmotionState,
                               weights: Dict[str, float]) -> EmotionState:
        """
        加权平均融合情绪
        """
        text_weight = weights["text_emotion"]
        speech_weight = weights["speech_emotion"]
        
        # 根据置信度调整权重
        confidence_adjusted_text_weight = text_weight * text_emotion.confidence
        confidence_adjusted_speech_weight = speech_weight * speech_emotion.confidence
        
        # 归一化权重
        total_weight = confidence_adjusted_text_weight + confidence_adjusted_speech_weight
        if total_weight == 0:
            total_weight = 1.0  # 避免除零
        
        norm_text_weight = confidence_adjusted_text_weight / total_weight
        norm_speech_weight = confidence_adjusted_speech_weight / total_weight
        
        # 加权融合
        fused_valence = (text_emotion.valence * norm_text_weight + 
                        speech_emotion.valence * norm_speech_weight)
        fused_arousal = (text_emotion.arousal * norm_text_weight + 
                        speech_emotion.arousal * norm_speech_weight)
        
        # 融合置信度（加权平均）
        fused_confidence = (text_emotion.confidence * norm_text_weight + 
                          speech_emotion.confidence * norm_speech_weight)
        
        return EmotionState(
            valence=fused_valence,
            arousal=fused_arousal,
            confidence=fused_confidence
        )
    
    def _calculate_fusion_confidence(self, 
                                   text_emotion: EmotionState,
                                   speech_emotion: EmotionState,
                                   weights: Dict[str, float]) -> float:
        """
        计算融合置信度
        
        考虑因素：
        1. 单个模态的置信度
        2. 模态之间的一致性
        3. 融合权重的平衡性
        """
        # 基础置信度（加权平均）
        base_confidence = (text_emotion.confidence * weights["text_emotion"] + 
                          speech_emotion.confidence * weights["speech_emotion"])
        
        # 一致性加分（模态之间距离越小，置信度越高）
        emotion_distance = text_emotion.distance_to(speech_emotion)
        consistency_bonus = max(0, 1.0 - emotion_distance / 2.0)  # 最大距离为2
        
        # 综合置信度
        final_confidence = base_confidence * (0.7 + 0.3 * consistency_bonus)
        
        return min(1.0, final_confidence)  # 限制在[0,1]范围
    
    def _assess_fusion_quality(self, multimodal_emotion: MultimodalEmotion) -> float:
        """
        评估融合质量
        """
        if not multimodal_emotion.fused_emotion:
            return 0.0
        
        quality_factors = []
        
        # 置信度因子
        confidence_score = multimodal_emotion.confidence_scores.get("fusion", 0.0)
        quality_factors.append(confidence_score)
        
        # 数据完整性因子
        modality_count = sum([
            1 if multimodal_emotion.text_emotion else 0,
            1 if multimodal_emotion.speech_emotion else 0
        ])
        completeness_score = modality_count / 2.0  # 最大为2个模态
        quality_factors.append(completeness_score)
        
        # 一致性因子（如果有两个模态）
        if (multimodal_emotion.text_emotion and multimodal_emotion.speech_emotion):
            distance = multimodal_emotion.text_emotion.distance_to(multimodal_emotion.speech_emotion)
            consistency_score = max(0, 1.0 - distance / 2.0)
            quality_factors.append(consistency_score)
        
        return np.mean(quality_factors)
    
    def get_sleep_conducive_target(self, 
                                  current_emotion: EmotionState,
                                  sleep_preference: str = "gentle") -> EmotionState:
        """
        根据当前情绪计算适合的睡前目标情绪
        
        Args:
            current_emotion: 当前情绪状态
            sleep_preference: 睡眠偏好 ("gentle", "deep", "natural")
            
        Returns:
            目标睡前情绪状态
        """
        # 基本睡前目标：低唤醒、轻微积极
        base_sleep_target = {
            "gentle": EmotionState(valence=0.2, arousal=-0.6, confidence=0.9),
            "deep": EmotionState(valence=0.0, arousal=-0.8, confidence=0.9),
            "natural": EmotionState(valence=0.1, arousal=-0.5, confidence=0.9)
        }
        
        target = base_sleep_target.get(sleep_preference, base_sleep_target["gentle"])
        
        # 根据当前情绪调整（避免过大跃变）
        current_distance = current_emotion.distance_to(target)
        
        if current_distance > 1.5:  # 距离过远，需要中间过渡
            # 创建中间目标
            intermediate_valence = current_emotion.valence * 0.4 + target.valence * 0.6
            intermediate_arousal = current_emotion.arousal * 0.4 + target.arousal * 0.6
            
            target = EmotionState(
                valence=intermediate_valence,
                arousal=intermediate_arousal,
                confidence=target.confidence
            )
        
        self.logger.info(f"计算睡前目标：{sleep_preference} -> V={target.valence:.2f}, A={target.arousal:.2f}")
        
        return target
    
    def analyze_emotion_trajectory(self, 
                                 emotions: List[EmotionState]) -> Dict[str, Any]:
        """
        分析情绪轨迹
        
        Returns:
            轨迹分析结果
        """
        if len(emotions) < 2:
            return {"error": "需要至少两个情绪点"}
        
        # 提取数值
        valences = [e.valence for e in emotions]
        arousals = [e.arousal for e in emotions]
        confidences = [e.confidence for e in emotions]
        
        # 基本统计
        analysis = {
            "trajectory_length": len(emotions),
            "valence_stats": {
                "start": valences[0],
                "end": valences[-1],
                "mean": np.mean(valences),
                "std": np.std(valences),
                "change": valences[-1] - valences[0]
            },
            "arousal_stats": {
                "start": arousals[0],
                "end": arousals[-1],
                "mean": np.mean(arousals),
                "std": np.std(arousals),
                "change": arousals[-1] - arousals[0]
            },
            "confidence_stats": {
                "mean": np.mean(confidences),
                "min": np.min(confidences),
                "std": np.std(confidences)
            }
        }
        
        # 轨迹特征
        analysis["trajectory_features"] = {
            "total_distance": self._calculate_total_distance(emotions),
            "smoothness": self._calculate_trajectory_smoothness_va(emotions),
            "directionality": self._calculate_directionality(emotions),
            "stability": self._calculate_stability(emotions)
        }
        
        # 情绪区域分布
        quadrant_counts = defaultdict(int)
        for emotion in emotions:
            quadrant, _ = self.classify_emotion_by_va(emotion.valence, emotion.arousal)
            quadrant_counts[quadrant.value] += 1
        
        analysis["quadrant_distribution"] = dict(quadrant_counts)
        
        return analysis
    
    def _calculate_total_distance(self, emotions: List[EmotionState]) -> float:
        """计算轨迹总距离"""
        total_distance = 0.0
        for i in range(1, len(emotions)):
            total_distance += emotions[i].distance_to(emotions[i-1])
        return total_distance
    
    def _calculate_trajectory_smoothness_va(self, emotions: List[EmotionState]) -> float:
        """计算V-A轨迹平滑度"""
        if len(emotions) < 3:
            return 1.0
        
        valences = [e.valence for e in emotions]
        arousals = [e.arousal for e in emotions]
        
        # 计算二阶差分
        v_second_diff = np.diff(np.diff(valences))
        a_second_diff = np.diff(np.diff(arousals))
        
        # 平滑度指标
        v_smoothness = 1.0 / (1.0 + np.var(v_second_diff))
        a_smoothness = 1.0 / (1.0 + np.var(a_second_diff))
        
        return (v_smoothness + a_smoothness) / 2.0
    
    def _calculate_directionality(self, emotions: List[EmotionState]) -> float:
        """计算轨迹方向性"""
        if len(emotions) < 2:
            return 0.0
        
        start = emotions[0]
        end = emotions[-1]
        
        # 直线距离 vs 实际距离
        direct_distance = start.distance_to(end)
        actual_distance = self._calculate_total_distance(emotions)
        
        if actual_distance == 0:
            return 1.0
        
        directionality = direct_distance / actual_distance
        return min(1.0, directionality)
    
    def _calculate_stability(self, emotions: List[EmotionState]) -> float:
        """计算轨迹稳定性"""
        if len(emotions) < 2:
            return 1.0
        
        distances = []
        for i in range(1, len(emotions)):
            distances.append(emotions[i].distance_to(emotions[i-1]))
        
        # 稳定性 = 1 / (1 + 距离方差)
        stability = 1.0 / (1.0 + np.var(distances))
        return stability
    
    def export_emotion_analysis(self, 
                              emotions: List[EmotionState],
                              analysis: Dict[str, Any],
                              filepath: str) -> None:
        """
        导出情绪分析结果
        """
        export_data = {
            "valence_arousal_model_version": "1.0",
            "timestamp": np.datetime64('now').astype(str),
            "model_config": self.config,
            "emotions_data": [
                {
                    "valence": e.valence,
                    "arousal": e.arousal,
                    "confidence": e.confidence,
                    "timestamp": e.timestamp
                } for e in emotions
            ],
            "analysis_results": analysis
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"情绪分析结果已导出到: {filepath}")

# 工具函数
def create_va_model(config_path: Optional[str] = None) -> ValenceArousalModel:
    """
    创建Valence-Arousal模型的便捷函数
    """
    config = None
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get("valence_arousal")
    
    return ValenceArousalModel(config)

def map_text_to_emotion(text: str, 
                       emotion_keywords: Dict[str, List[str]] = None) -> BasicEmotion:
    """
    简单的基于关键词的文本情绪映射（作为示例）
    
    实际使用中应该由更复杂的NLP模型来实现
    """
    if not emotion_keywords:
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "cheerful", "delighted"],
            "sad": ["sad", "depressed", "gloomy", "melancholy", "sorrow"],
            "angry": ["angry", "mad", "furious", "irritated", "rage"],
            "fear": ["afraid", "scared", "anxious", "worried", "fear"],
            "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil"]
        }
    
    text_lower = text.lower()
    
    for emotion_name, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return BasicEmotion(emotion_name)
    
    return BasicEmotion.CALM  # 默认返回平静

# 示例使用
if __name__ == "__main__":
    # 创建V-A模型
    va_model = ValenceArousalModel()
    
    # 模拟多模态情绪数据
    text_emotion = EmotionState(valence=-0.2, arousal=0.6, confidence=0.8)  # 焦虑
    speech_emotion = EmotionState(valence=-0.3, arousal=0.7, confidence=0.9)  # 焦虑
    
    # 融合情绪
    multimodal_result = va_model.fuse_multimodal_emotions(text_emotion, speech_emotion)
    
    # 分类情绪
    quadrant, basic_emotion = va_model.classify_emotion_by_va(
        multimodal_result.fused_emotion.valence,
        multimodal_result.fused_emotion.arousal
    )
    
    # 计算睡前目标
    sleep_target = va_model.get_sleep_conducive_target(
        multimodal_result.fused_emotion, "gentle"
    )
    
    print("Valence-Arousal模型演示:")
    print(f"融合情绪: V={multimodal_result.fused_emotion.valence:.2f}, A={multimodal_result.fused_emotion.arousal:.2f}")
    print(f"情绪分类: {quadrant.value} - {basic_emotion.value}")
    print(f"睡前目标: V={sleep_target.valence:.2f}, A={sleep_target.arousal:.2f}")
    print(f"融合置信度: {multimodal_result.confidence_scores['fusion']:.3f}")
