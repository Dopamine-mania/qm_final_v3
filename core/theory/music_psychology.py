"""
音乐心理学模块
基于2024年最新音乐心理学研究的音乐-情绪映射实现

核心理论：
1. BPM与情绪调节：基于EEG研究的节奏-情绪映射
2. 调性与效价：大调/小调对情绪的影响
3. 和声与心理影响：谐波共鸣的生理效应
4. 音色与情绪联想：不同乐器的心理效应

参考文献：
- Music tempo modulates emotional states as revealed through EEG insights (2024)
- Musical Key Characteristics & Emotions (2024)
- Effects of Musical Tempo on Musicians' and Non-musicians' Emotional Experience (2024)
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
from .valence_arousal import EmotionState, BasicEmotion
from .sleep_physiology import PhysiologicalState, MusicTherapyParameters

# 配置加载
try:
    with open(Path(__file__).parent.parent.parent / "configs" / "theory_params.yaml", "r", encoding="utf-8") as f:
        THEORY_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    THEORY_CONFIG = {
        "music_psychology": {
            "tempo_emotion_mapping": {
                "relaxation": [40, 60],
                "transition": [60, 90],
                "low_activation": [90, 106],
                "medium_activation": [106, 120]
            },
            "key_emotion_mapping": {
                "positive_major": ["C_major", "G_major", "D_major", "A_major"],
                "neutral_major": ["F_major", "Bb_major"],
                "contemplative_minor": ["A_minor", "E_minor", "B_minor"],
                "deep_minor": ["F#_minor", "C#_minor", "Bb_minor"]
            }
        }
    }

logger = logging.getLogger(__name__)

class MusicalKey(Enum):
    """
    音乐调性
    """
    # 大调
    C_MAJOR = "C_major"
    D_MAJOR = "D_major"
    E_MAJOR = "E_major"
    F_MAJOR = "F_major"
    G_MAJOR = "G_major"
    A_MAJOR = "A_major"
    B_MAJOR = "B_major"
    
    # 小调
    A_MINOR = "A_minor"
    B_MINOR = "B_minor"
    C_MINOR = "C_minor"
    D_MINOR = "D_minor"
    E_MINOR = "E_minor"
    F_MINOR = "F_minor"
    G_MINOR = "G_minor"
    FS_MINOR = "F#_minor"
    CS_MINOR = "C#_minor"
    BB_MINOR = "Bb_minor"

class TempoCategory(Enum):
    """
    节奏分类
    """
    GRAVE = "grave"           # 40-60 BPM - 缓慢、庄重
    LARGO = "largo"           # 45-60 BPM - 宽广、庄严
    ADAGIO = "adagio"         # 60-76 BPM - 慢板、深情
    ANDANTE = "andante"       # 76-108 BPM - 行板、安稳
    MODERATO = "moderato"     # 108-120 BPM - 中板、适中
    ALLEGRO = "allegro"       # 120-168 BPM - 快板、活泼
    PRESTO = "presto"         # 168-200 BPM - 急板、激烈

class InstrumentFamily(Enum):
    """
    乐器家族
    """
    STRINGS = "strings"       # 弦乐器
    WINDS = "winds"           # 管乐器
    BRASS = "brass"           # 铜管乐器
    PERCUSSION = "percussion" # 打击乐器
    KEYBOARD = "keyboard"     # 键盘乐器
    VOICE = "voice"           # 人声
    ELECTRONIC = "electronic" # 电子乐器
    AMBIENT = "ambient"       # 环境音效

@dataclass
class MusicalCharacteristics:
    """
    音乐特征
    """
    tempo_bpm: float                    # 节奏 (BPM)
    key: MusicalKey                     # 调性
    time_signature: Tuple[int, int]     # 拍号 (e.g., (4, 4))
    dynamics: str                       # 力度记号 (pp, p, mp, mf, f, ff)
    primary_instruments: List[InstrumentFamily]  # 主要乐器
    harmonic_complexity: float          # 和声复杂度 (0-1)
    melodic_range: Tuple[float, float]  # 旋律音域 (Hz)
    texture: str                        # 织体 (monophonic, homophonic, polyphonic)
    
    def get_tempo_category(self) -> TempoCategory:
        """获取节奏分类"""
        if self.tempo_bpm <= 60:
            return TempoCategory.GRAVE if self.tempo_bpm <= 50 else TempoCategory.LARGO
        elif self.tempo_bpm <= 76:
            return TempoCategory.ADAGIO
        elif self.tempo_bpm <= 108:
            return TempoCategory.ANDANTE
        elif self.tempo_bpm <= 120:
            return TempoCategory.MODERATO
        elif self.tempo_bpm <= 168:
            return TempoCategory.ALLEGRO
        else:
            return TempoCategory.PRESTO
    
    def is_major_key(self) -> bool:
        """判断是否为大调"""
        return "major" in self.key.value
    
    def get_emotional_valence_tendency(self) -> float:
        """根据调性获取情绪倾向"""
        if self.is_major_key():
            return 0.6  # 大调倾向积极
        else:
            return -0.4  # 小调倾向消极

@dataclass
class PsychoacousticProfile:
    """
    心理声学概谱
    """
    roughness: float           # 粗糙度 (0-1)
    brightness: float          # 明亮度 (0-1)
    warmth: float             # 温暖度 (0-1)
    spaciousness: float       # 空间感 (0-1)
    attack_sharpness: float   # 攻击锐度 (0-1)
    spectral_centroid: float  # 频谱质心 (Hz)
    spectral_rolloff: float   # 频谱卷降 (Hz)
    zero_crossing_rate: float # 过零率
    
    def get_overall_pleasantness(self) -> float:
        """计算整体愉悦度"""
        # 基于心理声学理论，计算整体的感知愉悦度
        pleasantness = (
            (1.0 - self.roughness) * 0.3 +      # 低粗糙度 = 高愉悦度
            self.warmth * 0.25 +                # 温暖度贡献
            (1.0 - self.attack_sharpness) * 0.2 + # 柔和的攻击
            self.spaciousness * 0.15 +          # 适度的空间感
            min(1.0, self.brightness) * 0.1     # 适度的明亮度
        )
        return max(0, min(1, pleasantness))

class MusicPsychologyModel:
    """
    音乐心理学模型
    
    结合最新的音乐心理学和神经科学研究，
    实现情绪到音乐特征的科学映射。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or THEORY_CONFIG["music_psychology"]
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 节奏-情绪映射
        self.tempo_emotion_mapping = self.config["tempo_emotion_mapping"]
        
        # 调性-情绪映射
        self.key_emotion_mapping = self.config["key_emotion_mapping"]
        
        # 初始化调性特征库
        self.key_characteristics = self._create_key_characteristics()
        
        # 初始化乐器心理效应库
        self.instrument_psychology = self._create_instrument_psychology()
        
        self.logger.info("音乐心理学模型初始化完成")
    
    def _create_key_characteristics(self) -> Dict[MusicalKey, Dict[str, float]]:
        """
        创建调性特征库
        
        基于历史上各调性的情感特征和现代研究
        """
        characteristics = {
            # 大调 - 整体较为积极
            MusicalKey.C_MAJOR: {
                "valence": 0.7, "arousal": 0.3, "brightness": 0.8, "warmth": 0.6,
                "description": "纯洁、单纯、广阔"
            },
            MusicalKey.G_MAJOR: {
                "valence": 0.8, "arousal": 0.4, "brightness": 0.9, "warmth": 0.7,
                "description": "明亮、快乐、活泼"
            },
            MusicalKey.D_MAJOR: {
                "valence": 0.8, "arousal": 0.5, "brightness": 0.9, "warmth": 0.6,
                "description": "辉煌、胜利、庄严"
            },
            MusicalKey.A_MAJOR: {
                "valence": 0.6, "arousal": 0.4, "brightness": 0.7, "warmth": 0.8,
                "description": "温暖、亲密、人性化"
            },
            MusicalKey.F_MAJOR: {
                "valence": 0.5, "arousal": 0.2, "brightness": 0.6, "warmth": 0.8,
                "description": "宁静、田园、温和"
            },
            
            # 小调 - 整体较为内省
            MusicalKey.A_MINOR: {
                "valence": -0.3, "arousal": 0.1, "brightness": 0.4, "warmth": 0.6,
                "description": "纯眉、内省、简朴"
            },
            MusicalKey.E_MINOR: {
                "valence": -0.4, "arousal": 0.2, "brightness": 0.3, "warmth": 0.5,
                "description": "深沉、悲伤、沉思"
            },
            MusicalKey.B_MINOR: {
                "valence": -0.5, "arousal": 0.3, "brightness": 0.3, "warmth": 0.4,
                "description": "孤独、消沉、神秘"
            },
            MusicalKey.FS_MINOR: {
                "valence": -0.6, "arousal": 0.4, "brightness": 0.2, "warmth": 0.3,
                "description": "阴暗、追恼、痛苦"
            },
            MusicalKey.D_MINOR: {
                "valence": -0.7, "arousal": 0.1, "brightness": 0.2, "warmth": 0.4,
                "description": "最悲伤的调性、深度抑郁"
            }
        }
        
        return characteristics
    
    def _create_instrument_psychology(self) -> Dict[InstrumentFamily, Dict[str, float]]:
        """
        创建乐器心理效应库
        """
        psychology = {
            InstrumentFamily.STRINGS: {
                "warmth": 0.8, "intimacy": 0.9, "expressiveness": 0.9,
                "soothing_effect": 0.8, "sleep_conducive": 0.7
            },
            InstrumentFamily.KEYBOARD: {
                "clarity": 0.9, "precision": 0.8, "versatility": 0.9,
                "soothing_effect": 0.7, "sleep_conducive": 0.8
            },
            InstrumentFamily.WINDS: {
                "breath_connection": 0.9, "flowing": 0.8, "organic": 0.8,
                "soothing_effect": 0.6, "sleep_conducive": 0.6
            },
            InstrumentFamily.AMBIENT: {
                "spaciousness": 0.9, "immersion": 0.9, "relaxation": 0.9,
                "soothing_effect": 0.9, "sleep_conducive": 0.9
            },
            InstrumentFamily.VOICE: {
                "human_connection": 1.0, "emotional_direct": 0.9, "intimacy": 0.8,
                "soothing_effect": 0.7, "sleep_conducive": 0.5  # 歌词可能干扰睡眠
            },
            InstrumentFamily.PERCUSSION: {
                "rhythm_emphasis": 1.0, "energy": 0.8, "grounding": 0.6,
                "soothing_effect": 0.3, "sleep_conducive": 0.2
            },
            InstrumentFamily.BRASS: {
                "power": 0.9, "brilliance": 0.8, "announcement": 0.9,
                "soothing_effect": 0.2, "sleep_conducive": 0.1
            }
        }
        
        return psychology
    
    def map_emotion_to_tempo(self, emotion_state: EmotionState) -> Tuple[float, TempoCategory]:
        """
        根据情绪状态映射到适合的节奏
        
        基于2024年EEG研究：节奏对情绪的调节作用
        """
        # 基于 arousal 水平决定基础节奏范围
        if emotion_state.arousal > 0.5:  # 高判醒
            tempo_range = self.tempo_emotion_mapping["medium_activation"]
        elif emotion_state.arousal > 0.0:  # 中判醒
            tempo_range = self.tempo_emotion_mapping["low_activation"]
        elif emotion_state.arousal > -0.5:  # 低判醒
            tempo_range = self.tempo_emotion_mapping["transition"]
        else:  # 极低判醒
            tempo_range = self.tempo_emotion_mapping["relaxation"]
        
        # 根据 valence 微调
        valence_adjustment = emotion_state.valence * 5  # -5 到 +5 BPM
        
        # 计算目标BPM
        base_tempo = np.mean(tempo_range)
        target_tempo = base_tempo + valence_adjustment
        
        # 限制在合理范围内
        target_tempo = max(tempo_range[0], min(tempo_range[1], target_tempo))
        
        # 确定节奏分类
        tempo_category = self._get_tempo_category_from_bpm(target_tempo)
        
        self.logger.info(f"情绪到节奏映射: V={emotion_state.valence:.2f}, A={emotion_state.arousal:.2f} -> {target_tempo:.1f} BPM ({tempo_category.value})")
        
        return target_tempo, tempo_category
    
    def _get_tempo_category_from_bpm(self, bpm: float) -> TempoCategory:
        """根据BPM获取节奏分类"""
        if bpm <= 60:
            return TempoCategory.GRAVE
        elif bpm <= 76:
            return TempoCategory.ADAGIO
        elif bpm <= 108:
            return TempoCategory.ANDANTE
        elif bpm <= 120:
            return TempoCategory.MODERATO
        elif bpm <= 168:
            return TempoCategory.ALLEGRO
        else:
            return TempoCategory.PRESTO
    
    def select_optimal_key(self, 
                          emotion_state: EmotionState,
                          target_emotion: Optional[EmotionState] = None) -> MusicalKey:
        """
        选择最佳调性
        
        Args:
            emotion_state: 当前情绪状态
            target_emotion: 目标情绪状态（可选）
            
        Returns:
            最优调性
        """
        # 如果有目标情绪，优先考虑目标
        reference_emotion = target_emotion if target_emotion else emotion_state
        
        best_key = MusicalKey.C_MAJOR  # 默认值
        min_distance = float('inf')
        
        for key, characteristics in self.key_characteristics.items():
            # 计算调性特征与目标情绪的匹配度
            key_valence = characteristics["valence"]
            key_arousal = characteristics["arousal"]
            
            # 欧式距离
            distance = math.sqrt(
                (reference_emotion.valence - key_valence) ** 2 +
                (reference_emotion.arousal - key_arousal) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                best_key = key
        
        self.logger.info(f"选择调性: {best_key.value} (匹配度: {1.0 - min_distance/2.0:.3f})")
        
        return best_key
    
    def recommend_instruments(self, 
                            target_emotion: EmotionState,
                            sleep_context: bool = True) -> List[InstrumentFamily]:
        """
        推荐适合的乐器
        
        Args:
            target_emotion: 目标情绪
            sleep_context: 是否为睡眠场景
            
        Returns:
            推荐的乐器家族列表
        """
        recommendations = []
        
        # 睡眠场景下的乐器选择
        if sleep_context:
            # 按睡眠适合度排序
            sleep_friendly = [
                (InstrumentFamily.AMBIENT, 0.9),
                (InstrumentFamily.KEYBOARD, 0.8),
                (InstrumentFamily.STRINGS, 0.7),
                (InstrumentFamily.WINDS, 0.6)
            ]
            
            # 根据情绪状态调整权重
            for instrument, base_score in sleep_friendly:
                psychology = self.instrument_psychology[instrument]
                
                # 情绪适配性调整
                if target_emotion.valence < -0.3:  # 消极情绪
                    score = base_score * psychology["soothing_effect"]
                elif target_emotion.arousal > 0.3:  # 高判醒
                    score = base_score * (psychology["soothing_effect"] + psychology.get("calming", 0.5)) / 2
                else:
                    score = base_score
                
                if score > 0.6:  # 阈值过滤
                    recommendations.append(instrument)
            
            # 确保至少有两种乐器
            if len(recommendations) < 2:
                recommendations.extend([InstrumentFamily.AMBIENT, InstrumentFamily.KEYBOARD])
                recommendations = list(set(recommendations))  # 去重
        
        else:
            # 非睡眠场景，根据情绪选择
            if target_emotion.arousal > 0.5:
                recommendations.extend([InstrumentFamily.KEYBOARD, InstrumentFamily.STRINGS, InstrumentFamily.PERCUSSION])
            elif target_emotion.valence > 0.3:
                recommendations.extend([InstrumentFamily.STRINGS, InstrumentFamily.WINDS, InstrumentFamily.KEYBOARD])
            else:
                recommendations.extend([InstrumentFamily.AMBIENT, InstrumentFamily.STRINGS, InstrumentFamily.KEYBOARD])
        
        self.logger.info(f"推荐乐器: {[i.value for i in recommendations]}")
        
        return recommendations
    
    def calculate_psychoacoustic_target(self, 
                                      emotion_state: EmotionState,
                                      sleep_optimized: bool = True) -> PsychoacousticProfile:
        """
        计算目标心理声学概谱
        
        Args:
            emotion_state: 目标情绪状态
            sleep_optimized: 是否为睡眠优化
            
        Returns:
            目标心理声学概谱
        """
        # 基础参数
        base_profile = PsychoacousticProfile(
            roughness=0.3,
            brightness=0.5,
            warmth=0.6,
            spaciousness=0.7,
            attack_sharpness=0.4,
            spectral_centroid=1000.0,
            spectral_rolloff=3000.0,
            zero_crossing_rate=0.1
        )
        
        # 根据情绪调整
        if emotion_state.valence < -0.3:  # 消极情绪
            base_profile.warmth += 0.2  # 增加温暖度
            base_profile.brightness -= 0.1  # 降低明亮度
            base_profile.attack_sharpness -= 0.2  # 更加柔和
        
        if emotion_state.arousal > 0.3:  # 高判醒
            base_profile.roughness -= 0.2  # 降低粗糙度
            base_profile.spaciousness += 0.1  # 增加空间感
        
        # 睡眠优化调整
        if sleep_optimized:
            base_profile.roughness = min(0.2, base_profile.roughness)  # 限制粗糙度
            base_profile.attack_sharpness = min(0.3, base_profile.attack_sharpness)  # 限制攻击锐度
            base_profile.warmth = max(0.6, base_profile.warmth)  # 保证温暖度
            base_profile.spaciousness = max(0.7, base_profile.spaciousness)  # 保证空间感
            
            # 调整频谱参数
            base_profile.spectral_centroid = min(800.0, base_profile.spectral_centroid)  # 低频偏向
            base_profile.spectral_rolloff = min(2500.0, base_profile.spectral_rolloff)  # 柔和高频
        
        # 归一化所有数值
        base_profile.roughness = max(0, min(1, base_profile.roughness))
        base_profile.brightness = max(0, min(1, base_profile.brightness))
        base_profile.warmth = max(0, min(1, base_profile.warmth))
        base_profile.spaciousness = max(0, min(1, base_profile.spaciousness))
        base_profile.attack_sharpness = max(0, min(1, base_profile.attack_sharpness))
        
        self.logger.info(f"心理声学目标: 温暖度={base_profile.warmth:.2f}, 空间感={base_profile.spaciousness:.2f}")
        
        return base_profile
    
    def generate_musical_prescription(self, 
                                    emotion_state: EmotionState,
                                    target_emotion: Optional[EmotionState] = None,
                                    duration_minutes: float = 20.0,
                                    sleep_context: bool = True) -> MusicalCharacteristics:
        """
        生成音乐处方
        
        综合考虑所有因素，生成一个完整的音乐特征配置
        """
        # 选择目标情绪（如果没有指定，使用当前情绪）
        working_emotion = target_emotion if target_emotion else emotion_state
        
        # 计算各个音乐参数
        tempo_bpm, tempo_category = self.map_emotion_to_tempo(working_emotion)
        optimal_key = self.select_optimal_key(emotion_state, target_emotion)
        recommended_instruments = self.recommend_instruments(working_emotion, sleep_context)
        
        # 计算其他特征
        # 拍号：睡眠场景偏好简单拍号
        time_signature = (4, 4) if sleep_context else (4, 4)  # 可以扩展更多选择
        
        # 力度：根据情绪和场景选择
        if sleep_context:
            dynamics = "pp" if working_emotion.arousal < -0.3 else "p"
        else:
            if working_emotion.arousal > 0.5:
                dynamics = "mf"
            elif working_emotion.arousal > 0.0:
                dynamics = "mp"
            else:
                dynamics = "p"
        
        # 和声复杂度：睡眠时倾向简单
        if sleep_context:
            harmonic_complexity = 0.3 + abs(working_emotion.valence) * 0.2  # 0.3-0.5
        else:
            harmonic_complexity = 0.5 + abs(working_emotion.arousal) * 0.3  # 0.5-0.8
        harmonic_complexity = max(0.1, min(0.9, harmonic_complexity))
        
        # 旋律音域：根据情绪调整
        if working_emotion.arousal < -0.3:  # 低判醒，低频域
            melodic_range = (200.0, 800.0)
        elif working_emotion.arousal > 0.3:  # 高判醒，广音域
            melodic_range = (300.0, 1200.0)
        else:  # 中判醒，中音域
            melodic_range = (250.0, 1000.0)
        
        # 织体：睡眠场景偏好主调音乐
        if sleep_context:
            texture = "homophonic"
        else:
            if harmonic_complexity > 0.6:
                texture = "polyphonic"
            else:
                texture = "homophonic"
        
        prescription = MusicalCharacteristics(
            tempo_bpm=tempo_bpm,
            key=optimal_key,
            time_signature=time_signature,
            dynamics=dynamics,
            primary_instruments=recommended_instruments,
            harmonic_complexity=harmonic_complexity,
            melodic_range=melodic_range,
            texture=texture
        )
        
        self.logger.info(f"音乐处方生成: {tempo_bpm:.1f}BPM {optimal_key.value} {dynamics} {texture}")
        
        return prescription
    
    def analyze_music_emotion_fit(self, 
                                characteristics: MusicalCharacteristics,
                                target_emotion: EmotionState) -> Dict[str, float]:
        """
        分析音乐特征与目标情绪的匹配度
        
        Returns:
            匹配度分析结果
        """
        analysis = {}
        
        # 节奏匹配度
        optimal_tempo, _ = self.map_emotion_to_tempo(target_emotion)
        tempo_diff = abs(characteristics.tempo_bpm - optimal_tempo)
        tempo_fit = max(0, 1.0 - tempo_diff / 30.0)  # 30 BPM为容差
        analysis["tempo_fit"] = tempo_fit
        
        # 调性匹配度
        key_characteristics = self.key_characteristics.get(characteristics.key, {})
        if key_characteristics:
            key_valence = key_characteristics.get("valence", 0)
            key_arousal = key_characteristics.get("arousal", 0)
            key_distance = math.sqrt(
                (target_emotion.valence - key_valence) ** 2 +
                (target_emotion.arousal - key_arousal) ** 2
            )
            key_fit = max(0, 1.0 - key_distance / 2.0)  # 最大距离为2
        else:
            key_fit = 0.5  # 未知调性的默认分数
        analysis["key_fit"] = key_fit
        
        # 乐器适配性
        recommended_instruments = self.recommend_instruments(target_emotion, sleep_context=True)
        instrument_overlap = len(set(characteristics.primary_instruments) & set(recommended_instruments))
        instrument_fit = instrument_overlap / max(len(recommended_instruments), 1)
        analysis["instrument_fit"] = instrument_fit
        
        # 复杂度适配性
        if target_emotion.arousal < -0.3:  # 低判醒，需要简单
            complexity_fit = 1.0 - characteristics.harmonic_complexity
        elif target_emotion.arousal > 0.3:  # 高判醒，可以复杂
            complexity_fit = characteristics.harmonic_complexity
        else:  # 中判醒，中等复杂度
            optimal_complexity = 0.5
            complexity_fit = 1.0 - abs(characteristics.harmonic_complexity - optimal_complexity)
        analysis["complexity_fit"] = complexity_fit
        
        # 综合匹配度
        weights = [0.3, 0.25, 0.25, 0.2]  # tempo, key, instrument, complexity
        fits = [tempo_fit, key_fit, instrument_fit, complexity_fit]
        overall_fit = sum(w * f for w, f in zip(weights, fits))
        analysis["overall_fit"] = overall_fit
        
        # 质量评估
        if overall_fit > 0.8:
            analysis["quality_rating"] = "excellent"
        elif overall_fit > 0.6:
            analysis["quality_rating"] = "good"
        elif overall_fit > 0.4:
            analysis["quality_rating"] = "fair"
        else:
            analysis["quality_rating"] = "poor"
        
        self.logger.info(f"音乐情绪匹配分析: 综合匹配度={overall_fit:.3f} ({analysis['quality_rating']})")
        
        return analysis
    
    def export_prescription_data(self, 
                               prescription: MusicalCharacteristics,
                               emotion_analysis: Dict[str, Any],
                               filepath: str) -> None:
        """
        导出音乐处方数据
        """
        export_data = {
            "music_psychology_model_version": "1.0",
            "timestamp": np.datetime64('now').astype(str),
            "model_config": self.config,
            "prescription": {
                "tempo_bpm": prescription.tempo_bpm,
                "key": prescription.key.value,
                "time_signature": prescription.time_signature,
                "dynamics": prescription.dynamics,
                "primary_instruments": [i.value for i in prescription.primary_instruments],
                "harmonic_complexity": prescription.harmonic_complexity,
                "melodic_range": prescription.melodic_range,
                "texture": prescription.texture
            },
            "emotion_analysis": emotion_analysis
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"音乐处方数据已导出到: {filepath}")

# 工具函数
def create_music_psychology_model(config_path: Optional[str] = None) -> MusicPsychologyModel:
    """
    创建音乐心理学模型的便捷函数
    """
    config = None
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        config = full_config.get("music_psychology")
    
    return MusicPsychologyModel(config)

# 示例使用
if __name__ == "__main__":
    # 创建音乐心理学模型
    music_model = MusicPsychologyModel()
    
    # 模拟情绪状态（焦虑、高判醒）
    current_emotion = EmotionState(valence=-0.2, arousal=0.6, confidence=0.9)
    target_emotion = EmotionState(valence=0.2, arousal=-0.6, confidence=0.9)  # 睡前目标
    
    # 生成音乐处方
    prescription = music_model.generate_musical_prescription(
        current_emotion, target_emotion, 
        duration_minutes=20.0, sleep_context=True
    )
    
    # 分析匹配度
    fit_analysis = music_model.analyze_music_emotion_fit(prescription, target_emotion)
    
    # 计算心理声学目标
    psychoacoustic_target = music_model.calculate_psychoacoustic_target(
        target_emotion, sleep_optimized=True
    )
    
    print("音乐心理学模型演示:")
    print(f"音乐处方: {prescription.tempo_bpm:.1f}BPM {prescription.key.value} {prescription.dynamics}")
    print(f"推荐乐器: {[i.value for i in prescription.primary_instruments]}")
    print(f"匹配度: {fit_analysis['overall_fit']:.3f} ({fit_analysis['quality_rating']})")
    print(f"心理声学: 温暖度={psychoacoustic_target.warmth:.2f}, 愉悦度={psychoacoustic_target.get_overall_pleasantness():.2f}")
