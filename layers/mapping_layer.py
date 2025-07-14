#!/usr/bin/env python3
"""
映射层 (Mapping Layer) - Layer 3

KG-MLP混合映射架构，核心功能包括：
1. 知识图谱驱动的专家知识编码
2. 深度学习的个性化映射学习
3. 27维情绪向量到音乐参数的映射
4. 混合融合策略和自适应权重
5. 睡眠治疗的领域特化优化

处理流程：
Fusion Layer → Mapping Layer → Generation Layer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
import json
from enum import Enum

from .base_layer import BaseLayer, LayerData, LayerConfig
from core.utils import (
    ConfigLoader, DataValidator, PerformanceMonitor, 
    get_project_root, normalize_vector, cosine_similarity
)

logger = logging.getLogger(__name__)

class MappingStrategy(Enum):
    """映射策略枚举"""
    KNOWLEDGE_GRAPH = "knowledge_graph"      # 纯知识图谱
    DEEP_LEARNING = "deep_learning"          # 纯深度学习
    HYBRID_FUSION = "hybrid_fusion"          # 混合融合
    ADAPTIVE_WEIGHT = "adaptive_weight"      # 自适应权重

@dataclass
class MappingLayerConfig(LayerConfig):
    """映射层配置"""
    # 基础配置
    input_emotion_dim: int = 27
    output_music_dim: int = 64  # 音乐参数维度
    
    # 映射策略
    mapping_strategy: str = "hybrid_fusion"
    kg_weight: float = 0.6      # 知识图谱权重
    mlp_weight: float = 0.4     # MLP权重
    
    # 知识图谱配置
    kg_enabled: bool = True
    kg_embedding_dim: int = 128
    kg_relation_types: int = 8
    
    # MLP配置
    mlp_enabled: bool = True
    mlp_hidden_dims: List[int] = None
    mlp_dropout: float = 0.2
    mlp_activation: str = "relu"
    
    # 睡眠治疗特化
    sleep_therapy_mode: bool = True
    circadian_adaptation: bool = True
    therapy_stage_aware: bool = True
    
    # 性能配置
    use_gpu: bool = True
    batch_size: int = 1
    max_processing_time: float = 100.0  # ms
    
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [256, 128, 64]

class MusicParameter:
    """单阶段音乐参数结构"""
    def __init__(self):
        # 基础音乐参数
        self.tempo_bpm: float = 60.0          # 节拍速度
        self.key_signature: str = "C_major"    # 调性
        self.time_signature: Tuple[int, int] = (4, 4)  # 拍号
        self.dynamics: str = "mp"              # 力度
        
        # 音色和织体
        self.instrument_weights: Dict[str, float] = {}  # 乐器权重
        self.texture_complexity: float = 0.5   # 织体复杂度
        self.harmonic_richness: float = 0.5    # 和声丰富度
        
        # 情绪表达
        self.valence_mapping: float = 0.0      # 效价映射
        self.arousal_mapping: float = 0.0      # 唤醒映射
        self.tension_level: float = 0.0        # 张力水平
        
        # 治疗特化
        self.iso_stage: str = "synchronization"  # ISO阶段
        self.stage_duration: float = 5.0         # 阶段持续时间（分钟）
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式便于传递给音乐生成API"""
        return {
            'tempo_bpm': self.tempo_bpm,
            'key_signature': self.key_signature,
            'time_signature': self.time_signature,
            'dynamics': self.dynamics,
            'instrument_weights': self.instrument_weights,
            'texture_complexity': self.texture_complexity,
            'harmonic_richness': self.harmonic_richness,
            'valence_mapping': self.valence_mapping,
            'arousal_mapping': self.arousal_mapping,
            'tension_level': self.tension_level,
            'iso_stage': self.iso_stage,
            'stage_duration': self.stage_duration
        }

class ISOThreeStageParams:
    """ISO三阶段音乐参数"""
    def __init__(self):
        self.match_stage: MusicParameter = MusicParameter()      # 匹配阶段
        self.guide_stage: MusicParameter = MusicParameter()      # 引导阶段  
        self.target_stage: MusicParameter = MusicParameter()     # 目标阶段
        
        # 设置阶段标识
        self.match_stage.iso_stage = "match"
        self.guide_stage.iso_stage = "guide"
        self.target_stage.iso_stage = "target"
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'match_stage': self.match_stage.to_dict(),
            'guide_stage': self.guide_stage.to_dict(),
            'target_stage': self.target_stage.to_dict()
        }

class KnowledgeGraphModule:
    """知识图谱模块"""
    
    def __init__(self, config: MappingLayerConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # 加载情绪-音乐映射知识
        self.emotion_music_knowledge = self._load_emotion_music_knowledge()
        
        # 构建知识图谱嵌入
        self.emotion_embeddings = self._build_emotion_embeddings()
        self.music_embeddings = self._build_music_embeddings()
        self.relation_embeddings = self._build_relation_embeddings()
        
        logger.info(f"知识图谱模块初始化完成，设备: {self.device}")
    
    def _load_emotion_music_knowledge(self) -> Dict[str, Any]:
        """加载情绪-音乐映射知识"""
        try:
            config_path = get_project_root() / "configs" / "emotion_27d.yaml"
            emotion_config = ConfigLoader.load_yaml(str(config_path))
            
            # 提取音乐治疗映射
            music_therapy_mapping = emotion_config.get('music_therapy_mapping', {})
            
            # 构建知识图谱
            knowledge = {
                'emotion_to_tempo': {},
                'emotion_to_key': {},
                'emotion_to_dynamics': {},
                'emotion_to_instruments': {},
                'emotion_relationships': emotion_config.get('emotion_relationships', {}),
                'therapy_strategies': music_therapy_mapping
            }
            
            # 基于现有配置构建映射规则
            all_emotions = {}
            all_emotions.update(emotion_config.get('base_emotions', {}))
            all_emotions.update(emotion_config.get('sleep_specific_emotions', {}))
            
            for emotion_key, emotion_data in all_emotions.items():
                valence = emotion_data.get('valence', 0.0)
                arousal = emotion_data.get('arousal', 0.0)
                
                # 基于效价-唤醒的音乐映射规则
                knowledge['emotion_to_tempo'][emotion_key] = self._valence_arousal_to_tempo(valence, arousal)
                knowledge['emotion_to_key'][emotion_key] = self._valence_arousal_to_key(valence, arousal)
                knowledge['emotion_to_dynamics'][emotion_key] = self._valence_arousal_to_dynamics(valence, arousal)
                knowledge['emotion_to_instruments'][emotion_key] = self._valence_arousal_to_instruments(valence, arousal)
            
            return knowledge
            
        except Exception as e:
            logger.error(f"加载情绪-音乐知识失败: {e}")
            return self._get_default_knowledge()
    
    def _valence_arousal_to_tempo(self, valence: float, arousal: float) -> float:
        """效价-唤醒到节拍速度的映射"""
        # 高唤醒 → 快节拍，低唤醒 → 慢节拍
        # 睡眠治疗：整体偏向较慢的节拍
        base_tempo = 60.0  # 睡眠治疗基础节拍
        arousal_factor = max(0.1, min(2.0, 1.0 + arousal * 0.8))  # 0.1-2.0倍数
        
        # 负效价情绪在睡眠治疗中需要更慢的节拍
        if valence < 0 and self.config.sleep_therapy_mode:
            arousal_factor *= 0.8
        
        return base_tempo * arousal_factor
    
    def _valence_arousal_to_key(self, valence: float, arousal: float) -> str:
        """效价-唤醒到调性的映射"""
        # 正效价 → 大调，负效价 → 小调
        # 高唤醒 → 升调，低唤醒 → 降调
        
        if valence >= 0:
            # 大调系
            if arousal > 0.3:
                return "D_major"  # 明亮
            elif arousal > -0.3:
                return "C_major"  # 平和
            else:
                return "F_major"  # 温暖
        else:
            # 小调系
            if arousal > 0.3:
                return "A_minor"  # 激烈
            elif arousal > -0.3:
                return "D_minor"  # 忧郁
            else:
                return "G_minor"  # 深沉
    
    def _valence_arousal_to_dynamics(self, valence: float, arousal: float) -> str:
        """效价-唤醒到力度的映射"""
        # 高唤醒 → 强力度，低唤醒 → 弱力度
        # 睡眠治疗：整体偏向较弱的力度
        
        if self.config.sleep_therapy_mode:
            if arousal > 0.5:
                return "mp"  # 中弱
            elif arousal > 0.0:
                return "p"   # 弱
            else:
                return "pp"  # 极弱
        else:
            if arousal > 0.5:
                return "mf"  # 中强
            elif arousal > -0.5:
                return "mp"  # 中弱
            else:
                return "p"   # 弱
    
    def _valence_arousal_to_instruments(self, valence: float, arousal: float) -> Dict[str, float]:
        """效价-唤醒到乐器权重的映射"""
        instruments = {
            'piano': 0.0,
            'strings': 0.0,
            'woodwinds': 0.0,
            'brass': 0.0,
            'percussion': 0.0,
            'harp': 0.0,
            'choir': 0.0,
            'ambient': 0.0
        }
        
        # 基于效价-唤醒的乐器选择
        if valence >= 0:  # 正效价
            instruments['piano'] = 0.6
            instruments['strings'] = 0.4
            if arousal > 0.3:
                instruments['woodwinds'] = 0.3
            else:
                instruments['harp'] = 0.3
        else:  # 负效价
            instruments['strings'] = 0.7
            instruments['piano'] = 0.3
            if arousal > 0.3:
                instruments['woodwinds'] = 0.2
            else:
                instruments['ambient'] = 0.4
        
        # 睡眠治疗模式调整
        if self.config.sleep_therapy_mode:
            # 增加舒缓乐器权重
            instruments['ambient'] = max(instruments['ambient'], 0.3)
            instruments['harp'] = max(instruments['harp'], 0.2)
            # 减少激烈乐器权重
            instruments['brass'] = min(instruments['brass'], 0.1)
            instruments['percussion'] = min(instruments['percussion'], 0.1)
        
        return instruments
    
    def _build_emotion_embeddings(self) -> torch.Tensor:
        """构建情绪嵌入"""
        # 27维情绪的嵌入表示
        emotion_embeddings = torch.randn(27, self.config.kg_embedding_dim).to(self.device)
        return emotion_embeddings
    
    def _build_music_embeddings(self) -> torch.Tensor:
        """构建音乐参数嵌入"""
        # 音乐参数的嵌入表示
        music_embeddings = torch.randn(self.config.output_music_dim, self.config.kg_embedding_dim).to(self.device)
        return music_embeddings
    
    def _build_relation_embeddings(self) -> torch.Tensor:
        """构建关系嵌入"""
        # 情绪-音乐关系的嵌入表示
        relation_embeddings = torch.randn(self.config.kg_relation_types, self.config.kg_embedding_dim).to(self.device)
        return relation_embeddings
    
    def _get_default_knowledge(self) -> Dict[str, Any]:
        """获取默认知识"""
        return {
            'emotion_to_tempo': {},
            'emotion_to_key': {},
            'emotion_to_dynamics': {},
            'emotion_to_instruments': {},
            'emotion_relationships': {},
            'therapy_strategies': {}
        }
    
    def map_emotion_to_music(self, emotion_vector: torch.Tensor) -> MusicParameter:
        """基于知识图谱映射情绪到音乐参数"""
        # 计算情绪嵌入
        emotion_embedding = torch.matmul(emotion_vector, self.emotion_embeddings)
        
        # 通过知识图谱关系计算音乐参数
        music_embedding = self._apply_kg_relations(emotion_embedding)
        
        # 转换为具体的音乐参数
        music_params = self._embedding_to_music_params(music_embedding, emotion_vector)
        
        return music_params
    
    def _apply_kg_relations(self, emotion_embedding: torch.Tensor) -> torch.Tensor:
        """应用知识图谱关系"""
        # 通过关系嵌入计算音乐嵌入
        # 这里使用简化的线性变换，实际可以使用更复杂的图神经网络
        
        # 关系变换
        relation_weights = torch.randn(self.config.kg_relation_types, self.config.kg_embedding_dim, self.config.kg_embedding_dim).to(self.device)
        
        # 应用多种关系
        music_embedding = torch.zeros(self.config.kg_embedding_dim).to(self.device)
        for i in range(self.config.kg_relation_types):
            relation_output = torch.matmul(emotion_embedding, relation_weights[i])
            music_embedding += relation_output / self.config.kg_relation_types
        
        return music_embedding
    
    def _embedding_to_music_params(self, music_embedding: torch.Tensor, emotion_vector: torch.Tensor) -> MusicParameter:
        """将音乐嵌入转换为具体的音乐参数"""
        music_params = MusicParameter()
        
        # 通过线性变换获取音乐参数
        param_weights = torch.randn(self.config.kg_embedding_dim, 13).to(self.device)  # 13个基础参数
        raw_params = torch.matmul(music_embedding, param_weights)
        
        # 映射到具体参数
        music_params.tempo_bpm = max(40.0, min(120.0, 60.0 + raw_params[0].item() * 20.0))
        music_params.valence_mapping = torch.tanh(raw_params[1]).item()
        music_params.arousal_mapping = torch.tanh(raw_params[2]).item()
        music_params.tension_level = torch.sigmoid(raw_params[3]).item()
        music_params.texture_complexity = torch.sigmoid(raw_params[4]).item()
        music_params.harmonic_richness = torch.sigmoid(raw_params[5]).item()
        music_params.therapy_intensity = torch.sigmoid(raw_params[6]).item()
        music_params.sleep_readiness = torch.sigmoid(raw_params[7]).item()
        
        # 乐器权重
        instrument_names = ['piano', 'strings', 'woodwinds', 'brass', 'percussion']
        instrument_weights = torch.softmax(raw_params[8:13], dim=0)
        for i, name in enumerate(instrument_names):
            music_params.instrument_weights[name] = instrument_weights[i].item()
        
        # 调性和力度通过规则映射
        avg_valence = torch.mean(emotion_vector * torch.tensor([
            # 这里需要根据27维情绪的效价权重
            -0.8, -0.6, -0.7, -0.6, 0.6, 0.8, 0.7, 0.5, 0.0,  # 基础9维
            -0.4, -0.7, -0.3, -0.2, -0.1, -0.5, -0.8, -0.2, -0.4,  # 睡眠情绪10-18
            -0.1, -0.3, -0.6, -0.5, -0.6, -0.4, -0.3, -0.5, -0.7   # 睡眠情绪19-27
        ]).to(self.device)).item()
        
        avg_arousal = torch.mean(emotion_vector * torch.tensor([
            # 这里需要根据27维情绪的唤醒权重
            0.7, 0.8, 0.3, -0.2, 0.4, 0.6, 0.5, -0.3, 0.0,  # 基础9维
            0.3, 0.6, -0.6, -0.4, 0.9, 0.4, 0.5, 0.8, 0.2,  # 睡眠情绪10-18
            -0.7, 0.7, 0.4, -0.1, 0.6, 0.3, 0.2, 0.4, 0.1   # 睡眠情绪19-27
        ]).to(self.device)).item()
        
        music_params.key_signature = self._valence_arousal_to_key(avg_valence, avg_arousal)
        music_params.dynamics = self._valence_arousal_to_dynamics(avg_valence, avg_arousal)
        
        return music_params

class MLPMappingModule(nn.Module):
    """MLP映射模块"""
    
    def __init__(self, config: MappingLayerConfig):
        super().__init__()
        self.config = config
        
        # 构建MLP网络
        layers = []
        input_dim = config.input_emotion_dim
        
        for hidden_dim in config.mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if config.mlp_activation == "relu":
                layers.append(nn.ReLU())
            elif config.mlp_activation == "gelu":
                layers.append(nn.GELU())
            elif config.mlp_activation == "tanh":
                layers.append(nn.Tanh())
            
            if config.mlp_dropout > 0:
                layers.append(nn.Dropout(config.mlp_dropout))
            
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, config.output_music_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 个性化适应层
        self.personal_adaptation = nn.Linear(config.output_music_dim, config.output_music_dim)
        
        # 治疗阶段感知层
        if config.therapy_stage_aware:
            self.therapy_stage_layer = nn.Linear(config.output_music_dim + 3, config.output_music_dim)  # +3 for ISO stages
        
        logger.info(f"MLP映射模块初始化完成，网络结构: {config.mlp_hidden_dims}")
    
    def forward(self, emotion_vector: torch.Tensor, therapy_stage: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 基础MLP映射
        music_features = self.mlp(emotion_vector)
        
        # 个性化适应
        adapted_features = self.personal_adaptation(music_features)
        
        # 治疗阶段感知
        if therapy_stage is not None and self.config.therapy_stage_aware:
            combined_features = torch.cat([adapted_features, therapy_stage], dim=-1)
            final_features = self.therapy_stage_layer(combined_features)
        else:
            final_features = adapted_features
        
        return final_features
    
    def extract_music_parameters(self, music_features: torch.Tensor) -> MusicParameter:
        """从特征向量提取音乐参数"""
        music_params = MusicParameter()
        
        # 基础参数映射
        music_params.tempo_bpm = max(40.0, min(120.0, 60.0 + music_features[0].item() * 30.0))
        music_params.valence_mapping = torch.tanh(music_features[1]).item()
        music_params.arousal_mapping = torch.tanh(music_features[2]).item()
        music_params.tension_level = torch.sigmoid(music_features[3]).item()
        music_params.texture_complexity = torch.sigmoid(music_features[4]).item()
        music_params.harmonic_richness = torch.sigmoid(music_features[5]).item()
        music_params.therapy_intensity = torch.sigmoid(music_features[6]).item()
        music_params.sleep_readiness = torch.sigmoid(music_features[7]).item()
        
        # 乐器权重
        instrument_features = music_features[8:16]
        instrument_weights = torch.softmax(instrument_features, dim=0)
        instrument_names = ['piano', 'strings', 'woodwinds', 'brass', 'percussion', 'harp', 'choir', 'ambient']
        
        for i, name in enumerate(instrument_names):
            music_params.instrument_weights[name] = instrument_weights[i].item()
        
        # 其他参数
        music_params.key_signature = self._features_to_key(music_features[16:20])
        music_params.dynamics = self._features_to_dynamics(music_features[20:24])
        
        return music_params
    
    def _features_to_key(self, key_features: torch.Tensor) -> str:
        """特征到调性的映射"""
        key_probs = torch.softmax(key_features, dim=0)
        key_names = ["C_major", "D_major", "F_major", "G_minor", "A_minor", "D_minor"]
        
        if len(key_names) > len(key_probs):
            key_names = key_names[:len(key_probs)]
        
        key_idx = torch.argmax(key_probs).item()
        return key_names[key_idx] if key_idx < len(key_names) else "C_major"
    
    def _features_to_dynamics(self, dynamics_features: torch.Tensor) -> str:
        """特征到力度的映射"""
        dynamics_probs = torch.softmax(dynamics_features, dim=0)
        dynamics_names = ["pp", "p", "mp", "mf"]
        
        if len(dynamics_names) > len(dynamics_probs):
            dynamics_names = dynamics_names[:len(dynamics_probs)]
        
        dynamics_idx = torch.argmax(dynamics_probs).item()
        return dynamics_names[dynamics_idx] if dynamics_idx < len(dynamics_names) else "mp"

class MappingLayer(BaseLayer):
    """映射层 - KG-MLP混合情绪到音乐参数映射"""
    
    def __init__(self, config: MappingLayerConfig):
        super().__init__(config)
        self.config = config
        self.layer_name = "mapping_layer"
        
        # 初始化设备
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"映射层使用设备: {self.device}")
        
        # 初始化知识图谱模块
        if config.kg_enabled:
            self.kg_module = KnowledgeGraphModule(config)
        else:
            self.kg_module = None
        
        # 初始化MLP模块
        if config.mlp_enabled:
            self.mlp_module = MLPMappingModule(config).to(self.device)
        else:
            self.mlp_module = None
        
        # 混合融合权重
        self.kg_weight = config.kg_weight
        self.mlp_weight = config.mlp_weight
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 数据验证器
        self.data_validator = DataValidator()
        
        logger.info(f"映射层初始化完成，策略: {config.mapping_strategy}")
    
    def _extract_emotion_vector(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """从输入数据提取情绪向量"""
        if 'emotion_analysis' in input_data:
            emotion_analysis = input_data['emotion_analysis']
            if 'emotion_vector' in emotion_analysis:
                emotion_vector = torch.tensor(emotion_analysis['emotion_vector'], dtype=torch.float32).to(self.device)
                return emotion_vector
        
        # 如果没有情绪向量，创建默认的中性向量
        logger.warning("未找到情绪向量，使用中性向量")
        neutral_vector = torch.zeros(self.config.input_emotion_dim, dtype=torch.float32).to(self.device)
        neutral_vector[8] = 1.0  # 中性情绪位置
        return neutral_vector
    
    def _extract_therapy_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取治疗上下文信息"""
        therapy_context = {
            'iso_stage': 'synchronization',
            'therapy_session_time': 0.0,
            'user_feedback': None,
            'circadian_phase': 'evening'
        }
        
        # 从输入数据中提取治疗上下文
        if 'therapy_context' in input_data:
            therapy_context.update(input_data['therapy_context'])
        
        return therapy_context
    
    def _kg_mapping(self, emotion_vector: torch.Tensor) -> MusicParameter:
        """知识图谱映射"""
        if self.kg_module is None:
            return MusicParameter()
        
        return self.kg_module.map_emotion_to_music(emotion_vector)
    
    def _mlp_mapping(self, emotion_vector: torch.Tensor, therapy_context: Dict[str, Any]) -> MusicParameter:
        """MLP映射"""
        if self.mlp_module is None:
            return MusicParameter()
        
        # 构建治疗阶段向量
        therapy_stage_vector = None
        if self.config.therapy_stage_aware:
            iso_stage = therapy_context.get('iso_stage', 'synchronization')
            therapy_stage_vector = torch.zeros(3).to(self.device)
            if iso_stage == 'synchronization':
                therapy_stage_vector[0] = 1.0
            elif iso_stage == 'guidance':
                therapy_stage_vector[1] = 1.0
            elif iso_stage == 'consolidation':
                therapy_stage_vector[2] = 1.0
        
        # MLP前向传播
        music_features = self.mlp_module(emotion_vector.unsqueeze(0), therapy_stage_vector.unsqueeze(0) if therapy_stage_vector is not None else None)
        
        # 提取音乐参数
        music_params = self.mlp_module.extract_music_parameters(music_features.squeeze(0))
        
        return music_params
    
    def _hybrid_fusion(self, kg_params: MusicParameter, mlp_params: MusicParameter) -> MusicParameter:
        """混合融合KG和MLP的结果"""
        fused_params = MusicParameter()
        
        # 线性融合基础参数
        fused_params.tempo_bpm = self.kg_weight * kg_params.tempo_bpm + self.mlp_weight * mlp_params.tempo_bpm
        fused_params.valence_mapping = self.kg_weight * kg_params.valence_mapping + self.mlp_weight * mlp_params.valence_mapping
        fused_params.arousal_mapping = self.kg_weight * kg_params.arousal_mapping + self.mlp_weight * mlp_params.arousal_mapping
        fused_params.tension_level = self.kg_weight * kg_params.tension_level + self.mlp_weight * mlp_params.tension_level
        fused_params.texture_complexity = self.kg_weight * kg_params.texture_complexity + self.mlp_weight * mlp_params.texture_complexity
        fused_params.harmonic_richness = self.kg_weight * kg_params.harmonic_richness + self.mlp_weight * mlp_params.harmonic_richness
        fused_params.therapy_intensity = self.kg_weight * kg_params.therapy_intensity + self.mlp_weight * mlp_params.therapy_intensity
        fused_params.sleep_readiness = self.kg_weight * kg_params.sleep_readiness + self.mlp_weight * mlp_params.sleep_readiness
        
        # 融合乐器权重
        all_instruments = set(kg_params.instrument_weights.keys()) | set(mlp_params.instrument_weights.keys())
        for instrument in all_instruments:
            kg_weight = kg_params.instrument_weights.get(instrument, 0.0)
            mlp_weight = mlp_params.instrument_weights.get(instrument, 0.0)
            fused_params.instrument_weights[instrument] = self.kg_weight * kg_weight + self.mlp_weight * mlp_weight
        
        # 分类参数采用权重选择
        if self.kg_weight > self.mlp_weight:
            fused_params.key_signature = kg_params.key_signature
            fused_params.dynamics = kg_params.dynamics
            fused_params.iso_stage = kg_params.iso_stage
        else:
            fused_params.key_signature = mlp_params.key_signature
            fused_params.dynamics = mlp_params.dynamics
            fused_params.iso_stage = mlp_params.iso_stage
        
        fused_params.time_signature = kg_params.time_signature  # 保持默认
        
        return fused_params
    
    def _generate_iso_three_stages(self, emotion_vector: torch.Tensor, therapy_context: Dict[str, Any]) -> ISOThreeStageParams:
        """生成ISO三阶段音乐参数"""
        iso_params = ISOThreeStageParams()
        
        # 获取基础情绪映射参数
        base_kg_params = self._kg_mapping(emotion_vector) if self.kg_module else MusicParameter()
        base_mlp_params = self._mlp_mapping(emotion_vector, therapy_context) if self.mlp_module else MusicParameter()
        base_params = self._hybrid_fusion(base_kg_params, base_mlp_params)
        
        # 1. 匹配阶段 (Match Stage) - 与当前情绪状态同步
        iso_params.match_stage = self._create_match_stage_params(base_params, emotion_vector, therapy_context)
        
        # 2. 引导阶段 (Guide Stage) - 逐步过渡到目标状态  
        iso_params.guide_stage = self._create_guide_stage_params(base_params, emotion_vector, therapy_context)
        
        # 3. 目标阶段 (Target Stage) - 达到睡眠准备状态
        iso_params.target_stage = self._create_target_stage_params(base_params, emotion_vector, therapy_context)
        
        return iso_params
    
    def _create_match_stage_params(self, base_params: MusicParameter, emotion_vector: torch.Tensor, therapy_context: Dict[str, Any]) -> MusicParameter:
        """创建匹配阶段参数 - 与用户当前情绪状态同步"""
        match_params = MusicParameter()
        
        # 复制基础参数
        match_params.tempo_bpm = base_params.tempo_bpm
        match_params.key_signature = base_params.key_signature
        match_params.dynamics = base_params.dynamics
        match_params.valence_mapping = base_params.valence_mapping
        match_params.arousal_mapping = base_params.arousal_mapping
        match_params.tension_level = base_params.tension_level
        match_params.instrument_weights = base_params.instrument_weights.copy()
        
        # 匹配阶段特化：保持与当前情绪的高度一致性
        match_params.iso_stage = "match"
        match_params.stage_duration = 2.0  # 2分钟匹配期
        match_params.therapy_intensity = 0.3  # 低治疗强度，主要是建立同步
        match_params.sleep_readiness = 0.2   # 低睡眠准备度
        
        # 强化当前情绪特征
        if abs(match_params.arousal_mapping) > 0.5:  # 高唤醒状态
            match_params.tempo_bpm = min(100.0, match_params.tempo_bpm * 1.1)  # 稍微增加节拍匹配
        
        return match_params
    
    def _create_guide_stage_params(self, base_params: MusicParameter, emotion_vector: torch.Tensor, therapy_context: Dict[str, Any]) -> MusicParameter:
        """创建引导阶段参数 - 逐步引导向目标状态过渡"""
        guide_params = MusicParameter()
        
        # 基于基础参数进行引导性调整
        guide_params.tempo_bpm = max(50.0, base_params.tempo_bpm * 0.85)  # 逐步降低节拍
        guide_params.key_signature = self._transition_to_calming_key(base_params.key_signature)
        guide_params.dynamics = self._transition_to_calming_dynamics(base_params.dynamics)
        
        # 引导过渡中的参数
        guide_params.valence_mapping = base_params.valence_mapping * 0.7 + 0.3 * 0.2  # 向中性偏正转移
        guide_params.arousal_mapping = base_params.arousal_mapping * 0.6  # 降低唤醒水平
        guide_params.tension_level = base_params.tension_level * 0.7  # 降低张力
        
        # 乐器过渡：增加舒缓乐器比重
        guide_params.instrument_weights = base_params.instrument_weights.copy()
        guide_params.instrument_weights['strings'] = min(0.8, guide_params.instrument_weights.get('strings', 0.0) + 0.2)
        guide_params.instrument_weights['harp'] = min(0.6, guide_params.instrument_weights.get('harp', 0.0) + 0.3)
        guide_params.instrument_weights['ambient'] = min(0.5, guide_params.instrument_weights.get('ambient', 0.0) + 0.2)
        
        # 引导阶段特化
        guide_params.iso_stage = "guide"
        guide_params.stage_duration = 6.0  # 6分钟引导期
        guide_params.therapy_intensity = 0.7  # 中高治疗强度
        guide_params.sleep_readiness = 0.5   # 中等睡眠准备度
        guide_params.texture_complexity = base_params.texture_complexity * 0.8  # 简化织体
        guide_params.harmonic_richness = base_params.harmonic_richness * 0.9   # 简化和声
        
        return guide_params
    
    def _create_target_stage_params(self, base_params: MusicParameter, emotion_vector: torch.Tensor, therapy_context: Dict[str, Any]) -> MusicParameter:
        """创建目标阶段参数 - 达到理想的睡眠准备状态"""
        target_params = MusicParameter()
        
        # 目标：深度放松的睡眠准备状态
        target_params.tempo_bpm = 45.0  # 固定的极慢节拍
        target_params.key_signature = "C_major"  # 最安全平和的调性
        target_params.dynamics = "pp"   # 极弱力度
        target_params.time_signature = (4, 4)  # 稳定拍号
        
        # 目标情绪状态：低唤醒、中性偏正效价
        target_params.valence_mapping = 0.2    # 轻微正效价
        target_params.arousal_mapping = -0.7   # 极低唤醒
        target_params.tension_level = 0.1      # 极低张力
        
        # 睡眠优化乐器配置
        target_params.instrument_weights = {
            'piano': 0.3,      # 温和钢琴
            'strings': 0.5,    # 主导弦乐
            'harp': 0.4,       # 舒缓竖琴
            'ambient': 0.6,    # 环境音效
            'woodwinds': 0.2,  # 轻柔木管
            'choir': 0.3,      # 天籁人声
            'brass': 0.0,      # 无铜管
            'percussion': 0.0  # 无打击乐
        }
        
        # 目标阶段特化
        target_params.iso_stage = "target"
        target_params.stage_duration = 7.0  # 7分钟巩固期
        target_params.therapy_intensity = 0.9  # 高治疗强度
        target_params.sleep_readiness = 0.9    # 高睡眠准备度
        target_params.texture_complexity = 0.2 # 极简织体
        target_params.harmonic_richness = 0.3  # 简单和声
        
        return target_params
    
    def _transition_to_calming_key(self, current_key: str) -> str:
        """调性向舒缓方向过渡"""
        # 大调系保持，小调系向相对大调或平行大调过渡
        key_transitions = {
            "A_minor": "C_major",    # 关系大调
            "D_minor": "F_major",    # 关系大调
            "G_minor": "Bb_major",   # 关系大调
            "E_minor": "G_major",    # 关系大调
            "B_minor": "D_major",    # 关系大调
            "F#_minor": "A_major",   # 关系大调
        }
        return key_transitions.get(current_key, current_key)
    
    def _transition_to_calming_dynamics(self, current_dynamics: str) -> str:
        """力度向舒缓方向过渡"""
        dynamics_transitions = {
            "ff": "mf",   # 极强→中强
            "f": "mp",    # 强→中弱
            "mf": "p",    # 中强→弱
            "mp": "p",    # 中弱→弱
            "p": "pp",    # 弱→极弱
            "pp": "pp"    # 保持极弱
        }
        return dynamics_transitions.get(current_dynamics, "p")
    
    def _apply_sleep_therapy_adaptation(self, music_params: MusicParameter, therapy_context: Dict[str, Any]) -> MusicParameter:
        """应用睡眠治疗适应（单阶段版本，保持向后兼容）"""
        if not self.config.sleep_therapy_mode:
            return music_params
        
        # 根据治疗阶段调整参数
        iso_stage = therapy_context.get('iso_stage', 'synchronization')
        
        if iso_stage == 'synchronization':
            # 同步阶段：与当前情绪状态匹配
            pass  # 保持原参数
        elif iso_stage == 'guidance':
            # 引导阶段：逐步向目标状态过渡
            music_params.tempo_bpm = max(50.0, music_params.tempo_bpm * 0.9)
            music_params.therapy_intensity = min(1.0, music_params.therapy_intensity * 1.1)
        elif iso_stage == 'consolidation':
            # 巩固阶段：稳定在目标状态
            music_params.tempo_bpm = max(45.0, music_params.tempo_bpm * 0.8)
            music_params.sleep_readiness = min(1.0, music_params.sleep_readiness * 1.2)
        
        # 昼夜节律适应
        if self.config.circadian_adaptation:
            circadian_phase = therapy_context.get('circadian_phase', 'evening')
            if circadian_phase == 'evening':
                music_params.tempo_bpm = max(45.0, music_params.tempo_bpm * 0.9)
                music_params.sleep_readiness = min(1.0, music_params.sleep_readiness * 1.1)
        
        return music_params
    
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """映射层处理实现"""
        self.performance_monitor.start_timer("mapping_layer_processing")
        
        try:
            # 验证输入数据
            if not input_data.data:
                raise ValueError("输入数据为空")
            
            # 提取情绪向量
            emotion_vector = self._extract_emotion_vector(input_data.data)
            
            # 提取治疗上下文
            therapy_context = self._extract_therapy_context(input_data.data)
            
            # 检查是否启用ISO三阶段模式
            use_iso_three_stages = therapy_context.get('enable_iso_three_stages', True)
            
            if use_iso_three_stages and self.config.sleep_therapy_mode:
                # 🎵 ISO三阶段音乐参数生成
                iso_params = self._generate_iso_three_stages(emotion_vector, therapy_context)
                
                # 计算三阶段映射置信度
                match_confidence = self._calculate_mapping_confidence(emotion_vector, iso_params.match_stage)
                guide_confidence = self._calculate_mapping_confidence(emotion_vector, iso_params.guide_stage) 
                target_confidence = self._calculate_mapping_confidence(emotion_vector, iso_params.target_stage)
                mapping_confidence = (match_confidence + guide_confidence + target_confidence) / 3.0
                
                # 创建ISO三阶段输出数据
                output_data = LayerData(
                    layer_name=self.layer_name,
                    timestamp=datetime.now(),
                    data={
                        'iso_three_stage_params': iso_params.to_dict(),
                        'mapping_info': {
                            'strategy': self.config.mapping_strategy,
                            'kg_weight': self.kg_weight,
                            'mlp_weight': self.mlp_weight,
                            'therapy_context': therapy_context,
                            'mapping_confidence': mapping_confidence,
                            'stage_confidences': {
                                'match': match_confidence,
                                'guide': guide_confidence,
                                'target': target_confidence
                            },
                            'iso_mode': True
                        },
                        'emotion_vector': emotion_vector.cpu().numpy().tolist(),
                        'processing_info': {
                            'kg_enabled': self.config.kg_enabled,
                            'mlp_enabled': self.config.mlp_enabled,
                            'sleep_therapy_mode': self.config.sleep_therapy_mode,
                            'iso_three_stages_enabled': True,
                            'total_therapy_duration': 15.0  # 2+6+7分钟
                        }
                    },
                    metadata={
                        'source_layer': input_data.layer_name,
                        'processing_device': str(self.device),
                        'mapping_strategy': self.config.mapping_strategy,
                        'output_format': 'iso_three_stages'
                    },
                    confidence=mapping_confidence
                )
            else:
                # 传统单阶段映射（向后兼容）
                if self.config.mapping_strategy == MappingStrategy.KNOWLEDGE_GRAPH.value:
                    music_params = self._kg_mapping(emotion_vector)
                elif self.config.mapping_strategy == MappingStrategy.DEEP_LEARNING.value:
                    music_params = self._mlp_mapping(emotion_vector, therapy_context)
                elif self.config.mapping_strategy == MappingStrategy.HYBRID_FUSION.value:
                    kg_params = self._kg_mapping(emotion_vector)
                    mlp_params = self._mlp_mapping(emotion_vector, therapy_context)
                    music_params = self._hybrid_fusion(kg_params, mlp_params)
                else:
                    raise ValueError(f"不支持的映射策略: {self.config.mapping_strategy}")
                
                # 应用睡眠治疗适应
                music_params = self._apply_sleep_therapy_adaptation(music_params, therapy_context)
                
                # 计算映射置信度
                mapping_confidence = self._calculate_mapping_confidence(emotion_vector, music_params)
                
                # 创建单阶段输出数据
                output_data = LayerData(
                    layer_name=self.layer_name,
                    timestamp=datetime.now(),
                    data={
                        'music_parameters': music_params.to_dict(),
                        'mapping_info': {
                            'strategy': self.config.mapping_strategy,
                            'kg_weight': self.kg_weight,
                            'mlp_weight': self.mlp_weight,
                            'therapy_context': therapy_context,
                            'mapping_confidence': mapping_confidence,
                            'iso_mode': False
                        },
                        'emotion_vector': emotion_vector.cpu().numpy().tolist(),
                        'processing_info': {
                            'kg_enabled': self.config.kg_enabled,
                            'mlp_enabled': self.config.mlp_enabled,
                            'sleep_therapy_mode': self.config.sleep_therapy_mode,
                            'iso_three_stages_enabled': False
                        }
                    },
                    metadata={
                        'source_layer': input_data.layer_name,
                        'processing_device': str(self.device),
                        'mapping_strategy': self.config.mapping_strategy,
                        'output_format': 'single_stage'
                    },
                    confidence=mapping_confidence
                )
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer("mapping_layer_processing")
            output_data.processing_time = processing_time
            
            # 记录统计信息
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            # 日志记录（区分ISO三阶段模式和单阶段模式）
            if use_iso_three_stages and self.config.sleep_therapy_mode:
                logger.info(f"映射层处理完成 (ISO三阶段) - "
                           f"匹配: {iso_params.match_stage.tempo_bpm:.1f}BPM → "
                           f"引导: {iso_params.guide_stage.tempo_bpm:.1f}BPM → "
                           f"目标: {iso_params.target_stage.tempo_bpm:.1f}BPM, "
                           f"置信度: {mapping_confidence:.3f}, 耗时: {processing_time*1000:.1f}ms")
            else:
                logger.info(f"映射层处理完成 - 节拍: {music_params.tempo_bpm:.1f}BPM, "
                           f"调性: {music_params.key_signature}, 置信度: {mapping_confidence:.3f}, "
                           f"耗时: {processing_time*1000:.1f}ms")
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"映射层处理失败: {e}")
            
            # 创建错误输出
            error_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'error': str(e),
                    'music_parameters': MusicParameter().to_dict(),  # 默认参数
                    'mapping_info': {
                        'strategy': self.config.mapping_strategy,
                        'error': True
                    }
                },
                metadata={'error': True, 'source_layer': input_data.layer_name},
                confidence=0.0
            )
            
            processing_time = self.performance_monitor.end_timer("mapping_layer_processing")
            error_data.processing_time = processing_time
            
            return error_data
    
    def _calculate_mapping_confidence(self, emotion_vector: torch.Tensor, music_params: MusicParameter) -> float:
        """计算映射置信度"""
        # 基于情绪向量的确定性
        emotion_entropy = -torch.sum(emotion_vector * torch.log(emotion_vector + 1e-8)).item()
        emotion_confidence = 1.0 / (1.0 + emotion_entropy)
        
        # 基于音乐参数的合理性
        param_confidence = 1.0
        
        # 检查是否有tempo_bpm属性（兼容MusicParameter和ISOThreeStageParams）
        if hasattr(music_params, 'tempo_bpm'):
            if music_params.tempo_bpm < 30 or music_params.tempo_bpm > 150:
                param_confidence *= 0.8
            if hasattr(music_params, 'therapy_intensity') and (music_params.therapy_intensity < 0.1 or music_params.therapy_intensity > 1.0):
                param_confidence *= 0.9
        
        # 综合置信度
        overall_confidence = (emotion_confidence * 0.6 + param_confidence * 0.4)
        
        return float(np.clip(overall_confidence, 0.0, 1.0))
    
    # ==================== 标准化接口（用户规范） ====================
    
    def get_kg_initial_mapping(self, emotion_vector: torch.Tensor) -> Dict[str, float]:
        """
        知识图谱初始映射 - 基于GEMS原理的规则集
        输入: 27维情绪向量 [0.0-1.0]
        输出: 8个标准音乐参数 [标准化数值]
        """
        # 将tensor转为numpy便于处理
        if isinstance(emotion_vector, torch.Tensor):
            emotion_values = emotion_vector.cpu().numpy()
        else:
            emotion_values = np.array(emotion_vector)
        
        # 找到主导情绪（最高强度）
        dominant_emotion_idx = np.argmax(emotion_values)
        dominant_intensity = emotion_values[dominant_emotion_idx]
        
        # 基于GEMS原理的音乐映射规则（简化版）
        # 27维情绪映射：0-8基础情绪，9-26睡眠特化情绪
        
        # 默认参数（中性状态）
        kg_params = {
            'tempo': 0.5,           # 0.0-1.0 映射到 60-120 BPM
            'mode': 0.0,            # 0.0=大调, 1.0=小调
            'dynamics': 0.5,        # 0.0-1.0 音量强度
            'harmony_consonance': 0.7,  # 0.0-1.0 和声协和度
            'timbre_preference': 0.5,   # 0.0-1.0 音色偏好
            'pitch_register': 0.5,      # 0.0-1.0 音高音域
            'density': 0.5,         # 0.0-1.0 密度
            'emotional_envelope_direction': 0.0  # -1.0到1.0 情绪方向
        }
        
        # GEMS原理规则集
        if dominant_emotion_idx == 1:  # fear_anxiety (焦虑)
            kg_params.update({
                'tempo': 0.7 + dominant_intensity * 0.2,  # 较快节拍
                'mode': 0.8,                              # 偏小调
                'dynamics': 0.3,                          # 较弱力度（睡眠治疗）
                'harmony_consonance': 0.3,                # 较不协和
                'timbre_preference': 0.2,                 # 柔和音色
                'pitch_register': 0.6,                    # 中高音域
                'density': 0.3,                           # 低密度
                'emotional_envelope_direction': -0.8      # 下降趋势
            })
        elif dominant_emotion_idx >= 13 and dominant_emotion_idx <= 15:  # hyperarousal (过度觉醒)
            kg_params.update({
                'tempo': 0.8 + dominant_intensity * 0.1,  # 快节拍
                'mode': 0.0,                              # 大调
                'dynamics': 0.2,                          # 很弱力度
                'harmony_consonance': 0.5,                # 中等协和
                'timbre_preference': 0.1,                 # 极柔和
                'pitch_register': 0.3,                    # 低音域
                'density': 0.2,                           # 极低密度
                'emotional_envelope_direction': -0.9      # 强烈下降
            })
        elif dominant_emotion_idx >= 18 and dominant_emotion_idx <= 20:  # peaceful (平静)
            kg_params.update({
                'tempo': 0.2,                             # 很慢节拍
                'mode': 0.0,                              # 大调
                'dynamics': 0.1,                          # 极弱力度
                'harmony_consonance': 0.9,                # 高度协和
                'timbre_preference': 0.0,                 # 最柔和音色
                'pitch_register': 0.2,                    # 低音域
                'density': 0.1,                           # 极低密度
                'emotional_envelope_direction': 0.1       # 微上升（积极）
            })
        elif dominant_emotion_idx >= 9 and dominant_emotion_idx <= 12:  # sleep_anxiety (睡眠焦虑)
            kg_params.update({
                'tempo': 0.4,                             # 中慢节拍
                'mode': 0.6,                              # 偏小调
                'dynamics': 0.2,                          # 弱力度
                'harmony_consonance': 0.6,                # 较协和
                'timbre_preference': 0.3,                 # 柔和音色
                'pitch_register': 0.4,                    # 中低音域
                'density': 0.3,                           # 低密度
                'emotional_envelope_direction': -0.6      # 下降趋势
            })
        
        # 根据睡眠治疗模式调整
        if self.config.sleep_therapy_mode:
            kg_params['tempo'] = min(kg_params['tempo'], 0.6)  # 限制最大节拍
            kg_params['dynamics'] = min(kg_params['dynamics'], 0.3)  # 限制音量
            kg_params['emotional_envelope_direction'] = min(kg_params['emotional_envelope_direction'], 0.0)  # 偏向下降
        
        return kg_params
    
    def apply_mlp_personalization(self, kg_parameters: Dict[str, float], 
                                  emotion_vector: torch.Tensor, 
                                  user_profile_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        MLP个性化微调机制
        输入: KG初始参数 + 情绪向量 + 用户偏好数据
        输出: 个性化调整后的音乐参数
        """
        personalized_params = kg_parameters.copy()
        
        # 模拟用户偏好数据（如果没有提供）
        if user_profile_data is None:
            user_profile_data = {
                'tempo_preference': 0.0,      # -0.2到0.2的偏好调整
                'mode_preference': 0.0,       # 调式偏好
                'volume_sensitivity': 1.0,    # 音量敏感度
                'harmony_preference': 0.0,    # 和声偏好
                'therapy_responsiveness': 1.0 # 治疗响应度
            }
        
        # 个性化调整逻辑
        
        # 1. 节拍个性化：基于用户历史偏好
        tempo_adjustment = user_profile_data.get('tempo_preference', 0.0)
        personalized_params['tempo'] = np.clip(
            personalized_params['tempo'] + tempo_adjustment, 0.0, 1.0
        )
        
        # 2. 调式个性化：根据用户情绪反应模式
        if user_profile_data.get('mode_preference', 0.0) != 0.0:
            mode_shift = user_profile_data['mode_preference'] * 0.1
            personalized_params['mode'] = np.clip(
                personalized_params['mode'] + mode_shift, 0.0, 1.0
            )
        
        # 3. 动态个性化：基于情绪强度和用户敏感度
        emotion_intensity = torch.max(emotion_vector).item()
        volume_sensitivity = user_profile_data.get('volume_sensitivity', 1.0)
        
        if emotion_intensity > 0.7:  # 高强度情绪
            personalized_params['dynamics'] *= (0.8 * volume_sensitivity)
        
        # 4. 和声个性化：根据用户音乐背景
        harmony_pref = user_profile_data.get('harmony_preference', 0.0)
        personalized_params['harmony_consonance'] = np.clip(
            personalized_params['harmony_consonance'] + harmony_pref * 0.2, 0.0, 1.0
        )
        
        # 5. 治疗响应度调整：根据用户治疗效果历史
        therapy_responsiveness = user_profile_data.get('therapy_responsiveness', 1.0)
        if therapy_responsiveness > 1.0:  # 响应度高的用户
            personalized_params['emotional_envelope_direction'] *= 1.2
            personalized_params['emotional_envelope_direction'] = np.clip(
                personalized_params['emotional_envelope_direction'], -1.0, 1.0
            )
        
        # 6. 昼夜节律调整
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 6:  # 深夜/凌晨
            personalized_params['tempo'] *= 0.9
            personalized_params['dynamics'] *= 0.8
        
        return personalized_params
    
    def map_emotion_to_music(self, emotion_vector: torch.Tensor, 
                           user_profile_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        主映射函数：情绪向量 → 标准化音乐参数
        
        Args:
            emotion_vector: 27维情绪向量 [0.0-1.0]
            user_profile_data: 用户偏好数据（可选）
            
        Returns:
            标准化音乐参数字典，包含8个标准参数
        """
        # 阶段1：知识图谱初始映射
        kg_parameters = self.get_kg_initial_mapping(emotion_vector)
        
        # 阶段2：MLP个性化微调
        final_parameters = self.apply_mlp_personalization(
            kg_parameters, emotion_vector, user_profile_data
        )
        
        # 确保所有参数在有效范围内
        for key, value in final_parameters.items():
            if key == 'emotional_envelope_direction':
                final_parameters[key] = np.clip(value, -1.0, 1.0)
            else:
                final_parameters[key] = np.clip(value, 0.0, 1.0)
        
        return final_parameters
    
    def convert_to_detailed_params(self, standard_params: Dict[str, float]) -> MusicParameter:
        """
        工具函数：将标准化参数转换为详细的MusicParameter对象
        用于与现有ISO三阶段功能兼容
        """
        music_param = MusicParameter()
        
        # 转换基础参数
        music_param.tempo_bpm = 60.0 + standard_params['tempo'] * 60.0  # 60-120 BPM
        music_param.key_signature = "C_major" if standard_params['mode'] < 0.5 else "A_minor"
        
        # 力度映射
        dynamics_map = ["pp", "p", "mp", "mf"]
        dynamics_idx = int(standard_params['dynamics'] * 3.99)
        music_param.dynamics = dynamics_map[min(dynamics_idx, 3)]
        
        # 情绪映射
        music_param.valence_mapping = 1.0 - standard_params['mode']  # 大调=正效价
        music_param.arousal_mapping = standard_params['tempo'] * 2.0 - 1.0  # -1到1
        music_param.tension_level = 1.0 - standard_params['harmony_consonance']
        
        # 织体和密度
        music_param.texture_complexity = standard_params['density']
        music_param.harmonic_richness = standard_params['harmony_consonance']
        
        return music_param
    
    # ==================== 原有功能保持不变 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取映射层状态"""
        base_status = super().get_status()
        
        # 添加映射层特有的状态信息
        mapping_status = {
            'mapping_strategy': self.config.mapping_strategy,
            'kg_enabled': self.config.kg_enabled,
            'mlp_enabled': self.config.mlp_enabled,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'sleep_therapy_mode': self.config.sleep_therapy_mode,
            'kg_weight': self.kg_weight,
            'mlp_weight': self.mlp_weight,
            'performance_stats': self.performance_monitor.get_all_stats()
        }
        
        base_status.update(mapping_status)
        return base_status