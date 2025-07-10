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
    """音乐参数结构"""
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
        self.therapy_intensity: float = 0.5      # 治疗强度
        self.sleep_readiness: float = 0.0        # 睡眠准备度
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
            'therapy_intensity': self.therapy_intensity,
            'sleep_readiness': self.sleep_readiness
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
    
    def _apply_sleep_therapy_adaptation(self, music_params: MusicParameter, therapy_context: Dict[str, Any]) -> MusicParameter:
        """应用睡眠治疗适应"""
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
            
            # 根据策略执行映射
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
            
            # 创建输出数据
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
                        'mapping_confidence': mapping_confidence
                    },
                    'emotion_vector': emotion_vector.cpu().numpy().tolist(),
                    'processing_info': {
                        'kg_enabled': self.config.kg_enabled,
                        'mlp_enabled': self.config.mlp_enabled,
                        'sleep_therapy_mode': self.config.sleep_therapy_mode
                    }
                },
                metadata={
                    'source_layer': input_data.layer_name,
                    'processing_device': str(self.device),
                    'mapping_strategy': self.config.mapping_strategy
                },
                confidence=mapping_confidence
            )
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer("mapping_layer_processing")
            output_data.processing_time = processing_time
            
            # 记录统计信息
            self.total_processed += 1
            self.total_processing_time += processing_time
            
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
        if music_params.tempo_bpm < 30 or music_params.tempo_bpm > 150:
            param_confidence *= 0.8
        if music_params.therapy_intensity < 0.1 or music_params.therapy_intensity > 1.0:
            param_confidence *= 0.9
        
        # 综合置信度
        overall_confidence = (emotion_confidence * 0.6 + param_confidence * 0.4)
        
        return float(np.clip(overall_confidence, 0.0, 1.0))
    
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