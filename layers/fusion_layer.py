#!/usr/bin/env python3
"""
融合层 (Fusion Layer) - Layer 2

27维细粒度情绪分类和多模态融合，核心功能包括：
1. 27维情绪空间的细粒度分类
2. 多模态情绪数据的融合处理
3. 情绪关系建模和转换预测
4. 置信度加权的混合融合算法
5. 睡眠专用情绪的特殊处理

处理流程：
Input Layer → Fusion Layer → Mapping Layer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

from .base_layer import BaseLayer, LayerData, LayerConfig
from core.utils import (
    ConfigLoader, DataValidator, PerformanceMonitor, 
    get_project_root, normalize_vector, cosine_similarity
)

logger = logging.getLogger(__name__)

@dataclass
class FusionLayerConfig(LayerConfig):
    """融合层配置"""
    # 情绪分类配置
    total_emotions: int = 27
    base_emotions: int = 9
    extended_emotions: int = 18
    
    # 多模态融合配置
    text_weight: float = 0.4
    audio_weight: float = 0.3
    video_weight: float = 0.3
    
    # 融合策略配置
    fusion_strategy: str = "confidence_weighted"  # ["simple", "confidence_weighted", "attention_based"]
    confidence_threshold: float = 0.6
    
    # 情绪关系建模
    enable_emotion_relationships: bool = True
    relationship_weight: float = 0.2
    
    # 性能配置
    use_gpu: bool = True
    batch_size: int = 1
    max_processing_time_ms: int = 150

class EmotionClassifier(nn.Module):
    """27维情绪分类器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_emotions: int = 27):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # 情绪分类头
        self.emotion_classifier = nn.Linear(hidden_dim // 4, num_emotions)
        
        # 置信度预测头
        self.confidence_predictor = nn.Linear(hidden_dim // 4, 1)
        
        # 情绪强度预测头
        self.intensity_predictor = nn.Linear(hidden_dim // 4, num_emotions)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 特征提取
        features = self.feature_extractor(x)
        
        # 情绪分类（使用softmax）
        emotion_logits = self.emotion_classifier(features)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # 置信度预测（使用sigmoid）
        confidence = torch.sigmoid(self.confidence_predictor(features))
        
        # 情绪强度预测（使用sigmoid）
        intensity = torch.sigmoid(self.intensity_predictor(features))
        
        return {
            'emotion_probs': emotion_probs,
            'confidence': confidence,
            'intensity': intensity,
            'features': features
        }

class MultiModalFusionModule:
    """多模态融合模块"""
    
    def __init__(self, config: FusionLayerConfig):
        self.config = config
        self.text_weight = config.text_weight
        self.audio_weight = config.audio_weight
        self.video_weight = config.video_weight
        
        # 归一化权重
        total_weight = self.text_weight + self.audio_weight + self.video_weight
        self.text_weight /= total_weight
        self.audio_weight /= total_weight
        self.video_weight /= total_weight
        
        logger.info(f"多模态权重 - 文本:{self.text_weight:.2f}, 音频:{self.audio_weight:.2f}, 视频:{self.video_weight:.2f}")
    
    def simple_fusion(self, modality_results: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """简单加权融合"""
        fused_emotions = None
        fused_confidence = None
        fused_intensity = None
        
        for modality, results in modality_results.items():
            if modality == 'text':
                weight = self.text_weight
            elif modality == 'audio':
                weight = self.audio_weight
            elif modality == 'video':
                weight = self.video_weight
            else:
                continue
                
            if fused_emotions is None:
                fused_emotions = weight * results['emotion_probs']
                fused_confidence = weight * results['confidence']
                fused_intensity = weight * results['intensity']
            else:
                fused_emotions += weight * results['emotion_probs']
                fused_confidence += weight * results['confidence']
                fused_intensity += weight * results['intensity']
        
        return {
            'emotion_probs': fused_emotions,
            'confidence': fused_confidence,
            'intensity': fused_intensity
        }
    
    def confidence_weighted_fusion(self, modality_results: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """置信度加权融合"""
        # 计算各模态的置信度权重
        confidences = {}
        for modality, results in modality_results.items():
            confidences[modality] = results['confidence'].item()
        
        # 归一化置信度
        total_confidence = sum(confidences.values())
        if total_confidence > 0:
            for modality in confidences:
                confidences[modality] /= total_confidence
        else:
            # 如果所有置信度都为0，使用均匀权重
            num_modalities = len(confidences)
            for modality in confidences:
                confidences[modality] = 1.0 / num_modalities
        
        # 应用置信度权重
        fused_emotions = None
        fused_confidence = None
        fused_intensity = None
        
        for modality, results in modality_results.items():
            weight = confidences[modality]
            
            if fused_emotions is None:
                fused_emotions = weight * results['emotion_probs']
                fused_confidence = weight * results['confidence']
                fused_intensity = weight * results['intensity']
            else:
                fused_emotions += weight * results['emotion_probs']
                fused_confidence += weight * results['confidence']
                fused_intensity += weight * results['intensity']
        
        return {
            'emotion_probs': fused_emotions,
            'confidence': fused_confidence,
            'intensity': fused_intensity,
            'modality_weights': confidences
        }
    
    def attention_based_fusion(self, modality_results: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """基于注意力机制的融合"""
        # 提取特征
        features = []
        for modality, results in modality_results.items():
            features.append(results['features'])
        
        # 计算注意力权重
        if len(features) > 1:
            feature_stack = torch.stack(features, dim=0)
            attention_scores = torch.softmax(torch.sum(feature_stack, dim=-1), dim=0)
        else:
            attention_scores = torch.ones(len(features)) / len(features)
        
        # 应用注意力权重
        fused_emotions = None
        fused_confidence = None
        fused_intensity = None
        
        for i, (modality, results) in enumerate(modality_results.items()):
            weight = attention_scores[i].item()
            
            if fused_emotions is None:
                fused_emotions = weight * results['emotion_probs']
                fused_confidence = weight * results['confidence']
                fused_intensity = weight * results['intensity']
            else:
                fused_emotions += weight * results['emotion_probs']
                fused_confidence += weight * results['confidence']
                fused_intensity += weight * results['intensity']
        
        return {
            'emotion_probs': fused_emotions,
            'confidence': fused_confidence,
            'intensity': fused_intensity,
            'attention_weights': attention_scores.tolist()
        }
    
    def fuse_modalities(self, modality_results: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """执行多模态融合"""
        # 处理复杂配置格式
        strategy = self.config.fusion_strategy
        if isinstance(strategy, dict):
            strategy_name = strategy.get('algorithm', 'confidence_weighted')
        else:
            strategy_name = strategy
        
        if strategy_name == "simple":
            return self.simple_fusion(modality_results)
        elif strategy_name in ["confidence_weighted", "hybrid_attention_fusion"]:
            return self.confidence_weighted_fusion(modality_results)
        elif strategy_name == "attention_based":
            return self.attention_based_fusion(modality_results)
        else:
            logger.warning(f"未知的融合策略: {strategy_name}，使用默认策略")
            return self.confidence_weighted_fusion(modality_results)

class EmotionRelationshipModule:
    """情绪关系建模模块"""
    
    def __init__(self, emotion_config: Dict[str, Any], device: str = 'cpu'):
        self.emotion_config = emotion_config
        self.emotion_relationships = emotion_config.get('emotion_relationships', {})
        self.device = device
        
        # 创建情绪ID映射
        self.emotion_id_map = {}
        self.id_emotion_map = {}
        
        # 基础情绪映射
        for emotion_name, emotion_data in emotion_config.get('base_emotions', {}).items():
            emotion_id = emotion_data['id'] - 1  # 转换为0-based索引
            self.emotion_id_map[emotion_name] = emotion_id
            self.id_emotion_map[emotion_id] = emotion_name
        
        # 睡眠专用情绪映射
        for emotion_name, emotion_data in emotion_config.get('sleep_specific_emotions', {}).items():
            emotion_id = emotion_data['id'] - 1  # 转换为0-based索引
            self.emotion_id_map[emotion_name] = emotion_id
            self.id_emotion_map[emotion_id] = emotion_name
        
        # 构建关系矩阵
        self.mutual_exclusion_matrix = self._build_mutual_exclusion_matrix()
        self.synergy_matrix = self._build_synergy_matrix()
        self.transition_matrix = self._build_transition_matrix()
        
        logger.info(f"情绪关系建模初始化完成，共{len(self.emotion_id_map)}种情绪")
    
    def _build_mutual_exclusion_matrix(self) -> torch.Tensor:
        """构建互斥关系矩阵"""
        num_emotions = len(self.emotion_id_map)
        matrix = torch.zeros(num_emotions, num_emotions, device=self.device)
        
        mutual_exclusive = self.emotion_relationships.get('mutually_exclusive', [])
        for emotion_pair in mutual_exclusive:
            if len(emotion_pair) == 2:
                emotion1, emotion2 = emotion_pair
                if emotion1 in self.emotion_id_map and emotion2 in self.emotion_id_map:
                    id1 = self.emotion_id_map[emotion1]
                    id2 = self.emotion_id_map[emotion2]
                    matrix[id1, id2] = 1.0
                    matrix[id2, id1] = 1.0
        
        return matrix
    
    def _build_synergy_matrix(self) -> torch.Tensor:
        """构建协同关系矩阵"""
        num_emotions = len(self.emotion_id_map)
        matrix = torch.zeros(num_emotions, num_emotions, device=self.device)
        
        synergistic = self.emotion_relationships.get('synergistic', [])
        for emotion_pair in synergistic:
            if len(emotion_pair) == 2:
                emotion1, emotion2 = emotion_pair
                if emotion1 in self.emotion_id_map and emotion2 in self.emotion_id_map:
                    id1 = self.emotion_id_map[emotion1]
                    id2 = self.emotion_id_map[emotion2]
                    matrix[id1, id2] = 1.0
                    matrix[id2, id1] = 1.0
        
        return matrix
    
    def _build_transition_matrix(self) -> torch.Tensor:
        """构建转换关系矩阵"""
        num_emotions = len(self.emotion_id_map)
        matrix = torch.zeros(num_emotions, num_emotions, device=self.device)
        
        transition_pairs = self.emotion_relationships.get('transition_pairs', [])
        for transition in transition_pairs:
            from_emotion = transition.get('from')
            to_emotion = transition.get('to')
            probability = transition.get('probability', 0.5)
            
            if (from_emotion in self.emotion_id_map and 
                to_emotion in self.emotion_id_map):
                from_id = self.emotion_id_map[from_emotion]
                to_id = self.emotion_id_map[to_emotion]
                matrix[from_id, to_id] = probability
        
        return matrix
    
    def apply_relationship_constraints(self, emotion_probs: torch.Tensor, relationship_weight: float = 0.2) -> torch.Tensor:
        """应用情绪关系约束"""
        adjusted_probs = emotion_probs.clone()
        
        # 确保emotion_probs是正确的形状 (batch_size, num_emotions)
        if len(emotion_probs.shape) == 1:
            emotion_probs = emotion_probs.unsqueeze(0)
        
        # 对于矩阵乘法，我们需要转置emotion_probs为 (num_emotions, batch_size)
        emotion_probs_t = emotion_probs.transpose(0, 1)  # (27, 1)
        
        # 应用协同关系增强: (27, 27) x (27, 1) = (27, 1)
        synergy_boost = torch.matmul(self.synergy_matrix, emotion_probs_t)
        synergy_boost = synergy_boost.transpose(0, 1)  # 转回 (1, 27)
        adjusted_probs += relationship_weight * synergy_boost
        
        # 应用互斥关系抑制: (27, 27) x (27, 1) = (27, 1)
        exclusion_suppression = torch.matmul(self.mutual_exclusion_matrix, emotion_probs_t)
        exclusion_suppression = exclusion_suppression.transpose(0, 1)  # 转回 (1, 27)
        adjusted_probs -= relationship_weight * exclusion_suppression
        
        # 确保概率非负并重新归一化
        adjusted_probs = torch.clamp(adjusted_probs, min=0.0)
        adjusted_probs = F.softmax(adjusted_probs, dim=-1)
        
        return adjusted_probs
    
    def predict_emotion_transition(self, current_emotion_probs: torch.Tensor) -> Dict[str, float]:
        """预测情绪转换"""
        # 确保输入张量形状正确
        if len(current_emotion_probs.shape) > 1:
            current_emotion_probs = current_emotion_probs.squeeze()
        
        # 计算转换概率
        transition_probs = torch.matmul(self.transition_matrix, current_emotion_probs)
        
        # 确保transition_probs是1维张量
        if len(transition_probs.shape) > 1:
            transition_probs = transition_probs.squeeze()
        
        # 找到最可能的转换
        top_transitions = torch.topk(transition_probs, k=min(3, len(transition_probs)))
        
        transitions = {}
        for i, (prob, emotion_id) in enumerate(zip(top_transitions.values, top_transitions.indices)):
            if prob > 0.1:  # 只考虑概率大于0.1的转换
                emotion_name = self.id_emotion_map.get(emotion_id.item(), f"emotion_{emotion_id.item()}")
                transitions[emotion_name] = prob.item()
        
        return transitions

class FusionLayer(BaseLayer):
    """融合层 - 27维情绪分类和多模态融合"""
    
    def __init__(self, config: FusionLayerConfig):
        super().__init__(config)
        self.config = config
        self.layer_name = "fusion_layer"
        
        # 加载情绪配置
        self.emotion_config = self._load_emotion_config()
        
        # 初始化设备
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"融合层使用设备: {self.device}")
        
        # 初始化分类器
        self.text_classifier = EmotionClassifier(768, num_emotions=config.total_emotions).to(self.device)  # BERT特征维度
        self.audio_classifier = EmotionClassifier(512, num_emotions=config.total_emotions).to(self.device)  # 音频特征维度
        self.video_classifier = EmotionClassifier(512, num_emotions=config.total_emotions).to(self.device)  # 视频特征维度
        
        # 初始化融合模块
        self.fusion_module = MultiModalFusionModule(config)
        
        # 初始化情绪关系建模
        if config.enable_emotion_relationships:
            self.relationship_module = EmotionRelationshipModule(self.emotion_config, device=str(self.device))
        else:
            self.relationship_module = None
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 数据验证器
        self.data_validator = DataValidator()
        
        logger.info(f"融合层初始化完成，支持{config.total_emotions}维情绪分类")
    
    def _load_emotion_config(self) -> Dict[str, Any]:
        """加载情绪配置"""
        try:
            config_path = get_project_root() / "configs" / "emotion_27d.yaml"
            return ConfigLoader.load_yaml(str(config_path))
        except Exception as e:
            logger.error(f"加载情绪配置失败: {e}")
            return {}
    
    def _extract_features(self, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """从输入数据提取特征"""
        features = {}
        
        # 提取文本特征
        if 'text' in input_data:
            text_data = input_data['text']
            if isinstance(text_data, str) and len(text_data.strip()) > 0:
                # 简化的文本特征提取（基于关键词）
                text_features = self._extract_text_features(text_data)
                features['text'] = text_features
                logger.debug(f"提取文本特征: {text_features.shape}")
            else:
                logger.warning(f"文本数据无效: {text_data}")
        
        # 提取音频特征
        if 'audio' in input_data:
            audio_data = input_data['audio']
            if isinstance(audio_data, np.ndarray):
                # 这里应该使用音频特征提取器
                # 为了简化，我们使用随机特征
                audio_features = torch.randn(1, 512).to(self.device)
                features['audio'] = audio_features
                logger.debug(f"提取音频特征: {audio_features.shape}")
        
        # 提取视频特征
        if 'video' in input_data:
            video_data = input_data['video']
            if isinstance(video_data, np.ndarray):
                # 这里应该使用视频特征提取器
                # 为了简化，我们使用随机特征
                video_features = torch.randn(1, 512).to(self.device)
                features['video'] = video_features
                logger.debug(f"提取视频特征: {video_features.shape}")
        
        return features
    
    def _extract_text_features(self, text: str) -> torch.Tensor:
        """增强的文本特征提取"""
        # 27维情绪关键词映射（更细粒度）
        emotion_keywords = {
            # 基础情绪（9维）
            'anger': ['愤怒', '生气', '烦躁', '恼火', '愤恨', '激怒'],
            'fear_anxiety': ['焦虑', '紧张', '担心', '害怕', '不安', '恐惧', '惊慌'],
            'disgust': ['厌恶', '恶心', '反感', '讨厌', '憎恶'],
            'sadness': ['悲伤', '沮丧', '失落', '难过', '抑郁', '消沉'],
            'amusement': ['有趣', '好玩', '逗乐', '娱乐', '幽默'],
            'joy': ['开心', '快乐', '高兴', '愉快', '兴奋', '欣喜'],
            'inspiration': ['启发', '激励', '鼓舞', '振奋', '感动'],
            'tenderness': ['温柔', '温暖', '体贴', '关爱', '柔情'],
            'neutral': ['平常', '一般', '普通', '正常', '平淡'],
            
            # 睡眠专用情绪（18维）
            'rumination': ['反刍', '思考', '纠结', '想太多', '胡思乱想', '钻牛角尖'],
            'sleep_anxiety': ['睡眠焦虑', '睡不着', '失眠', '难眠', '睡觉焦虑'],
            'physical_fatigue': ['身体疲惫', '体力不支', '身体累', '肌肉酸痛', '体力透支'],
            'mental_fatigue': ['精神疲惫', '脑子累', '思维疲劳', '心理疲惫', '精神透支'],
            'hyperarousal': ['过度觉醒', '精神亢奋', '太兴奋', '睡不下去', '大脑活跃'],
            'bedtime_worry': ['就寝担忧', '睡前担心', '床上焦虑', '躺下就担心'],
            'sleep_dread': ['睡眠恐惧', '怕睡觉', '睡觉恐惧', '害怕入睡'],
            'racing_thoughts': ['思维奔逸', '想法很多', '脑子转不停', '思绪万千'],
            'somatic_tension': ['躯体紧张', '身体紧绷', '肌肉紧张', '全身紧张'],
            'emotional_numbness': ['情感麻木', '没感觉', '情绪冷淡', '无所谓'],
            'restless_energy': ['不安能量', '躁动', '坐立不安', '精力过剩'],
            'sleep_frustration': ['睡眠挫败', '睡不着很烦', '失眠挫折', '对睡眠失望'],
            'bedtime_loneliness': ['就寝孤独', '一个人睡觉', '夜晚孤单', '睡前孤独'],
            'anticipatory_anxiety': ['预期焦虑', '担心明天', '提前担心', '未来焦虑'],
            'sleep_perfectionism': ['睡眠完美主义', '必须睡好', '睡觉要求高', '对睡眠苛刻'],
            'bedroom_discomfort': ['卧室不适', '环境不舒服', '床不舒服', '房间问题'],
            'sleep_monitoring_anxiety': ['睡眠监控焦虑', '担心睡眠质量', '过度关注睡眠'],
            'morning_dread': ['晨起恐惧', '不想起床', '害怕早上', '起床焦虑'],
            
            # 正面睡眠状态
            'peaceful': ['平静', '安详', '宁静', '祥和', '安宁'],
            'relaxed': ['放松', '舒适', '松弛', '自在', '惬意'],
            'drowsy': ['困倦', '想睡', '疲倦', '倦意', '睡意'],
            'ready_for_sleep': ['准备睡觉', '想要休息', '可以睡了', '准备入睡']
        }
        
        # 计算特征向量（增强版）
        text_lower = text.lower().replace('，', '').replace('。', '').replace('！', '').replace('？', '')
        features = []
        
        # 对每个情绪类别计算匹配分数
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            # 计算关键词匹配分数（考虑权重）
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # 根据关键词长度给予不同权重
                    weight = len(keyword) / 10.0  # 更长的关键词权重更高
                    score += weight
            emotion_scores[emotion] = score
            features.append(score)
        
        # 添加语义特征
        semantic_features = [
            len(text),  # 文本长度
            text.count('很'),  # 强化词
            text.count('非常'),  # 强化词
            text.count('特别'),  # 强化词
            text.count('就是'),  # 确定词
            text.count('感觉'),  # 感受词
            text.count('觉得'),  # 感受词
            1 if '睡不着' in text_lower else 0,  # 失眠指标
            1 if '疲惫' in text_lower else 0,  # 疲劳指标
            1 if '焦虑' in text_lower else 0,  # 焦虑指标
            1 if '平静' in text_lower else 0,  # 平静指标
            1 if '大脑' in text_lower or '脑子' in text_lower else 0,  # 大脑活动
            1 if '身体' in text_lower else 0,  # 身体状态
        ]
        features.extend(semantic_features)
        
        # 情绪强度计算
        max_emotion_score = max(emotion_scores.values()) if emotion_scores.values() else 0
        emotion_intensity = min(max_emotion_score, 5.0) / 5.0  # 标准化到0-1
        features.append(emotion_intensity)
        
        # 情绪极性（正面/负面）
        positive_emotions = ['joy', 'peaceful', 'relaxed', 'drowsy', 'ready_for_sleep', 'tenderness', 'inspiration']
        negative_emotions = ['anger', 'fear_anxiety', 'sadness', 'sleep_anxiety', 'sleep_frustration']
        
        positive_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)
        
        if positive_score + negative_score > 0:
            polarity = (positive_score - negative_score) / (positive_score + negative_score)
        else:
            polarity = 0.0
        features.append(polarity)
        
        # 填充到768维（模拟BERT特征）
        while len(features) < 768:
            # 使用更复杂的填充策略
            remaining = 768 - len(features)
            if remaining > len(features):
                features.extend(features)
            else:
                features.extend(features[:remaining])
        
        features = features[:768]  # 截断到768维
        
        # 增强归一化（防止全零向量）
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 如果特征向量全为零，添加小的随机噪声
        if torch.sum(torch.abs(features_tensor)) < 1e-6:
            features_tensor += torch.randn_like(features_tensor) * 0.01
        
        features_tensor = F.normalize(features_tensor, p=2, dim=1)
        
        return features_tensor
    
    def _classify_emotions(self, features: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """执行情绪分类"""
        results = {}
        
        # 文本情绪分类
        if 'text' in features:
            text_result = self.text_classifier(features['text'])
            results['text'] = text_result
            logger.debug(f"文本情绪分类完成，置信度: {text_result['confidence'].item():.3f}")
        
        # 音频情绪分类
        if 'audio' in features:
            audio_result = self.audio_classifier(features['audio'])
            results['audio'] = audio_result
            logger.debug(f"音频情绪分类完成，置信度: {audio_result['confidence'].item():.3f}")
        
        # 视频情绪分类
        if 'video' in features:
            video_result = self.video_classifier(features['video'])
            results['video'] = video_result
            logger.debug(f"视频情绪分类完成，置信度: {video_result['confidence'].item():.3f}")
        
        return results
    
    def _interpret_emotion_results(self, fused_results: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """解释情绪分类结果"""
        emotion_probs = fused_results['emotion_probs'].detach().cpu().numpy()
        confidence = fused_results['confidence'].item()
        intensity = fused_results['intensity'].detach().cpu().numpy()
        
        # 确保emotion_probs是1维数组
        if len(emotion_probs.shape) > 1:
            emotion_probs = emotion_probs.flatten()
        
        # 确保intensity是1维数组并且与emotion_probs长度匹配
        if len(intensity.shape) > 1:
            intensity = intensity.flatten()
        
        # 如果intensity长度不匹配，使用emotion_probs作为强度的近似
        if len(intensity) != len(emotion_probs):
            intensity = emotion_probs.copy()
        
        # 找到最可能的情绪
        top_emotion_id = np.argmax(emotion_probs)
        top_emotion_prob = emotion_probs[top_emotion_id]
        
        # 获取情绪名称
        emotion_name = self.relationship_module.id_emotion_map.get(top_emotion_id, f"emotion_{top_emotion_id}") if self.relationship_module else f"emotion_{top_emotion_id}"
        
        # 获取前3个最可能的情绪
        top_3_indices = np.argsort(emotion_probs)[-3:][::-1]
        top_3_emotions = []
        for idx in top_3_indices:
            name = self.relationship_module.id_emotion_map.get(idx, f"emotion_{idx}") if self.relationship_module else f"emotion_{idx}"
            top_3_emotions.append({
                'name': name,
                'probability': float(emotion_probs[idx]),
                'intensity': float(intensity[idx]) if idx < len(intensity) else float(emotion_probs[idx])
            })
        
        # 预测情绪转换
        transitions = {}
        if self.relationship_module:
            transitions = self.relationship_module.predict_emotion_transition(fused_results['emotion_probs'])
        
        return {
            'primary_emotion': {
                'name': emotion_name,
                'probability': float(top_emotion_prob),
                'intensity': float(intensity[top_emotion_id]) if top_emotion_id < len(intensity) else float(top_emotion_prob)
            },
            'top_3_emotions': top_3_emotions,
            'overall_confidence': confidence,
            'emotion_vector': emotion_probs.tolist(),
            'intensity_vector': intensity.tolist(),
            'predicted_transitions': transitions
        }
    
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """融合层处理实现"""
        self.performance_monitor.start_timer("fusion_layer_processing")
        
        try:
            # 验证输入数据
            if not input_data.data:
                raise ValueError("输入数据为空")
            
            # 提取特征
            features = self._extract_features(input_data.data)
            if not features or len(features) == 0:
                logger.warning(f"输入数据无有效模态，数据内容: {input_data.data}")
                # 检查是否有多模态数据结构
                if 'multimodal_data' in input_data.data:
                    multimodal = input_data.data['multimodal_data']
                    if 'text' in multimodal and multimodal['text'].get('text'):
                        # 直接处理文本数据
                        text_content = multimodal['text']['text']
                        features = {'text': self._extract_text_features(text_content)}
                        logger.info(f"从多模态数据中提取文本特征: {text_content[:50]}...")
                    else:
                        raise ValueError("多模态数据中无有效文本")
                elif 'text' in input_data.data:
                    features = {'text': torch.randn(1, 768).to(self.device)}
                    logger.info("使用默认文本特征进行处理")
                else:
                    raise ValueError("无法提取有效特征")
            
            # 执行情绪分类
            classification_results = self._classify_emotions(features)
            
            # 多模态融合
            fused_results = self.fusion_module.fuse_modalities(classification_results)
            
            # 应用情绪关系约束
            if self.relationship_module and self.config.enable_emotion_relationships:
                fused_results['emotion_probs'] = self.relationship_module.apply_relationship_constraints(
                    fused_results['emotion_probs'], 
                    self.config.relationship_weight
                )
            
            # 解释结果
            emotion_analysis = self._interpret_emotion_results(fused_results)
            
            # 计算置信度
            overall_confidence = float(fused_results['confidence'])
            
            # 创建输出数据
            output_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'emotion_analysis': emotion_analysis,
                    'raw_features': {k: v.detach().cpu().numpy().tolist() for k, v in features.items()},
                    'fusion_strategy': self.config.fusion_strategy,
                    'multimodal_weights': fused_results.get('modality_weights', {}),
                    'processing_info': {
                        'num_modalities': len(features),
                        'total_emotions': self.config.total_emotions,
                        'relationship_modeling': self.config.enable_emotion_relationships
                    }
                },
                metadata={
                    'source_layer': input_data.layer_name,
                    'processing_device': str(self.device),
                    'fusion_method': self.config.fusion_strategy
                },
                confidence=overall_confidence
            )
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer("fusion_layer_processing")
            output_data.processing_time = processing_time
            
            # 记录统计信息
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            logger.info(f"融合层处理完成 - 主要情绪: {emotion_analysis['primary_emotion']['name']}, "
                       f"置信度: {overall_confidence:.3f}, 耗时: {processing_time*1000:.1f}ms")
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"融合层处理失败: {e}")
            
            # 创建错误输出
            error_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'error': str(e),
                    'emotion_analysis': {
                        'primary_emotion': {'name': 'neutral', 'probability': 1.0, 'intensity': 0.0},
                        'overall_confidence': 0.0
                    }
                },
                metadata={'error': True, 'source_layer': input_data.layer_name},
                confidence=0.0
            )
            
            processing_time = self.performance_monitor.end_timer("fusion_layer_processing")
            error_data.processing_time = processing_time
            
            return error_data
    
    def get_emotion_info(self, emotion_name: str) -> Dict[str, Any]:
        """获取情绪信息"""
        # 查找基础情绪
        base_emotions = self.emotion_config.get('base_emotions', {})
        if emotion_name in base_emotions:
            return base_emotions[emotion_name]
        
        # 查找睡眠专用情绪
        sleep_emotions = self.emotion_config.get('sleep_specific_emotions', {})
        if emotion_name in sleep_emotions:
            return sleep_emotions[emotion_name]
        
        return {}
    
    def get_therapy_priority(self, emotion_name: str) -> str:
        """获取治疗优先级"""
        emotion_info = self.get_emotion_info(emotion_name)
        return emotion_info.get('therapy_priority', 'low')
    
    def get_sleep_impact(self, emotion_name: str) -> str:
        """获取睡眠影响"""
        emotion_info = self.get_emotion_info(emotion_name)
        return emotion_info.get('sleep_impact', 'neutral')
    
    def get_status(self) -> Dict[str, Any]:
        """获取融合层状态"""
        base_status = super().get_status()
        
        # 添加融合层特有的状态信息
        fusion_status = {
            'emotion_dimensions': self.config.total_emotions,
            'fusion_strategy': self.config.fusion_strategy,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'relationship_modeling': self.config.enable_emotion_relationships,
            'performance_stats': self.performance_monitor.get_all_stats()
        }
        
        base_status.update(fusion_status)
        return base_status