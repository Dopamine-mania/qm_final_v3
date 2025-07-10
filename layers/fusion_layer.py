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
        if self.config.fusion_strategy == "simple":
            return self.simple_fusion(modality_results)
        elif self.config.fusion_strategy == "confidence_weighted":
            return self.confidence_weighted_fusion(modality_results)
        elif self.config.fusion_strategy == "attention_based":
            return self.attention_based_fusion(modality_results)
        else:
            logger.warning(f"未知的融合策略: {self.config.fusion_strategy}，使用默认策略")
            return self.confidence_weighted_fusion(modality_results)

class EmotionRelationshipModule:
    """情绪关系建模模块"""
    
    def __init__(self, emotion_config: Dict[str, Any]):
        self.emotion_config = emotion_config
        self.emotion_relationships = emotion_config.get('emotion_relationships', {})
        
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
        matrix = torch.zeros(num_emotions, num_emotions)
        
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
        matrix = torch.zeros(num_emotions, num_emotions)
        
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
        matrix = torch.zeros(num_emotions, num_emotions)
        
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
        
        # 应用协同关系增强
        synergy_boost = torch.matmul(self.synergy_matrix, emotion_probs)
        adjusted_probs += relationship_weight * synergy_boost
        
        # 应用互斥关系抑制
        exclusion_suppression = torch.matmul(self.mutual_exclusion_matrix, emotion_probs)
        adjusted_probs -= relationship_weight * exclusion_suppression
        
        # 确保概率非负并重新归一化
        adjusted_probs = torch.clamp(adjusted_probs, min=0.0)
        adjusted_probs = F.softmax(adjusted_probs, dim=-1)
        
        return adjusted_probs
    
    def predict_emotion_transition(self, current_emotion_probs: torch.Tensor) -> Dict[str, float]:
        """预测情绪转换"""
        # 计算转换概率
        transition_probs = torch.matmul(self.transition_matrix, current_emotion_probs)
        
        # 找到最可能的转换
        top_transitions = torch.topk(transition_probs, k=3)
        
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
            self.relationship_module = EmotionRelationshipModule(self.emotion_config)
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
        """简化的文本特征提取"""
        # 睡眠相关情绪关键词
        emotion_keywords = {
            'anxiety': ['焦虑', '紧张', '担心', '害怕', '不安'],
            'sadness': ['悲伤', '沮丧', '失落', '难过', '抑郁'],
            'anger': ['愤怒', '生气', '烦躁', '恼火', '愤恨'],
            'fear': ['恐惧', '害怕', '惊慌', '胆怯', '畏惧'],
            'joy': ['开心', '快乐', '高兴', '愉快', '兴奋'],
            'relaxation': ['放松', '平静', '宁静', '安详', '舒适'],
            'insomnia': ['失眠', '睡不着', '难眠', '辗转反侧'],
            'drowsiness': ['困倦', '想睡', '疲惫', '倦意', '睡意'],
            'restlessness': ['不安', '躁动', '烦躁', '坐立不安'],
            'peace': ['平和', '安宁', '祥和', '宁静', '安静']
        }
        
        # 计算特征向量
        text_lower = text.lower()
        features = []
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            features.append(score)
        
        # 添加文本长度和其他统计特征
        features.extend([
            len(text),  # 文本长度
            text.count('。'),  # 句子数
            text.count('！'),  # 感叹号数
            text.count('？'),  # 问号数
        ])
        
        # 填充到768维（模拟BERT特征）
        while len(features) < 768:
            features.extend(features[:min(10, 768 - len(features))])
        
        features = features[:768]  # 截断到768维
        
        # 归一化
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
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
        emotion_probs = fused_results['emotion_probs'].cpu().numpy()
        confidence = fused_results['confidence'].item()
        intensity = fused_results['intensity'].cpu().numpy()
        
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
                'intensity': float(intensity[idx])
            })
        
        # 预测情绪转换
        transitions = {}
        if self.relationship_module:
            transitions = self.relationship_module.predict_emotion_transition(fused_results['emotion_probs'])
        
        return {
            'primary_emotion': {
                'name': emotion_name,
                'probability': float(top_emotion_prob),
                'intensity': float(intensity[top_emotion_id])
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
                    'raw_features': {k: v.cpu().numpy().tolist() for k, v in features.items()},
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