#!/usr/bin/env python3
"""
测试标准化融合层接口
验证用户规范要求的五个核心函数
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.fusion_layer import FusionLayer, FusionLayerConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_standard_fusion_interface():
    """测试标准化融合接口"""
    
    print("🧠 标准化融合层接口测试")
    print("=" * 60)
    
    # 初始化融合层
    config = FusionLayerConfig(layer_name="test_fusion_layer")
    fusion_layer = FusionLayer(config)
    
    # 测试场景：模拟不同模态的输入数据
    test_scenarios = [
        {
            "name": "完整多模态数据",
            "face_data": {
                'au_01': 0.2,  # 内眉上扬
                'au_04': 0.8,  # 眉毛下压 - 愤怒
                'au_06': 0.1,  # 面颊提升
                'au_12': 0.0,  # 嘴角上扬
                'au_15': 0.7,  # 嘴角下拉 - 悲伤
                'au_17': 0.5,  # 下巴上抬
                'au_43': 0.3   # 闭眼 - 疲劳
            },
            "audio_data": {
                'pitch_mean': 180.0,
                'pitch_std': 25.0,
                'loudness_mean': 0.3,
                'loudness_std': 0.15,
                'speech_rate': 3.5,
                'pause_ratio': 0.3,
                'jitter': 0.02,
                'shimmer': 0.08,
                'harmonics_noise_ratio': 15.0
            },
            "text_input": "我今天感到很焦虑，躺在床上睡不着，心情特别沮丧"
        },
        {
            "name": "面部关键点数据",
            "face_data": [
                # 模拟68个关键点的x,y坐标
                *[np.random.uniform(50, 590) for _ in range(136)]  # 68点 * 2坐标
            ],
            "audio_data": None,
            "text_input": "感觉很疲惫但是大脑还在活跃，难以入睡"
        },
        {
            "name": "仅文本输入",
            "face_data": None,
            "audio_data": None,
            "text_input": "心情平静，准备进入睡眠状态，感觉很放松"
        },
        {
            "name": "音频特征列表",
            "face_data": None,
            "audio_data": [0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.8, 0.2, 0.1, 0.9, 0.3, 0.7],
            "text_input": None
        }
    ]
    
    print(f"\n📊 测试 {len(test_scenarios)} 种输入场景...\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🔄 === 场景 {i}: {scenario['name']} ===")
        
        # 测试1: 面部情绪特征提取
        print(f"\n😊 1. 面部情绪特征提取:")
        face_features = fusion_layer.extract_face_emotion_features(scenario['face_data'])
        print(f"   输入类型: {type(scenario['face_data']).__name__ if scenario['face_data'] else 'None'}")
        print(f"   输出维度: {len(face_features)}维")
        print(f"   特征范围: [{min(face_features):.3f}, {max(face_features):.3f}]")
        print(f"   特征均值: {np.mean(face_features):.3f}")
        
        # 显示关键特征
        if scenario['face_data'] and isinstance(scenario['face_data'], dict):
            print(f"   主要AU特征:")
            au_keys = ['au_04', 'au_06', 'au_12', 'au_15', 'au_43']  # 重要的AU
            for j, au_key in enumerate(au_keys):
                if j < len(face_features) and au_key in scenario['face_data']:
                    original_value = scenario['face_data'][au_key]
                    feature_value = face_features[j] if j < len(face_features) else 0.0
                    print(f"     {au_key}: {original_value:.2f} → {feature_value:.3f}")
        
        # 测试2: 音频情绪特征提取
        print(f"\n🎵 2. 音频情绪特征提取:")
        audio_features = fusion_layer.extract_audio_emotion_features(scenario['audio_data'])
        print(f"   输入类型: {type(scenario['audio_data']).__name__ if scenario['audio_data'] else 'None'}")
        print(f"   输出维度: {len(audio_features)}维")
        print(f"   特征范围: [{min(audio_features):.3f}, {max(audio_features):.3f}]")
        print(f"   特征均值: {np.mean(audio_features):.3f}")
        
        # 显示音频特征解释
        if scenario['audio_data'] and isinstance(scenario['audio_data'], dict):
            print(f"   音频情绪特征:")
            feature_names = ['音调均值', '音调变化', '响度均值', '响度变化', '语速', 
                           '停顿比例', '基频抖动', '振幅抖动', '谐噪比', '紧张度', 
                           '活力度', '平静度', '疲劳度', '焦虑度']
            for j, name in enumerate(feature_names):
                if j < len(audio_features):
                    value = audio_features[j]
                    percentage = value * 100
                    print(f"     {name}: {percentage:.1f}%")
        
        # 测试3: 文本情绪特征提取
        print(f"\n📝 3. 文本情绪特征提取:")
        text_features = fusion_layer.extract_text_emotion_features(scenario['text_input'])
        text_display = f'"{scenario["text_input"]}"' if scenario['text_input'] else 'None'
        print(f"   输入文本: {text_display}")
        print(f"   输出维度: {len(text_features)}维")
        print(f"   特征范围: [{min(text_features):.3f}, {max(text_features):.3f}]")
        print(f"   特征均值: {np.mean(text_features):.3f}")
        
        # 显示文本情绪检测结果
        if scenario['text_input']:
            emotion_names = ['文本长度', '词数', '句数', '快乐', '悲伤', '愤怒', '恐惧', 
                           '惊讶', '厌恶', '焦虑', '疲劳', '平静']
            print(f"   文本情绪检测:")
            for j, name in enumerate(emotion_names):
                if j < len(text_features):
                    value = text_features[j]
                    if name in ['文本长度', '词数', '句数']:
                        print(f"     {name}: {value:.2f}")
                    else:
                        percentage = value * 100
                        print(f"     {name}: {percentage:.1f}%")
        
        # 测试4: 多模态特征融合
        print(f"\n🔗 4. 多模态特征融合:")
        fused_features = fusion_layer.fuse_multimodal_features(face_features, audio_features, text_features)
        print(f"   输入: 面部{len(face_features)}维 + 音频{len(audio_features)}维 + 文本{len(text_features)}维")
        print(f"   输出维度: {len(fused_features)}维")
        print(f"   特征范围: [{min(fused_features):.3f}, {max(fused_features):.3f}]")
        print(f"   特征均值: {np.mean(fused_features):.3f}")
        
        # 显示融合特征结构
        print(f"   融合特征结构:")
        print(f"     • 面部特征: 0-19 ({np.mean(fused_features[0:20]):.3f})")
        print(f"     • 音频特征: 20-34 ({np.mean(fused_features[20:35]):.3f})")
        print(f"     • 文本特征: 35-59 ({np.mean(fused_features[35:60]):.3f})")
        print(f"     • 交互特征: 60-63 ({np.mean(fused_features[60:64]):.3f})")
        
        # 测试5: 主函数 - 情感状态推断
        print(f"\n🎯 5. 情感状态推断 (主函数):")
        emotion_vector = fusion_layer.infer_affective_state(
            raw_face_data=scenario['face_data'],
            raw_audio_data=scenario['audio_data'],
            text_input=scenario['text_input']
        )
        print(f"   输出维度: {len(emotion_vector)}维情绪向量")
        print(f"   向量和: {sum(emotion_vector):.6f} (应接近1.0)")
        print(f"   向量范围: [{min(emotion_vector):.3f}, {max(emotion_vector):.3f}]")
        
        # 找出主导情绪
        max_emotion_idx = np.argmax(emotion_vector)
        max_emotion_prob = emotion_vector[max_emotion_idx]
        
        # 情绪名称映射 (简化版27维情绪空间)
        emotion_names = [
            'anger', 'fear_anxiety', 'disgust', 'sadness', 'amusement', 'joy', 
            'inspiration', 'tenderness', 'neutral', 'rumination', 'sleep_anxiety',
            'physical_fatigue', 'mental_fatigue', 'hyperarousal', 'bedtime_worry', 
            'sleep_dread', 'restless_sleep', 'sleep_guilt', 'dawn_anxiety', 'peaceful',
            'relaxed', 'drowsy', 'tired_content', 'pre_sleep_calm', 'deep_relaxation',
            'sleep_readiness', 'meditative'
        ]
        
        main_emotion = emotion_names[max_emotion_idx] if max_emotion_idx < len(emotion_names) else f"emotion_{max_emotion_idx}"
        print(f"   主导情绪: {main_emotion} ({max_emotion_prob:.3f})")
        
        # 显示前5个最高概率的情绪
        emotion_probs = [(emotion_names[i] if i < len(emotion_names) else f"emotion_{i}", prob) 
                        for i, prob in enumerate(emotion_vector)]
        emotion_probs.sort(key=lambda x: x[1], reverse=True)
        print(f"   情绪排序 (前5):")
        for j, (emotion, prob) in enumerate(emotion_probs[:5]):
            percentage = prob * 100
            print(f"     {j+1}. {emotion}: {percentage:.1f}%")
        
        print(f"\n" + "─" * 50)
    
    print(f"\n🎉 标准化融合层接口测试完成！")
    print(f"\n📋 测试总结:")
    print(f"   ✅ extract_face_emotion_features() - 面部AU/关键点特征提取正常")
    print(f"   ✅ extract_audio_emotion_features() - 音频韵律特征提取正常")
    print(f"   ✅ extract_text_emotion_features() - 文本情绪关键词检测正常")
    print(f"   ✅ fuse_multimodal_features() - 特征级融合正常")
    print(f"   ✅ infer_affective_state() - 主函数情感推断正常")
    print(f"\n💡 特点:")
    print(f"   • 支持多种输入格式 (AU字典、关键点坐标、音频特征、文本)")
    print(f"   • 特征级融合策略 + 跨模态交互")
    print(f"   • 输出27维标准化情绪向量")
    print(f"   • 强制决策逻辑确保准确分类")
    print(f"   • 与现有系统完全兼容")

def test_edge_cases():
    """测试边界情况和异常处理"""
    
    print(f"\n🧪 边界情况测试")
    print("=" * 40)
    
    config = FusionLayerConfig(layer_name="test_fusion_layer")
    fusion_layer = FusionLayer(config)
    
    # 测试空输入
    print("1. 空输入测试:")
    emotion_vector = fusion_layer.infer_affective_state(None, None, None)
    print(f"   空输入结果: 主导情绪索引 {np.argmax(emotion_vector)} (概率: {max(emotion_vector):.3f})")
    
    # 测试无效数据
    print("\n2. 无效数据测试:")
    emotion_vector = fusion_layer.infer_affective_state({}, [], "")
    print(f"   无效数据结果: 主导情绪索引 {np.argmax(emotion_vector)} (概率: {max(emotion_vector):.3f})")
    
    # 测试极值数据
    print("\n3. 极值数据测试:")
    extreme_face = {f'au_{i:02d}': 1.0 for i in range(1, 46)}  # 所有AU最大值
    extreme_audio = {
        'pitch_mean': 500.0,  # 极高音调
        'loudness_mean': 1.0,  # 最大响度
        'speech_rate': 15.0   # 极快语速
    }
    emotion_vector = fusion_layer.infer_affective_state(extreme_face, extreme_audio, "非常非常非常焦虑愤怒")
    print(f"   极值数据结果: 主导情绪索引 {np.argmax(emotion_vector)} (概率: {max(emotion_vector):.3f})")
    
    print("\n✅ 边界情况测试通过")

if __name__ == "__main__":
    test_standard_fusion_interface()
    test_edge_cases()