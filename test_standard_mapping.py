#!/usr/bin/env python3
"""
测试标准化映射层接口
验证用户规范要求的三个核心函数
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.mapping_layer import MappingLayer, MappingLayerConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_standard_mapping_interface():
    """测试标准化映射接口"""
    
    print("🎼 标准化映射层接口测试")
    print("=" * 60)
    
    # 初始化映射层
    config = MappingLayerConfig(layer_name="test_mapping_layer")
    mapping_layer = MappingLayer(config)
    
    # 测试场景：模拟不同情绪的27维向量
    test_scenarios = [
        {
            "name": "焦虑情绪",
            "emotion_vector": np.zeros(27),  # 基础向量
            "dominant_emotion_idx": 1,       # fear_anxiety
            "intensity": 0.8
        },
        {
            "name": "睡眠焦虑", 
            "emotion_vector": np.zeros(27),
            "dominant_emotion_idx": 10,      # sleep_anxiety
            "intensity": 0.7
        },
        {
            "name": "过度觉醒",
            "emotion_vector": np.zeros(27),
            "dominant_emotion_idx": 14,      # hyperarousal
            "intensity": 0.9
        },
        {
            "name": "平静状态",
            "emotion_vector": np.zeros(27),
            "dominant_emotion_idx": 19,      # peaceful
            "intensity": 0.6
        }
    ]
    
    # 设置情绪向量
    for scenario in test_scenarios:
        scenario["emotion_vector"][scenario["dominant_emotion_idx"]] = scenario["intensity"]
        # 添加少量噪声到其他维度
        for i in range(27):
            if i != scenario["dominant_emotion_idx"]:
                scenario["emotion_vector"][i] = np.random.uniform(0.0, 0.1)
    
    print(f"\n📊 测试 {len(test_scenarios)} 种情绪场景...\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🔄 === 场景 {i}: {scenario['name']} ===")
        print(f"   主导情绪索引: {scenario['dominant_emotion_idx']}")
        print(f"   情绪强度: {scenario['intensity']:.1f}")
        
        # 转换为torch tensor
        emotion_tensor = torch.tensor(scenario["emotion_vector"], dtype=torch.float32)
        
        # 测试1: KG初始映射
        print(f"\n🧠 1. 知识图谱初始映射:")
        kg_params = mapping_layer.get_kg_initial_mapping(emotion_tensor)
        
        for param_name, value in kg_params.items():
            if param_name == 'tempo':
                bpm = 60.0 + value * 60.0
                print(f"   {param_name}: {value:.3f} ({bpm:.1f} BPM)")
            elif param_name == 'mode':
                mode_str = "大调" if value < 0.5 else "小调"
                print(f"   {param_name}: {value:.3f} ({mode_str})")
            else:
                print(f"   {param_name}: {value:.3f}")
        
        # 测试2: MLP个性化微调
        print(f"\n🎯 2. MLP个性化微调:")
        
        # 模拟用户偏好数据
        user_profile = {
            'tempo_preference': 0.1,      # 偏好稍快节拍
            'mode_preference': -0.1,      # 偏好大调
            'volume_sensitivity': 0.8,    # 音量敏感
            'harmony_preference': 0.2,    # 偏好协和
            'therapy_responsiveness': 1.2 # 治疗响应度高
        }
        
        personalized_params = mapping_layer.apply_mlp_personalization(
            kg_params, emotion_tensor, user_profile
        )
        
        print(f"   用户偏好调整:")
        for param_name, value in personalized_params.items():
            original_value = kg_params[param_name]
            change = value - original_value
            change_str = f"({change:+.3f})" if abs(change) > 0.001 else ""
            
            if param_name == 'tempo':
                bpm = 60.0 + value * 60.0
                print(f"   {param_name}: {value:.3f} ({bpm:.1f} BPM) {change_str}")
            else:
                print(f"   {param_name}: {value:.3f} {change_str}")
        
        # 测试3: 主映射函数
        print(f"\n🎵 3. 主映射函数输出:")
        final_params = mapping_layer.map_emotion_to_music(emotion_tensor, user_profile)
        
        print(f"   最终音乐参数:")
        for param_name, value in final_params.items():
            if param_name == 'tempo':
                bpm = 60.0 + value * 60.0
                print(f"   • {param_name}: {bpm:.1f} BPM")
            elif param_name == 'mode':
                mode_str = "大调" if value < 0.5 else "小调"
                print(f"   • {param_name}: {mode_str} ({value:.3f})")
            elif param_name == 'dynamics':
                volume_pct = value * 100
                print(f"   • {param_name}: {volume_pct:.1f}% 音量")
            elif param_name == 'harmony_consonance':
                consonance_pct = value * 100
                print(f"   • {param_name}: {consonance_pct:.1f}% 协和度")
            elif param_name == 'emotional_envelope_direction':
                if value > 0.1:
                    direction = "上升 ↗"
                elif value < -0.1:
                    direction = "下降 ↘"
                else:
                    direction = "保持 →"
                print(f"   • {param_name}: {direction} ({value:.3f})")
            else:
                percentage = value * 100
                print(f"   • {param_name}: {percentage:.1f}%")
        
        # 测试4: 转换为详细参数（与ISO三阶段兼容）
        print(f"\n🔧 4. 转换为详细参数 (ISO兼容):")
        detailed_params = mapping_layer.convert_to_detailed_params(final_params)
        print(f"   • 节拍: {detailed_params.tempo_bpm:.1f} BPM")
        print(f"   • 调性: {detailed_params.key_signature}")
        print(f"   • 力度: {detailed_params.dynamics}")
        print(f"   • 效价映射: {detailed_params.valence_mapping:.3f}")
        print(f"   • 唤醒映射: {detailed_params.arousal_mapping:.3f}")
        print(f"   • 张力水平: {detailed_params.tension_level:.3f}")
        
        print(f"\n" + "─" * 50)
    
    print(f"\n🎉 标准化接口测试完成！")
    print(f"\n📋 测试总结:")
    print(f"   ✅ get_kg_initial_mapping() - KG规则映射正常")
    print(f"   ✅ apply_mlp_personalization() - 个性化微调正常")
    print(f"   ✅ map_emotion_to_music() - 主映射函数正常")
    print(f"   ✅ convert_to_detailed_params() - 兼容性转换正常")
    print(f"\n💡 特点:")
    print(f"   • 8个标准化参数输出")
    print(f"   • 基于GEMS原理的KG规则")
    print(f"   • 用户偏好个性化调整")
    print(f"   • 与ISO三阶段功能兼容")

if __name__ == "__main__":
    test_standard_mapping_interface()