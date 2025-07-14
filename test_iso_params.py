#!/usr/bin/env python3
"""
测试ISO三阶段参数生成
验证情绪映射到三阶段音乐参数的详细输出
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.input_layer import InputLayer, InputLayerConfig
from layers.fusion_layer import FusionLayer, FusionLayerConfig  
from layers.mapping_layer import MappingLayer, MappingLayerConfig
from layers.base_layer import LayerData
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_iso_three_stages():
    """测试ISO三阶段参数生成"""
    
    print("🎵 ISO三阶段音乐参数生成测试")
    print("=" * 50)
    
    # 初始化层
    input_config = InputLayerConfig(layer_name="input_layer")
    fusion_config = FusionLayerConfig(layer_name="fusion_layer")
    mapping_config = MappingLayerConfig(layer_name="mapping_layer")
    
    input_layer = InputLayer(input_config)
    fusion_layer = FusionLayer(fusion_config)
    mapping_layer = MappingLayer(mapping_config)
    
    # 测试场景
    test_scenarios = [
        {"text": "我今天感到很焦虑，躺在床上睡不着", "expected_emotion": "sleep_anxiety"},
        {"text": "感觉很疲惫但是大脑还在活跃，难以入睡", "expected_emotion": "hyperarousal"},
        {"text": "心情平静，准备进入睡眠状态", "expected_emotion": "peaceful"}
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔄 === 场景 {i}: {scenario['text']} ===")
        
        # 输入层处理
        input_data = LayerData(
            layer_name="test_input",
            timestamp=datetime.now(),
            data={
                "multimodal_data": {
                    "text": {
                        "text": scenario["text"],
                        "features": {
                            "emotion_keywords": [],
                            "semantic_features": {
                                "sentiment_polarity": 0.0,
                                "subjectivity": 0.0,
                                "complexity": 0.05
                            },
                            "text_length": len(scenario["text"]),
                            "sentence_count": 1
                        }
                    }
                },
                "mode": "text_only"
            }
        )
        
        input_result = await input_layer._process_impl(input_data)
        print(f"✅ 输入层完成")
        
        # 融合层处理
        fusion_result = await fusion_layer._process_impl(input_result)
        print(f"✅ 融合层完成 - 主要情绪: {fusion_result.data.get('emotion_analysis', {}).get('primary_emotion', 'unknown')}")
        
        # 映射层处理 
        mapping_result = await mapping_layer._process_impl(fusion_result)
        print(f"✅ 映射层完成")
        
        # 检查是否有ISO三阶段参数
        if 'iso_three_stage_params' in mapping_result.data:
            iso_params = mapping_result.data['iso_three_stage_params']
            print(f"\n🎯 ISO三阶段参数:")
            
            # 匹配阶段
            match_stage = iso_params['match_stage']
            print(f"  🎼 匹配阶段 (Match):")
            print(f"    节拍: {match_stage['tempo_bpm']:.1f} BPM")
            print(f"    调性: {match_stage['key_signature']}")
            print(f"    力度: {match_stage['dynamics']}")
            print(f"    效价: {match_stage['valence_mapping']:.2f}")
            print(f"    唤醒: {match_stage['arousal_mapping']:.2f}")
            print(f"    治疗强度: {match_stage.get('therapy_intensity', 0.0):.2f}")
            print(f"    睡眠准备: {match_stage.get('sleep_readiness', 0.0):.2f}")
            print(f"    持续时间: {match_stage['stage_duration']:.1f}分钟")
            
            # 引导阶段
            guide_stage = iso_params['guide_stage'] 
            print(f"  🌟 引导阶段 (Guide):")
            print(f"    节拍: {guide_stage['tempo_bpm']:.1f} BPM")
            print(f"    调性: {guide_stage['key_signature']}")
            print(f"    力度: {guide_stage['dynamics']}")
            print(f"    效价: {guide_stage['valence_mapping']:.2f}")
            print(f"    唤醒: {guide_stage['arousal_mapping']:.2f}")
            print(f"    治疗强度: {guide_stage.get('therapy_intensity', 0.0):.2f}")
            print(f"    睡眠准备: {guide_stage.get('sleep_readiness', 0.0):.2f}")
            print(f"    持续时间: {guide_stage['stage_duration']:.1f}分钟")
            
            # 目标阶段
            target_stage = iso_params['target_stage']
            print(f"  🎯 目标阶段 (Target):")
            print(f"    节拍: {target_stage['tempo_bpm']:.1f} BPM")
            print(f"    调性: {target_stage['key_signature']}")
            print(f"    力度: {target_stage['dynamics']}")
            print(f"    效价: {target_stage['valence_mapping']:.2f}")
            print(f"    唤醒: {target_stage['arousal_mapping']:.2f}")
            print(f"    治疗强度: {target_stage.get('therapy_intensity', 0.0):.2f}")
            print(f"    睡眠准备: {target_stage.get('sleep_readiness', 0.0):.2f}")
            print(f"    持续时间: {target_stage['stage_duration']:.1f}分钟")
            
            # 三阶段过渡分析
            match_bpm = match_stage['tempo_bpm']
            guide_bpm = guide_stage['tempo_bpm'] 
            target_bpm = target_stage['tempo_bpm']
            
            print(f"\n📊 三阶段过渡分析:")
            print(f"    BPM变化: {match_bpm:.1f} → {guide_bpm:.1f} → {target_bpm:.1f}")
            print(f"    总降幅: {match_bpm - target_bpm:.1f} BPM")
            print(f"    总治疗时长: {match_stage['stage_duration'] + guide_stage['stage_duration'] + target_stage['stage_duration']:.1f}分钟")
            
            # 乐器配置
            print(f"\n🎻 目标阶段乐器配置:")
            for instrument, weight in target_stage['instrument_weights'].items():
                if weight > 0:
                    print(f"    {instrument}: {weight:.1f}")
        else:
            print("❌ 未找到ISO三阶段参数")
        
        print(f"📈 映射置信度: {mapping_result.confidence:.3f}")
        
    print(f"\n🎉 ISO三阶段测试完成！")

if __name__ == "__main__":
    asyncio.run(test_iso_three_stages())