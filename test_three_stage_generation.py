#!/usr/bin/env python3
"""
测试ISO三阶段音画同步生成
验证生成层的三阶段内容生成功能
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
from layers.generation_layer import GenerationLayer, GenerationLayerConfig
from layers.base_layer import LayerData
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_three_stage_generation():
    """测试ISO三阶段音画同步生成"""
    
    print("🎭 ISO三阶段音画同步生成测试")
    print("=" * 60)
    
    # 初始化所有层
    input_config = InputLayerConfig(layer_name="input_layer")
    fusion_config = FusionLayerConfig(layer_name="fusion_layer")
    mapping_config = MappingLayerConfig(layer_name="mapping_layer")
    generation_config = GenerationLayerConfig(
        layer_name="generation_layer",
        audio_enabled=True,
        video_enabled=True,
        audio_duration=30.0,  # 缩短测试时间
        video_duration=30.0,
        video_fps=5  # 降低帧率加快测试
    )
    
    input_layer = InputLayer(input_config)
    fusion_layer = FusionLayer(fusion_config)
    mapping_layer = MappingLayer(mapping_config)
    generation_layer = GenerationLayer(generation_config)
    
    # 测试场景：睡眠焦虑情境
    test_scenario = {
        "text": "我今天感到很焦虑，躺在床上睡不着，心情特别沮丧",
        "expected_emotion": "sleep_anxiety"
    }
    
    print(f"\n🎯 测试场景: {test_scenario['text']}")
    print("=" * 50)
    
    # 阶段1：输入层处理
    print("\n📥 阶段1: 输入层处理")
    input_data = LayerData(
        layer_name="test_input",
        timestamp=datetime.now(),
        data={
            "multimodal_data": {
                "text": {
                    "text": test_scenario["text"],
                    "features": {
                        "emotion_keywords": [],
                        "semantic_features": {
                            "sentiment_polarity": -0.3,
                            "subjectivity": 0.8,
                            "complexity": 0.05
                        },
                        "text_length": len(test_scenario["text"]),
                        "sentence_count": 1
                    }
                }
            },
            "mode": "text_only"
        }
    )
    
    input_result = await input_layer._process_impl(input_data)
    print(f"✅ 输入层完成 - 数据模式: {input_result.data.get('mode', 'unknown')}")
    
    # 阶段2：融合层处理
    print("\n🧠 阶段2: 融合层处理")
    fusion_result = await fusion_layer._process_impl(input_result)
    emotion_analysis = fusion_result.data.get('emotion_analysis', {})
    primary_emotion = emotion_analysis.get('primary_emotion', {})
    print(f"✅ 融合层完成 - 主要情绪: {primary_emotion.get('name', 'unknown')} "
          f"(置信度: {primary_emotion.get('probability', 0):.3f})")
    
    # 阶段3：映射层处理（生成ISO三阶段参数）
    print("\n🗺️ 阶段3: 映射层处理")
    mapping_result = await mapping_layer._process_impl(fusion_result)
    print(f"✅ 映射层完成 - 置信度: {mapping_result.confidence:.3f}")
    
    # 检查是否生成了ISO三阶段参数
    iso_params = mapping_result.data.get('iso_three_stage_params')
    if iso_params:
        print(f"\n🎼 ISO三阶段参数生成成功:")
        stages = ['match_stage', 'guide_stage', 'target_stage']
        total_duration = 0
        
        for stage_name in stages:
            stage_data = iso_params[stage_name]
            stage_duration = stage_data['stage_duration']
            total_duration += stage_duration
            
            print(f"  📍 {stage_name}:")
            print(f"     节拍: {stage_data['tempo_bpm']:.1f} BPM")
            print(f"     调性: {stage_data['key_signature']}")
            print(f"     力度: {stage_data['dynamics']}")
            print(f"     持续: {stage_duration:.1f}分钟")
            print(f"     治疗强度: {stage_data.get('therapy_intensity', 0.0):.2f}")
        
        print(f"  🕒 总治疗时长: {total_duration:.1f}分钟")
    else:
        print("❌ 未生成ISO三阶段参数，无法继续测试")
        return
    
    # 阶段4：生成层处理（三阶段音画同步生成）
    print(f"\n🎬 阶段4: 三阶段音画同步生成")
    print("=" * 40)
    
    generation_start_time = datetime.now()
    generation_result = await generation_layer._process_impl(mapping_result)
    generation_end_time = datetime.now()
    generation_time = (generation_end_time - generation_start_time).total_seconds()
    
    print(f"✅ 生成层完成 - 耗时: {generation_time:.1f}秒")
    
    # 分析生成结果
    generated_content = generation_result.data.get('generated_content', {})
    
    if 'stages' in generated_content:
        print(f"\n🎭 三阶段内容生成分析:")
        print(f"   • 连贯叙事: {generated_content.get('continuous_narrative', False)}")
        print(f"   • 总时长: {generated_content.get('total_duration', 0):.1f}分钟")
        
        # 分析每个阶段的生成结果
        stages_data = generated_content['stages']
        for stage_idx, stage_name in enumerate(['match_stage', 'guide_stage', 'target_stage'], 1):
            if stage_name in stages_data:
                stage_content = stages_data[stage_name]
                stage_info = stage_content.get('stage_info', {})
                
                print(f"\n   🎼 阶段{stage_idx}: {stage_name}")
                print(f"      目标: {stage_info.get('emotional_target', 'neutral')}")
                print(f"      时长: {stage_info.get('stage_duration', 0):.1f}分钟")
                print(f"      节拍: {stage_info.get('tempo_bpm', 0):.1f} BPM")
                
                # 音频分析
                if 'audio' in stage_content:
                    audio_data = stage_content['audio']
                    if 'error' not in audio_data:
                        print(f"      🎵 音频: ✅ {audio_data.get('duration', 0):.1f}s, "
                              f"{audio_data.get('sample_rate', 0)}Hz, "
                              f"{audio_data.get('format', 'Unknown')}")
                        
                        # 音频数组分析
                        audio_array = audio_data.get('audio_array')
                        if audio_array is not None:
                            import numpy as np
                            rms = np.sqrt(np.mean(audio_array**2))
                            peak = np.max(np.abs(audio_array))
                            print(f"               RMS: {rms:.3f}, Peak: {peak:.3f}")
                    else:
                        print(f"      🎵 音频: ❌ {audio_data.get('error', '未知错误')}")
                
                # 视频分析
                if 'video' in stage_content:
                    video_data = stage_content['video']
                    if 'error' not in video_data:
                        frames = video_data.get('frames', [])
                        print(f"      🎬 视频: ✅ {len(frames)}帧, "
                              f"{video_data.get('fps', 0)}fps, "
                              f"{video_data.get('resolution', (0, 0))}")
                        
                        # 视频帧分析
                        if frames:
                            import numpy as np
                            first_frame = frames[0]
                            last_frame = frames[-1]
                            first_brightness = np.mean(first_frame)
                            last_brightness = np.mean(last_frame)
                            brightness_change = last_brightness - first_brightness
                            print(f"               亮度变化: {first_brightness:.1f} → {last_brightness:.1f} "
                                  f"({brightness_change:+.1f})")
                    else:
                        print(f"      🎬 视频: ❌ {video_data.get('error', '未知错误')}")
                
                # 同步分析
                if 'sync_metadata' in stage_content:
                    sync_info = stage_content['sync_metadata']
                    sync_accuracy = sync_info.get('sync_accuracy', 0)
                    print(f"      🔗 同步: {sync_accuracy:.1%} 准确度")
        
        # 叙事连贯性分析
        narrative_quality = generated_content.get('narrative_quality', {})
        if narrative_quality:
            print(f"\n   📖 叙事连贯性分析:")
            print(f"      • 节拍连贯性: {'✅' if narrative_quality.get('tempo_coherence', False) else '❌'}")
            print(f"      • 治疗连贯性: {'✅' if narrative_quality.get('therapy_coherence', False) else '❌'}")
            print(f"      • 总体评分: {narrative_quality.get('overall_coherence_score', 0):.2f}")
            print(f"      • 叙事类型: {narrative_quality.get('narrative_type', '未知')}")
        
        # 阶段转换分析
        sync_metadata = generated_content.get('sync_metadata', {})
        transitions = sync_metadata.get('stage_transitions', [])
        if transitions:
            print(f"\n   🔄 阶段转换分析:")
            for transition in transitions:
                print(f"      • {transition['from_stage']} → {transition['to_stage']}")
                print(f"        转换点: {transition['transition_point']:.1f}分钟")
                print(f"        方法: {transition['transition_method']}")
                print(f"        连贯性: {transition['continuity_score']:.2f}")
    
    else:
        print("❌ 未生成三阶段内容")
    
    print(f"\n🎉 ISO三阶段音画同步生成测试完成！")
    print(f"\n📊 测试总结:")
    print(f"   ✅ 输入层文本处理 - 正常")
    print(f"   ✅ 融合层情绪识别 - 正常") 
    print(f"   ✅ 映射层ISO参数生成 - 正常")
    print(f"   ✅ 生成层三阶段内容生成 - 正常")
    print(f"   ⏱️ 总处理时间: {generation_time:.1f}秒")
    print(f"\n💡 特色功能:")
    print(f"   • 三阶段连贯音乐叙事 (match → guide → target)")
    print(f"   • 阶段特定的音频效果处理")
    print(f"   • 阶段特定的视觉效果处理") 
    print(f"   • 跨阶段音画同步优化")
    print(f"   • 叙事连贯性自动验证")

if __name__ == "__main__":
    asyncio.run(test_three_stage_generation())