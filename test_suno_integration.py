#!/usr/bin/env python3
"""
测试Suno API集成
快速验证三阶段音乐叙事生成功能
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from layers.generation_layer import GenerationLayer, GenerationLayerConfig, MusicParameter
from layers.base_layer import LayerData
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_suno_integration():
    """测试Suno API集成"""
    
    logger.info("🚀 开始测试Suno API集成...")
    
    # 创建生成层配置
    config = GenerationLayerConfig(
        layer_name="test_generation_layer",
        audio_enabled=True,
        video_enabled=False,  # 先只测试音频
        audio_duration=120.0,  # 2分钟
        generation_strategy="hybrid"
    )
    
    # 初始化生成层
    generation_layer = GenerationLayer(config)
    
    # 模拟从融合层来的情绪数据和映射层的音乐参数
    test_input_data = LayerData(
        layer_name="test_input",
        timestamp=datetime.now(),
        data={
            'emotion_analysis': {
                'primary_emotion': {'name': '焦虑', 'confidence': 0.85},
                'emotion_vector': [0.2, 0.8, -0.3, 0.1]  # 示例27维向量的一部分
            },
            'music_parameters': {
                'tempo_bpm': 80.0,
                'key_signature': 'C_major',
                'valence_mapping': -0.3,  # 负面情绪
                'arousal_mapping': 0.6,   # 高唤醒
                'iso_stage': 'synchronization'
            }
        },
        metadata={'test_run': True},
        confidence=0.85
    )
    
    try:
        # 测试生成过程
        logger.info("🎵 开始生成三阶段音乐...")
        result = await generation_layer._process_impl(test_input_data)
        
        # 检查结果
        if result.data.get('error'):
            logger.error(f"❌ 生成失败: {result.data['error']}")
            return False
        
        generated_content = result.data.get('generated_content', {})
        audio_content = generated_content.get('audio', {})
        
        if 'error' in audio_content:
            logger.error(f"❌ 音频生成失败: {audio_content['error']}")
            return False
        
        # 输出结果信息
        logger.info("✅ 生成成功！")
        logger.info(f"   时长: {audio_content.get('duration', 0)}秒")
        logger.info(f"   采样率: {audio_content.get('sample_rate', 0)}Hz")
        logger.info(f"   声道数: {audio_content.get('channels', 0)}")
        logger.info(f"   三阶段叙事: {audio_content.get('three_stage_narrative', False)}")
        logger.info(f"   使用备用方案: {audio_content.get('fallback_used', '未知')}")
        
        # 显示阶段提示词（如果有）
        stage_prompts = audio_content.get('stage_prompts', {})
        if stage_prompts:
            logger.info("📝 生成的阶段提示词:")
            for stage, prompt in stage_prompts.items():
                logger.info(f"   {stage}: {prompt[:100]}...")
        
        # 检查音频数组
        audio_array = audio_content.get('audio_array')
        if audio_array is not None:
            logger.info(f"   音频数组形状: {audio_array.shape}")
            logger.info(f"   音频数据类型: {audio_array.dtype}")
        
        logger.info("🎉 Suno API集成测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_emotions():
    """测试多种情绪场景"""
    
    test_emotions = [
        {'name': '焦虑', 'valence': -0.3, 'arousal': 0.6},
        {'name': '疲惫', 'valence': -0.1, 'arousal': -0.4},
        {'name': '中性', 'valence': 0.0, 'arousal': 0.0},
        {'name': '烦躁', 'valence': -0.5, 'arousal': 0.8},
    ]
    
    logger.info(f"🎭 测试{len(test_emotions)}种不同情绪场景...")
    
    # 创建生成层
    config = GenerationLayerConfig(
        layer_name="multi_emotion_test_layer",
        audio_enabled=True,
        video_enabled=False,
        audio_duration=60.0  # 缩短到1分钟加快测试
    )
    generation_layer = GenerationLayer(config)
    
    success_count = 0
    
    for i, emotion in enumerate(test_emotions, 1):
        logger.info(f"🔄 测试场景 {i}/{len(test_emotions)}: {emotion['name']}")
        
        test_data = LayerData(
            layer_name="emotion_test",
            timestamp=datetime.now(),
            data={
                'emotion_analysis': {
                    'primary_emotion': {'name': emotion['name'], 'confidence': 0.80},
                },
                'music_parameters': {
                    'tempo_bpm': 70.0,
                    'valence_mapping': emotion['valence'],
                    'arousal_mapping': emotion['arousal'],
                }
            },
            metadata={'emotion_test': True},
            confidence=0.80
        )
        
        try:
            result = await generation_layer._process_impl(test_data)
            
            if not result.data.get('error'):
                audio_content = result.data.get('generated_content', {}).get('audio', {})
                if 'error' not in audio_content:
                    success_count += 1
                    logger.info(f"   ✅ {emotion['name']} - 生成成功")
                else:
                    logger.warning(f"   ⚠️ {emotion['name']} - 音频生成失败")
            else:
                logger.warning(f"   ⚠️ {emotion['name']} - 处理失败")
                
        except Exception as e:
            logger.error(f"   ❌ {emotion['name']} - 异常: {e}")
    
    logger.info(f"🎯 多情绪测试完成: {success_count}/{len(test_emotions)} 成功")
    return success_count == len(test_emotions)

async def main():
    """主测试函数"""
    logger.info("🧪 开始Suno API集成测试套件...")
    
    # 基础功能测试
    test1_result = await test_suno_integration()
    
    # 多情绪场景测试
    test2_result = await test_multiple_emotions()
    
    # 汇总结果
    if test1_result and test2_result:
        logger.info("🎉 所有测试通过！Suno API集成成功！")
        logger.info("💡 下一步可以:")
        logger.info("   1. 获取真实的Suno API密钥")
        logger.info("   2. 替换_simulate_suno_response为真实API调用")
        logger.info("   3. 测试完整的端到端流程")
        return True
    else:
        logger.error("❌ 部分测试失败，需要检查问题")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)