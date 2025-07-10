#!/usr/bin/env python3
"""
融合层测试脚本

测试27维情绪分类和多模态融合功能，包括：
1. 融合层初始化测试
2. 情绪分类测试
3. 多模态融合测试
4. 情绪关系建模测试
5. 端到端处理测试
"""

import sys
import os
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_fusion_layer_imports():
    """测试融合层模块导入"""
    print("🔍 测试融合层模块导入...")
    
    try:
        from layers.fusion_layer import FusionLayer, FusionLayerConfig
        from layers.fusion_layer import EmotionClassifier, MultiModalFusionModule, EmotionRelationshipModule
        print("✅ 融合层模块导入成功")
        return True
        
    except Exception as e:
        print(f"❌ 融合层模块导入失败: {e}")
        return False

def test_fusion_layer_config():
    """测试融合层配置"""
    print("\n🔍 测试融合层配置...")
    
    try:
        from layers.fusion_layer import FusionLayerConfig
        
        # 创建默认配置
        config = FusionLayerConfig()
        print(f"✅ 默认配置创建成功: {config.total_emotions}维情绪空间")
        
        # 创建自定义配置
        custom_config = FusionLayerConfig(
            total_emotions=27,
            fusion_strategy="confidence_weighted",
            enable_emotion_relationships=True,
            use_gpu=False  # 测试时禁用GPU
        )
        print(f"✅ 自定义配置创建成功: 融合策略={custom_config.fusion_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ 融合层配置测试失败: {e}")
        return False

def test_emotion_config_loading():
    """测试情绪配置加载"""
    print("\n🔍 测试情绪配置加载...")
    
    try:
        from core.utils import ConfigLoader, get_project_root
        
        # 加载27维情绪配置
        config_path = get_project_root() / "configs" / "emotion_27d.yaml"
        emotion_config = ConfigLoader.load_yaml(str(config_path))
        
        total_dimensions = emotion_config['emotion_space']['total_dimensions']
        base_emotions = len(emotion_config['base_emotions'])
        sleep_emotions = len(emotion_config['sleep_specific_emotions'])
        
        print(f"✅ 情绪配置加载成功: 总维度={total_dimensions}, 基础情绪={base_emotions}, 睡眠情绪={sleep_emotions}")
        
        # 验证情绪关系
        relationships = emotion_config['emotion_relationships']
        print(f"✅ 情绪关系配置: 互斥关系={len(relationships.get('mutually_exclusive', []))}, "
              f"协同关系={len(relationships.get('synergistic', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 情绪配置加载失败: {e}")
        return False

def test_fusion_layer_initialization():
    """测试融合层初始化"""
    print("\n🔍 测试融合层初始化...")
    
    try:
        from layers.fusion_layer import FusionLayer, FusionLayerConfig
        
        # 创建配置
        config = FusionLayerConfig(
            total_emotions=27,
            use_gpu=False,  # 测试时禁用GPU
            enable_emotion_relationships=True
        )
        
        # 初始化融合层
        fusion_layer = FusionLayer(config)
        print(f"✅ 融合层初始化成功: {fusion_layer.layer_name}")
        
        # 获取状态
        status = fusion_layer.get_status()
        print(f"✅ 层状态获取成功: 情绪维度={status['emotion_dimensions']}")
        
        # 关闭融合层
        fusion_layer.shutdown()
        print("✅ 融合层关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 融合层初始化失败: {e}")
        return False

def test_emotion_relationship_module():
    """测试情绪关系建模模块"""
    print("\n🔍 测试情绪关系建模模块...")
    
    try:
        from layers.fusion_layer import EmotionRelationshipModule
        from core.utils import ConfigLoader, get_project_root
        
        # 加载情绪配置
        config_path = get_project_root() / "configs" / "emotion_27d.yaml"
        emotion_config = ConfigLoader.load_yaml(str(config_path))
        
        # 初始化关系模块
        relationship_module = EmotionRelationshipModule(emotion_config)
        print(f"✅ 情绪关系模块初始化成功")
        
        # 测试情绪ID映射
        emotion_count = len(relationship_module.emotion_id_map)
        print(f"✅ 情绪映射构建成功: {emotion_count}种情绪")
        
        # 测试关系矩阵
        mutual_exclusion_matrix = relationship_module.mutual_exclusion_matrix
        synergy_matrix = relationship_module.synergy_matrix
        transition_matrix = relationship_module.transition_matrix
        
        print(f"✅ 关系矩阵构建成功: 互斥={mutual_exclusion_matrix.shape}, "
              f"协同={synergy_matrix.shape}, 转换={transition_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 情绪关系建模测试失败: {e}")
        return False

def test_multimodal_fusion():
    """测试多模态融合"""
    print("\n🔍 测试多模态融合...")
    
    try:
        from layers.fusion_layer import MultiModalFusionModule, FusionLayerConfig
        import torch
        
        # 创建配置
        config = FusionLayerConfig(
            fusion_strategy="confidence_weighted",
            text_weight=0.4,
            audio_weight=0.3,
            video_weight=0.3
        )
        
        # 初始化融合模块
        fusion_module = MultiModalFusionModule(config)
        print("✅ 多模态融合模块初始化成功")
        
        # 创建模拟的模态结果
        modality_results = {
            'text': {
                'emotion_probs': torch.rand(1, 27),
                'confidence': torch.tensor([[0.8]]),
                'intensity': torch.rand(1, 27),
                'features': torch.rand(1, 128)
            },
            'audio': {
                'emotion_probs': torch.rand(1, 27),
                'confidence': torch.tensor([[0.6]]),
                'intensity': torch.rand(1, 27),
                'features': torch.rand(1, 128)
            }
        }
        
        # 执行融合
        fused_results = fusion_module.fuse_modalities(modality_results)
        print(f"✅ 多模态融合完成: 融合置信度={fused_results['confidence'].item():.3f}")
        
        # 测试不同的融合策略
        config.fusion_strategy = "simple"
        fusion_module = MultiModalFusionModule(config)
        simple_results = fusion_module.fuse_modalities(modality_results)
        print(f"✅ 简单融合策略测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 多模态融合测试失败: {e}")
        return False

async def test_end_to_end_processing():
    """测试端到端处理"""
    print("\n🔍 测试端到端处理...")
    
    try:
        from layers.fusion_layer import FusionLayer, FusionLayerConfig
        from layers.base_layer import LayerData
        
        # 创建融合层
        config = FusionLayerConfig(
            use_gpu=False,  # 测试时禁用GPU
            enable_emotion_relationships=True
        )
        fusion_layer = FusionLayer(config)
        
        # 创建测试输入数据
        test_data = LayerData(
            layer_name="input_layer",
            timestamp=datetime.now(),
            data={
                'text': "我今天感到很焦虑，躺在床上翻来覆去睡不着，总是担心明天的工作",
                'multimodal_data': {
                    'text_features': np.random.randn(768).tolist(),
                    'quality_score': 0.85
                }
            },
            metadata={
                'source': 'test',
                'modalities': ['text']
            },
            confidence=0.8
        )
        
        # 异步处理
        result = await fusion_layer.process(test_data)
        
        # 验证结果
        emotion_analysis = result.data['emotion_analysis']
        primary_emotion = emotion_analysis['primary_emotion']
        
        print(f"✅ 端到端处理成功:")
        print(f"   - 主要情绪: {primary_emotion['name']}")
        print(f"   - 情绪概率: {primary_emotion['probability']:.3f}")
        print(f"   - 整体置信度: {emotion_analysis['overall_confidence']:.3f}")
        print(f"   - 处理时间: {result.processing_time*1000:.1f}ms")
        
        # 获取情绪信息
        emotion_info = fusion_layer.get_emotion_info(primary_emotion['name'])
        if emotion_info:
            therapy_priority = emotion_info.get('therapy_priority', 'unknown')
            sleep_impact = emotion_info.get('sleep_impact', 'unknown')
            print(f"   - 治疗优先级: {therapy_priority}")
            print(f"   - 睡眠影响: {sleep_impact}")
        
        # 关闭融合层
        fusion_layer.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ 端到端处理测试失败: {e}")
        return False

async def test_pipeline_integration():
    """测试管道集成"""
    print("\n🔍 测试管道集成...")
    
    try:
        from layers.input_layer import InputLayer, InputLayerConfig
        from layers.fusion_layer import FusionLayer, FusionLayerConfig
        from layers.base_layer import LayerPipeline, LayerData
        from datetime import datetime
        
        # 创建输入层
        input_config = InputLayerConfig(
            text_enabled=True,
            audio_enabled=False,
            video_enabled=False
        )
        input_layer = InputLayer(input_config)
        
        # 创建融合层
        fusion_config = FusionLayerConfig(
            use_gpu=False,
            enable_emotion_relationships=True
        )
        fusion_layer = FusionLayer(fusion_config)
        
        # 创建管道
        pipeline = LayerPipeline([input_layer, fusion_layer])
        print("✅ 输入层+融合层管道创建成功")
        
        # 添加文本输入到输入层
        input_layer.add_text_input("我感到很焦虑，总是担心睡不着觉")
        
        # 创建测试数据
        test_data = LayerData(
            layer_name="test_pipeline",
            timestamp=datetime.now(),
            data={"test_input": "管道集成测试"},
            metadata={"source": "pipeline_test"}
        )
        
        # 执行管道处理
        result = await pipeline.process(test_data)
        
        # 验证结果来自融合层
        if result.layer_name == "fusion_layer":
            emotion_analysis = result.data.get('emotion_analysis', {})
            primary_emotion = emotion_analysis.get('primary_emotion', {})
            print(f"✅ 管道处理成功: 最终情绪={primary_emotion.get('name', 'unknown')}")
        else:
            print(f"⚠️ 管道处理结果层级不正确: {result.layer_name}")
        
        # 获取管道状态
        pipeline_status = pipeline.get_pipeline_status()
        print(f"✅ 管道状态: 成功率={pipeline_status['success_rate']:.2f}")
        
        # 关闭层
        input_layer.shutdown()
        fusion_layer.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ 管道集成测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始融合层测试")
    print("=" * 60)
    
    tests = [
        ("融合层模块导入", test_fusion_layer_imports),
        ("融合层配置", test_fusion_layer_config),
        ("情绪配置加载", test_emotion_config_loading),
        ("融合层初始化", test_fusion_layer_initialization),
        ("情绪关系建模", test_emotion_relationship_module),
        ("多模态融合", test_multimodal_fusion),
        ("端到端处理", test_end_to_end_processing),
        ("管道集成", test_pipeline_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            # 同步测试
            if not asyncio.iscoroutinefunction(test_func):
                if test_func():
                    passed += 1
                else:
                    failed += 1
            # 异步测试
            else:
                if await test_func():
                    passed += 1
                else:
                    failed += 1
                    
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            failed += 1
    
    # 测试结果
    print("\n" + "=" * 60)
    print(f"📊 融合层测试结果: 通过={passed}, 失败={failed}")
    
    if failed == 0:
        print("🎉 所有测试通过！融合层功能正常")
        return True
    else:
        print(f"⚠️ {failed}个测试失败，请检查相关问题")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)