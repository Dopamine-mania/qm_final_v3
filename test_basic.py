#!/usr/bin/env python3
"""
qm_final3 基础测试脚本

验证项目基础架构是否正常工作，包括：
1. 导入测试
2. 配置加载测试
3. 层初始化测试
4. 基础功能测试
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试核心模块导入
        from core.utils import ConfigLoader, DataValidator, PerformanceMonitor
        print("✅ 核心工具模块导入成功")
        
        # 测试层模块导入
        from layers.base_layer import BaseLayer, LayerData, LayerConfig, LayerPipeline
        print("✅ 基础层模块导入成功")
        
        from layers.input_layer import InputLayer, InputLayerConfig
        print("✅ 输入层模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🔍 测试配置加载...")
    
    try:
        from core.utils import ConfigLoader
        
        # 测试YAML配置加载
        config_path = "configs/six_layer_architecture.yaml"
        if os.path.exists(config_path):
            config = ConfigLoader.load_yaml(config_path)
            print(f"✅ 主配置加载成功: {config['system']['name']}")
        else:
            print(f"⚠️ 主配置文件不存在: {config_path}")
        
        # 测试情绪配置加载
        emotion_config_path = "configs/emotion_27d.yaml"
        if os.path.exists(emotion_config_path):
            emotion_config = ConfigLoader.load_yaml(emotion_config_path)
            print(f"✅ 情绪配置加载成功: {emotion_config['emotion_space']['total_dimensions']}维")
        else:
            print(f"⚠️ 情绪配置文件不存在: {emotion_config_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_layer_initialization():
    """测试层初始化"""
    print("\n🔍 测试层初始化...")
    
    try:
        from layers.input_layer import InputLayer, InputLayerConfig
        
        # 创建输入层配置
        config = InputLayerConfig(
            text_enabled=True,
            audio_enabled=False,  # 暂时禁用音频
            video_enabled=False,  # 暂时禁用视频
            max_processing_time_ms=100
        )
        
        # 初始化输入层
        input_layer = InputLayer(config)
        print(f"✅ 输入层初始化成功: {input_layer.layer_name}")
        
        # 获取层状态
        status = input_layer.get_status()
        print(f"✅ 层状态获取成功: 处理次数={status['total_processed']}")
        
        # 关闭输入层
        input_layer.shutdown()
        print("✅ 输入层关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 层初始化失败: {e}")
        return False

def test_data_processing():
    """测试数据处理"""
    print("\n🔍 测试数据处理...")
    
    try:
        from layers.base_layer import LayerData
        from layers.input_layer import InputLayer, InputLayerConfig
        from datetime import datetime
        import asyncio
        
        # 创建输入层
        config = InputLayerConfig(
            text_enabled=True,
            audio_enabled=False,
            video_enabled=False,
            max_processing_time_ms=100
        )
        input_layer = InputLayer(config)
        
        # 添加文本输入
        test_text = "我今天感到很焦虑，躺在床上翻来覆去睡不着"
        input_layer.add_text_input(test_text)
        print(f"✅ 文本输入添加成功: {test_text}")
        
        # 创建测试数据
        test_data = LayerData(
            layer_name="test",
            timestamp=datetime.now(),
            data={"test_input": "测试数据"},
            metadata={"source": "test"}
        )
        
        # 异步处理数据
        async def process_test():
            result = await input_layer.process(test_data)
            return result
        
        # 运行异步处理
        result = asyncio.run(process_test())
        print(f"✅ 数据处理成功: 置信度={result.confidence:.2f}")
        
        # 关闭输入层
        input_layer.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return False

def test_pipeline():
    """测试管道"""
    print("\n🔍 测试管道...")
    
    try:
        from layers.base_layer import LayerPipeline, LayerData
        from layers.input_layer import InputLayer, InputLayerConfig
        from datetime import datetime
        import asyncio
        
        # 创建输入层
        config = InputLayerConfig(
            text_enabled=True,
            audio_enabled=False,
            video_enabled=False,
            max_processing_time_ms=100
        )
        input_layer = InputLayer(config)
        
        # 创建管道
        pipeline = LayerPipeline([input_layer])
        print("✅ 管道创建成功")
        
        # 创建测试数据
        test_data = LayerData(
            layer_name="test",
            timestamp=datetime.now(),
            data={"test_input": "管道测试数据"},
            metadata={"source": "pipeline_test"}
        )
        
        # 异步处理管道
        async def process_pipeline():
            result = await pipeline.process(test_data)
            return result
        
        # 运行管道处理
        result = asyncio.run(process_pipeline())
        print(f"✅ 管道处理成功: 层={result.layer_name}")
        
        # 获取管道状态
        pipeline_status = pipeline.get_pipeline_status()
        print(f"✅ 管道状态获取成功: 成功率={pipeline_status['success_rate']:.2f}")
        
        # 关闭输入层
        input_layer.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ 管道测试失败: {e}")
        return False

def test_utilities():
    """测试工具函数"""
    print("\n🔍 测试工具函数...")
    
    try:
        from core.utils import (
            DataValidator, PerformanceMonitor, 
            normalize_vector, cosine_similarity,
            format_duration, calculate_memory_usage
        )
        import numpy as np
        
        # 测试数据验证
        validator = DataValidator()
        
        # 测试文本验证
        text_valid = validator.validate_text("测试文本")
        print(f"✅ 文本验证: {text_valid}")
        
        # 测试音频验证
        audio_data = np.random.randn(1000)
        audio_valid = validator.validate_audio(audio_data)
        print(f"✅ 音频验证: {audio_valid}")
        
        # 测试性能监控
        monitor = PerformanceMonitor()
        monitor.start_timer("test_operation")
        
        # 模拟操作
        import time
        time.sleep(0.01)
        
        elapsed = monitor.end_timer("test_operation")
        print(f"✅ 性能监控: 耗时={elapsed*1000:.2f}ms")
        
        # 测试向量操作
        vec1 = np.random.randn(10)
        vec2 = np.random.randn(10)
        
        vec1_norm = normalize_vector(vec1)
        similarity = cosine_similarity(vec1, vec2)
        print(f"✅ 向量操作: 相似度={similarity:.3f}")
        
        # 测试时间格式化
        duration_str = format_duration(0.123)
        print(f"✅ 时间格式化: {duration_str}")
        
        # 测试内存使用
        memory_info = calculate_memory_usage()
        if memory_info:
            print(f"✅ 内存监控: {memory_info.get('rss_mb', 0):.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始qm_final3基础测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("配置加载", test_config_loading),
        ("层初始化", test_layer_initialization),
        ("数据处理", test_data_processing),
        ("管道测试", test_pipeline),
        ("工具函数", test_utilities)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            failed += 1
    
    # 测试结果
    print("\n" + "=" * 50)
    print(f"📊 测试结果: 通过={passed}, 失败={failed}")
    
    if failed == 0:
        print("🎉 所有测试通过！qm_final3基础架构运行正常")
        return True
    else:
        print(f"⚠️ {failed}个测试失败，请检查相关问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)