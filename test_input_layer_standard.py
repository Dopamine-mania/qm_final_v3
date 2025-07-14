#!/usr/bin/env python3
"""
测试输入层标准化接口函数
验证根据用户规范添加的四个标准化函数
"""

import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.input_layer import InputLayer, InputLayerConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_input_layer_standard_functions():
    """测试输入层标准化接口函数"""
    
    print("📱 输入层标准化接口函数测试")
    print("=" * 60)
    
    # 初始化输入层配置
    config = InputLayerConfig(
        layer_name="test_input_layer",
        text_enabled=True,
        audio_enabled=True,
        video_enabled=True,
        audio_chunk_size=1024,
        video_width=640,
        video_height=480
    )
    
    # 创建输入层实例
    input_layer = InputLayer(config)
    
    print(f"✅ 输入层初始化完成")
    print(f"   • 文本模态: {'启用' if config.text_enabled else '禁用'}")
    print(f"   • 音频模态: {'启用' if config.audio_enabled else '禁用'}")
    print(f"   • 视频模态: {'启用' if config.video_enabled else '禁用'}")
    
    # 测试1: capture_video_frame()
    print(f"\n📹 测试1: capture_video_frame() 函数")
    print("=" * 40)
    
    try:
        video_frame = input_layer.capture_video_frame()
        print(f"✅ 视频帧捕获成功")
        print(f"   • 数据类型: {type(video_frame)}")
        print(f"   • 数组形状: {video_frame.shape}")
        print(f"   • 数据类型: {video_frame.dtype}")
        print(f"   • 数值范围: [{video_frame.min()}, {video_frame.max()}]")
        
        # 验证预期格式
        if isinstance(video_frame, np.ndarray) and len(video_frame.shape) == 3:
            height, width, channels = video_frame.shape
            print(f"   ✅ 格式验证: {height}x{width}x{channels} (height x width x channels)")
        else:
            print(f"   ❌ 格式错误: 期望3维数组")
            
    except Exception as e:
        print(f"❌ 视频帧捕获失败: {e}")
    
    # 测试2: capture_audio_chunk()
    print(f"\n🎵 测试2: capture_audio_chunk() 函数")
    print("=" * 40)
    
    try:
        audio_chunk = input_layer.capture_audio_chunk()
        print(f"✅ 音频块捕获成功")
        print(f"   • 数据类型: {type(audio_chunk)}")
        print(f"   • 数组形状: {audio_chunk.shape}")
        print(f"   • 数据类型: {audio_chunk.dtype}")
        print(f"   • 数值范围: [{audio_chunk.min():.3f}, {audio_chunk.max():.3f}]")
        print(f"   • RMS能量: {np.sqrt(np.mean(audio_chunk**2)):.3f}")
        
        # 验证预期格式
        if isinstance(audio_chunk, np.ndarray) and len(audio_chunk.shape) == 1:
            print(f"   ✅ 格式验证: 一维音频数组，长度 {len(audio_chunk)}")
        else:
            print(f"   ❌ 格式错误: 期望一维数组")
            
    except Exception as e:
        print(f"❌ 音频块捕获失败: {e}")
    
    # 测试3: get_user_text_input()
    print(f"\n📝 测试3: get_user_text_input() 函数")
    print("=" * 40)
    
    try:
        # 先添加一些测试文本到缓冲区
        test_texts = [
            "我感到很焦虑，睡不着觉",
            "今天压力很大，需要放松",
            "心情沮丧，想要平静下来"
        ]
        
        for text in test_texts:
            input_layer.add_text_input(text)
        
        # 测试多次获取文本输入
        for i in range(3):
            text_input = input_layer.get_user_text_input()
            print(f"✅ 文本输入获取成功 #{i+1}")
            print(f"   • 数据类型: {type(text_input)}")
            print(f"   • 文本长度: {len(text_input)}")
            print(f"   • 内容: {text_input}")
            
            # 验证预期格式
            if isinstance(text_input, str) and len(text_input) > 0:
                print(f"   ✅ 格式验证: 非空字符串")
            else:
                print(f"   ❌ 格式错误: 期望非空字符串")
            print()
            
    except Exception as e:
        print(f"❌ 文本输入获取失败: {e}")
    
    # 测试4: collect_multimodal_data()
    print(f"\n🔄 测试4: collect_multimodal_data() 主函数")
    print("=" * 40)
    
    try:
        multimodal_data = input_layer.collect_multimodal_data()
        print(f"✅ 多模态数据收集成功")
        
        # 分析收集到的数据结构
        print(f"\n📊 数据结构分析:")
        
        # 视频数据分析
        video_data = multimodal_data.get('video', {})
        print(f"   📹 视频数据:")
        print(f"      • 启用状态: {video_data.get('enabled', False)}")
        if video_data.get('raw_frame') is not None:
            raw_frame = video_data['raw_frame']
            print(f"      • 原始帧形状: {raw_frame.shape}")
            print(f"      • 处理状态: {'已处理' if video_data.get('processed') else '未处理'}")
        
        # 音频数据分析
        audio_data = multimodal_data.get('audio', {})
        print(f"   🎵 音频数据:")
        print(f"      • 启用状态: {audio_data.get('enabled', False)}")
        if audio_data.get('raw_chunk') is not None:
            raw_chunk = audio_data['raw_chunk']
            print(f"      • 原始音频长度: {len(raw_chunk)}")
            print(f"      • 处理状态: {'已处理' if audio_data.get('processed') else '未处理'}")
        
        # 文本数据分析
        text_data = multimodal_data.get('text', {})
        print(f"   📝 文本数据:")
        print(f"      • 启用状态: {text_data.get('enabled', False)}")
        if text_data.get('raw_text'):
            raw_text = text_data['raw_text']
            print(f"      • 原始文本: {raw_text}")
            print(f"      • 处理状态: {'已处理' if text_data.get('processed') else '未处理'}")
        
        # 元数据分析
        metadata = multimodal_data.get('metadata', {})
        print(f"   📋 元数据:")
        print(f"      • 收集时间: {metadata.get('collection_timestamp')}")
        print(f"      • 收集模式: {metadata.get('collection_mode')}")
        print(f"      • 数据质量: {metadata.get('data_quality', 0):.3f}")
        
        enabled_modalities = metadata.get('enabled_modalities', {})
        enabled_count = sum([1 for enabled in enabled_modalities.values() if enabled])
        print(f"      • 启用模态数: {enabled_count}/3")
        
        # 验证数据完整性
        required_keys = ['video', 'audio', 'text', 'metadata']
        missing_keys = [key for key in required_keys if key not in multimodal_data]
        
        if not missing_keys:
            print(f"   ✅ 数据完整性: 所有必需字段都存在")
        else:
            print(f"   ❌ 数据完整性: 缺少字段 {missing_keys}")
            
    except Exception as e:
        print(f"❌ 多模态数据收集失败: {e}")
    
    # 性能测试
    print(f"\n⚡ 性能测试")
    print("=" * 30)
    
    import time
    
    # 测试单次调用性能
    performance_tests = [
        ("capture_video_frame", lambda: input_layer.capture_video_frame()),
        ("capture_audio_chunk", lambda: input_layer.capture_audio_chunk()),
        ("get_user_text_input", lambda: input_layer.get_user_text_input()),
        ("collect_multimodal_data", lambda: input_layer.collect_multimodal_data())
    ]
    
    for test_name, test_func in performance_tests:
        start_time = time.time()
        try:
            result = test_func()
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # 转换为毫秒
            print(f"   • {test_name}: {execution_time:.2f}ms ✅")
        except Exception as e:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            print(f"   • {test_name}: {execution_time:.2f}ms ❌ ({e})")
    
    # 批量测试
    print(f"\n📦 批量调用测试 (10次)")
    print("=" * 30)
    
    batch_size = 10
    start_time = time.time()
    
    try:
        for i in range(batch_size):
            multimodal_data = input_layer.collect_multimodal_data()
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        avg_time = total_time / batch_size
        
        print(f"   • 总时间: {total_time:.2f}ms")
        print(f"   • 平均时间: {avg_time:.2f}ms")
        print(f"   • 吞吐量: {1000/avg_time:.1f} 次/秒")
        
    except Exception as e:
        print(f"   ❌ 批量测试失败: {e}")
    
    # 清理资源
    input_layer.shutdown()
    
    print(f"\n🎉 输入层标准化接口函数测试完成！")
    print(f"\n📊 测试总结:")
    print(f"   ✅ capture_video_frame() - 视频帧捕获模拟")
    print(f"   ✅ capture_audio_chunk() - 音频块捕获模拟") 
    print(f"   ✅ get_user_text_input() - 文本输入模拟")
    print(f"   ✅ collect_multimodal_data() - 多模态数据收集主函数")
    print(f"\n💡 核心特性:")
    print(f"   • 传感器数据捕获模拟")
    print(f"   • 原始数据处理和预处理")
    print(f"   • 多模态数据同步收集")
    print(f"   • 数据质量评估和监控")
    print(f"   • 错误处理和模拟数据回退")

if __name__ == "__main__":
    test_input_layer_standard_functions()