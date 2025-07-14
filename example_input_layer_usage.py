#!/usr/bin/env python3
"""
输入层标准化接口使用示例
展示如何使用新添加的四个标准化函数
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.input_layer import InputLayer, InputLayerConfig

def main():
    """输入层标准化接口使用示例"""
    
    print("🚀 输入层标准化接口使用示例")
    print("=" * 50)
    
    # 1. 创建输入层配置
    config = InputLayerConfig(
        layer_name="example_input",
        text_enabled=True,
        audio_enabled=True,
        video_enabled=True
    )
    
    # 2. 初始化输入层
    input_layer = InputLayer(config)
    print("✅ 输入层初始化完成")
    
    # 3. 使用标准化接口函数
    
    # 视频帧捕获
    print("\n📹 捕获视频帧:")
    video_frame = input_layer.capture_video_frame()
    print(f"   视频帧形状: {video_frame.shape}")
    print(f"   数据类型: {video_frame.dtype}")
    
    # 音频块捕获
    print("\n🎵 捕获音频块:")
    audio_chunk = input_layer.capture_audio_chunk()
    print(f"   音频块长度: {len(audio_chunk)}")
    print(f"   音频RMS: {np.sqrt(np.mean(audio_chunk**2)):.3f}")
    
    # 文本输入获取
    print("\n📝 获取文本输入:")
    text_input = input_layer.get_user_text_input()
    print(f"   文本内容: {text_input}")
    
    # 多模态数据收集（主要接口）
    print("\n🔄 收集多模态数据:")
    multimodal_data = input_layer.collect_multimodal_data()
    
    # 分析收集到的数据
    video_enabled = multimodal_data['video']['enabled']
    audio_enabled = multimodal_data['audio']['enabled']
    text_enabled = multimodal_data['text']['enabled']
    data_quality = multimodal_data['metadata']['data_quality']
    
    print(f"   视频模态: {'✅' if video_enabled else '❌'}")
    print(f"   音频模态: {'✅' if audio_enabled else '❌'}")
    print(f"   文本模态: {'✅' if text_enabled else '❌'}")
    print(f"   数据质量: {data_quality:.3f}")
    
    # 4. 清理资源
    input_layer.shutdown()
    print("\n✅ 示例运行完成")

if __name__ == "__main__":
    main()