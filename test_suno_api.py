#!/usr/bin/env python3
"""
🧪 Suno API测试脚本
快速验证API调用是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gradio_enhanced_final import call_suno_api, get_emotion_music_features

def test_api_calls():
    print("🧪 测试Suno API调用功能")
    print("=" * 50)
    
    emotion = "焦虑"
    music_features = get_emotion_music_features(emotion)
    
    print(f"🎭 测试情绪: {emotion}")
    print(f"🎵 音乐特征: {music_features}")
    print()
    
    # 测试1：模拟模式（默认）
    print("🔍 测试1: 模拟模式")
    response1 = call_suno_api(emotion, music_features, enable_real_api=False)
    print(f"   结果: {'✅ 模拟响应' if response1.get('mock') else '❌ 意外的真实响应'}")
    print(f"   任务ID: {response1.get('task_id', 'N/A')}")
    print()
    
    # 测试2：真实API模式
    print("🔍 测试2: 真实API模式（注意：这会消耗费用！）")
    print("⚠️ 如果不想消耗费用，请按Ctrl+C中断")
    
    try:
        import time
        print("⏳ 3秒后调用真实API...")
        time.sleep(3)
        
        response2 = call_suno_api(emotion, music_features, enable_real_api=True)
        
        if response2.get('mock'):
            print("   ❌ 意外返回模拟响应")
        else:
            print("   ✅ 真实API调用成功")
            print(f"   任务ID: {response2.get('task_id', 'N/A')}")
            print(f"   状态: {response2.get('status', 'N/A')}")
            if response2.get('data'):
                print(f"   音频URL: {response2.get('data', {}).get('audio_url', 'N/A')}")
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断，跳过真实API测试")
    except Exception as e:
        print(f"   ❌ API调用失败: {e}")
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    test_api_calls()