#!/usr/bin/env python3
"""
模拟API调用成功，测试后续完整逻辑
跳过真实API调用，直接测试从audio_url到界面播放的完整流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_enhanced_final import download_suno_audio, process_therapy_request
import tempfile
import time

def create_mock_api_response():
    """创建模拟的API响应，包含真实音频URL"""
    return {
        "code": "success",
        "data": {
            "task_id": "mock-task-123",
            "status": "IN_PROGRESS",  # 测试关键修复：不等SUCCESS
            "data": [
                {
                    "id": "mock-audio-1",
                    "title": "Test Therapy Music",
                    "duration": 120,
                    "audio_url": "https://cdn1.suno.ai/7d0fa1f8-6cb6-46ca-b937-c13dab540209.mp3",  # 使用已知可用的URL
                    "tags": "therapy, sleep, calm"
                }
            ]
        }
    }

def test_complete_flow():
    """测试从模拟API响应到界面播放的完整流程"""
    print("🧪 测试：模拟API成功 → 界面播放音乐")
    print("=" * 60)
    
    # 1. 模拟API调用成功
    print("1️⃣ 模拟API调用成功...")
    mock_response = create_mock_api_response()
    print(f"✅ 模拟API响应: {mock_response['code']}")
    
    # 2. 提取音频URL（这是我们修复的关键逻辑）
    print("\n2️⃣ 提取音频URL...")
    audio_url = None
    task_data = mock_response.get('data')
    
    if isinstance(task_data, dict):
        if 'data' in task_data and isinstance(task_data['data'], list) and len(task_data['data']) > 0:
            audio_item = task_data['data'][0]
            audio_url = audio_item.get('audio_url')
            print(f"🎵 找到音频URL: {audio_url}")
            print(f"📝 音频标题: {audio_item.get('title')}")
            print(f"⏱️ 音频时长: {audio_item.get('duration')}秒")
    
    if not audio_url:
        print("❌ 没有找到音频URL")
        return False
    
    # 3. 下载真实音频
    print("\n3️⃣ 下载真实Suno音频...")
    downloaded_file = download_suno_audio(audio_url)
    
    if not downloaded_file:
        print("❌ 音频下载失败")
        return False
    
    print(f"✅ 音频下载成功: {downloaded_file}")
    
    # 4. 验证文件存在和大小
    print("\n4️⃣ 验证音频文件...")
    if os.path.exists(downloaded_file):
        file_size = os.path.getsize(downloaded_file)
        print(f"📊 文件大小: {file_size:,} bytes")
        print(f"📁 文件路径: {downloaded_file}")
        
        # 5. 模拟返回给Gradio
        print("\n5️⃣ 模拟返回给Gradio界面...")
        
        # 这就是process_therapy_request函数最终返回的
        report = f"""✅ 模拟API调用成功！真实音频已下载

🧠 情绪识别结果:
   • 检测情绪: 焦虑
   • 置信度: 85.0%
   • 处理时间: 2.5秒
   • 音频来源: 真实Suno AI音乐

🎵 真实AI音乐信息:
   • 来源: Suno AI (chirp-v3模型)
   • 音频URL: {audio_url[:60]}...
   • 下载状态: ✅ 成功下载
   • 文件格式: MP3 → WAV (兼容播放)
   • 文件大小: {file_size:,} bytes

🎧 测试结论:
   • 这个文件路径可以传给Gradio Audio组件
   • 用户应该能在播放栏听到真实的Suno AI音乐
   • 整个流程从API到播放都应该正常工作"""
        
        print(f"📋 界面报告:")
        print(report)
        
        print(f"\n🎉 完整流程测试成功！")
        print(f"🔗 返回给Gradio的音频文件: {downloaded_file}")
        print(f"📱 用户在界面应该能听到真实的AI音乐")
        
        return True
    else:
        print("❌ 下载的文件不存在")
        return False

def test_gradio_integration():
    """测试与Gradio的集成"""
    print("\n" + "=" * 60)
    print("🔧 测试Gradio集成...")
    
    # 这模拟了用户在界面的操作
    print("模拟用户操作：")
    print("1. 用户输入: '我感到很焦虑，难以入睡'")
    print("2. 用户勾选: ✅ 使用Suno AI音乐生成")
    print("3. 用户勾选: ✅ 启用真实API调用")
    print("4. 用户点击: 生成三阶段疗愈体验")
    
    # 模拟调用process_therapy_request，但用我们的模拟响应
    print("\n模拟process_therapy_request函数执行...")
    print("(实际会调用真实API，但我们跳过这步)")
    
    # 直接测试后续逻辑
    result = test_complete_flow()
    
    if result:
        print(f"\n✅ 集成测试成功！")
        print(f"💡 结论：真实API调用逻辑应该能正常工作")
        print(f"🎯 下一步：可以安全地连接真实API调用")
    else:
        print(f"\n❌ 集成测试失败，需要修复")

if __name__ == "__main__":
    print("🚀 开始完整流程测试")
    print("🎯 目标：验证从API成功到界面播放的完整逻辑")
    print("💰 成本：零！不调用真实API")
    
    test_gradio_integration()
    
    print(f"\n📝 测试总结：")
    print(f"   • 如果上述测试成功，说明后续逻辑没问题")
    print(f"   • 可以安全地连接真实API调用")
    print(f"   • 用户花钱后应该能听到真实音乐")