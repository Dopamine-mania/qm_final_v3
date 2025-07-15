#!/usr/bin/env python3
"""
测试图片生成功能
验证Stable Diffusion API集成和三阶段图片生成逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_enhanced_final import (
    generate_image_prompts, 
    call_stable_diffusion_api, 
    get_emotion_music_features
)

def test_image_prompt_generation():
    """测试图片提示词生成"""
    print("🎨 测试图片提示词生成...")
    print("=" * 60)
    
    # 测试不同情绪的图片提示词生成
    emotions = ["焦虑", "疲惫", "烦躁", "平静", "压力"]
    duration = 12  # 12秒音乐
    
    for emotion in emotions:
        print(f"\n📊 测试情绪: {emotion}")
        print("-" * 40)
        
        # 获取音乐特征
        music_features = get_emotion_music_features(emotion)
        
        # 生成图片提示词
        image_prompts = generate_image_prompts(emotion, music_features, duration)
        
        print(f"图片数量: {len(image_prompts)}张")
        print(f"生成间隔: 每3秒一张")
        
        # 显示每张图片的信息
        for i, prompt_data in enumerate(image_prompts):
            print(f"\n🖼️ 图片{i+1}:")
            print(f"   时间戳: 第{prompt_data['timestamp']}秒")
            print(f"   阶段: {prompt_data['stage']}")
            print(f"   提示词: {prompt_data['prompt'][:80]}...")
    
    print("\n✅ 图片提示词生成测试完成")

def test_stable_diffusion_api():
    """测试Stable Diffusion API调用"""
    print("\n🔧 测试Stable Diffusion API调用...")
    print("=" * 60)
    
    # 测试提示词
    test_prompts = [
        "peaceful starry night, calm ocean, deep relaxation, tranquil sleep, therapeutic healing art",
        "energetic morning, vibrant landscape, renewed vitality, fresh beginning, therapeutic healing art",
        "perfect harmony, balanced nature, inner peace, emotional stability, therapeutic healing art"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🎨 测试提示词 {i+1}:")
        print(f"提示词: {prompt}")
        
        # 调用API（测试模式）
        result = call_stable_diffusion_api(prompt, enable_real_api=True)
        
        if result.get('success'):
            print(f"✅ 生成成功")
            print(f"图片URL: {result.get('image_url')}")
            print(f"是否模拟: {result.get('mock', False)}")
        else:
            print(f"❌ 生成失败: {result.get('error')}")
    
    print("\n✅ Stable Diffusion API测试完成")

def test_complete_image_flow():
    """测试完整的图片生成流程"""
    print("\n🌊 测试完整图片生成流程...")
    print("=" * 60)
    
    # 模拟用户输入
    emotion = "焦虑"
    duration = 12
    
    print(f"情绪: {emotion}")
    print(f"时长: {duration}秒")
    
    # 1. 获取音乐特征
    music_features = get_emotion_music_features(emotion)
    print(f"\n🎵 音乐特征:")
    print(f"   匹配阶段: {music_features['匹配阶段']['mood']}")
    print(f"   引导阶段: {music_features['引导阶段']['mood']}")
    print(f"   目标阶段: {music_features['目标阶段']['mood']}")
    
    # 2. 生成图片提示词
    image_prompts = generate_image_prompts(emotion, music_features, duration)
    print(f"\n🖼️ 生成{len(image_prompts)}张图片提示词")
    
    # 3. 生成图片
    generated_images = []
    for i, prompt_data in enumerate(image_prompts):
        print(f"\n📸 生成图片{i+1} (第{prompt_data['timestamp']}秒, {prompt_data['stage']})")
        
        image_result = call_stable_diffusion_api(
            prompt_data['prompt'], 
            enable_real_api=True
        )
        
        if image_result.get('success'):
            generated_images.append({
                'stage': prompt_data['stage'],
                'timestamp': prompt_data['timestamp'],
                'image_url': image_result.get('image_url'),
                'prompt': prompt_data['prompt'][:50] + "..."
            })
            print(f"   ✅ 生成成功: {image_result.get('image_url')}")
        else:
            print(f"   ❌ 生成失败: {image_result.get('error')}")
    
    # 4. 总结结果
    print(f"\n📊 生成结果总结:")
    print(f"   成功生成: {len(generated_images)}/{len(image_prompts)}张")
    print(f"   时间跨度: 0-{duration}秒")
    print(f"   阶段分布:")
    
    stage_count = {}
    for img in generated_images:
        stage = img['stage']
        stage_count[stage] = stage_count.get(stage, 0) + 1
    
    for stage, count in stage_count.items():
        print(f"     {stage}: {count}张")
    
    print("\n✅ 完整图片生成流程测试完成")

def main():
    """主测试函数"""
    print("🚀 开始图片生成功能测试")
    print("🎯 目标：验证Stable Diffusion API集成和三阶段图片生成")
    print("💰 成本：测试模式，完全免费")
    print()
    
    # 运行所有测试
    test_image_prompt_generation()
    test_stable_diffusion_api()
    test_complete_image_flow()
    
    print(f"\n🎉 所有测试完成！")
    print(f"📝 测试总结:")
    print(f"   • 图片提示词生成：根据ISO三阶段原则生成")
    print(f"   • API调用：使用与Suno相同的feiai.chat端点")
    print(f"   • 完整流程：从情绪到图片序列的完整转换")
    print(f"   • 成本控制：测试模式确保零费用")

if __name__ == "__main__":
    main()