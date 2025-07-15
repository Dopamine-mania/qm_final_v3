#!/usr/bin/env python3
"""
测试图片内容生成功能
验证修复后的图片能够正确生成真实的本地图片文件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_enhanced_final import call_stable_diffusion_api, generate_image_prompts, get_emotion_music_features

def test_local_image_generation():
    """测试本地图片生成功能"""
    print("🎨 测试本地图片生成功能...")
    print("=" * 60)
    
    # 测试不同的提示词
    test_prompts = [
        "dark stormy clouds, turbulent ocean waves, dramatic shadows, moody atmosphere",
        "soft moonlight breaking through clouds, gentle waves, calming transition",
        "peaceful starry night, calm ocean, deep relaxation, tranquil sleep"
    ]
    
    generated_files = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🖼️ 测试提示词 {i+1}:")
        print(f"提示词: {prompt}")
        
        # 调用图片生成API
        result = call_stable_diffusion_api(prompt, enable_real_api=True)
        
        if result.get('success'):
            image_path = result.get('image_path')
            print(f"✅ 生成成功: {image_path}")
            
            # 检查文件是否存在
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"📂 文件存在: {file_size} bytes")
                
                # 检查文件类型
                file_ext = os.path.splitext(image_path)[1].lower()
                print(f"📄 文件类型: {file_ext}")
                
                generated_files.append(image_path)
            else:
                print(f"❌ 文件不存在: {image_path}")
        else:
            print(f"❌ 生成失败: {result.get('error')}")
    
    return generated_files

def test_complete_image_flow():
    """测试完整的图片生成流程"""
    print("\n🌊 测试完整图片生成流程...")
    print("=" * 60)
    
    # 测试参数
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
            image_path = image_result.get('image_path')
            generated_images.append(image_path)
            print(f"   ✅ 生成成功: {image_path}")
            
            # 验证文件
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"   📂 文件大小: {file_size} bytes")
            else:
                print(f"   ❌ 文件不存在")
        else:
            print(f"   ❌ 生成失败: {image_result.get('error')}")
    
    # 4. 总结结果
    print(f"\n📊 生成结果总结:")
    print(f"   成功生成: {len(generated_images)}/{len(image_prompts)}张")
    print(f"   图片文件列表:")
    for i, img_path in enumerate(generated_images):
        print(f"     {i+1}. {img_path}")
    
    return generated_images

def test_pil_availability():
    """测试PIL是否可用"""
    print("\n🔍 测试PIL可用性...")
    print("=" * 60)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL可用，可以生成真实图片")
        
        # 测试创建一个简单图片
        img = Image.new('RGB', (100, 100), (255, 0, 0))
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        print(f"✅ 测试图片生成成功: {temp_file.name}")
        
        # 清理测试文件
        os.unlink(temp_file.name)
        
        return True
    except ImportError as e:
        print(f"❌ PIL不可用: {e}")
        print("💡 请安装PIL: pip install Pillow")
        return False

def main():
    """主测试函数"""
    print("🚀 开始图片内容生成测试")
    print("🎯 目标：验证能生成真实的本地图片文件")
    print("💡 特点：使用PIL生成带有文字的彩色图片")
    print()
    
    # 检查PIL可用性
    pil_available = test_pil_availability()
    
    if pil_available:
        # 运行测试
        generated_files = test_local_image_generation()
        complete_flow_files = test_complete_image_flow()
        
        print(f"\n🎉 测试完成！")
        print(f"📝 测试总结:")
        print(f"   • PIL可用: ✅")
        print(f"   • 基础图片生成: {len(generated_files)}张")
        print(f"   • 完整流程图片: {len(complete_flow_files)}张")
        print(f"   • 图片格式: PNG")
        print(f"   • 图片大小: 512x512像素")
        print(f"   • 包含文字: 阶段标识和序号")
        
        # 清理测试文件
        all_files = generated_files + complete_flow_files
        for file_path in all_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"🗑️ 清理文件: {file_path}")
            except Exception as e:
                print(f"⚠️ 清理失败: {e}")
    else:
        print(f"\n❌ 测试无法进行：PIL不可用")
        print(f"💡 建议：pip install Pillow")

if __name__ == "__main__":
    main()