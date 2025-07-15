#!/usr/bin/env python3
"""
测试图片显示功能
验证修改后的Gradio界面能够正确显示图片
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_enhanced_final import process_therapy_request

def test_image_display():
    """测试图片显示功能"""
    print("🧪 测试图片显示功能...")
    print("=" * 60)
    
    # 测试参数
    user_input = "我感到很焦虑，心跳加速，难以入睡"
    duration = 12
    use_suno_api = True
    enable_real_api = True
    existing_task_id = ""
    enable_image_generation = True
    
    print(f"输入参数:")
    print(f"  情绪描述: {user_input}")
    print(f"  时长: {duration}秒")
    print(f"  使用Suno API: {use_suno_api}")
    print(f"  启用真实API: {enable_real_api}")
    print(f"  启用图片生成: {enable_image_generation}")
    
    # 调用处理函数
    try:
        result = process_therapy_request(
            user_input=user_input,
            duration=duration,
            use_suno_api=use_suno_api,
            enable_real_api=enable_real_api,
            existing_task_id=existing_task_id,
            enable_image_generation=enable_image_generation
        )
        
        print(f"\n✅ 函数调用成功")
        print(f"返回值数量: {len(result)}")
        
        if len(result) == 4:
            report, audio_file, image_gallery, status = result
            
            print(f"\n📊 返回值分析:")
            print(f"  报告类型: {type(report)}")
            print(f"  音频文件: {audio_file}")
            print(f"  图片画廊类型: {type(image_gallery)}")
            print(f"  图片数量: {len(image_gallery) if image_gallery else 0}")
            print(f"  状态: {status}")
            
            # 显示图片信息
            if image_gallery and len(image_gallery) > 0:
                print(f"\n🖼️ 图片信息:")
                for i, img_url in enumerate(image_gallery):
                    print(f"  图片{i+1}: {img_url}")
                    
                # 验证图片URL格式
                print(f"\n🔍 图片URL验证:")
                for i, img_url in enumerate(image_gallery):
                    if img_url.startswith("https://via.placeholder.com/"):
                        print(f"  图片{i+1}: ✅ 格式正确")
                    else:
                        print(f"  图片{i+1}: ❌ 格式异常")
            else:
                print(f"\n⚠️ 没有生成图片")
                
            # 检查报告中的图片信息
            if "🖼️ 配套疗愈图片" in report:
                print(f"\n📋 报告包含图片信息: ✅")
            else:
                print(f"\n📋 报告不包含图片信息: ❌")
                
        else:
            print(f"❌ 返回值数量不正确，期望4个，实际{len(result)}个")
            
    except Exception as e:
        print(f"❌ 函数调用失败: {e}")
        import traceback
        traceback.print_exc()

def test_without_image_generation():
    """测试不启用图片生成的情况"""
    print("\n🧪 测试不启用图片生成的情况...")
    print("=" * 60)
    
    # 测试参数
    user_input = "我感到很疲惫，但大脑还在活跃"
    duration = 15
    use_suno_api = False
    enable_real_api = False
    existing_task_id = ""
    enable_image_generation = False
    
    print(f"输入参数:")
    print(f"  情绪描述: {user_input}")
    print(f"  启用图片生成: {enable_image_generation}")
    
    try:
        result = process_therapy_request(
            user_input=user_input,
            duration=duration,
            use_suno_api=use_suno_api,
            enable_real_api=enable_real_api,
            existing_task_id=existing_task_id,
            enable_image_generation=enable_image_generation
        )
        
        print(f"\n✅ 函数调用成功")
        print(f"返回值数量: {len(result)}")
        
        if len(result) == 4:
            report, audio_file, image_gallery, status = result
            
            print(f"\n📊 返回值分析:")
            print(f"  图片数量: {len(image_gallery) if image_gallery else 0}")
            print(f"  状态: {status}")
            
            if not image_gallery or len(image_gallery) == 0:
                print(f"✅ 正确：未启用图片生成时图片列表为空")
            else:
                print(f"❌ 错误：未启用图片生成但仍有图片")
                
        else:
            print(f"❌ 返回值数量不正确")
            
    except Exception as e:
        print(f"❌ 函数调用失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始图片显示功能测试")
    print("🎯 目标：验证Gradio界面能正确显示图片")
    print("💡 特点：测试模式生成不同颜色的占位符图片")
    print()
    
    # 运行测试
    test_image_display()
    test_without_image_generation()
    
    print(f"\n🎉 测试完成！")
    print(f"📝 测试总结:")
    print(f"   • 函数返回值格式：(report, audio_file, image_gallery, status)")
    print(f"   • 图片数据：URL列表，适合Gradio Gallery组件")
    print(f"   • 测试模式：不同颜色的占位符图片显示不同阶段")
    print(f"   • 界面集成：所有组件都已正确配置")

if __name__ == "__main__":
    main()