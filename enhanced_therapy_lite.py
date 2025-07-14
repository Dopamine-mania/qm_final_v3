#!/usr/bin/env python3
"""
🌙 轻量级增强三阶段疗愈系统
优化性能，避免卡顿，快速响应
"""

import numpy as np
import sys
import os
import time
from pathlib import Path

def generate_lightweight_therapy_audio(duration=10, sample_rate=22050, emotion="焦虑"):
    """生成轻量级疗愈音频（性能优化版本）"""
    print(f"🎵 生成{duration}秒轻量级疗愈音频 (针对{emotion}情绪)")
    
    # 减少采样率和时长以提升性能
    duration = min(duration, 15)  # 最大15秒
    sample_rate = 22050  # 降低采样率
    
    # 简化的情绪参数
    emotion_params = {
        "焦虑": {"sync_freq": 440, "guide_freq": 330, "consolidate_freq": 220},
        "疲惫": {"sync_freq": 380, "guide_freq": 280, "consolidate_freq": 200},
        "烦躁": {"sync_freq": 460, "guide_freq": 350, "consolidate_freq": 240},
        "平静": {"sync_freq": 400, "guide_freq": 320, "consolidate_freq": 210},
        "压力": {"sync_freq": 480, "guide_freq": 360, "consolidate_freq": 230}
    }
    
    params = emotion_params.get(emotion, emotion_params["焦虑"])
    
    # 三阶段时间分配
    stage1_duration = duration * 0.3
    stage2_duration = duration * 0.4
    stage3_duration = duration * 0.3
    
    print(f"🎵 三阶段时长: 同步期{stage1_duration:.1f}s → 引导期{stage2_duration:.1f}s → 巩固期{stage3_duration:.1f}s")
    
    # 生成音频数组（简化版）
    total_samples = int(sample_rate * duration)
    audio_mono = np.zeros(total_samples)
    
    # 时间轴
    t = np.linspace(0, duration, total_samples)
    
    # 第一阶段：同步期
    stage1_mask = t <= stage1_duration
    stage1_audio = 0.3 * np.sin(2 * np.pi * params['sync_freq'] * t[stage1_mask])
    audio_mono[stage1_mask] = stage1_audio
    
    # 第二阶段：引导期（简化的线性过渡）
    stage2_mask = (t > stage1_duration) & (t <= stage1_duration + stage2_duration)
    stage2_t = t[stage2_mask] - stage1_duration
    transition_progress = stage2_t / stage2_duration
    
    # 线性频率过渡
    current_freq = params['sync_freq'] + (params['guide_freq'] - params['sync_freq']) * transition_progress
    stage2_audio = 0.25 * np.sin(2 * np.pi * current_freq * stage2_t)
    audio_mono[stage2_mask] = stage2_audio
    
    # 第三阶段：巩固期
    stage3_mask = t > stage1_duration + stage2_duration
    stage3_t = t[stage3_mask] - stage1_duration - stage2_duration
    stage3_audio = 0.15 * np.sin(2 * np.pi * params['consolidate_freq'] * stage3_t)
    # 添加简单的衰减
    stage3_audio *= np.exp(-stage3_t / 8)
    audio_mono[stage3_mask] = stage3_audio
    
    # 简化的立体声处理
    stereo_audio = np.column_stack([audio_mono, audio_mono])
    
    # 归一化
    stereo_audio = stereo_audio / np.max(np.abs(stereo_audio) + 1e-10) * 0.7
    
    # 简化的淡入淡出
    fade_samples = int(0.1 * sample_rate)  # 0.1秒淡入淡出
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        stereo_audio[:fade_samples] *= fade_in[:, np.newaxis]
        if len(stereo_audio) > fade_samples:
            stereo_audio[-fade_samples:] *= fade_out[:, np.newaxis]
    
    return stereo_audio.astype(np.float32), sample_rate, params

def save_audio_simple(audio_array, sample_rate, output_path):
    """简化的音频保存"""
    try:
        # 尝试使用scipy
        from scipy.io import wavfile
        audio_int = (audio_array * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int)
        return True
    except ImportError:
        # 保存为numpy数组
        np.save(output_path.replace('.wav', '.npy'), audio_array)
        return False

def detect_emotion_simple(user_input):
    """简化的情绪检测"""
    emotion_keywords = {
        "焦虑": ["焦虑", "紧张", "担心", "不安"],
        "疲惫": ["疲惫", "累", "疲劳", "困倦"],
        "烦躁": ["烦躁", "烦恼", "易怒", "急躁"],
        "平静": ["平静", "放松", "安静", "宁静"],
        "压力": ["压力", "紧迫", "负担", "重压"]
    }
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in user_input for keyword in keywords):
            return emotion, 0.9
    
    return "焦虑", 0.85  # 默认

def main():
    """主函数 - 轻量级版本"""
    print("🚀 启动轻量级增强三阶段疗愈系统...")
    print("⚡ 性能优化版本 - 快速响应，避免卡顿")
    print("🎯 三阶段：同步期(30%) → 引导期(40%) → 巩固期(30%)")
    print("=" * 50)
    
    # 快速测试场景
    test_scenarios = [
        "我感到很焦虑，心跳加速",
        "我很疲惫，无法放松",
        "我感到烦躁不安"
    ]
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # 快速测试模式
            user_input = test_scenarios[0]
            duration = 5
        elif sys.argv[1] == "--demo":
            # 多场景演示
            print("🎬 轻量级多场景演示...")
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            for i, scenario in enumerate(test_scenarios, 1):
                print(f"\n🎬 场景 {i}: {scenario}")
                
                start_time = time.time()
                emotion, confidence = detect_emotion_simple(scenario)
                print(f"🎯 检测情绪: {emotion} ({confidence:.0%})")
                
                # 生成音频
                audio_array, sample_rate, params = generate_lightweight_therapy_audio(
                    duration=8, emotion=emotion
                )
                
                # 保存音频
                audio_path = output_dir / f"lite_therapy_{emotion}_{i}.wav"
                success = save_audio_simple(audio_array, sample_rate, str(audio_path))
                
                processing_time = time.time() - start_time
                print(f"✅ 完成! 处理时间: {processing_time:.1f}秒")
                print(f"📁 保存至: {audio_path}")
                
                if i < len(test_scenarios):
                    print("⏳ 准备下一个场景...")
                    time.sleep(0.5)
            
            print(f"\n🎉 所有场景完成! 总计{len(test_scenarios)}个音频文件")
            return
        else:
            user_input = " ".join(sys.argv[1:])
            duration = 10
    else:
        # 默认场景
        user_input = "我感到很焦虑，心跳加速，难以入睡"
        duration = 10
    
    print(f"\n💭 用户输入: {user_input}")
    print(f"⏱️ 疗愈时长: {duration}秒")
    print("🔄 开始处理...")
    
    start_time = time.time()
    
    # 情绪识别
    emotion, confidence = detect_emotion_simple(user_input)
    print(f"🎯 检测情绪: {emotion} (置信度: {confidence:.0%})")
    
    # 生成音频
    print("🎵 生成轻量级疗愈音频...")
    audio_array, sample_rate, params = generate_lightweight_therapy_audio(
        duration=duration, emotion=emotion
    )
    
    # 保存音频
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%H%M%S")
    audio_path = output_dir / f"lite_therapy_{emotion}_{timestamp}.wav"
    
    print("💾 保存音频文件...")
    success = save_audio_simple(audio_array, sample_rate, str(audio_path))
    
    processing_time = time.time() - start_time
    
    # 输出结果
    print("\n" + "=" * 50)
    print("✅ 轻量级增强疗愈音频生成完成!")
    print(f"""
🧠 处理结果:
   情绪类型: {emotion}
   置信度: {confidence:.0%}
   处理时间: {processing_time:.1f}秒

🎵 音频信息:
   时长: {duration}秒
   采样率: {sample_rate}Hz
   声道: 立体声
   
🌊 三阶段频率:
   同步期: {params['sync_freq']}Hz
   引导期: {params['guide_freq']}Hz  
   巩固期: {params['consolidate_freq']}Hz

📁 输出文件:
   {audio_path}
   格式: {'WAV' if success else 'NumPy数组'}

🎧 使用建议:
   - 佩戴耳机聆听
   - 跟随音频节奏呼吸
   - 感受三阶段转换
""")
    
    print("🌙 轻量级疗愈体验完成!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)