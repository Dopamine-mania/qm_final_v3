#!/usr/bin/env python3
"""
🔍 调试版增强疗愈系统
找出原版本卡顿的原因
"""

import numpy as np
import sys
import os
import time
from pathlib import Path

def debug_step(step_name, func, *args, **kwargs):
    """调试步骤包装器"""
    print(f"🔍 开始: {step_name}")
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"✅ 完成: {step_name} - 耗时: {end_time - start_time:.2f}秒")
        return result
    except Exception as e:
        end_time = time.time()
        print(f"❌ 错误: {step_name} - 耗时: {end_time - start_time:.2f}秒")
        print(f"   错误信息: {e}")
        raise

def debug_generate_audio(duration=20, sample_rate=44100, emotion="焦虑"):
    """调试音频生成过程"""
    print(f"🎵 调试音频生成: {duration}秒, {sample_rate}Hz, {emotion}")
    
    # 步骤1: 参数设置
    def setup_params():
        emotion_params = {
            "焦虑": {
                "sync_freq": 440, "guide_freq": 330, "consolidate_freq": 220,
                "sync_intensity": 0.4, "guide_intensity": 0.25, "consolidate_intensity": 0.15,
                "transition_type": "exponential"
            }
        }
        return emotion_params.get(emotion, emotion_params["焦虑"])
    
    params = debug_step("参数设置", setup_params)
    
    # 步骤2: 创建音频数组
    def create_audio_array():
        total_samples = int(sample_rate * duration)
        print(f"   总样本数: {total_samples:,}")
        if total_samples > 2000000:  # 超过200万样本
            print("   ⚠️ 警告: 样本数过多，可能导致内存问题")
        audio_array = np.zeros(total_samples)
        return audio_array, total_samples
    
    audio_array, total_samples = debug_step("创建音频数组", create_audio_array)
    
    # 步骤3: 生成时间轴
    def create_timeline():
        t_total = np.linspace(0, duration, total_samples)
        return t_total
    
    t_total = debug_step("生成时间轴", create_timeline)
    
    # 步骤4: 第一阶段音频
    def generate_stage1():
        stage1_duration = duration * 0.3
        stage1_mask = t_total <= stage1_duration
        stage1_time = t_total[stage1_mask]
        print(f"   第一阶段样本数: {len(stage1_time):,}")
        
        stage1_audio = params['sync_intensity'] * np.sin(2 * np.pi * params['sync_freq'] * stage1_time)
        
        # 添加焦虑特征
        if emotion == "焦虑":
            tremolo = 0.1 * np.sin(2 * np.pi * 5 * stage1_time)
            stage1_audio *= (1 + tremolo)
        
        audio_array[stage1_mask] = stage1_audio
        return stage1_duration
    
    stage1_duration = debug_step("第一阶段音频", generate_stage1)
    
    # 步骤5: 第二阶段音频（最复杂）
    def generate_stage2():
        stage2_duration = duration * 0.4
        stage2_start = stage1_duration
        stage2_end = stage2_start + stage2_duration
        stage2_mask = (t_total > stage2_start) & (t_total <= stage2_end)
        stage2_time = t_total[stage2_mask] - stage2_start
        print(f"   第二阶段样本数: {len(stage2_time):,}")
        
        # 过渡计算
        transition_progress = stage2_time / stage2_duration
        
        # 选择过渡函数
        if params['transition_type'] == "exponential":
            transition_curve = 1 - np.exp(-3 * transition_progress)
        else:
            transition_curve = transition_progress
        
        # 动态频率变化
        current_freq = params['sync_freq'] + (params['guide_freq'] - params['sync_freq']) * transition_curve
        current_intensity = params['sync_intensity'] + (params['guide_intensity'] - params['sync_intensity']) * transition_curve
        
        stage2_audio = current_intensity * np.sin(2 * np.pi * current_freq * stage2_time)
        
        # 添加和谐泛音
        harmonic1 = 0.3 * current_intensity * np.sin(2 * np.pi * current_freq * 2 * stage2_time)
        harmonic2 = 0.2 * current_intensity * np.sin(2 * np.pi * current_freq * 3 * stage2_time)
        stage2_audio += harmonic1 + harmonic2
        
        audio_array[stage2_mask] = stage2_audio
        return stage2_end
    
    stage2_end = debug_step("第二阶段音频", generate_stage2)
    
    # 步骤6: 第三阶段音频
    def generate_stage3():
        stage3_mask = t_total > stage2_end
        stage3_time = t_total[stage3_mask] - stage2_end
        print(f"   第三阶段样本数: {len(stage3_time):,}")
        
        # 频率过渡
        consolidate_transition = np.exp(-stage3_time / 3)
        final_freq = params['guide_freq'] + (params['consolidate_freq'] - params['guide_freq']) * (1 - consolidate_transition)
        final_intensity = params['consolidate_intensity'] * np.exp(-stage3_time / 10)
        
        stage3_audio = final_intensity * np.sin(2 * np.pi * final_freq * stage3_time)
        
        # 添加自然音效
        nature_sound = 0.05 * np.random.normal(0, 1, len(stage3_time))
        wave_sound = 0.1 * final_intensity * np.sin(2 * np.pi * 0.3 * stage3_time)
        stage3_audio += nature_sound + wave_sound
        
        audio_array[stage3_mask] = stage3_audio
    
    debug_step("第三阶段音频", generate_stage3)
    
    # 步骤7: 立体声处理
    def create_stereo():
        left_channel = audio_array.copy()
        right_channel = audio_array.copy()
        
        # 立体声延迟
        stereo_delay = int(0.01 * sample_rate)
        print(f"   立体声延迟样本数: {stereo_delay}")
        
        if len(right_channel) > stereo_delay:
            right_channel[stereo_delay:] = audio_array[:-stereo_delay]
        
        # 混响（可能是性能瓶颈）
        print(f"   开始混响计算...")
        reverb_length = int(0.5 * sample_rate)
        print(f"   混响长度: {reverb_length}")
        
        # 这里可能是卡顿的原因
        reverb_impulse = np.exp(-np.linspace(0, 2, reverb_length))
        reverb = 0.1 * np.convolve(audio_array, reverb_impulse, mode='same')
        print(f"   混响计算完成")
        
        left_channel += reverb
        right_channel += reverb * 0.8
        
        stereo_audio = np.column_stack([left_channel, right_channel])
        
        # 归一化
        stereo_audio = stereo_audio / np.max(np.abs(stereo_audio)) * 0.8
        
        return stereo_audio
    
    stereo_audio = debug_step("立体声处理", create_stereo)
    
    # 步骤8: 淡入淡出
    def apply_fade():
        fade_samples = int(0.5 * sample_rate)
        print(f"   淡入淡出样本数: {fade_samples}")
        
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        stereo_audio[:fade_samples] *= fade_in[:, np.newaxis]
        stereo_audio[-fade_samples:] *= fade_out[:, np.newaxis]
        
        return stereo_audio.astype(np.float32)
    
    final_audio = debug_step("淡入淡出", apply_fade)
    
    return final_audio, sample_rate, params

def main():
    """主函数 - 调试版本"""
    print("🔍 启动调试版增强疗愈系统...")
    print("🎯 目标：找出性能瓶颈和卡顿原因")
    print("=" * 50)
    
    # 测试不同参数
    test_cases = [
        {"duration": 5, "sample_rate": 22050, "name": "轻量级测试"},
        {"duration": 10, "sample_rate": 44100, "name": "标准测试"},
        {"duration": 20, "sample_rate": 44100, "name": "完整测试"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}: {test_case['name']}")
        print(f"   参数: {test_case['duration']}秒, {test_case['sample_rate']}Hz")
        
        try:
            start_time = time.time()
            audio_array, sample_rate, params = debug_generate_audio(
                duration=test_case['duration'],
                sample_rate=test_case['sample_rate'],
                emotion="焦虑"
            )
            total_time = time.time() - start_time
            
            print(f"✅ 测试用例 {i} 完成")
            print(f"   总耗时: {total_time:.2f}秒")
            print(f"   音频形状: {audio_array.shape}")
            print(f"   内存占用: {audio_array.nbytes / 1024 / 1024:.1f}MB")
            
            # 保存调试音频
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            try:
                from scipy.io import wavfile
                audio_path = output_dir / f"debug_{test_case['name'].replace(' ', '_')}.wav"
                audio_int = (audio_array * 32767).astype(np.int16)
                wavfile.write(str(audio_path), sample_rate, audio_int)
                print(f"   保存至: {audio_path}")
            except ImportError:
                print("   scipy未安装，跳过保存")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户中断测试用例 {i}")
            break
        except Exception as e:
            print(f"❌ 测试用例 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🔍 调试完成!")
    print("💡 性能优化建议:")
    print("   1. 降低采样率 (44100 → 22050)")
    print("   2. 减少音频时长 (20s → 10s)")
    print("   3. 简化混响计算")
    print("   4. 优化和谐泛音生成")

if __name__ == "__main__":
    main()