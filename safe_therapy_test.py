#!/usr/bin/env python3
"""
🛡️ 超级安全测试版本
确保不会卡死，可以随时退出
"""

import numpy as np
import sys
import time
import signal
from pathlib import Path

# 设置信号处理器，确保可以安全退出
def signal_handler(signum, frame):
    print('\n🛑 收到退出信号，安全退出...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def ultra_safe_audio_gen(duration=3, sample_rate=16000):
    """超级安全的音频生成，绝对不会卡死"""
    print(f"🔒 安全生成{duration}秒音频")
    
    # 极限安全参数
    duration = min(duration, 5)  # 最大5秒
    sample_rate = min(sample_rate, 22050)  # 最大22kHz
    
    total_samples = int(sample_rate * duration)
    print(f"   样本数: {total_samples:,}")
    
    if total_samples > 150000:  # 超过15万样本就警告
        print("   ⚠️ 样本数较大，降低到安全范围")
        duration = 3
        sample_rate = 16000
        total_samples = int(sample_rate * duration)
    
    # 检查内存
    estimated_memory = total_samples * 8 / 1024 / 1024  # MB
    print(f"   预计内存: {estimated_memory:.1f}MB")
    
    if estimated_memory > 5:  # 超过5MB就减少
        print("   ⚠️ 内存占用过大，进一步优化")
        duration = 2
        sample_rate = 16000
        total_samples = int(sample_rate * duration)
    
    # 分段生成，避免一次性计算过多
    print("   🔄 分段生成音频...")
    
    # 三阶段时间分配
    stage1_len = int(total_samples * 0.3)
    stage2_len = int(total_samples * 0.4)
    stage3_len = total_samples - stage1_len - stage2_len
    
    # 第一阶段：简单正弦波
    print("   🎵 第一阶段...")
    t1 = np.linspace(0, duration * 0.3, stage1_len)
    stage1 = 0.3 * np.sin(2 * np.pi * 440 * t1)
    
    # 检查是否被中断
    if len(stage1) == 0:
        print("   ⚠️ 第一阶段生成异常")
        return None
    
    # 第二阶段：线性频率变化
    print("   🎵 第二阶段...")
    t2 = np.linspace(0, duration * 0.4, stage2_len)
    freq_change = 440 + (330 - 440) * (t2 / (duration * 0.4))
    stage2 = 0.25 * np.sin(2 * np.pi * freq_change * t2)
    
    # 第三阶段：低频放松
    print("   🎵 第三阶段...")
    t3 = np.linspace(0, duration * 0.3, stage3_len)
    stage3 = 0.15 * np.sin(2 * np.pi * 220 * t3) * np.exp(-t3)
    
    # 合并（最安全的方式）
    print("   🔗 合并音频...")
    try:
        audio_mono = np.concatenate([stage1, stage2, stage3])
    except Exception as e:
        print(f"   ❌ 合并失败: {e}")
        return None
    
    # 简单立体声（不做复杂处理）
    print("   🎧 创建立体声...")
    audio_stereo = np.column_stack([audio_mono, audio_mono])
    
    # 简单归一化
    max_val = np.max(np.abs(audio_stereo))
    if max_val > 0:
        audio_stereo = audio_stereo / max_val * 0.7
    
    return audio_stereo.astype(np.float32), sample_rate

def safe_save_audio(audio, sample_rate, filename):
    """安全保存音频"""
    try:
        from scipy.io import wavfile
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int)
        return True
    except:
        np.save(filename.replace('.wav', '.npy'), audio)
        return False

def main():
    """主函数 - 超级安全版本"""
    print("🛡️ 超级安全测试版本启动")
    print("🔒 确保不会卡死，随时可以Ctrl+C退出")
    print("=" * 40)
    
    # 快速测试用例
    test_cases = [
        {"duration": 2, "sample_rate": 16000, "name": "迷你测试"},
        {"duration": 3, "sample_rate": 22050, "name": "小型测试"},
        {"duration": 5, "sample_rate": 22050, "name": "标准测试"}
    ]
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 测试 {i}: {test['name']}")
        print(f"   参数: {test['duration']}秒, {test['sample_rate']}Hz")
        
        # 添加超时保护
        start_time = time.time()
        timeout = 10  # 10秒超时
        
        try:
            # 生成音频
            result = ultra_safe_audio_gen(test['duration'], test['sample_rate'])
            
            if result is None:
                print(f"   ❌ 测试 {i} 失败")
                continue
            
            audio, sr = result
            generation_time = time.time() - start_time
            
            if generation_time > timeout:
                print(f"   ⏰ 测试 {i} 超时 ({generation_time:.1f}秒)")
                continue
            
            # 保存音频
            filename = output_dir / f"safe_test_{i}.wav"
            success = safe_save_audio(audio, sr, str(filename))
            
            print(f"   ✅ 测试 {i} 完成!")
            print(f"   ⏱️ 用时: {generation_time:.2f}秒")
            print(f"   📁 保存: {filename}")
            print(f"   🎵 格式: {'WAV' if success else 'NPY'}")
            print(f"   📊 大小: {audio.shape}")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户中断测试 {i}")
            break
        except Exception as e:
            print(f"   ❌ 测试 {i} 异常: {e}")
            continue
    
    print("\n🎉 安全测试完成!")
    print("💡 如果这个版本都正常，说明原版本确实有性能问题")
    print("🔧 建议使用 enhanced_therapy_lite.py 作为替代方案")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序被用户中断，安全退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        sys.exit(1)