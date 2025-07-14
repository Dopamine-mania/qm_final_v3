#!/usr/bin/env python3
"""
🎵 直接播放Suno音乐
简单的音乐播放器，无需Web界面
"""

import os
import subprocess
import platform

def play_music():
    """播放Suno音乐"""
    audio_file = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final3/previous_suno_fdd1b90b.mp3"
    
    if not os.path.exists(audio_file):
        print("❌ 音乐文件不存在")
        return
    
    print("🎵 播放Suno AI生成的音乐: 'Whisper of the Moon'")
    print("📊 音乐信息:")
    print("   • 时长: 2分44秒")
    print("   • 模型: Chirp-v4")
    print("   • 风格: 宁静睡眠音乐")
    print("   • 特色: 指弹吉他 + 钢琴和弦 + 环境音")
    print("=" * 50)
    
    # 根据操作系统选择播放方式
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", audio_file])
            print("✅ 在系统默认播放器中打开音乐")
        elif system == "Windows":
            os.startfile(audio_file)
            print("✅ 在系统默认播放器中打开音乐")
        elif system == "Linux":
            subprocess.run(["xdg-open", audio_file])
            print("✅ 在系统默认播放器中打开音乐")
        else:
            print(f"⚠️ 未知操作系统: {system}")
            print(f"📂 请手动播放: {audio_file}")
            
    except Exception as e:
        print(f"❌ 播放失败: {e}")
        print(f"📂 请手动播放: {audio_file}")

if __name__ == "__main__":
    play_music()