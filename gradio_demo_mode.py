#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - 演示模式
不需要真实API，使用模拟数据展示完整功能
"""

import gradio as gr
import numpy as np
import cv2
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
import json

# 生成模拟音频数据
def generate_mock_audio(duration=15, sample_rate=44100):
    """生成模拟的三阶段音频数据"""
    print(f"🎵 生成模拟音频: {duration}秒, 采样率: {sample_rate}Hz")
    
    # 创建三阶段音频
    stage_duration = duration // 3
    t = np.linspace(0, stage_duration, int(sample_rate * stage_duration))
    
    # 第一阶段：同步期 - 较高频率，匹配用户情绪
    stage1_freq = 440  # A4
    stage1 = 0.3 * np.sin(2 * np.pi * stage1_freq * t) * np.exp(-t/5)
    
    # 第二阶段：引导期 - 逐渐降低频率
    stage2_freq = 330  # E4
    stage2 = 0.2 * np.sin(2 * np.pi * stage2_freq * t) * np.exp(-t/8)
    
    # 第三阶段：巩固期 - 低频放松音
    stage3_freq = 220  # A3
    stage3 = 0.1 * np.sin(2 * np.pi * stage3_freq * t) * np.exp(-t/12)
    
    # 合并三阶段
    audio_array = np.concatenate([stage1, stage2, stage3])
    
    # 添加白噪声和环境音
    noise = 0.05 * np.random.normal(0, 1, len(audio_array))
    audio_array = audio_array + noise
    
    # 归一化
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array.astype(np.float32), sample_rate

# 生成模拟视频帧
def generate_mock_video_frames(duration=15, fps=30):
    """生成模拟的疗愈视频帧"""
    print(f"🎬 生成模拟视频: {duration}秒, {fps}fps")
    
    frame_count = int(duration * fps)
    frames = []
    
    width, height = 640, 480
    
    for i in range(frame_count):
        # 创建渐变背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 时间进度
        progress = i / frame_count
        
        # 颜色变化：从蓝色到紫色到深蓝
        if progress < 0.33:  # 第一阶段：同步期
            blue = int(255 * (1 - progress * 3))
            green = int(100 * progress * 3)
            red = int(50 * progress * 3)
        elif progress < 0.66:  # 第二阶段：引导期
            stage_progress = (progress - 0.33) / 0.33
            blue = int(100 + 155 * (1 - stage_progress))
            green = int(100 * (1 - stage_progress))
            red = int(50 + 100 * stage_progress)
        else:  # 第三阶段：巩固期
            stage_progress = (progress - 0.66) / 0.34
            blue = int(50 + 50 * (1 - stage_progress))
            green = int(20 * (1 - stage_progress))
            red = int(20 + 30 * (1 - stage_progress))
        
        # 填充渐变背景
        for y in range(height):
            for x in range(width):
                # 径向渐变
                center_x, center_y = width // 2, height // 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                gradient = 1 - (distance / max_distance)
                
                frame[y, x] = [
                    int(blue * gradient),
                    int(green * gradient),
                    int(red * gradient)
                ]
        
        # 添加呼吸效果圆圈
        breathing_radius = 50 + 30 * np.sin(progress * 4 * np.pi)
        cv2.circle(frame, (width//2, height//2), int(breathing_radius), (255, 255, 255), 2)
        
        # 添加文字指示
        if progress < 0.33:
            stage_text = "同步期 - 匹配情绪"
        elif progress < 0.66:
            stage_text = "引导期 - 情绪转换"
        else:
            stage_text = "巩固期 - 深度放松"
        
        cv2.putText(frame, stage_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames, fps

# 创建音画同步视频
def create_demo_video(audio_array, sample_rate, video_frames, fps):
    """创建演示用的音画同步视频"""
    try:
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "demo_audio.wav")
        video_path = os.path.join(temp_dir, "demo_video.mp4")
        output_path = os.path.join(temp_dir, "demo_synchronized.mp4")
        
        # 保存音频
        import soundfile as sf
        sf.write(audio_path, audio_array, sample_rate)
        print(f"✅ 演示音频保存到: {audio_path}")
        
        # 创建视频
        if len(video_frames) > 0:
            frame_height, frame_width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            for frame in video_frames:
                out.write(frame)
            
            out.release()
            print(f"✅ 演示视频保存到: {video_path}")
            
            # 使用ffmpeg合成（如果可用）
            try:
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    '-shortest',
                    output_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ 演示音画同步视频生成成功: {output_path}")
                    return output_path
                else:
                    print(f"⚠️ ffmpeg不可用，返回纯视频文件")
                    return video_path
            except:
                print(f"⚠️ ffmpeg不可用，返回纯视频文件")
                return video_path
        else:
            print("❌ 没有视频帧")
            return None
            
    except Exception as e:
        print(f"❌ 创建演示视频时出错: {e}")
        return None

# 演示模式的情绪处理
def demo_emotion_processing(user_input):
    """演示模式的情绪处理"""
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, None, "输入太短"
    
    print(f"🔄 演示模式处理情绪输入: {user_input}")
    
    # 模拟情绪识别
    emotions = ["焦虑", "疲惫", "烦躁", "平静", "压力"]
    detected_emotion = "焦虑"  # 简单起见，固定识别为焦虑
    
    if "焦虑" in user_input or "紧张" in user_input:
        detected_emotion = "焦虑"
    elif "累" in user_input or "疲" in user_input:
        detected_emotion = "疲惫"
    elif "烦" in user_input or "躁" in user_input:
        detected_emotion = "烦躁"
    elif "平静" in user_input or "还好" in user_input:
        detected_emotion = "平静"
    elif "压力" in user_input or "忙" in user_input:
        detected_emotion = "压力"
    
    # 模拟置信度
    confidence = 0.85
    
    # 生成模拟音频和视频
    print("🎵 生成三阶段治疗音频...")
    audio_array, sample_rate = generate_mock_audio(duration=15)
    
    print("🎬 生成同步视觉内容...")
    video_frames, fps = generate_mock_video_frames(duration=15, fps=30)
    
    # 创建音画同步视频
    print("🎬 合成音画同步视频...")
    video_path = create_demo_video(audio_array, sample_rate, video_frames, fps)
    
    # 组织返回信息
    emotion_info = f"🧠 识别情绪: {detected_emotion}\n置信度: {confidence:.1%}\n情绪类型: 睡前{detected_emotion}状态"
    
    if video_path:
        info_text = f"""✅ 演示模式音画同步视频生成成功！

🎵 音频信息:
  - 时长: 15秒
  - 采样率: {sample_rate}Hz
  - 三阶段设计: 同步→引导→巩固

🎬 视频信息:
  - 帧数: {len(video_frames)}帧
  - 帧率: {fps}fps
  - 分辨率: 640x480

📝 注意: 这是演示模式，使用模拟数据
真实版本需要配置Suno API和其他商业API"""
        
        return emotion_info, video_path, (sample_rate, audio_array), info_text
    else:
        return emotion_info, None, (sample_rate, audio_array), "❌ 视频生成失败"

def create_demo_interface():
    """创建演示界面"""
    with gr.Blocks(title="🌙 睡眠疗愈AI - 演示模式") as app:
        gr.HTML("<h1 style='text-align: center;'>🌙 睡眠疗愈AI - 演示模式</h1>")
        gr.HTML("<p style='text-align: center; color: orange;'>⚠️ 演示模式：使用模拟数据，不需要真实API</p>")
        
        with gr.Row():
            with gr.Column():
                emotion_input = gr.Textbox(
                    label="💭 描述您的感受",
                    placeholder="例如：我感到很焦虑，心跳加速，难以入睡...",
                    lines=4
                )
                
                process_btn = gr.Button("🎬 演示：生成音画同步疗愈视频", variant="primary")
            
            with gr.Column():
                emotion_result = gr.Textbox(label="🧠 情绪识别结果", lines=3)
                
                # 音画同步视频输出
                video_output = gr.Video(label="🎬 演示：音画同步疗愈视频")
                
                # 音频输出
                audio_output = gr.Audio(label="🎵 演示：三阶段疗愈音频")
                
                info_output = gr.Textbox(label="📊 演示信息", lines=8)
        
        # 使用说明
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 10px;">
            <h3>🎯 演示模式说明</h3>
            <p><strong>🎵 音频生成：</strong> 模拟三阶段音乐叙事 (同步→引导→巩固)</p>
            <p><strong>🎬 视频生成：</strong> 模拟疗愈视觉内容，颜色渐变+呼吸效果</p>
            <p><strong>🧠 情绪识别：</strong> 基于关键词的简单情绪分类</p>
            <p><strong>⚠️ 注意：</strong> 真实版本需要配置Suno API等商业服务</p>
        </div>
        """)
        
        # 绑定事件
        process_btn.click(
            demo_emotion_processing,
            inputs=emotion_input,
            outputs=[emotion_result, video_output, audio_output, info_output]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动演示模式...")
    
    app = create_demo_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()