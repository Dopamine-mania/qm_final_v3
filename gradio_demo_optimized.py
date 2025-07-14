#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - 优化演示模式
支持用户选择：纯音乐模式 vs 音画结合模式
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

def generate_therapy_audio(duration=15, sample_rate=44100, emotion="焦虑"):
    """生成疗愈音频（快速版本）"""
    print(f"🎵 生成{duration}秒疗愈音频 (针对{emotion}情绪)")
    
    # 根据情绪调整频率参数
    emotion_params = {
        "焦虑": {"sync": 440, "guide": 330, "consolidate": 220},
        "疲惫": {"sync": 380, "guide": 280, "consolidate": 200},
        "烦躁": {"sync": 460, "guide": 350, "consolidate": 240},
        "平静": {"sync": 400, "guide": 320, "consolidate": 210},
        "压力": {"sync": 480, "guide": 360, "consolidate": 230}
    }
    
    params = emotion_params.get(emotion, emotion_params["焦虑"])
    
    # 三阶段音频生成
    stage_duration = duration // 3
    t = np.linspace(0, stage_duration, int(sample_rate * stage_duration))
    
    # 第一阶段：同步期
    print(f"🎵 第一阶段-同步期: {params['sync']}Hz")
    stage1 = 0.3 * np.sin(2 * np.pi * params['sync'] * t) * np.exp(-t/5)
    
    # 第二阶段：引导期
    print(f"🎵 第二阶段-引导期: {params['guide']}Hz")
    stage2 = 0.2 * np.sin(2 * np.pi * params['guide'] * t) * np.exp(-t/8)
    
    # 第三阶段：巩固期
    print(f"🎵 第三阶段-巩固期: {params['consolidate']}Hz")
    stage3 = 0.1 * np.sin(2 * np.pi * params['consolidate'] * t) * np.exp(-t/12)
    
    # 合并三阶段
    audio_array = np.concatenate([stage1, stage2, stage3])
    
    # 添加白噪声和自然音效
    print("🎵 添加自然音效...")
    noise = 0.03 * np.random.normal(0, 1, len(audio_array))
    
    # 添加轻微的双声道效果
    if len(audio_array.shape) == 1:
        # 创建立体声
        left_channel = audio_array + noise
        right_channel = audio_array + 0.05 * np.sin(2 * np.pi * 100 * np.linspace(0, duration, len(audio_array)))
        audio_array = np.column_stack([left_channel, right_channel])
    
    # 归一化
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array.astype(np.float32), sample_rate, params

def generate_simple_visual(duration=15, fps=30):
    """生成简化视觉内容（快速版本）"""
    print(f"🎬 生成简化视觉内容 ({duration}秒, {fps}fps)")
    
    # 降低帧率和分辨率以提高速度
    actual_fps = 15  # 降低帧率
    frame_count = int(duration * actual_fps)
    frames = []
    width, height = 480, 320  # 降低分辨率
    
    for i in range(frame_count):
        if i % 15 == 0:  # 每秒更新一次
            current_second = i // actual_fps
            print(f"🎬 渲染: 第{current_second+1}秒")
        
        # 创建简单的渐变背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 时间进度
        progress = i / frame_count
        
        # 简化的颜色变化
        if progress < 0.33:
            color = [int(255 * (1 - progress * 3)), int(100 * progress * 3), 50]
        elif progress < 0.66:
            stage_progress = (progress - 0.33) / 0.33
            color = [int(100 * (1 - stage_progress)), int(100 * (1 - stage_progress)), int(150 * stage_progress)]
        else:
            color = [30, 30, int(100 * (1 - (progress - 0.66) / 0.34))]
        
        # 填充背景
        frame[:] = color
        
        # 简单的中心圆圈
        radius = int(50 + 20 * np.sin(progress * 4 * np.pi))
        cv2.circle(frame, (width//2, height//2), radius, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames, actual_fps

def create_audio_video_sync(audio_array, sample_rate, video_frames, fps):
    """快速音画同步合成"""
    try:
        print("🎬 开始快速音画同步合成...")
        
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_path = os.path.join(temp_dir, "video.mp4")
        output_path = os.path.join(temp_dir, "synchronized.mp4")
        
        # 保存音频
        import soundfile as sf
        sf.write(audio_path, audio_array, sample_rate)
        
        # 创建视频
        if len(video_frames) > 0:
            frame_height, frame_width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            for frame in video_frames:
                out.write(frame)
            
            out.release()
            
            # 快速ffmpeg合成
            try:
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-loglevel', 'quiet',
                    '-i', video_path, '-i', audio_path,
                    '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-c:a', 'aac', '-shortest', output_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True)
                if result.returncode == 0:
                    print(f"✅ 快速音画同步完成")
                    return output_path
                else:
                    return video_path
            except:
                return video_path
        else:
            return None
            
    except Exception as e:
        print(f"❌ 音画同步失败: {e}")
        return None

def process_emotion_optimized(user_input, output_mode, duration):
    """优化的情绪处理函数"""
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, None, "输入太短"
    
    try:
        start_time = time.time()
        
        # 情绪识别
        print("🧠 步骤1/3: 情绪识别分析")
        emotions = {
            "焦虑": {"confidence": 0.85, "type": "睡前焦虑状态", "color": "蓝紫色"},
            "疲惫": {"confidence": 0.82, "type": "身体疲惫状态", "color": "暖橙色"},
            "烦躁": {"confidence": 0.88, "type": "情绪烦躁状态", "color": "红紫色"},
            "平静": {"confidence": 0.75, "type": "相对平静状态", "color": "绿蓝色"},
            "压力": {"confidence": 0.90, "type": "心理压力状态", "color": "深蓝色"}
        }
        
        detected_emotion = "焦虑"
        for emotion_key in emotions.keys():
            if emotion_key in user_input:
                detected_emotion = emotion_key
                break
        
        emotion_info = emotions[detected_emotion]
        
        # 生成音频
        print("🎵 步骤2/3: 生成三阶段疗愈音频")
        audio_array, sample_rate, audio_params = generate_therapy_audio(
            duration=duration, 
            emotion=detected_emotion
        )
        
        # 根据输出模式决定是否生成视频
        video_output = None
        if output_mode == "音画结合":
            print("🎬 步骤3/3: 生成音画同步视频")
            video_frames, fps = generate_simple_visual(duration=duration)
            video_output = create_audio_video_sync(audio_array, sample_rate, video_frames, fps)
        else:
            print("🎵 步骤3/3: 纯音乐模式 - 跳过视频生成")
        
        processing_time = time.time() - start_time
        
        # 组织返回信息
        emotion_result = f"""🧠 情绪识别结果:
情绪类型: {detected_emotion}
置信度: {emotion_info['confidence']:.1%}
状态描述: {emotion_info['type']}
视觉主题: {emotion_info['color']}调
处理时间: {processing_time:.1f}秒"""
        
        # 音频信息
        audio_info = f"""✅ 疗愈音频生成完成！

🎵 音频详情:
  - 时长: {duration}秒
  - 采样率: {sample_rate}Hz
  - 声道: 立体声
  - 针对情绪: {detected_emotion}

🎼 三阶段频率设计:
  - 同步期: {audio_params['sync']}Hz (匹配{detected_emotion}情绪)
  - 引导期: {audio_params['guide']}Hz (逐步引导转换)
  - 巩固期: {audio_params['consolidate']}Hz (深度放松状态)

⚡ 性能信息:
  - 输出模式: {output_mode}
  - 处理时间: {processing_time:.1f}秒
  - 系统状态: 正常
  
🎧 使用建议:
  - 佩戴耳机获得最佳效果
  - 在安静环境中聆听
  - 跟随音乐节奏调整呼吸
  - 让音乐引导您进入放松状态"""
        
        if output_mode == "音画结合" and video_output:
            audio_info += f"""

🎬 视频信息:
  - 视频文件: 已生成音画同步视频
  - 分辨率: 480x320 (优化性能)
  - 帧率: 15fps (优化性能)
  - 视觉效果: {emotion_info['color']}调渐变 + 呼吸引导"""
        
        audio_info += f"""

📝 技术说明:
  - 演示模式：使用数学合成音频
  - 真实版本：将使用Suno AI等商业API
  - 音频质量：CD级别 (44.1kHz/16-bit)
  - 疗愈原理：基于ISO音乐治疗原则"""
        
        return emotion_result, video_output, (sample_rate, audio_array), audio_info
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理错误: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None, None, f"错误详情: {traceback.format_exc()}"

def create_optimized_interface():
    """创建优化的用户界面"""
    with gr.Blocks(
        title="🌙 睡眠疗愈AI - 优化演示",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1>🌙 睡眠疗愈AI - 优化演示</h1>
            <p>个性化三阶段音乐疗愈系统</p>
            <p style="color: #ffeb3b;">✨ 支持纯音乐模式和音画结合模式</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💭 情绪输入")
                
                # 情绪选择
                emotion_examples = gr.Dropdown(
                    choices=[
                        "😰 我感到很焦虑，心跳加速，难以入睡",
                        "😴 我很疲惫，但大脑还在活跃，无法放松",
                        "😤 我感到烦躁不安，容易被小事影响",
                        "😌 我比较平静，但希望更深层的放松",
                        "🤯 最近压力很大，总是感到紧张"
                    ],
                    label="🎭 快速选择情绪",
                    value="😰 我感到很焦虑，心跳加速，难以入睡"
                )
                
                emotion_input = gr.Textbox(
                    label="✍️ 或详细描述您的感受",
                    placeholder="描述您的情绪状态...",
                    lines=3,
                    value="我感到很焦虑，心跳加速，难以入睡"
                )
                
                gr.Markdown("### ⚙️ 输出设置")
                
                # 输出模式选择
                output_mode = gr.Radio(
                    choices=["纯音乐", "音画结合"],
                    value="纯音乐",
                    label="🎬 输出模式"
                )
                
                # 时长选择
                duration = gr.Slider(
                    minimum=10, maximum=60, value=15, step=5,
                    label="⏱️ 音频时长（秒）"
                )
                
                process_btn = gr.Button("🎵 开始生成疗愈内容", variant="primary", size="lg")
                
                # 模式说明
                gr.HTML("""
                <div style="margin-top: 15px; padding: 10px; background-color: #f0f8ff; border-radius: 8px;">
                    <strong>📊 模式对比：</strong><br>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                        <div>
                            <strong>🎵 纯音乐模式</strong><br>
                            • 处理时间: 3-5秒<br>
                            • 专注音乐疗愈<br>
                            • 适合纯音乐爱好者
                        </div>
                        <div>
                            <strong>🎬 音画结合模式</strong><br>
                            • 处理时间: 15-20秒<br>
                            • 视听双重疗愈<br>
                            • 适合多媒体体验
                        </div>
                    </div>
                </div>
                """)
            
            with gr.Column(scale=3):
                gr.Markdown("### 🎬 生成结果")
                
                # 情绪识别结果
                emotion_result = gr.Textbox(
                    label="🧠 情绪识别结果",
                    lines=6,
                    interactive=False
                )
                
                # 视频输出（条件显示）
                video_output = gr.Video(
                    label="🎬 音画同步疗愈视频",
                    height=300,
                    visible=True
                )
                
                # 音频输出（主要输出）
                audio_output = gr.Audio(
                    label="🎵 三阶段疗愈音频",
                    type="numpy"
                )
                
                # 详细信息
                info_output = gr.Textbox(
                    label="📊 详细信息",
                    lines=18,
                    interactive=False
                )
        
        # 使用指南
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px; border: 1px solid #e9ecef;">
            <h3>🎯 使用指南</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div>
                    <h4>🎵 纯音乐模式</h4>
                    <ul>
                        <li>快速生成（3-5秒）</li>
                        <li>专注于音乐疗愈效果</li>
                        <li>适合日常使用</li>
                        <li>节省处理时间</li>
                    </ul>
                </div>
                <div>
                    <h4>🎬 音画结合模式</h4>
                    <ul>
                        <li>完整体验（15-20秒）</li>
                        <li>视听双重疗愈</li>
                        <li>适合演示展示</li>
                        <li>沉浸式体验</li>
                    </ul>
                </div>
                <div>
                    <h4>🎧 最佳实践</h4>
                    <ul>
                        <li>佩戴耳机聆听</li>
                        <li>选择安静环境</li>
                        <li>跟随音乐调整呼吸</li>
                        <li>让音乐引导放松</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
        # 事件绑定
        def update_input_from_dropdown(selected):
            return selected.split(" ", 1)[1] if " " in selected else selected
        
        emotion_examples.change(
            update_input_from_dropdown,
            inputs=emotion_examples,
            outputs=emotion_input
        )
        
        process_btn.click(
            process_emotion_optimized,
            inputs=[emotion_input, output_mode, duration],
            outputs=[emotion_result, video_output, audio_output, info_output]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动优化演示模式...")
    print("🎵 支持纯音乐模式（快速）和音画结合模式（完整）")
    print("⚡ 纯音乐模式：3-5秒完成")
    print("🎬 音画结合模式：15-20秒完成")
    
    app = create_optimized_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()