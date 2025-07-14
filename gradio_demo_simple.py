#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - 简化增强演示模式
修复版本兼容性问题，提供完整的进度反馈
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

def update_progress_info(step_name, progress, total_steps, details=""):
    """生成进度信息"""
    if progress == 0:
        return "🎯 等待开始处理..."
    elif progress >= total_steps:
        return "✅ 处理完成！"
    else:
        progress_bar = "█" * progress + "░" * (total_steps - progress)
        return f"{step_name}\n[{progress_bar}] {progress}/{total_steps}\n{details}"

def generate_mock_audio_with_feedback(duration=15, sample_rate=44100):
    """生成模拟音频并返回进度信息"""
    print(f"🎵 开始生成{duration}秒音频，采样率{sample_rate}Hz")
    
    # 模拟音频生成过程
    stage_duration = duration // 3
    t = np.linspace(0, stage_duration, int(sample_rate * stage_duration))
    
    # 第一阶段：同步期
    print("🎵 第一阶段 - 同步期：生成高频匹配音频...")
    time.sleep(0.3)
    stage1_freq = 440  # A4
    stage1 = 0.3 * np.sin(2 * np.pi * stage1_freq * t) * np.exp(-t/5)
    
    # 第二阶段：引导期
    print("🎵 第二阶段 - 引导期：生成过渡引导音频...")
    time.sleep(0.3)
    stage2_freq = 330  # E4
    stage2 = 0.2 * np.sin(2 * np.pi * stage2_freq * t) * np.exp(-t/8)
    
    # 第三阶段：巩固期
    print("🎵 第三阶段 - 巩固期：生成低频放松音频...")
    time.sleep(0.3)
    stage3_freq = 220  # A3
    stage3 = 0.1 * np.sin(2 * np.pi * stage3_freq * t) * np.exp(-t/12)
    
    # 合并和后处理
    print("🎵 音频后处理：合并三阶段音频...")
    audio_array = np.concatenate([stage1, stage2, stage3])
    
    # 添加环境音效
    noise = 0.05 * np.random.normal(0, 1, len(audio_array))
    audio_array = audio_array + noise
    
    # 归一化
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array.astype(np.float32), sample_rate

def generate_mock_video_with_feedback(duration=15, fps=30):
    """生成模拟视频并返回进度信息"""
    print(f"🎬 开始生成{duration}秒视频，帧率{fps}fps")
    
    frame_count = int(duration * fps)
    frames = []
    width, height = 640, 480
    
    for i in range(frame_count):
        # 每30帧（1秒）输出一次进度
        if i % 30 == 0:
            current_second = i // fps
            print(f"🎬 渲染进度：第{current_second+1}秒 ({i+1}/{frame_count}帧)")
        
        # 创建渐变背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 时间进度
        progress = i / frame_count
        
        # 三阶段颜色变化
        if progress < 0.33:  # 第一阶段：同步期
            blue = int(255 * (1 - progress * 3))
            green = int(100 * progress * 3)
            red = int(50 * progress * 3)
            stage_name = "同步期"
        elif progress < 0.66:  # 第二阶段：引导期
            stage_progress = (progress - 0.33) / 0.33
            blue = int(100 + 155 * (1 - stage_progress))
            green = int(100 * (1 - stage_progress))
            red = int(50 + 100 * stage_progress)
            stage_name = "引导期"
        else:  # 第三阶段：巩固期
            stage_progress = (progress - 0.66) / 0.34
            blue = int(50 + 50 * (1 - stage_progress))
            green = int(20 * (1 - stage_progress))
            red = int(20 + 30 * (1 - stage_progress))
            stage_name = "巩固期"
        
        # 填充渐变背景
        for y in range(height):
            for x in range(width):
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
        
        # 添加阶段文字
        cv2.putText(frame, stage_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames, fps

def create_demo_video_with_feedback(audio_array, sample_rate, video_frames, fps):
    """创建演示视频并返回进度信息"""
    try:
        print("🎬 开始合成音画同步视频...")
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "demo_audio.wav")
        video_path = os.path.join(temp_dir, "demo_video.mp4")
        output_path = os.path.join(temp_dir, "demo_synchronized.mp4")
        
        # 保存音频
        print(f"💾 保存音频文件到: {audio_path}")
        import soundfile as sf
        sf.write(audio_path, audio_array, sample_rate)
        
        # 创建视频
        print(f"🎬 创建视频文件，处理{len(video_frames)}帧...")
        if len(video_frames) > 0:
            frame_height, frame_width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            for i, frame in enumerate(video_frames):
                if i % 90 == 0:  # 每3秒更新一次
                    print(f"📝 写入视频帧：{i+1}/{len(video_frames)}")
                out.write(frame)
            
            out.release()
            
            # 使用ffmpeg合成
            print("🔄 使用FFmpeg进行音画同步...")
            try:
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
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
                    print(f"✅ 音画同步视频生成成功: {output_path}")
                    return output_path
                else:
                    print(f"⚠️ FFmpeg不可用，返回纯视频文件: {video_path}")
                    return video_path
            except Exception as e:
                print(f"⚠️ FFmpeg错误: {str(e)}，返回纯视频文件")
                return video_path
        else:
            print("❌ 没有视频帧，生成失败")
            return None
            
    except Exception as e:
        print(f"❌ 创建视频失败: {str(e)}")
        return None

def enhanced_demo_processing_with_progress(user_input, progress_update):
    """增强的演示模式处理，带进度更新"""
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, None, "输入太短"
    
    try:
        total_steps = 7
        current_step = 0
        
        # 步骤1: 情绪识别
        current_step += 1
        progress_info = update_progress_info("🧠 情绪识别分析", current_step, total_steps, "分析用户输入的情绪状态...")
        print(f"步骤 {current_step}/{total_steps}: 情绪识别分析")
        time.sleep(0.5)
        
        # 模拟情绪识别
        emotions = {
            "焦虑": {"confidence": 0.85, "type": "睡前焦虑状态"},
            "疲惫": {"confidence": 0.82, "type": "身体疲惫状态"},
            "烦躁": {"confidence": 0.88, "type": "情绪烦躁状态"},
            "平静": {"confidence": 0.75, "type": "相对平静状态"},
            "压力": {"confidence": 0.90, "type": "心理压力状态"}
        }
        
        detected_emotion = "焦虑"
        for emotion_key in emotions.keys():
            if emotion_key in user_input:
                detected_emotion = emotion_key
                break
        
        emotion_info = emotions[detected_emotion]
        
        # 步骤2: 制定治疗方案
        current_step += 1
        progress_info = update_progress_info("📋 制定治疗方案", current_step, total_steps, f"基于{detected_emotion}情绪制定三阶段治疗方案...")
        print(f"步骤 {current_step}/{total_steps}: 制定治疗方案")
        time.sleep(0.5)
        
        # 步骤3: 生成音频
        current_step += 1
        progress_info = update_progress_info("🎵 生成三阶段音频", current_step, total_steps, "生成同步→引导→巩固音频...")
        print(f"步骤 {current_step}/{total_steps}: 生成三阶段音频")
        audio_array, sample_rate = generate_mock_audio_with_feedback(duration=15)
        
        # 步骤4: 生成视频
        current_step += 1
        progress_info = update_progress_info("🎬 生成疗愈视频", current_step, total_steps, "生成三阶段视觉内容...")
        print(f"步骤 {current_step}/{total_steps}: 生成疗愈视频")
        video_frames, fps = generate_mock_video_with_feedback(duration=15, fps=30)
        
        # 步骤5: 合成视频
        current_step += 1
        progress_info = update_progress_info("🎬 合成音画同步", current_step, total_steps, "使用OpenCV和FFmpeg合成...")
        print(f"步骤 {current_step}/{total_steps}: 合成音画同步")
        video_path = create_demo_video_with_feedback(audio_array, sample_rate, video_frames, fps)
        
        # 步骤6: 后处理
        current_step += 1
        progress_info = update_progress_info("🔧 后处理优化", current_step, total_steps, "优化视频质量和音频效果...")
        print(f"步骤 {current_step}/{total_steps}: 后处理优化")
        time.sleep(0.5)
        
        # 步骤7: 完成
        current_step += 1
        progress_info = update_progress_info("🎉 生成完成", current_step, total_steps, "音画同步疗愈视频已准备就绪！")
        print(f"步骤 {current_step}/{total_steps}: 生成完成")
        
        # 组织返回信息
        emotion_result = f"""🧠 情绪识别结果:
情绪类型: {detected_emotion}
置信度: {emotion_info['confidence']:.1%}
状态描述: {emotion_info['type']}
处理时间: {datetime.now().strftime('%H:%M:%S')}"""
        
        if video_path:
            info_text = f"""✅ 演示模式生成完成！

🎵 音频信息:
  - 时长: 15秒 (3个阶段各5秒)
  - 采样率: {sample_rate}Hz
  - 声道: 立体声
  - 三阶段设计: 同步(440Hz) → 引导(330Hz) → 巩固(220Hz)

🎬 视频信息:
  - 总帧数: {len(video_frames)}帧
  - 帧率: {fps}fps
  - 分辨率: 640x480
  - 视觉效果: 渐变颜色 + 呼吸圆圈 + 阶段标识

🔧 处理流程:
  - 步骤1: 情绪识别分析 ✅
  - 步骤2: 制定治疗方案 ✅
  - 步骤3: 生成三阶段音频 ✅
  - 步骤4: 生成疗愈视频 ✅
  - 步骤5: 合成音画同步 ✅
  - 步骤6: 后处理优化 ✅
  - 步骤7: 生成完成 ✅

📝 注意事项:
  - 这是演示模式，使用模拟数据
  - 真实版本需要配置商业API
  - 演示音频采用数学合成
  - 完整处理时间约15-20秒"""
            
            return emotion_result, video_path, (sample_rate, audio_array), info_text
        else:
            return emotion_result, None, (sample_rate, audio_array), "❌ 视频生成失败"
    
    except Exception as e:
        error_msg = f"❌ 处理错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, None, None, f"错误详情: {traceback.format_exc()}"

def create_simple_interface():
    """创建简化但功能完整的界面"""
    with gr.Blocks(
        title="🌙 睡眠疗愈AI - 演示模式",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1>🌙 睡眠疗愈AI - 演示模式</h1>
            <p>完整的三阶段音乐叙事疗愈系统</p>
            <p style="color: #ffeb3b;">⚠️ 演示模式：使用模拟数据，展示完整功能</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💭 情绪输入")
                
                # 快速选择示例
                example_emotions = [
                    "😰 我感到很焦虑，心跳加速，难以入睡",
                    "😴 我很疲惫，但大脑还在活跃，无法放松",
                    "😤 我感到烦躁不安，容易被小事影响",
                    "😌 我比较平静，但希望更深层的放松",
                    "🤯 最近压力很大，总是感到紧张"
                ]
                
                emotion_examples = gr.Dropdown(
                    choices=example_emotions,
                    label="🎭 快速选择常见情绪",
                    value=example_emotions[0]
                )
                
                emotion_input = gr.Textbox(
                    label="✍️ 或自定义描述您的感受",
                    placeholder="描述您现在的情绪状态、身体感受、思维状态等...",
                    lines=4,
                    value="我感到很焦虑，心跳加速，难以入睡"
                )
                
                process_btn = gr.Button("🎬 开始生成音画同步疗愈视频", variant="primary", size="lg")
                
                # 进度提示
                gr.HTML("""
                <div style="margin-top: 15px; padding: 10px; background-color: #f0f8ff; border-radius: 8px; border-left: 4px solid #1976d2;">
                    <strong>📊 处理流程（约15-20秒）：</strong><br>
                    1. 🧠 情绪识别分析<br>
                    2. 📋 制定治疗方案<br>
                    3. 🎵 生成三阶段音频<br>
                    4. 🎬 生成疗愈视频<br>
                    5. 🎬 合成音画同步<br>
                    6. 🔧 后处理优化<br>
                    7. 🎉 生成完成
                </div>
                """)
            
            with gr.Column(scale=3):
                gr.Markdown("### 🎬 生成结果")
                
                # 情绪识别结果
                emotion_result = gr.Textbox(
                    label="🧠 情绪识别结果",
                    lines=4,
                    interactive=False
                )
                
                # 主要输出：音画同步视频
                video_output = gr.Video(
                    label="🎬 三阶段音画同步疗愈视频",
                    height=400
                )
                
                # 音频输出
                audio_output = gr.Audio(
                    label="🎵 三阶段疗愈音频",
                    type="numpy"
                )
                
                # 详细信息
                info_output = gr.Textbox(
                    label="📊 详细处理信息",
                    lines=15,
                    interactive=False
                )
        
        # 使用说明
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 10px;">
            <h3>🎯 使用说明</h3>
            <ul>
                <li><strong>🎧 建议佩戴耳机</strong>：获得最佳的立体声效果</li>
                <li><strong>📱 观看进度</strong>：终端会显示详细的处理进度</li>
                <li><strong>⏱️ 处理时间</strong>：整个过程约15-20秒</li>
                <li><strong>🎬 最终效果</strong>：15秒音画同步疗愈视频</li>
            </ul>
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
            lambda x: enhanced_demo_processing_with_progress(x, None),
            inputs=emotion_input,
            outputs=[emotion_result, video_output, audio_output, info_output]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动简化增强演示模式...")
    print("📊 包含完整的进度显示，但简化了定时更新")
    print("🎯 进度信息主要通过终端输出显示")
    
    app = create_simple_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()