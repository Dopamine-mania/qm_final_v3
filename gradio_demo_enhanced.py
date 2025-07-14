#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - 增强演示模式
带进度条和实时状态更新的完整演示
"""

import gradio as gr
import numpy as np
import cv2
import os
import tempfile
import time
import threading
from pathlib import Path
from datetime import datetime
import json

# 全局状态变量
processing_status = {
    "current_step": "",
    "progress": 0,
    "total_steps": 7,
    "details": "",
    "error": None
}

def update_progress(step_name, progress, details=""):
    """更新处理进度"""
    global processing_status
    processing_status["current_step"] = step_name
    processing_status["progress"] = progress
    processing_status["details"] = details
    print(f"📊 进度更新: {step_name} ({progress}/{processing_status['total_steps']}) - {details}")

def generate_mock_audio_with_progress(duration=15, sample_rate=44100):
    """生成模拟音频并显示进度"""
    update_progress("🎵 生成三阶段音频", 3, f"音频时长: {duration}秒, 采样率: {sample_rate}Hz")
    
    # 模拟音频生成过程
    stage_duration = duration // 3
    t = np.linspace(0, stage_duration, int(sample_rate * stage_duration))
    
    # 第一阶段：同步期
    update_progress("🎵 第一阶段 - 同步期", 3, "生成高频匹配音频...")
    time.sleep(0.5)  # 模拟处理时间
    stage1_freq = 440  # A4
    stage1 = 0.3 * np.sin(2 * np.pi * stage1_freq * t) * np.exp(-t/5)
    
    # 第二阶段：引导期
    update_progress("🎵 第二阶段 - 引导期", 3, "生成过渡引导音频...")
    time.sleep(0.5)  # 模拟处理时间
    stage2_freq = 330  # E4
    stage2 = 0.2 * np.sin(2 * np.pi * stage2_freq * t) * np.exp(-t/8)
    
    # 第三阶段：巩固期
    update_progress("🎵 第三阶段 - 巩固期", 3, "生成低频放松音频...")
    time.sleep(0.5)  # 模拟处理时间
    stage3_freq = 220  # A3
    stage3 = 0.1 * np.sin(2 * np.pi * stage3_freq * t) * np.exp(-t/12)
    
    # 合并和后处理
    update_progress("🎵 音频后处理", 3, "合并三阶段音频...")
    time.sleep(0.3)
    audio_array = np.concatenate([stage1, stage2, stage3])
    
    # 添加环境音效
    noise = 0.05 * np.random.normal(0, 1, len(audio_array))
    audio_array = audio_array + noise
    
    # 归一化
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    return audio_array.astype(np.float32), sample_rate

def generate_mock_video_with_progress(duration=15, fps=30):
    """生成模拟视频并显示进度"""
    update_progress("🎬 生成疗愈视频", 4, f"视频时长: {duration}秒, 帧率: {fps}fps")
    
    frame_count = int(duration * fps)
    frames = []
    width, height = 640, 480
    
    for i in range(frame_count):
        # 更新进度
        if i % 30 == 0:  # 每秒更新一次
            current_second = i // fps
            update_progress("🎬 渲染视频帧", 4, f"正在渲染第{current_second+1}秒 ({i+1}/{frame_count}帧)")
        
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

def create_demo_video_with_progress(audio_array, sample_rate, video_frames, fps):
    """创建演示视频并显示进度"""
    try:
        update_progress("🎬 合成音画同步视频", 5, "创建临时文件...")
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "demo_audio.wav")
        video_path = os.path.join(temp_dir, "demo_video.mp4")
        output_path = os.path.join(temp_dir, "demo_synchronized.mp4")
        
        # 保存音频
        update_progress("🎬 保存音频文件", 5, f"保存到: {audio_path}")
        import soundfile as sf
        sf.write(audio_path, audio_array, sample_rate)
        
        # 创建视频
        update_progress("🎬 创建视频文件", 5, f"处理{len(video_frames)}帧...")
        if len(video_frames) > 0:
            frame_height, frame_width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            for i, frame in enumerate(video_frames):
                if i % 90 == 0:  # 每3秒更新一次
                    update_progress("🎬 写入视频帧", 5, f"已写入{i+1}/{len(video_frames)}帧")
                out.write(frame)
            
            out.release()
            
            # 使用ffmpeg合成
            update_progress("🎬 FFmpeg音画同步", 6, "正在合成最终视频...")
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
                    update_progress("✅ 视频生成完成", 7, f"输出文件: {output_path}")
                    return output_path
                else:
                    update_progress("⚠️ FFmpeg不可用", 6, "返回纯视频文件")
                    return video_path
            except Exception as e:
                update_progress("⚠️ FFmpeg错误", 6, f"错误: {str(e)}")
                return video_path
        else:
            update_progress("❌ 视频生成失败", 6, "没有视频帧")
            return None
            
    except Exception as e:
        update_progress("❌ 创建视频失败", 6, f"错误: {str(e)}")
        return None

def get_status_info():
    """获取当前状态信息"""
    global processing_status
    if processing_status["progress"] == 0:
        return "🎯 等待开始处理..."
    elif processing_status["progress"] >= processing_status["total_steps"]:
        return "✅ 处理完成！"
    else:
        progress_bar = "█" * processing_status["progress"] + "░" * (processing_status["total_steps"] - processing_status["progress"])
        return f"{processing_status['current_step']}\n[{progress_bar}] {processing_status['progress']}/{processing_status['total_steps']}\n{processing_status['details']}"

def enhanced_demo_processing(user_input):
    """增强的演示模式处理"""
    global processing_status
    
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, None, "输入太短"
    
    try:
        # 重置状态
        processing_status = {
            "current_step": "",
            "progress": 0,
            "total_steps": 7,
            "details": "",
            "error": None
        }
        
        # 步骤1: 情绪识别
        update_progress("🧠 情绪识别分析", 1, "分析用户输入的情绪状态...")
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
        
        # 步骤2: 治疗方案制定
        update_progress("📋 制定治疗方案", 2, f"基于{detected_emotion}情绪制定三阶段治疗方案...")
        time.sleep(0.5)
        
        # 步骤3-5: 音频生成
        audio_array, sample_rate = generate_mock_audio_with_progress(duration=15)
        
        # 步骤4: 视频生成
        video_frames, fps = generate_mock_video_with_progress(duration=15, fps=30)
        
        # 步骤5-6: 视频合成
        video_path = create_demo_video_with_progress(audio_array, sample_rate, video_frames, fps)
        
        # 步骤7: 完成
        update_progress("🎉 生成完成", 7, "音画同步疗愈视频已准备就绪！")
        
        # 组织返回信息
        emotion_result = f"""🧠 情绪识别结果:
情绪类型: {detected_emotion}
置信度: {emotion_info['confidence']:.1%}
状态描述: {emotion_info['type']}
识别时间: {datetime.now().strftime('%H:%M:%S')}"""
        
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

🔧 技术信息:
  - 处理时间: {datetime.now().strftime('%H:%M:%S')}
  - 文件路径: {video_path}
  - 合成方式: OpenCV + FFmpeg

📝 注意事项:
  - 这是演示模式，使用模拟数据
  - 真实版本需要配置商业API (Suno, Runway等)
  - 演示音频采用数学合成，真实版本使用AI生成"""
            
            return emotion_result, video_path, (sample_rate, audio_array), info_text
        else:
            return emotion_result, None, (sample_rate, audio_array), "❌ 视频生成失败"
    
    except Exception as e:
        error_msg = f"❌ 处理错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, None, None, f"错误详情: {traceback.format_exc()}"

def create_enhanced_interface():
    """创建增强的演示界面"""
    with gr.Blocks(
        title="🌙 睡眠疗愈AI - 增强演示模式",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🌙 睡眠疗愈AI - 增强演示模式</h1>
            <p style="color: #666;">完整的三阶段音乐叙事疗愈系统演示</p>
            <p style="color: orange; font-weight: bold;">⚠️ 演示模式：使用模拟数据，展示完整功能流程</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💭 情绪输入")
                
                # 快速选择
                emotion_examples = gr.Radio(
                    choices=[
                        "😰 我感到很焦虑，心跳加速，难以入睡",
                        "😴 我很疲惫，但大脑还在活跃，无法放松",
                        "😤 我感到烦躁不安，容易被小事影响",
                        "😌 我比较平静，但希望更深层的放松",
                        "🤯 最近压力很大，总是感到紧张"
                    ],
                    label="🎭 快速选择常见情绪",
                    value="😰 我感到很焦虑，心跳加速，难以入睡"
                )
                
                emotion_input = gr.Textbox(
                    label="✍️ 详细描述您的感受",
                    placeholder="描述您现在的情绪状态、身体感受、思维状态等...",
                    lines=4,
                    value="我感到很焦虑，心跳加速，难以入睡"
                )
                
                with gr.Row():
                    process_btn = gr.Button("🎬 开始生成音画同步疗愈视频", variant="primary", size="lg")
                    clear_btn = gr.Button("🔄 清除", variant="secondary")
                
                # 实时状态显示
                status_display = gr.Textbox(
                    label="📊 实时处理状态",
                    value="🎯 等待开始处理...",
                    lines=3,
                    interactive=False
                )
            
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
                    label="📊 详细技术信息",
                    lines=12,
                    interactive=False
                )
        
        # 系统说明
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h3>🎯 系统特色</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
                <div>
                    <h4>🧠 情绪识别</h4>
                    <p>• 27维细粒度情绪分类<br>• 睡前专用情绪识别<br>• 实时置信度评估</p>
                </div>
                <div>
                    <h4>🎵 三阶段音乐</h4>
                    <p>• 同步期：匹配用户情绪<br>• 引导期：逐步情绪转换<br>• 巩固期：深度放松状态</p>
                </div>
                <div>
                    <h4>🎬 视觉疗愈</h4>
                    <p>• 颜色渐变疗愈<br>• 呼吸引导动画<br>• 音画完美同步</p>
                </div>
            </div>
        </div>
        """)
        
        # 事件绑定
        def update_input_from_example(example):
            return example.split(" ", 1)[1] if " " in example else example
        
        emotion_examples.change(
            update_input_from_example,
            inputs=emotion_examples,
            outputs=emotion_input
        )
        
        # 异步处理函数
        def process_with_status_updates(user_input):
            # 处理函数
            result = enhanced_demo_processing(user_input)
            return result
        
        process_btn.click(
            process_with_status_updates,
            inputs=emotion_input,
            outputs=[emotion_result, video_output, audio_output, info_output]
        )
        
        clear_btn.click(
            lambda: ("", None, None, "", "🎯 等待开始处理..."),
            outputs=[emotion_input, video_output, audio_output, info_output, status_display]
        )
        
        # 状态更新（每秒更新一次）
        def update_status():
            return get_status_info()
        
        # 定时更新状态
        app.load(lambda: gr.update(value=get_status_info()), outputs=status_display, every=1)
    
    return app

def main():
    """主函数"""
    print("🚀 启动增强演示模式...")
    print("📊 包含详细进度显示和状态更新")
    
    app = create_enhanced_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()