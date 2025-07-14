#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - 修复版本
直接从生成层获取音视频数据，输出音画同步视频
"""

import gradio as gr
import asyncio
import sys
import time
import numpy as np
import cv2
import os
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from main import QMFinal3System
from layers.base_layer import LayerData

# 全局系统实例
system = None

def init_system():
    """初始化系统"""
    global system
    try:
        system = QMFinal3System()
        return "✅ 系统初始化完成！"
    except Exception as e:
        return f"❌ 系统初始化失败: {str(e)}"

def process_emotion_debug(user_input):
    """处理情绪输入并生成音画同步视频"""
    global system
    
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, None, "输入太短"
    
    if not system:
        return "❌ 请先点击'初始化系统'按钮", None, None, "系统未初始化"
    
    try:
        print(f"🔄 开始处理情绪输入: {user_input}")
        
        # 创建输入数据
        input_data = LayerData(
            layer_name="gradio_fixed",
            timestamp=datetime.now(),
            data={"test_input": user_input},
            metadata={"source": "gradio_fixed", "user_input": user_input}
        )
        
        # 添加文本输入到输入层
        if system.layers:
            input_layer = system.layers[0]
            if hasattr(input_layer, 'add_text_input'):
                input_layer.add_text_input(user_input)
        
        # 通过真正的系统管道处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(system.pipeline.process(input_data))
            
            # 直接从生成层获取数据
            generation_layer = None
            for layer in system.layers:
                if hasattr(layer, '__class__') and 'GenerationLayer' in str(layer.__class__):
                    generation_layer = layer
                    break
            
            if generation_layer:
                print("✅ 找到生成层，开始提取音视频数据...")
                
                # 从生成层的缓存中获取数据
                if hasattr(generation_layer, 'layer_cache'):
                    cache_data = generation_layer.layer_cache
                    print(f"📦 生成层缓存数据: {list(cache_data.keys())}")
                    
                    # 查找最新的生成内容
                    latest_content = None
                    for key, value in cache_data.items():
                        if hasattr(value, 'data') and 'generated_content' in value.data:
                            latest_content = value.data['generated_content']
                            break
                    
                    if latest_content:
                        print("🎬 找到生成内容，开始合成音画同步视频...")
                        
                        # 提取音频
                        audio_content = latest_content.get('audio', {})
                        audio_array = audio_content.get('audio_array')
                        sample_rate = audio_content.get('sample_rate', 44100)
                        
                        # 提取视频
                        video_content = latest_content.get('video', {})
                        video_frames = video_content.get('frames', [])
                        fps = video_content.get('fps', 30)
                        
                        print(f"🎵 音频数据: {type(audio_array)}, 采样率: {sample_rate}")
                        print(f"🎬 视频数据: {len(video_frames)} 帧, FPS: {fps}")
                        
                        # 合成音画同步视频
                        if audio_array is not None and len(video_frames) > 0:
                            video_path = create_synchronized_video(audio_array, sample_rate, video_frames, fps)
                            
                            if video_path:
                                emotion_name = audio_content.get('emotion_name', '复合情绪')
                                emotion_info = f"🧠 识别情绪: {emotion_name}\n🎵 音频: {len(audio_array)/sample_rate:.1f}秒\n🎬 视频: {len(video_frames)}帧"
                                
                                info_text = f"✅ 音画同步视频生成成功!\n文件路径: {video_path}\n音频时长: {len(audio_array)/sample_rate:.1f}秒\n视频帧数: {len(video_frames)}\n帧率: {fps}fps"
                                
                                return emotion_info, video_path, None, info_text
                            else:
                                return "❌ 视频合成失败", None, None, "无法合成音画同步视频"
                        else:
                            return "⚠️ 音频或视频数据不完整", None, None, f"音频: {type(audio_array)}, 视频帧: {len(video_frames)}"
                    else:
                        return "❌ 未找到生成内容", None, None, "生成层缓存为空"
                else:
                    return "❌ 生成层缓存不存在", None, None, "无法访问生成层缓存"
            else:
                return "❌ 未找到生成层", None, None, "系统架构错误"
                
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 处理失败: {str(e)}", None, None, f"错误详情: {traceback.format_exc()}"
        finally:
            loop.close()
        
    except Exception as e:
        import traceback
        return f"❌ 处理错误: {str(e)}", None, None, f"错误详情: {traceback.format_exc()}"

def create_synchronized_video(audio_array, sample_rate, video_frames, fps):
    """创建音画同步视频"""
    try:
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_path = os.path.join(temp_dir, "video.mp4")
        output_path = os.path.join(temp_dir, "synchronized_video.mp4")
        
        # 保存音频
        import soundfile as sf
        sf.write(audio_path, audio_array, sample_rate)
        print(f"✅ 音频保存到: {audio_path}")
        
        # 创建视频
        if len(video_frames) > 0:
            frame_height, frame_width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
            
            for frame in video_frames:
                # 确保帧是RGB格式
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # RGB转BGR (OpenCV使用BGR)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                else:
                    # 如果是灰度图，转换为BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(frame_bgr)
            
            out.release()
            print(f"✅ 视频保存到: {video_path}")
            
            # 使用ffmpeg合成音画同步视频
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
                print(f"✅ 音画同步视频生成成功: {output_path}")
                return output_path
            else:
                print(f"❌ ffmpeg合成失败: {result.stderr}")
                return None
        else:
            print("❌ 没有视频帧")
            return None
            
    except Exception as e:
        print(f"❌ 创建同步视频时出错: {e}")
        return None

def create_interface():
    """创建界面"""
    with gr.Blocks(title="🌙 睡眠疗愈AI - 修复版本") as app:
        gr.HTML("<h1 style='text-align: center;'>🌙 睡眠疗愈AI - 音画同步版本</h1>")
        
        with gr.Row():
            init_btn = gr.Button("🚀 初始化系统", variant="primary")
            init_status = gr.Textbox(label="系统状态", lines=2)
        
        with gr.Row():
            with gr.Column():
                emotion_input = gr.Textbox(
                    label="💭 描述您的感受",
                    placeholder="例如：我感到很焦虑，心跳加速，难以入睡...",
                    lines=4
                )
                
                process_btn = gr.Button("🎬 生成音画同步疗愈视频", variant="primary")
            
            with gr.Column():
                emotion_result = gr.Textbox(label="🧠 情绪识别结果", lines=3)
                
                # 音画同步视频输出
                video_output = gr.Video(label="🎬 音画同步疗愈视频")
                
                # 单独的音频输出（备用）
                audio_output = gr.Audio(label="🎵 音频（备用）")
                
                info_output = gr.Textbox(label="📊 详细信息", lines=6)
        
        # 绑定事件
        init_btn.click(init_system, outputs=init_status)
        process_btn.click(
            process_emotion_debug,
            inputs=emotion_input,
            outputs=[emotion_result, video_output, audio_output, info_output]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动音画同步版本...")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()