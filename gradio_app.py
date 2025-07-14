#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - Gradio版本
基于三阶段音乐叙事的睡前情绪疗愈Web应用

用法: python gradio_app.py
"""

import gradio as gr
import asyncio
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from main import QMFinal3System
from layers.base_layer import LayerData

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局系统实例
system = None

def init_system():
    """初始化系统"""
    global system
    if system is None:
        try:
            system = QMFinal3System()
            return "✅ 系统初始化完成！"
        except Exception as e:
            return f"❌ 系统初始化失败: {e}"
    return "✅ 系统已就绪"

async def process_emotion_input(user_input: str):
    """处理用户情绪输入"""
    global system
    
    if not system:
        return "❌ 系统未初始化", None, "请先初始化系统"
    
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, ""
    
    try:
        # 创建输入数据
        input_data = LayerData(
            layer_name="gradio_interface",
            timestamp=datetime.now(),
            data={"test_input": user_input},
            metadata={"source": "gradio_app", "user_input": user_input}
        )
        
        # 添加文本输入到输入层
        if system.layers:
            input_layer = system.layers[0]
            if hasattr(input_layer, 'add_text_input'):
                input_layer.add_text_input(user_input)
        
        # 通过管道处理
        result = await system.pipeline.process(input_data)
        
        # 提取情绪识别结果
        emotion_info = "未知情绪"
        confidence = 0.0
        audio_info = "暂无音频生成"
        
        # 从管道历史中获取情绪信息
        if hasattr(system.pipeline, 'layer_results'):
            for layer_result in system.pipeline.layer_results:
                if (hasattr(layer_result, 'data') and 
                    'emotion_analysis' in layer_result.data):
                    analysis = layer_result.data['emotion_analysis']
                    primary_emotion = analysis.get('primary_emotion', {})
                    emotion_info = f"主要情绪: {primary_emotion.get('name', '未知')}"
                    confidence = layer_result.confidence
                    break
        
        # 查找生成的音频内容
        if hasattr(system.pipeline, 'layer_results'):
            for layer_result in system.pipeline.layer_results:
                if (hasattr(layer_result, 'data') and 
                    'generated_content' in layer_result.data):
                    generated_content = layer_result.data['generated_content']
                    audio_content = generated_content.get('audio', {})
                    if audio_content and 'audio_array' in audio_content:
                        duration = audio_content.get('duration', 0)
                        sample_rate = audio_content.get('sample_rate', 44100)
                        three_stage = audio_content.get('three_stage_narrative', False)
                        audio_info = f"✅ 生成成功!\n时长: {duration:.0f}秒\n采样率: {sample_rate}Hz\n三阶段叙事: {'是' if three_stage else '否'}"
                        
                        # 返回音频数组用于播放
                        audio_array = audio_content.get('audio_array')
                        if audio_array is not None and isinstance(audio_array, np.ndarray):
                            if audio_array.dtype != np.float32:
                                audio_array = audio_array.astype(np.float32)
                            if np.max(np.abs(audio_array)) > 0:
                                audio_array = audio_array / np.max(np.abs(audio_array))
                            return f"{emotion_info}\n置信度: {confidence:.1%}", (sample_rate, audio_array), audio_info
                    break
        
        return f"{emotion_info}\n置信度: {confidence:.1%}", None, audio_info
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        return f"❌ 处理失败: {e}", None, "处理过程中出现错误"

def gradio_process_emotion(user_input):
    """Gradio同步包装函数"""
    global system
    
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, None, "输入太短"
    
    if not system:
        return "❌ 请先点击'初始化系统'按钮", None, None, "系统未初始化"
    
    try:
        # 同步方式处理，避免asyncio问题
        from layers.base_layer import LayerData
        from datetime import datetime
        
        # 创建输入数据
        input_data = LayerData(
            layer_name="gradio_interface",
            timestamp=datetime.now(),
            data={"test_input": user_input},
            metadata={"source": "gradio_app", "user_input": user_input}
        )
        
        # 添加文本输入到输入层
        if system.layers:
            input_layer = system.layers[0]
            if hasattr(input_layer, 'add_text_input'):
                input_layer.add_text_input(user_input)
        
        # 使用真正的后端系统处理
        print(f"🔄 开始处理情绪输入: {user_input[:50]}...")
        
        # 通过真正的系统管道处理
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(system.pipeline.process(input_data))
            
            # 提取情绪识别结果
            emotion_info = "🧠 情绪识别中..."
            confidence = 0.0
            audio_array = None
            sample_rate = 44100
            
            # 从管道历史中获取情绪信息
            if hasattr(system.pipeline, 'layer_results'):
                for layer_result in system.pipeline.layer_results:
                    if (hasattr(layer_result, 'data') and 
                        'emotion_analysis' in layer_result.data):
                        analysis = layer_result.data['emotion_analysis']
                        primary_emotion = analysis.get('primary_emotion', {})
                        emotion_name = primary_emotion.get('name', '未知')
                        confidence = layer_result.confidence
                        emotion_info = f"🧠 主要情绪: {emotion_name}\n置信度: {confidence:.1%}"
                        print(f"✅ 情绪识别完成: {emotion_name}, 置信度: {confidence:.1%}")
                        break
            
            # 查找生成的音频内容
            audio_info = "🎵 正在生成三阶段音乐..."
            
            if hasattr(system.pipeline, 'layer_results'):
                for layer_result in system.pipeline.layer_results:
                    if (hasattr(layer_result, 'data') and 
                        'generated_content' in layer_result.data):
                        generated_content = layer_result.data['generated_content']
                        audio_content = generated_content.get('audio', {})
                        
                        if audio_content and 'audio_array' in audio_content:
                            duration = audio_content.get('duration', 0)
                            sample_rate = audio_content.get('sample_rate', 44100)
                            three_stage = audio_content.get('three_stage_narrative', False)
                            
                            # 获取音频数组
                            audio_array = audio_content.get('audio_array')
                            if audio_array is not None and isinstance(audio_array, np.ndarray):
                                if audio_array.dtype != np.float32:
                                    audio_array = audio_array.astype(np.float32)
                                if np.max(np.abs(audio_array)) > 0:
                                    audio_array = audio_array / np.max(np.abs(audio_array))
                                
                                audio_info = f"🎵 三阶段音乐生成完成!\n⏱️ 时长: {duration:.0f}秒\n🔊 采样率: {sample_rate}Hz\n📖 三阶段叙事: {'✅' if three_stage else '❌'}"
                                
                                # 显示阶段信息
                                stage_prompts = audio_content.get('stage_prompts', {})
                                if stage_prompts:
                                    audio_info += "\n\n🎼 音乐阶段设计:"
                                    for stage, prompt in stage_prompts.items():
                                        audio_info += f"\n• {stage}: {prompt[:100]}..."
                                
                                print(f"✅ 音频生成完成: {duration:.0f}秒, {sample_rate}Hz")
                                break
                        else:
                            # 如果没有真正的音频，显示错误信息
                            if 'error' in audio_content:
                                audio_info = f"❌ 音频生成失败: {audio_content['error']}"
                            else:
                                audio_info = "⚠️ 音频生成中，请稍候..."
                        break
            
            # 返回结果 (emotion_result, audio_output, video_output, audio_info)
            if audio_array is not None:
                return emotion_info, (sample_rate, audio_array), None, audio_info
            else:
                return emotion_info, None, None, audio_info
                
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

# 预设情绪选项
emotion_presets = {
    "😰 焦虑紧张": "我感到很焦虑，心跳加速，难以平静下来，脑子里总是想着各种担心的事情",
    "😴 疲惫困倦": "我感到非常疲惫，身体很累，但是大脑还在活跃，难以入睡",
    "😤 烦躁不安": "我感到很烦躁，心情不好，容易被小事影响，无法集中注意力",
    "😌 相对平静": "我的心情比较平静，但希望能进入更深层的放松状态，为睡眠做准备",
    "🤯 压力山大": "最近压力很大，学习工作任务重，总是感到时间不够用，内心很紧张"
}

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="🌙 睡眠疗愈AI",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue"
        ),
        css="""
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .main-title {
            text-align: center;
            color: white;
            font-size: 3rem;
            margin: 2rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.9);
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        """
    ) as app:
        
        gr.HTML("""
        <div class="main-title">🌙 睡眠疗愈AI</div>
        <div class="subtitle">基于情绪识别的三阶段音乐叙事疗愈系统</div>
        """)
        
        # 系统初始化
        with gr.Row():
            init_btn = gr.Button("🚀 初始化系统", variant="primary")
            init_status = gr.Textbox(label="系统状态", interactive=False)
        
        init_btn.click(init_system, outputs=init_status)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 💭 描述您现在的感受")
                
                # 预设情绪选择
                emotion_dropdown = gr.Dropdown(
                    choices=list(emotion_presets.keys()),
                    label="🎭 快速选择预设情绪",
                    value=None
                )
                
                # 情绪输入框
                emotion_input = gr.Textbox(
                    label="✍️ 详细描述您的感受",
                    placeholder="例如：我今天工作压力很大，心情有些焦虑，躺在床上总是想东想西，无法入睡...",
                    lines=4
                )
                
                # 处理按钮
                process_btn = gr.Button("🧠 开始情绪分析与音乐生成", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 您的疗愈内容")
                
                # 情绪识别结果
                emotion_result = gr.Textbox(
                    label="🧠 情绪识别结果",
                    lines=3,
                    interactive=False
                )
                
                # 生成的音频
                audio_output = gr.Audio(
                    label="🎵 三阶段疗愈音乐",
                    type="numpy"
                )
                
                # 生成的视频（如果有）
                video_output = gr.Video(
                    label="🖼️ 疗愈视觉内容",
                    visible=True
                )
                
                # 音频信息
                audio_info = gr.Textbox(
                    label="📊 音频/视频信息",
                    lines=6,
                    interactive=False
                )
        
        # 使用指南
        with gr.Row():
            gr.Markdown("""
            ### 💡 使用建议
            1. **🎧 佩戴耳机**：获得最佳的立体声效果
            2. **🌙 调暗灯光**：创造适合睡眠的环境  
            3. **🧘‍♀️ 放松身体**：找到舒适的姿势
            4. **🎵 专注聆听**：跟随音乐的三阶段引导
            5. **😴 自然入睡**：让音乐引导您进入梦乡
            
            ### 🔮 AI疗愈原理
            - **🎯 三阶段音乐叙事**: 匹配→引导→巩固
            - **🧠 27维情绪识别**: 识别细粒度的睡前情绪状态
            - **🎼 智能音乐生成**: 基于Suno AI的音乐创作，符合音乐治疗ISO原则
            """)
        
        # 预设情绪选择事件
        def update_input_from_preset(preset):
            if preset:
                return emotion_presets[preset]
            return ""
        
        emotion_dropdown.change(
            update_input_from_preset,
            inputs=emotion_dropdown,
            outputs=emotion_input
        )
        
        # 处理按钮事件
        process_btn.click(
            gradio_process_emotion,
            inputs=emotion_input,
            outputs=[emotion_result, audio_output, video_output, audio_info]
        )
    
    return app

def main():
    """主函数"""
    # 创建界面
    app = create_interface()
    
    # 启动应用
    print("🚀 正在启动睡眠疗愈AI...")
    print("🌐 启动后将自动生成公共访问链接...")
    
    # 使用share=True生成公共链接
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 这是关键！自动生成公共链接
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()