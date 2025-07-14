#!/usr/bin/env python3
"""
🎵 快速测试Suno音乐播放
简化版界面，专门用来播放之前成功生成的Suno音乐
"""

import gradio as gr
import os

def load_previous_suno_music():
    """加载之前成功生成的Suno音乐"""
    audio_file_path = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final3/previous_suno_fdd1b90b.mp3"
    
    if os.path.exists(audio_file_path):
        report = """🎵 成功加载之前的Suno AI音乐！

🎼 音乐信息:
   • 标题: "Whisper of the Moon"
   • 时长: 约2分44秒 (164秒)
   • 模型: Chirp-v4 (Suno最新模型)
   • 风格: 宁静睡眠音乐
   • 标签: sleep, soft, acoustic, soothing
   
🎹 音乐特色:
   • 指弹吉他与温柔钢琴和弦
   • 环境弦乐的微妙嗡鸣声
   • 多层次柔和音响
   • 专为睡前放松设计
   
🌙 疗愈效果:
   • 深度放松: acoustic fingerpicking营造安全感
   • 情绪稳定: 温和的钢琴和弦带来平静
   • 助眠引导: 环境音效帮助大脑放松
   • 持续疗愈: 2分44秒完整的放松体验
   
🎧 使用建议:
   • 佩戴耳机获得最佳立体声效果
   • 调至舒适音量 (建议50-70%)
   • 在安静环境中聆听
   • 闭眼跟随音乐进入放松状态
   
✨ 这是真实的Suno AI生成音乐，展示了AI音乐疗愈的实际效果！
🌟 任务ID: fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc"""
        
        return report, audio_file_path
    else:
        return "❌ 未找到之前的音乐文件", None

def create_interface():
    """创建简化界面"""
    with gr.Blocks(title="🎵 Suno音乐播放测试") as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1>🎵 Suno AI音乐播放测试</h1>
            <p>播放之前成功生成的真实AI音乐</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                load_btn = gr.Button(
                    "🎵 播放之前成功生成的Suno音乐",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                info_output = gr.Textbox(
                    label="📊 音乐信息",
                    lines=20,
                    interactive=False
                )
                
                audio_output = gr.Audio(
                    label="🎵 Suno AI音乐",
                    type="filepath"
                )
        
        load_btn.click(
            load_previous_suno_music,
            inputs=[],
            outputs=[info_output, audio_output]
        )
    
    return app

def main():
    print("🎵 启动Suno音乐播放测试")
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7870,
        share=True,
        debug=False
    )

if __name__ == "__main__":
    main()