#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - Gradio调试版本
"""

import gradio as gr
import asyncio
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 全局系统实例
system = None

def init_system():
    """初始化系统"""
    global system
    try:
        from main import QMFinal3System
        system = QMFinal3System()
        return "✅ 系统初始化完成！可以开始使用了。"
    except Exception as e:
        return f"❌ 系统初始化失败: {str(e)}\n{traceback.format_exc()}"

def simple_test():
    """简单测试函数"""
    return "🎉 测试成功！系统运行正常。"

def process_emotion_simple(user_input):
    """简化的情绪处理函数"""
    global system
    
    if not user_input or len(user_input.strip()) < 5:
        return "⚠️ 请输入至少5个字符的情绪描述", None, "输入太短"
    
    if not system:
        return "❌ 请先点击'初始化系统'按钮", None, "系统未初始化"
    
    try:
        # 简单的模拟处理
        return f"✅ 收到您的情绪描述: {user_input[:50]}...", None, "功能开发中，请稍后测试完整版本"
        
    except Exception as e:
        return f"❌ 处理错误: {str(e)}", None, f"错误详情: {traceback.format_exc()}"

def create_debug_interface():
    """创建调试界面"""
    
    with gr.Blocks(title="🌙 睡眠疗愈AI - 调试版本") as app:
        
        gr.HTML("<h1 style='text-align: center;'>🌙 睡眠疗愈AI - 调试版本</h1>")
        
        with gr.Row():
            with gr.Column():
                # 系统初始化
                init_btn = gr.Button("🚀 初始化系统", variant="primary")
                init_status = gr.Textbox(label="系统状态", lines=3)
                
                # 简单测试
                test_btn = gr.Button("🧪 测试功能")
                test_result = gr.Textbox(label="测试结果")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💭 情绪输入测试")
                
                emotion_input = gr.Textbox(
                    label="描述您的感受",
                    placeholder="例如：我感到很焦虑...",
                    lines=3
                )
                
                process_btn = gr.Button("🧠 处理情绪", variant="secondary")
            
            with gr.Column():
                gr.Markdown("### 📊 处理结果")
                
                emotion_result = gr.Textbox(label="情绪识别", lines=2)
                audio_result = gr.Audio(label="音频输出")
                debug_info = gr.Textbox(label="调试信息", lines=3)
        
        # 绑定事件
        init_btn.click(init_system, outputs=init_status)
        test_btn.click(simple_test, outputs=test_result)
        process_btn.click(
            process_emotion_simple,
            inputs=emotion_input,
            outputs=[emotion_result, audio_result, debug_info]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动调试版本...")
    
    app = create_debug_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()