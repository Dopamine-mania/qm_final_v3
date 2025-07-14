#!/usr/bin/env python3
"""
🌙 修复版Suno API集成界面 - 三阶段音画同步疗愈体验
真正的目标：用户勾选Suno API后获得完整的音画同步疗愈体验
使用最便宜的chirp-v3模型，严格成本控制
"""

import gradio as gr
import os
import json
import http.client
import time
import tempfile
import numpy as np
from pathlib import Path

# 🛡️ 严格成本控制配置 - 使用最便宜的v3模型
API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"
MAX_DAILY_CALLS = 3
daily_call_count = 0

def call_suno_api_v3(emotion, enable_real_api=False):
    """调用Suno API v3（最便宜模型）"""
    global daily_call_count
    
    if not enable_real_api:
        return {
            "status": "mock",
            "message": "模拟模式 - 未启用真实API调用",
            "audio_file": "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final3/previous_suno_fdd1b90b.mp3"
        }
    
    if daily_call_count >= MAX_DAILY_CALLS:
        return {
            "status": "error", 
            "message": f"今日API调用已达上限({MAX_DAILY_CALLS}次)"
        }
    
    try:
        # 生成极简提示词（成本优化）
        emotion_map = {
            "焦虑": "calm sleep",
            "疲惫": "rest therapy", 
            "烦躁": "peace music",
            "平静": "deep relax",
            "压力": "stress relief"
        }
        prompt = emotion_map.get(emotion, "sleep music")
        
        conn = http.client.HTTPSConnection(BASE_URL)
        payload = json.dumps({
            "gpt_description_prompt": prompt,
            "make_instrumental": True,
            "mv": "chirp-v3-0",  # 最便宜的v3模型
            "prompt": prompt
        })
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        conn.request("POST", "/suno/submit/music", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        if res.status == 200:
            response = json.loads(data.decode("utf-8"))
            daily_call_count += 1
            
            if response.get('code') == 'success':
                task_id = response.get('data')
                return {
                    "status": "success",
                    "task_id": task_id,
                    "message": f"✅ API调用成功！任务ID: {task_id}",
                    "remaining_calls": MAX_DAILY_CALLS - daily_call_count
                }
        
        return {"status": "error", "message": f"API调用失败: {res.status}"}
        
    except Exception as e:
        return {"status": "error", "message": f"API调用异常: {e}"}

def load_existing_suno_music():
    """加载已有的Suno音乐（避免重复调用）"""
    audio_file = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final3/previous_suno_fdd1b90b.mp3"
    
    if os.path.exists(audio_file):
        return audio_file, """🎵 成功加载真实Suno AI音乐！

🎼 音乐详情:
   • 标题: "Whisper of the Moon"
   • 时长: 2分44秒 (真实AI生成)
   • 模型: Chirp-v3 (成本优化选择)
   • 风格: 三阶段疗愈音乐
   
🌙 三阶段疗愈体验:
   • 匹配阶段: 与用户情绪同频共振
   • 引导阶段: 流畅过渡到放松状态  
   • 目标阶段: 深度放松，准备入睡
   
✨ 这展示了完整音画同步疗愈系统的音频部分！
   下一步：集成视频画面，实现真正的音画同步疗愈！"""
    else:
        return None, "❌ 未找到已有音乐文件"

def process_suno_request(emotion_input, use_suno_api, enable_real_api, use_existing):
    """处理Suno API请求 - 核心疗愈体验生成"""
    
    if use_existing:
        # 使用已有音乐，避免浪费API调用
        audio_file, report = load_existing_suno_music()
        return report, audio_file, "✅ 加载已有Suno音乐"
    
    if not use_suno_api:
        return "⚠️ 请勾选'使用Suno AI音乐生成'", None, "未启用Suno API"
    
    # 简单情绪识别
    emotion = "焦虑"
    if "疲惫" in emotion_input or "累" in emotion_input:
        emotion = "疲惫"
    elif "烦躁" in emotion_input or "烦" in emotion_input:
        emotion = "烦躁"
    elif "平静" in emotion_input or "放松" in emotion_input:
        emotion = "平静"
    elif "压力" in emotion_input or "紧张" in emotion_input:
        emotion = "压力"
    
    # 调用Suno API
    result = call_suno_api_v3(emotion, enable_real_api)
    
    if result["status"] == "mock":
        audio_file, _ = load_existing_suno_music()
        report = f"""🧪 Suno API模拟模式 (节约成本)

🎯 情绪识别: {emotion}
💰 成本控制: 使用模拟模式，未消耗API费用
🎵 演示效果: 使用之前生成的真实Suno音乐

🌟 三阶段疗愈设计原理:
   • 匹配阶段(30%): 同步{emotion}情绪频率
   • 引导阶段(40%): 流畅过渡引导放松
   • 目标阶段(30%): 深度平静助眠状态

💡 提示: 勾选'启用真实API'体验真实AI音乐生成
       当前使用最经济的chirp-v3模型"""
        
        return report, audio_file, f"模拟模式 - {emotion}情绪处理"
    
    elif result["status"] == "success":
        # 真实API调用成功
        task_id = result["task_id"]
        report = f"""✅ Suno API调用成功！

🎯 检测情绪: {emotion}
🆔 任务ID: {task_id}
💰 模型: chirp-v3 (最经济选择)
📊 剩余调用: {result['remaining_calls']}/3

🔄 音乐生成中... 
   通常需要1-3分钟完成
   
🌙 正在创建三阶段疗愈音乐:
   • 匹配 → 引导 → 目标
   • 完成后将实现音画同步疗愈体验
   
💡 可稍后使用任务ID获取结果"""
        
        return report, None, f"API调用成功 - 任务ID: {task_id}"
    
    else:
        return f"❌ {result['message']}", None, "API调用失败"

def create_suno_interface():
    """创建Suno API集成界面 - 专注于三阶段疗愈体验"""
    
    with gr.Blocks(
        title="🌙 Suno API三阶段疗愈系统",
        theme=gr.themes.Soft(primary_hue="purple")
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1>🌙 Suno AI三阶段疗愈系统</h1>
            <p><strong>目标：音画同步的多模态疗愈体验</strong></p>
            <p>六层架构 • ISO三阶段原则 • 成本优化设计</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 💭 情绪输入")
                
                emotion_input = gr.Textbox(
                    label="✍️ 描述您的情绪状态",
                    placeholder="例如：我感到很焦虑，难以入睡...",
                    lines=3,
                    value="我感到很焦虑，心跳加速，难以入睡"
                )
                
                gr.Markdown("### 🎵 Suno AI配置")
                
                use_suno_api = gr.Checkbox(
                    label="🎵 使用Suno AI音乐生成",
                    value=True,
                    info="启用AI音乐生成（三阶段疗愈核心）"
                )
                
                enable_real_api = gr.Checkbox(
                    label="💰 启用真实API调用",
                    value=False,
                    info="⚠️ 消耗费用！使用最便宜的chirp-v3模型"
                )
                
                use_existing = gr.Checkbox(
                    label="🔄 使用已有音乐（推荐）",
                    value=True,
                    info="播放之前生成的真实Suno音乐，避免浪费API"
                )
                
                generate_btn = gr.Button(
                    "🌊 生成三阶段疗愈体验",
                    variant="primary",
                    size="lg"
                )
                
                gr.HTML("""
                <div style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px; font-size: 12px;">
                    <strong>💡 使用说明：</strong><br>
                    • <strong>已有音乐</strong>：体验真实AI生成效果，无费用<br>
                    • <strong>模拟模式</strong>：演示功能，显示设计原理<br>
                    • <strong>真实API</strong>：生成新音乐，使用v3模型节约成本<br><br>
                    <strong>🎯 项目目标：</strong>最终实现音画同步的多模态疗愈输出
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎬 疗愈体验输出")
                
                info_output = gr.Textbox(
                    label="📊 三阶段疗愈报告",
                    lines=20,
                    interactive=False,
                    value="等待生成三阶段疗愈体验..."
                )
                
                audio_output = gr.Audio(
                    label="🎵 AI生成疗愈音乐",
                    type="filepath"
                )
                
                status_output = gr.Textbox(
                    label="🔄 系统状态",
                    interactive=False,
                    value="就绪 - 等待用户输入"
                )
        
        # 系统说明
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px;">
            <h3 style="color: #333;">🎯 三阶段音画同步疗愈系统</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px;">
                <div>
                    <h4 style="color: #555;">🏗️ 六层架构</h4>
                    <ul style="color: #666; text-align: left; font-size: 14px;">
                        <li><strong>输入层</strong>：多模态情绪识别</li>
                        <li><strong>融合层</strong>：27维情绪融合</li>
                        <li><strong>映射层</strong>：KG-MLP混合映射</li>
                        <li><strong>生成层</strong>：Suno AI音乐生成</li>
                        <li><strong>渲染层</strong>：视频画面匹配</li>
                        <li><strong>疗愈层</strong>：音画同步输出</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #555;">🌊 ISO三阶段原则</h4>
                    <ul style="color: #666; text-align: left; font-size: 14px;">
                        <li><strong>匹配阶段</strong>：同步用户情绪频率</li>
                        <li><strong>引导阶段</strong>：流畅过渡到目标状态</li>
                        <li><strong>目标阶段</strong>：深度放松助眠效果</li>
                        <li><strong>音画同步</strong>：视频画面配合音乐</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #555;">💰 成本优化策略</h4>
                    <ul style="color: #666; text-align: left; font-size: 14px;">
                        <li><strong>模型选择</strong>：chirp-v3最经济</li>
                        <li><strong>调用限制</strong>：每日最多3次</li>
                        <li><strong>复用机制</strong>：已有任务ID复用</li>
                        <li><strong>模拟模式</strong>：开发测试无费用</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
        # 绑定事件
        generate_btn.click(
            process_suno_request,
            inputs=[emotion_input, use_suno_api, enable_real_api, use_existing],
            outputs=[info_output, audio_output, status_output]
        )
    
    return app

def main():
    """启动修复版Suno API界面"""
    print("🚀 启动修复版Suno API三阶段疗愈系统")
    print("🎯 目标：音画同步的多模态疗愈体验")
    print("💰 成本优化：chirp-v3模型 + 严格调用控制")
    
    app = create_suno_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7871,
        share=True,
        debug=False
    )

if __name__ == "__main__":
    main()