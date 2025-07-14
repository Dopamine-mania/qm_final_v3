#!/usr/bin/env python3
"""
🌙 睡眠疗愈AI - 情绪音乐治疗系统
基于三阶段音乐叙事的睡前情绪疗愈Web应用

用法: streamlit run app.py
"""

import streamlit as st
import asyncio
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import base64
from io import BytesIO

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from main import QMFinal3System
from layers.base_layer import LayerData

# 页面配置
st.set_page_config(
    page_title="🌙 睡眠疗愈AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主体样式 */
    .main {
        padding-top: 2rem;
    }
    
    /* 标题样式 */
    .title {
        text-align: center;
        font-size: 3rem;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        color: white;
    }
    
    .card-content {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* 输入框样式 */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 1.1rem;
        padding: 1rem;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 情绪显示样式 */
    .emotion-display {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* 成功消息样式 */
    .success-message {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 音视频播放器样式 */
    .video-container {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'result' not in st.session_state:
    st.session_state.result = None

def init_system():
    """初始化系统"""
    if st.session_state.system is None:
        try:
            with st.spinner("🚀 正在初始化睡眠疗愈AI系统..."):
                st.session_state.system = QMFinal3System()
            st.success("✅ 系统初始化完成！")
            return True
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {e}")
            return False
    return True

async def process_emotion_input(user_input: str):
    """处理用户情绪输入"""
    try:
        # 创建输入数据
        input_data = LayerData(
            layer_name="web_interface",
            timestamp=datetime.now(),
            data={"test_input": user_input},
            metadata={"source": "web_app", "user_input": user_input}
        )
        
        # 添加文本输入到输入层
        if st.session_state.system.layers:
            input_layer = st.session_state.system.layers[0]
            if hasattr(input_layer, 'add_text_input'):
                input_layer.add_text_input(user_input)
        
        # 通过管道处理
        result = await st.session_state.system.pipeline.process(input_data)
        return result
        
    except Exception as e:
        st.error(f"处理过程中出现错误: {e}")
        return None

def display_emotion_result(result):
    """显示情绪识别结果"""
    if not result or not result.data:
        return
    
    # 尝试从结果中提取情绪信息
    emotion_info = {}
    
    # 从管道历史中获取情绪信息
    if hasattr(st.session_state.system.pipeline, 'layer_results'):
        for layer_result in st.session_state.system.pipeline.layer_results:
            if (hasattr(layer_result, 'data') and 
                'emotion_analysis' in layer_result.data):
                analysis = layer_result.data['emotion_analysis']
                emotion_info = {
                    'primary_emotion': analysis.get('primary_emotion', {}),
                    'confidence': layer_result.confidence,
                    'layer_name': layer_result.layer_name
                }
                break
    
    if emotion_info:
        emotion_name = emotion_info.get('primary_emotion', {}).get('name', '未知')
        confidence = emotion_info.get('confidence', 0.0)
        
        st.markdown(f"""
        <div class="emotion-display">
            <h3>🧠 情绪识别结果</h3>
            <p><strong>主要情绪:</strong> {emotion_name}</p>
            <p><strong>置信度:</strong> {confidence:.1%}</p>
            <p><strong>处理层:</strong> {emotion_info.get('layer_name', '未知')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        return emotion_name, confidence
    
    return None, 0.0

def display_generated_content(result):
    """显示生成的音视频内容"""
    if not result or not result.data:
        return
    
    # 查找生成的内容
    generated_content = None
    
    # 从管道历史中获取生成内容
    if hasattr(st.session_state.system.pipeline, 'layer_results'):
        for layer_result in st.session_state.system.pipeline.layer_results:
            if (hasattr(layer_result, 'data') and 
                'generated_content' in layer_result.data):
                generated_content = layer_result.data['generated_content']
                break
    
    if generated_content:
        st.markdown("### 🎬 您的专属疗愈内容")
        
        # 显示音频内容
        audio_content = generated_content.get('audio', {})
        if audio_content and 'audio_array' in audio_content:
            st.markdown("#### 🎵 三阶段疗愈音乐")
            
            # 显示音乐信息
            duration = audio_content.get('duration', 0)
            sample_rate = audio_content.get('sample_rate', 44100)
            three_stage = audio_content.get('three_stage_narrative', False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("时长", f"{duration:.0f}秒")
            with col2:
                st.metric("采样率", f"{sample_rate}Hz")
            with col3:
                st.metric("三阶段叙事", "✅" if three_stage else "❌")
            
            # 尝试播放音频
            audio_array = audio_content.get('audio_array')
            if audio_array is not None and isinstance(audio_array, np.ndarray):
                try:
                    # 转换音频为可播放格式
                    if audio_array.dtype != np.float32:
                        audio_array = audio_array.astype(np.float32)
                    
                    # 确保音频在合理范围内
                    if np.max(np.abs(audio_array)) > 0:
                        audio_array = audio_array / np.max(np.abs(audio_array))
                    
                    # Streamlit音频播放器
                    st.audio(audio_array, sample_rate=sample_rate)
                    
                    # 显示阶段信息
                    stage_prompts = audio_content.get('stage_prompts', {})
                    if stage_prompts:
                        with st.expander("📝 查看三阶段音乐设计"):
                            for stage, prompt in stage_prompts.items():
                                st.markdown(f"**{stage.title()}阶段:**")
                                st.write(prompt[:200] + "..." if len(prompt) > 200 else prompt)
                                st.markdown("---")
                    
                except Exception as e:
                    st.warning(f"音频播放器加载失败: {e}")
                    st.info("💡 音频已生成，但当前环境无法播放。文件已保存到outputs/目录。")
        
        # 显示视频内容
        video_content = generated_content.get('video', {})
        if video_content and 'frames' in video_content:
            st.markdown("#### 🖼️ 疗愈视觉内容")
            
            frames = video_content.get('frames', [])
            fps = video_content.get('fps', 30)
            
            if frames:
                st.write(f"生成了 {len(frames)} 帧视频，帧率: {fps}fps")
                
                # 显示几帧预览
                cols = st.columns(min(5, len(frames)))
                for i, frame in enumerate(frames[:5]):
                    with cols[i]:
                        if isinstance(frame, np.ndarray):
                            st.image(frame, caption=f"第{i+1}帧", use_column_width=True)
        
        return True
    
    return False

def main():
    """主应用"""
    # 标题
    st.markdown('<h1 class="title">🌙 睡眠疗愈AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">基于情绪识别的三阶段音乐叙事疗愈系统</p>', unsafe_allow_html=True)
    
    # 初始化系统
    if not init_system():
        st.stop()
    
    # 主界面布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-content">
                <h2>💭 描述您现在的感受</h2>
                <p>请用文字描述您当前的情绪状态，我们将为您生成个性化的睡前疗愈音乐。</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 预设情绪选项
        st.markdown("#### 🎭 快速选择或自定义描述")
        
        emotion_presets = {
            "😰 焦虑紧张": "我感到很焦虑，心跳加速，难以平静下来，脑子里总是想着各种担心的事情",
            "😴 疲惫困倦": "我感到非常疲惫，身体很累，但是大脑还在活跃，难以入睡",
            "😤 烦躁不安": "我感到很烦躁，心情不好，容易被小事影响，无法集中注意力",
            "😌 相对平静": "我的心情比较平静，但希望能进入更深层的放松状态，为睡眠做准备",
            "🤯 压力山大": "最近压力很大，学习工作任务重，总是感到时间不够用，内心很紧张"
        }
        
        selected_preset = st.selectbox("选择预设情绪描述:", ["自定义输入"] + list(emotion_presets.keys()))
        
        if selected_preset != "自定义输入":
            user_input = emotion_presets[selected_preset]
            st.text_area("您的情绪描述:", value=user_input, height=100, key="preset_input")
        else:
            user_input = st.text_area(
                "请详细描述您的感受:",
                placeholder="例如：我今天工作压力很大，心情有些焦虑，躺在床上总是想东想西，无法入睡...",
                height=100
            )
        
        # 处理按钮
        if st.button("🧠 开始情绪分析与音乐生成", type="primary", disabled=st.session_state.processing):
            if user_input and len(user_input.strip()) > 5:
                st.session_state.processing = True
                st.session_state.result = None
                
                with st.spinner("🔄 正在分析您的情绪并生成专属疗愈内容..."):
                    # 显示处理进度
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 模拟处理进度
                    stages = [
                        "🧠 分析情绪状态...",
                        "🎵 设计音乐参数...", 
                        "🎼 生成三阶段音乐...",
                        "🖼️ 创建视觉内容...",
                        "✨ 完成疗愈内容..."
                    ]
                    
                    for i, stage in enumerate(stages):
                        status_text.text(stage)
                        progress_bar.progress((i + 1) / len(stages))
                        time.sleep(0.5)  # 模拟处理时间
                    
                    # 实际处理
                    result = asyncio.run(process_emotion_input(user_input))
                    st.session_state.result = result
                    st.session_state.processing = False
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 处理完成！")
                    
                    if result:
                        st.markdown('<div class="success-message">🎉 您的专属疗愈内容已生成！</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error("❌ 处理失败，请稍后重试")
            else:
                st.warning("⚠️ 请输入至少5个字符的情绪描述")
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-content">
                <h2>🎬 您的疗愈内容</h2>
                <p>基于您的情绪状态，我们将生成个性化的三阶段音画疗愈内容。</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示结果
        if st.session_state.result:
            # 显示情绪识别结果
            emotion_name, confidence = display_emotion_result(st.session_state.result)
            
            # 显示生成的内容
            if display_generated_content(st.session_state.result):
                st.markdown("---")
                st.markdown("### 💡 使用建议")
                st.markdown("""
                1. **🎧 佩戴耳机**：获得最佳的立体声效果
                2. **🌙 调暗灯光**：创造适合睡眠的环境
                3. **🧘‍♀️ 放松身体**：找到舒适的姿势
                4. **🎵 专注聆听**：跟随音乐的三阶段引导
                5. **😴 自然入睡**：让音乐引导您进入梦乡
                """)
            else:
                st.info("💭 完成情绪分析后，您的专属疗愈内容将在这里显示")
        else:
            # 显示使用说明
            st.markdown("""
            ### 🔮 AI疗愈原理
            
            **🎯 三阶段音乐叙事**
            - **匹配阶段**: 音乐与您当前情绪同步
            - **引导阶段**: 逐步过渡到平静状态  
            - **巩固阶段**: 建立稳定的睡前状态
            
            **🧠 27维情绪识别**
            - 识别细粒度的睡前情绪状态
            - 基于心理学和音乐治疗理论
            - 个性化的情绪-音乐映射
            
            **🎼 智能音乐生成**
            - 基于Suno AI的音乐创作
            - 符合音乐治疗ISO原则
            - 连贯的情绪转换叙事
            """)
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌟 睡眠疗愈AI v3.0 | 基于六层架构的情绪音乐治疗系统</p>
        <p>💖 祝您拥有美好的睡眠体验</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()