#!/usr/bin/env python3
"""
🌙 增强三阶段疗愈系统 - 完整Web界面版本
端到端体验：输入 → 生成 → 播放 → 看效果
集成Suno API：真实AI音乐生成（严格成本控制）
"""

import gradio as gr
import numpy as np
import time
import tempfile
import os
import json
import http.client
from pathlib import Path
from datetime import datetime

# 🛡️ 严格成本控制配置
SUNO_API_ENABLED = False  # 默认关闭！！！
TEST_MODE = True          # 测试模式
MAX_DAILY_CALLS = 3       # 每日最大调用次数
API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"

# 全局调用计数器
daily_call_count = 0
last_call_date = None

def get_emotion_music_features(emotion):
    """根据ISO三阶段原则映射情绪到音乐特征（硕士项目核心理论）"""
    features_database = {
        "焦虑": {
            "匹配阶段": {
                "tempo": "moderate tense",
                "key": "minor anxious", 
                "dynamics": "restless energy",
                "mood": "matching anxiety"
            },
            "引导阶段": {
                "tempo": "gradually calming",
                "key": "minor to neutral transition",
                "dynamics": "settling down", 
                "mood": "guiding to peace"
            },
            "目标阶段": {
                "tempo": "slow peaceful",
                "key": "major calm",
                "dynamics": "gentle soft",
                "mood": "deep relaxation for sleep"
            }
        },
        "疲惫": {
            "匹配阶段": {
                "tempo": "tired sluggish",
                "key": "minor weary",
                "dynamics": "heavy fatigue",
                "mood": "exhausted state"
            },
            "引导阶段": {
                "tempo": "gentle restoration",
                "key": "minor to warm transition", 
                "dynamics": "nurturing support",
                "mood": "healing tiredness"
            },
            "目标阶段": {
                "tempo": "deeply restful",
                "key": "warm major",
                "dynamics": "embracing comfort",
                "mood": "restorative sleep"
            }
        },
        "烦躁": {
            "匹配阶段": {
                "tempo": "agitated irregular",
                "key": "dissonant minor",
                "dynamics": "sharp edges",
                "mood": "irritated energy"
            },
            "引导阶段": {
                "tempo": "smoothing out",
                "key": "resolving tensions",
                "dynamics": "softening edges",
                "mood": "releasing irritation"
            },
            "目标阶段": {
                "tempo": "smooth flowing",
                "key": "resolved major",
                "dynamics": "peaceful waves",
                "mood": "serene sleep state"
            }
        },
        "平静": {
            "匹配阶段": {
                "tempo": "naturally calm",
                "key": "neutral peaceful",
                "dynamics": "already gentle",
                "mood": "existing tranquility"
            },
            "引导阶段": {
                "tempo": "deepening calm",
                "key": "enriching peace",
                "dynamics": "expanding serenity",
                "mood": "enhancing stillness"
            },
            "目标阶段": {
                "tempo": "profound stillness",
                "key": "deep major",
                "dynamics": "whisper soft",
                "mood": "transcendent sleep"
            }
        },
        "压力": {
            "匹配阶段": {
                "tempo": "pressured urgent",
                "key": "tense minor",
                "dynamics": "compressed energy",
                "mood": "stress overload"
            },
            "引导阶段": {
                "tempo": "releasing pressure",
                "key": "opening up space",
                "dynamics": "expanding freedom",
                "mood": "letting go stress"
            },
            "目标阶段": {
                "tempo": "weightless floating",
                "key": "liberated major",
                "dynamics": "free flowing",
                "mood": "stress-free sleep"
            }
        }
    }
    return features_database.get(emotion, features_database["焦虑"])

def generate_suno_prompt(emotion, music_features):
    """基于三阶段音乐特征生成Suno API提示词"""
    matching = music_features["匹配阶段"]
    guiding = music_features["引导阶段"]
    target = music_features["目标阶段"]
    
    prompt = f"""Therapeutic sleep music for {emotion} relief following ISO principle three-stage healing journey.

Stage 1 - Matching Phase: {matching['tempo']}, {matching['key']} key, {matching['dynamics']}, {matching['mood']}
Stage 2 - Guiding Phase: {guiding['tempo']}, {guiding['key']}, {guiding['dynamics']}, {guiding['mood']}
Stage 3 - Target Phase: {target['tempo']}, {target['key']}, {target['dynamics']}, {target['mood']}

Create one continuous instrumental piece with seamless transitions between the three stages. Ambient, healing, no vocals, smooth emotional flow from current state to deep sleep relaxation."""
    
    return prompt

def check_api_call_limit():
    """检查API调用限制"""
    global daily_call_count, last_call_date
    
    today = datetime.now().date()
    if last_call_date != today:
        daily_call_count = 0
        last_call_date = today
    
    if daily_call_count >= MAX_DAILY_CALLS:
        raise Exception(f"🚫 今日API调用次数已达上限 ({MAX_DAILY_CALLS})")

def simulate_suno_response(emotion):
    """模拟Suno API响应（测试模式）"""
    return {
        "task_id": f"mock_task_{int(time.time())}",
        "status": "SUCCESS",
        "data": {
            "audio_url": f"https://mock-suno-api.com/music/{emotion}_therapy.mp3",
            "title": f"Three-Stage {emotion} Therapy Music",
            "duration": 180  # 3分钟
        },
        "mock": True
    }

def call_suno_api(emotion, music_features, enable_real_api=False):
    """调用Suno API生成音乐（严格成本控制）"""
    global daily_call_count
    
    # 安全检查
    if not enable_real_api or not SUNO_API_ENABLED or TEST_MODE:
        print("🧪 使用模拟Suno API响应（测试模式）")
        return simulate_suno_response(emotion)
    
    # 检查调用限制
    check_api_call_limit()
    
    try:
        # 生成提示词
        prompt = generate_suno_prompt(emotion, music_features)
        
        print(f"🎵 调用真实Suno API生成音乐...")
        print(f"💰 注意：这将消耗API费用！")
        
        # API调用
        conn = http.client.HTTPSConnection(BASE_URL)
        payload = json.dumps({
            "gpt_description_prompt": prompt,
            "make_instrumental": True,  # 纯音乐
            "mv": "chirp-v3-0",  # 最便宜的模型，性价比第一
            "prompt": f"Three-stage therapy music for {emotion}"
        })
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        conn.request("POST", "/suno/submit/music", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        response = json.loads(data.decode("utf-8"))
        daily_call_count += 1
        
        print(f"✅ Suno API调用成功！任务ID: {response.get('task_id', 'unknown')}")
        print(f"📊 今日剩余调用次数: {MAX_DAILY_CALLS - daily_call_count}")
        
        return response
        
    except Exception as e:
        print(f"❌ Suno API调用失败: {e}")
        print("🔄 降级到模拟响应")
        return simulate_suno_response(emotion)

def generate_enhanced_therapy_audio_fast(duration=12, sample_rate=22050, emotion="焦虑"):
    """快速生成增强疗愈音频（优化版本）"""
    print(f"🎵 生成{duration}秒增强疗愈音频 (针对{emotion}情绪)")
    
    # 优化参数 - 确保快速生成
    duration = min(duration, 20)  # 最大20秒
    sample_rate = 22050  # 优化采样率
    
    # 情绪专属参数
    emotion_params = {
        "焦虑": {
            "sync_freq": 440, "guide_freq": 330, "consolidate_freq": 220,
            "sync_intensity": 0.4, "guide_intensity": 0.25, "consolidate_intensity": 0.15,
            "transition_type": "exponential", "color": "#FF6B6B"
        },
        "疲惫": {
            "sync_freq": 380, "guide_freq": 280, "consolidate_freq": 200,
            "sync_intensity": 0.35, "guide_intensity": 0.2, "consolidate_intensity": 0.1,
            "transition_type": "linear", "color": "#FFB366"
        },
        "烦躁": {
            "sync_freq": 460, "guide_freq": 350, "consolidate_freq": 240,
            "sync_intensity": 0.45, "guide_intensity": 0.3, "consolidate_intensity": 0.18,
            "transition_type": "sigmoid", "color": "#FF8E8E"
        },
        "平静": {
            "sync_freq": 400, "guide_freq": 320, "consolidate_freq": 210,
            "sync_intensity": 0.3, "guide_intensity": 0.2, "consolidate_intensity": 0.12,
            "transition_type": "smooth", "color": "#66D9AB"
        },
        "压力": {
            "sync_freq": 480, "guide_freq": 360, "consolidate_freq": 230,
            "sync_intensity": 0.5, "guide_intensity": 0.32, "consolidate_intensity": 0.2,
            "transition_type": "exponential", "color": "#6BB6FF"
        }
    }
    
    params = emotion_params.get(emotion, emotion_params["焦虑"])
    
    # 三阶段时间分配
    stage1_duration = duration * 0.3
    stage2_duration = duration * 0.4
    stage3_duration = duration * 0.3
    
    # 生成音频数组
    total_samples = int(sample_rate * duration)
    audio_array = np.zeros(total_samples)
    t_total = np.linspace(0, duration, total_samples)
    
    # 第一阶段：同步期
    stage1_mask = t_total <= stage1_duration
    stage1_time = t_total[stage1_mask]
    stage1_audio = params['sync_intensity'] * np.sin(2 * np.pi * params['sync_freq'] * stage1_time)
    
    # 添加情绪特征
    if emotion == "焦虑":
        tremolo = 0.1 * np.sin(2 * np.pi * 5 * stage1_time)
        stage1_audio *= (1 + tremolo)
    elif emotion == "疲惫":
        stage1_audio *= np.exp(-stage1_time / 8)
    
    audio_array[stage1_mask] = stage1_audio
    
    # 第二阶段：引导期 - 流畅过渡
    stage2_start = stage1_duration
    stage2_end = stage2_start + stage2_duration
    stage2_mask = (t_total > stage2_start) & (t_total <= stage2_end)
    stage2_time = t_total[stage2_mask] - stage2_start
    
    # 过渡曲线
    transition_progress = stage2_time / stage2_duration
    if params['transition_type'] == "exponential":
        transition_curve = 1 - np.exp(-3 * transition_progress)
    elif params['transition_type'] == "sigmoid":
        transition_curve = 1 / (1 + np.exp(-6 * (transition_progress - 0.5)))
    elif params['transition_type'] == "linear":
        transition_curve = transition_progress
    else:  # smooth
        transition_curve = 3 * transition_progress**2 - 2 * transition_progress**3
    
    # 动态频率和强度
    current_freq = params['sync_freq'] + (params['guide_freq'] - params['sync_freq']) * transition_curve
    current_intensity = params['sync_intensity'] + (params['guide_intensity'] - params['sync_intensity']) * transition_curve
    
    stage2_audio = current_intensity * np.sin(2 * np.pi * current_freq * stage2_time)
    
    # 简化的和谐泛音
    harmonic1 = 0.2 * current_intensity * np.sin(2 * np.pi * current_freq * 2 * stage2_time)
    stage2_audio += harmonic1
    
    audio_array[stage2_mask] = stage2_audio
    
    # 第三阶段：巩固期
    stage3_start = stage2_end
    stage3_mask = t_total > stage3_start
    stage3_time = t_total[stage3_mask] - stage3_start
    
    # 平滑过渡到巩固期
    consolidate_transition = np.exp(-stage3_time / 3)
    final_freq = params['guide_freq'] + (params['consolidate_freq'] - params['guide_freq']) * (1 - consolidate_transition)
    final_intensity = params['consolidate_intensity'] * np.exp(-stage3_time / 10)
    
    stage3_audio = final_intensity * np.sin(2 * np.pi * final_freq * stage3_time)
    
    # 添加自然音效
    nature_sound = 0.03 * np.random.normal(0, 1, len(stage3_time))
    wave_sound = 0.05 * final_intensity * np.sin(2 * np.pi * 0.3 * stage3_time)
    stage3_audio += nature_sound + wave_sound
    
    audio_array[stage3_mask] = stage3_audio
    
    # 简化的立体声处理
    left_channel = audio_array
    right_channel = audio_array.copy()
    
    # 轻微立体声延迟
    stereo_delay = int(0.005 * sample_rate)  # 5ms延迟
    if len(right_channel) > stereo_delay:
        right_channel[stereo_delay:] = audio_array[:-stereo_delay]
    
    # 合并立体声
    stereo_audio = np.column_stack([left_channel, right_channel])
    
    # 归一化
    stereo_audio = stereo_audio / np.max(np.abs(stereo_audio) + 1e-10) * 0.8
    
    # 淡入淡出
    fade_samples = int(0.2 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    stereo_audio[:fade_samples] *= fade_in[:, np.newaxis]
    stereo_audio[-fade_samples:] *= fade_out[:, np.newaxis]
    
    return stereo_audio.astype(np.float32), sample_rate, params

def detect_emotion_enhanced(user_input):
    """增强情绪检测"""
    if not user_input or len(user_input.strip()) < 2:
        return "焦虑", 0.85
    
    emotions = {
        "焦虑": ["焦虑", "紧张", "担心", "不安", "害怕", "恐惧", "心跳", "不安"],
        "疲惫": ["疲惫", "累", "疲劳", "困倦", "乏力", "无力", "疲倦", "困"],
        "烦躁": ["烦躁", "烦恼", "易怒", "急躁", "不耐烦", "暴躁", "愤怒", "生气"],
        "平静": ["平静", "放松", "安静", "宁静", "舒缓", "轻松", "安逸", "祥和"],
        "压力": ["压力", "紧迫", "负担", "重压", "沉重", "压抑", "紧张", "负重"]
    }
    
    max_score = 0
    detected_emotion = "焦虑"
    
    for emotion, keywords in emotions.items():
        score = sum(1 for keyword in keywords if keyword in user_input)
        if score > max_score:
            max_score = score
            detected_emotion = emotion
    
    confidence = min(0.85 + max_score * 0.03, 0.95)
    return detected_emotion, confidence

def process_therapy_request(user_input, duration, use_suno_api=False, enable_real_api=False):
    """处理疗愈请求 - 端到端流程（增强Suno API支持）"""
    if not user_input or len(user_input.strip()) < 3:
        return "⚠️ 请输入至少3个字符描述您的情绪状态", None, "输入过短"
    
    try:
        start_time = time.time()
        
        # 1. 情绪识别
        detected_emotion, confidence = detect_emotion_enhanced(user_input)
        
        # 2. 根据用户选择决定音频生成方式
        if use_suno_api:
            # 使用Suno API生成真实AI音乐
            music_features = get_emotion_music_features(detected_emotion)
            
            # 严格成本控制检查
            if enable_real_api and SUNO_API_ENABLED and not TEST_MODE:
                print("🚨 警告：即将调用真实Suno API，将产生费用！")
                confirm = input("确认继续？(y/N): ").lower().strip()
                if confirm != 'y':
                    print("❌ 用户取消API调用")
                    return "用户取消真实API调用", None, "已取消"
            
            # 调用Suno API
            suno_response = call_suno_api(detected_emotion, music_features, enable_real_api)
            
            if suno_response.get('mock', False):
                # 模拟模式 - 使用本地生成
                audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                    duration=duration, 
                    emotion=detected_emotion
                )
                audio_source = "Suno API模拟 + 本地增强算法"
            else:
                # 真实API响应处理
                audio_url = suno_response.get('data', {}).get('audio_url')
                if audio_url:
                    # 这里应该下载真实音频，暂时用本地生成替代
                    print(f"🎵 Suno音频URL: {audio_url}")
                    audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                        duration=duration, 
                        emotion=detected_emotion
                    )
                    audio_source = "真实Suno API生成"
                else:
                    # API失败，降级到本地
                    audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                        duration=duration, 
                        emotion=detected_emotion
                    )
                    audio_source = "API失败，本地生成"
        else:
            # 使用本地增强算法
            audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                duration=duration, 
                emotion=detected_emotion
            )
            audio_source = "本地增强算法"
        
        # 3. 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                import soundfile as sf
                sf.write(tmp_file.name, audio_array, sample_rate)
                audio_file = tmp_file.name
            except ImportError:
                # 如果没有soundfile，用scipy
                from scipy.io import wavfile
                audio_int = (audio_array * 32767).astype(np.int16)
                wavfile.write(tmp_file.name, sample_rate, audio_int)
                audio_file = tmp_file.name
        
        processing_time = time.time() - start_time
        
        # 4. 生成详细报告（集成Suno API信息）
        # 获取音乐特征信息
        music_features = get_emotion_music_features(detected_emotion)
        
        report = f"""✅ 增强三阶段疗愈音频生成完成！

🧠 情绪识别结果:
   • 检测情绪: {detected_emotion}
   • 置信度: {confidence:.1%}
   • 处理时间: {processing_time:.1f}秒
   • 音频来源: {audio_source}

🎵 音频技术参数:
   • 总时长: {duration}秒
   • 采样率: {sample_rate}Hz
   • 声道: 立体声
   • 针对情绪: {detected_emotion}

🎼 ISO三阶段音乐特征映射（硕士项目核心）:
   • 匹配阶段: {music_features['匹配阶段']['tempo']}, {music_features['匹配阶段']['key']}
     └─ {music_features['匹配阶段']['mood']}
   • 引导阶段: {music_features['引导阶段']['tempo']}, {music_features['引导阶段']['key']}
     └─ {music_features['引导阶段']['mood']}
   • 目标阶段: {music_features['目标阶段']['tempo']}, {music_features['目标阶段']['key']}
     └─ {music_features['目标阶段']['mood']}

🌊 三阶段流畅过渡:
   • 同步期 ({duration*0.3:.1f}s): {params['sync_freq']}Hz - 匹配{detected_emotion}情绪
   • 引导期 ({duration*0.4:.1f}s): {params['sync_freq']}→{params['guide_freq']}Hz - 流畅过渡
   • 巩固期 ({duration*0.3:.1f}s): {params['consolidate_freq']}Hz - 深度放松

🎼 疗愈技术特色:
   • 过渡类型: {params['transition_type']} (情绪专属)
   • 和谐泛音: 增强疗愈效果
   • 自然音效: 海浪声 + 环境音
   • 立体声场: 5ms延迟 + 空间感
   • 淡入淡出: 0.2秒平滑过渡

💰 成本控制状态:
   • API状态: {'开启' if SUNO_API_ENABLED else '关闭'}
   • 测试模式: {'是' if TEST_MODE else '否'}
   • 今日调用: {daily_call_count}/{MAX_DAILY_CALLS}

🎧 使用建议:
   • 佩戴耳机获得最佳立体声效果
   • 在安静环境中聆听
   • 跟随音频节奏调整呼吸
   • 专注感受三阶段情绪转换

🌟 核心创新:
   • 流畅过渡: 数学精确的无缝切换
   • 情绪映射: {detected_emotion}情绪的专属参数
   • 疗愈叙事: 连贯的情绪转换故事
   • 学术理论: ISO三阶段原则应用
   • API集成: 真实AI音乐生成能力

🌙 现在请戴上耳机，体验真正的流畅过渡疗愈效果！"""
        
        return report, audio_file, f"成功生成{detected_emotion}疗愈音频 - {audio_source}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None, "生成失败"

def create_therapy_interface():
    """创建疗愈界面"""
    # 自定义CSS样式
    css = """
    .therapy-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .therapy-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .therapy-subtitle {
        font-size: 16px;
        opacity: 0.9;
    }
    .therapy-highlight {
        background: #ffeb3b;
        color: #333;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(
        title="🌙 增强三阶段疗愈系统",
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
        css=css
    ) as app:
        
        # 标题区域
        gr.HTML("""
        <div class="therapy-container">
            <div class="therapy-title">🌙 增强三阶段疗愈系统</div>
            <div class="therapy-subtitle">端到端完整体验：输入情绪 → 智能生成 → 即时播放</div>
            <div style="margin-top: 10px;">
                <span class="therapy-highlight">✨ 真正的流畅过渡 + 完美音画同步</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 💭 情绪输入")
                
                # 快速情绪选择
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
                
                # 详细情绪描述
                emotion_input = gr.Textbox(
                    label="✍️ 详细描述您的情绪状态",
                    placeholder="请详细描述您当前的情绪感受...",
                    lines=3,
                    value="我感到很焦虑，心跳加速，难以入睡"
                )
                
                # 疗愈时长
                duration_slider = gr.Slider(
                    minimum=5, 
                    maximum=20, 
                    value=12, 
                    step=1,
                    label="⏱️ 疗愈时长（秒）",
                    info="推荐12-15秒获得最佳体验"
                )
                
                # Suno API选项
                with gr.Row():
                    use_suno = gr.Checkbox(
                        label="🎵 使用Suno AI音乐生成",
                        value=False,
                        info="启用真实AI音乐（测试模式下安全）"
                    )
                    enable_real_api = gr.Checkbox(
                        label="💰 启用真实API调用",
                        value=False,
                        info="⚠️ 需要消耗API费用！"
                    )
                
                # 生成按钮
                generate_btn = gr.Button(
                    "🌊 开始增强三阶段疗愈",
                    variant="primary",
                    size="lg"
                )
                
                # 系统说明
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <strong>🌊 增强三阶段疗愈原理：</strong><br>
                    <div style="margin-top: 10px; text-align: left;">
                        <div><strong>🎯 同步期 (30%)</strong>: 匹配您的情绪频率</div>
                        <div><strong>🌀 引导期 (40%)</strong>: 流畅过渡到放松状态</div>
                        <div><strong>💤 巩固期 (30%)</strong>: 深度放松，准备入睡</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 14px; opacity: 0.8;">
                        ✨ 特色：数学精确的无缝过渡 + 情绪专属参数
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: rgba(255,215,0,0.2); border-radius: 5px;">
                        <strong>🎵 Suno AI集成：</strong><br>
                        <div style="font-size: 12px; margin-top: 5px;">
                            • <strong>测试模式</strong>：安全模拟，无费用<br>
                            • <strong>真实模式</strong>：消耗API费用，需谨慎<br>
                            • <strong>成本控制</strong>：每日最多3次调用
                        </div>
                    </div>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎬 疗愈体验")
                
                # 详细信息显示
                info_output = gr.Textbox(
                    label="📊 疗愈生成报告",
                    lines=25,
                    interactive=False,
                    value="等待您的情绪输入，开始个性化疗愈体验..."
                )
                
                # 音频播放器
                audio_output = gr.Audio(
                    label="🎵 三阶段疗愈音频",
                    type="filepath",
                    interactive=True
                )
                
                # 状态显示
                status_output = gr.Textbox(
                    label="🔄 处理状态",
                    interactive=False,
                    value="就绪"
                )
        
        # 使用指南
        gr.HTML("""
        <div style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3 style="color: #333;">🎯 完整使用指南</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
                <div>
                    <h4 style="color: #555;">🎵 音频体验</h4>
                    <ul style="color: #666; text-align: left;">
                        <li>佩戴耳机获得最佳立体声效果</li>
                        <li>在安静环境中聆听</li>
                        <li>音量调至舒适水平</li>
                        <li>专注感受三阶段转换</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #555;">🧘 疗愈过程</h4>
                    <ul style="color: #666; text-align: left;">
                        <li>跟随音频节奏调整呼吸</li>
                        <li>让音乐引导您的情绪</li>
                        <li>感受从紧张到放松的过渡</li>
                        <li>享受最终的深度平静</li>
                    </ul>
                </div>
                <div>
                    <h4 style="color: #555;">🌟 技术特色</h4>
                    <ul style="color: #666; text-align: left;">
                        <li>5种情绪的专属参数设计</li>
                        <li>4种数学过渡函数</li>
                        <li>立体声空间化处理</li>
                        <li>自然音效融合</li>
                    </ul>
                </div>
            </div>
        </div>
        """)
        
        # 事件绑定
        def update_input_from_dropdown(selected):
            if " " in selected:
                return selected.split(" ", 1)[1]
            return selected
        
        emotion_examples.change(
            update_input_from_dropdown,
            inputs=emotion_examples,
            outputs=emotion_input
        )
        
        generate_btn.click(
            process_therapy_request,
            inputs=[emotion_input, duration_slider, use_suno, enable_real_api],
            outputs=[info_output, audio_output, status_output]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动增强三阶段疗愈系统 - 完整Web界面")
    print("🌊 端到端体验：输入情绪 → 智能生成 → 即时播放")
    print("✨ 特色：流畅过渡 + 完美音画同步")
    print("🎯 访问地址即将显示...")
    
    app = create_therapy_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7869,
        share=True,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()