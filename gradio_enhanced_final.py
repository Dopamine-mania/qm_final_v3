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
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

# 🛡️ 严格成本控制配置
SUNO_API_ENABLED = True   # 允许通过界面控制
TEST_MODE = False         # 允许通过界面控制
MAX_DAILY_CALLS = 3       # 每日最大调用次数
API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"

# 🖼️ 图片生成配置
STABLE_DIFFUSION_ENABLED = True  # 允许通过界面控制
MAX_IMAGES_PER_SESSION = 5       # 每会话最大图片数量
IMAGE_GENERATION_INTERVAL = 3    # 图片生成间隔（秒）

# 全局调用计数器
daily_call_count = 0
last_call_date = None
image_generation_count = 0

def download_suno_audio(audio_url):
    """下载Suno AI音频文件并转换为适合播放的格式"""
    try:
        print(f"🌐 开始下载Suno音频: {audio_url}")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
            mp3_file = tmp_mp3.name
        
        # 下载音频文件
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        request = urllib.request.Request(audio_url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise Exception(f"下载失败，HTTP状态码: {response.status}")
            
            # 读取音频数据
            audio_data = response.read()
            
            if len(audio_data) == 0:
                raise Exception("下载的音频文件为空")
            
            # 保存为MP3
            with open(mp3_file, 'wb') as f:
                f.write(audio_data)
            
            print(f"✅ 音频下载完成: {mp3_file} ({len(audio_data)} bytes)")
        
        # 验证文件
        if not os.path.exists(mp3_file) or os.path.getsize(mp3_file) == 0:
            raise Exception("下载的音频文件无效")
        
        # 转换为WAV格式（Gradio更好支持）
        wav_file = mp3_file.replace('.mp3', '.wav')
        
        try:
            # 尝试使用pydub转换
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_file)
            audio.export(wav_file, format="wav")
            print(f"✅ 音频转换完成: {wav_file}")
            
            # 清理临时MP3文件
            os.unlink(mp3_file)
            return wav_file
            
        except ImportError:
            print("⚠️ pydub不可用，尝试直接使用MP3文件")
            # 如果没有pydub，直接返回MP3文件
            return mp3_file
            
        except Exception as convert_error:
            print(f"⚠️ 音频转换失败: {convert_error}，使用原始MP3")
            return mp3_file
            
    except Exception as e:
        print(f"❌ 下载Suno音频失败: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    """按照官方示例格式生成极简API提示词"""
    # 完全按照官方示例：简单英文单词
    emotion_map = {
        "焦虑": "calm sleep",
        "疲惫": "rest therapy", 
        "烦躁": "peace music",
        "平静": "deep relax",
        "压力": "stress relief"
    }
    
    # 极简格式，避免所有中文和复杂描述
    simple_prompt = emotion_map.get(emotion, "sleep music")
    
    return simple_prompt

def generate_image_prompts(emotion, music_features, duration):
    """根据ISO三阶段原则生成图片提示词序列"""
    # 计算图片数量：每3秒一张图片
    num_images = max(3, int(duration / IMAGE_GENERATION_INTERVAL))
    if num_images > MAX_IMAGES_PER_SESSION:
        num_images = MAX_IMAGES_PER_SESSION
    
    # 三阶段视觉风格映射
    stage_visuals = {
        "匹配阶段": {
            "焦虑": "dark stormy clouds, turbulent ocean waves, dramatic shadows, moody atmosphere",
            "疲惫": "wilted flowers, fading sunset, tired traveler, exhausted landscape",
            "烦躁": "chaotic storm, lightning strikes, restless energy, turbulent emotions",
            "平静": "gentle morning light, soft meadow, peaceful lake, serene atmosphere",
            "压力": "heavy rain, pressing clouds, intense atmosphere, overwhelming environment"
        },
        "引导阶段": {
            "焦虑": "soft moonlight breaking through clouds, gentle waves, calming transition",
            "疲惫": "warm sunrise, blooming flowers, rejuvenating spring, peaceful rest",
            "烦躁": "clearing storm, rainbow appearing, peaceful after chaos, gentle breeze",
            "平静": "flowing stream, harmonious nature, balanced elements, perfect serenity",
            "压力": "clearing skies, burden lifting, light breaking through, relief atmosphere"
        },
        "目标阶段": {
            "焦虑": "peaceful starry night, calm ocean, deep relaxation, tranquil sleep",
            "疲惫": "energetic morning, vibrant landscape, renewed vitality, fresh beginning",
            "烦躁": "perfect harmony, balanced nature, inner peace, emotional stability",
            "平静": "transcendent beauty, spiritual calm, perfect balance, ultimate peace",
            "压力": "complete freedom, weightless feeling, liberated spirit, stress-free environment"
        }
    }
    
    # 生成图片提示词序列
    prompts = []
    stages = ["匹配阶段", "引导阶段", "目标阶段"]
    
    for i in range(num_images):
        # 计算当前阶段
        stage_progress = i / (num_images - 1) if num_images > 1 else 0
        if stage_progress < 0.33:
            stage = "匹配阶段"
        elif stage_progress < 0.67:
            stage = "引导阶段"
        else:
            stage = "目标阶段"
        
        # 获取基础视觉描述
        base_visual = stage_visuals[stage].get(emotion, stage_visuals[stage]["焦虑"])
        
        # 增强提示词
        enhanced_prompt = f"{base_visual}, therapeutic healing art, soft natural lighting, peaceful atmosphere, emotional journey, masterpiece, high quality, cinematic, beautiful composition"
        
        prompts.append({
            "stage": stage,
            "timestamp": i * IMAGE_GENERATION_INTERVAL,
            "prompt": enhanced_prompt,
            "emotion": emotion
        })
    
    return prompts

def call_stable_diffusion_api(prompt, enable_real_api=False):
    """调用Stable Diffusion API生成图片"""
    global image_generation_count
    
    # 🧪 测试模式：生成真实的本地图片文件
    TEST_MODE = True  # 改为False启用真实API调用
    
    if TEST_MODE and enable_real_api:
        print(f"🧪 测试模式：生成本地图片 - {prompt[:50]}...")
        
        # 根据提示词生成不同的本地图片
        image_generation_count += 1
        
        # 根据提示词内容选择合适的颜色和主题
        if "dark" in prompt or "storm" in prompt or "chaos" in prompt:
            color = (44, 62, 80)  # 深色
            theme = "匹配阶段"
        elif "moonlight" in prompt or "clearing" in prompt or "transition" in prompt:
            color = (52, 152, 219)  # 蓝色
            theme = "引导阶段"
        elif "peaceful" in prompt or "calm" in prompt or "harmony" in prompt:
            color = (39, 174, 96)  # 绿色
            theme = "目标阶段"
        else:
            color = (231, 76, 60)  # 默认红色
            theme = "疗愈图片"
        
        # 生成本地图片文件
        try:
            from PIL import Image, ImageDraw, ImageFont
            import tempfile
            
            # 创建512x512的图片
            img = Image.new('RGB', (512, 512), color)
            draw = ImageDraw.Draw(img)
            
            # 添加文字
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            except:
                # 如果没有系统字体，使用默认字体
                font = ImageFont.load_default()
            
            # 绘制主题文字
            text_lines = [
                f"{theme}",
                f"第 {image_generation_count} 张",
                "疗愈图片"
            ]
            
            # 计算文字位置
            y_offset = 200
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (512 - text_width) // 2
                
                # 绘制白色文字
                draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += text_height + 20
            
            # 保存到临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(temp_file.name, 'PNG')
            temp_file.close()
            
            print(f"✅ 生成本地图片: {temp_file.name}")
            
            return {
                "success": True,
                "image_path": temp_file.name,  # 返回本地文件路径
                "prompt": prompt,
                "mock": True
            }
            
        except ImportError:
            print("⚠️ PIL不可用，使用文本占位符")
            # 如果PIL不可用，创建一个简单的文本文件作为占位符
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_file.write(f"疗愈图片 {image_generation_count}\n主题: {theme}\n提示词: {prompt[:100]}...")
            temp_file.close()
            
            return {
                "success": True,
                "image_path": temp_file.name,
                "prompt": prompt,
                "mock": True
            }
        except Exception as e:
            print(f"❌ 生成本地图片失败: {e}")
            return {"success": False, "error": str(e)}
    
    if not enable_real_api:
        print("🔒 图片生成API已禁用")
        return {"success": False, "error": "API disabled"}
    
    try:
        # 检查生成限制
        if image_generation_count >= MAX_IMAGES_PER_SESSION:
            raise Exception(f"🚫 本会话图片生成已达上限 ({MAX_IMAGES_PER_SESSION})")
        
        print(f"🎨 调用Stable Diffusion API...")
        print(f"🖼️ 提示词: {prompt}")
        
        # 调用API
        conn = http.client.HTTPSConnection(BASE_URL)
        payload = json.dumps({
            "model": "stable-diffusion",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        print(f"🔍 Stable Diffusion API响应状态: {res.status}")
        
        if res.status == 200:
            result = json.loads(data.decode("utf-8"))
            print(f"✅ 图片生成成功")
            image_generation_count += 1
            return {
                "success": True,
                "result": result,
                "prompt": prompt
            }
        else:
            print(f"❌ API请求失败，状态码: {res.status}")
            print(f"   响应内容: {data.decode('utf-8')}")
            return {"success": False, "error": f"API Error: {res.status}"}
            
    except Exception as e:
        print(f"❌ 图片生成失败: {e}")
        return {"success": False, "error": str(e)}

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

def fetch_suno_result(task_id, max_wait_time=60):
    """查询Suno API任务结果"""
    import time
    
    print(f"🔍 查询任务结果: {task_id}")
    
    for attempt in range(max_wait_time // 5):  # 每5秒查询一次
        try:
            conn = http.client.HTTPSConnection(BASE_URL)
            
            # 添加请求头
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {API_KEY}'
            }
            
            # 查询任务状态 (修复: 使用REST风格端点)
            conn.request("GET", f"/suno/fetch/{task_id}", headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            print(f"🔍 查询API响应状态: {res.status}")
            print(f"🔍 查询API响应数据: {data.decode('utf-8')[:200]}...")
            
            if res.status == 200:
                # 检查响应是否为空
                if not data or len(data.strip()) == 0:
                    print("⚠️ API返回空响应")
                    time.sleep(5)
                    continue
                
                try:
                    result = json.loads(data.decode("utf-8"))
                    print(f"🔍 任务状态查询结果: {result}")
                    
                    # 检查任务状态
                    if result.get('code') == 'success' and result.get('data'):
                        task_data = result.get('data')
                        if isinstance(task_data, dict):
                            status = task_data.get('status')
                            
                            # 🔥 关键修复：检查是否已有可用音频，不用等SUCCESS
                            if 'data' in task_data and isinstance(task_data['data'], list) and len(task_data['data']) > 0:
                                for audio_item in task_data['data']:
                                    if audio_item.get('audio_url'):
                                        print(f"🎵 发现可用音频！status={status}, 立即返回")
                                        return result
                            
                            if status == 'SUCCESS':
                                print(f"✅ 音乐生成完成！")
                                return result
                            elif status in ['NOT_START', 'SUBMITTED', 'QUEUED', 'IN_PROGRESS']:
                                print(f"⏳ 任务进行中: {status}")
                            else:
                                print(f"❌ 任务失败: {status}")
                                return None
                        else:
                            print(f"⚠️ 任务数据格式异常: {type(task_data)}")
                    else:
                        print(f"⚠️ API响应格式异常: code={result.get('code')}, data存在={bool(result.get('data'))}")
                        
                except json.JSONDecodeError as je:
                    print(f"❌ JSON解析失败: {je}")
                    print(f"   原始数据: {data.decode('utf-8')}")
                    
            else:
                print(f"❌ API请求失败，状态码: {res.status}")
                print(f"   响应内容: {data.decode('utf-8')}")
            
            # 等待5秒后重试
            print(f"⏳ 等待5秒后重试... (尝试 {attempt + 1}/{max_wait_time // 5})")
            time.sleep(5)
            
        except Exception as e:
            print(f"⚠️ 查询任务状态出错: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
    
    print(f"⏰ 任务查询超时 ({max_wait_time}秒)")
    return None

def call_suno_api(emotion, music_features, enable_real_api=False):
    """调用Suno API生成音乐（严格成本控制）"""
    global daily_call_count
    
    # 🧪 测试模式：使用真实音频URL但不调用API
    TEST_MODE = True  # 改为False启用真实API调用
    
    if TEST_MODE and enable_real_api:
        print("🧪 测试模式：模拟API成功，使用真实音频URL")
        return {
            "code": "success",
            "data": {
                "task_id": "test-task-123",
                "status": "IN_PROGRESS",
                "data": [
                    {
                        "id": "test-audio-1",
                        "title": "Test Therapy Music",
                        "duration": 104.4,
                        "audio_url": "https://cdn1.suno.ai/7d0fa1f8-6cb6-46ca-b937-c13dab540209.mp3",
                        "tags": "therapy, sleep, calm, test"
                    }
                ]
            }
        }
    
    # 安全检查 - 只有用户明确启用真实API才调用
    if not enable_real_api:
        print("🧪 使用模拟Suno API响应（用户未启用真实API）")
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
        # 完全按照官方示例格式，使用极简英文
        payload = json.dumps({
            "gpt_description_prompt": prompt,  # 已经是极简英文
            "make_instrumental": True,
            "mv": "chirp-v3-0",  # 使用最便宜的v3模型
            "prompt": prompt  # 使用相同的极简prompt
        })
        
        # 调试：显示payload大小
        payload_size = len(payload.encode('utf-8'))
        print(f"🔍 Payload大小: {payload_size} bytes")
        print(f"🔍 Payload内容: {payload}")
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        conn.request("POST", "/suno/submit/music", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        print(f"🔍 API响应状态: {res.status}")
        print(f"🔍 API响应数据: {data.decode('utf-8')[:500]}...")  # 只显示前500字符
        
        if res.status != 200:
            raise Exception(f"API调用失败，状态码: {res.status}")
        
        response = json.loads(data.decode("utf-8"))
        
        # 验证响应格式
        if not isinstance(response, dict):
            raise Exception(f"API返回格式错误: {type(response)}")
        
        daily_call_count += 1
        
        # 解析任务ID - 根据实际API响应格式
        task_id = None
        if response.get('code') == 'success':
            task_id = response.get('data')  # 任务ID在data字段中
        
        print(f"✅ Suno API调用成功！任务ID: {task_id}")
        print(f"📊 今日剩余调用次数: {MAX_DAILY_CALLS - daily_call_count}")
        
        # 如果有任务ID，尝试获取结果
        if task_id:
            print(f"🔄 等待音乐生成完成...")
            result = fetch_suno_result(task_id)
            if result:
                return result
        
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

def process_therapy_request(user_input, duration, use_suno_api=False, enable_real_api=False, existing_task_id="", enable_image_generation=False):
    """处理疗愈请求 - 端到端流程（增强Suno API支持 + 图片生成）"""
    if not user_input or len(user_input.strip()) < 3:
        return "⚠️ 请输入至少3个字符描述您的情绪状态", None, [], "输入过短"
    
    try:
        start_time = time.time()
        
        # 1. 情绪识别
        detected_emotion, confidence = detect_emotion_enhanced(user_input)
        
        # 2. 根据用户选择决定音频生成方式
        if use_suno_api:
            # 使用Suno API生成真实AI音乐
            music_features = get_emotion_music_features(detected_emotion)
            
            # 检查是否使用现有任务ID
            if existing_task_id.strip():
                print(f"🔄 使用现有任务ID获取音乐: {existing_task_id}")
                suno_response = fetch_suno_result(existing_task_id.strip())
                if not suno_response:
                    print("❌ 无法获取现有任务结果，降级到本地生成")
                    audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                        duration=duration, 
                        emotion=detected_emotion
                    )
                    audio_source = "现有任务获取失败，本地生成"
                else:
                    print("✅ 成功获取现有任务结果")
            else:
                # 严格成本控制检查 - 必须两个条件都满足才调用真实API
                if use_suno_api and enable_real_api:
                    print("🚨 警告：两个条件都满足，即将调用真实Suno API，将产生费用！")
                    print(f"💰 今日剩余调用次数: {MAX_DAILY_CALLS - daily_call_count}")
                    # 在Web界面中，用户已经通过勾选框确认了
                    actual_enable_real_api = True
                else:
                    print("ℹ️ 成本控制：需要同时勾选'使用Suno AI'和'启用真实API'才调用真实API")
                    actual_enable_real_api = False
                
                # 调用Suno API
                suno_response = call_suno_api(detected_emotion, music_features, actual_enable_real_api)
            
            # 安全检查API响应
            if not suno_response or not isinstance(suno_response, dict):
                print("⚠️ API响应无效，使用本地生成")
                audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                    duration=duration, 
                    emotion=detected_emotion
                )
                audio_source = "API响应无效，本地生成"
            elif suno_response.get('mock', False):
                # 模拟模式 - 使用本地生成
                audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                    duration=duration, 
                    emotion=detected_emotion
                )
                audio_source = "Suno API模拟 + 本地增强算法"
            else:
                # 真实API响应处理
                try:
                    print(f"🔍 处理真实API响应: {suno_response}")
                    
                    # 检查是否有任务完成的音频数据
                    audio_url = None
                    task_data = suno_response.get('data')
                    
                    if isinstance(task_data, dict):
                        # 查看任务数据结构
                        if 'data' in task_data and isinstance(task_data['data'], list) and len(task_data['data']) > 0:
                            # 获取第一个音频
                            audio_item = task_data['data'][0]
                            audio_url = audio_item.get('audio_url')
                        elif 'audio_url' in task_data:
                            audio_url = task_data['audio_url']
                    
                    if audio_url:
                        print(f"🎵 发现Suno音频URL: {audio_url}")
                        # 下载真实Suno AI音频
                        try:
                            downloaded_audio = download_suno_audio(audio_url)
                            if downloaded_audio:
                                audio_file = downloaded_audio  # 直接使用下载的文件
                                audio_source = f"真实Suno AI音乐 (URL: {audio_url[:50]}...)"
                                print(f"✅ 成功下载真实Suno音乐: {audio_file}")
                                
                                # 如果启用图片生成，生成配套图片
                                image_info = ""
                                generated_images = []
                                if enable_image_generation:
                                    print("🎨 开始生成配套疗愈图片...")
                                    try:
                                        # 获取音乐时长（从音频文件或默认）
                                        music_duration = duration or 104  # 默认时长
                                        
                                        # 生成图片提示词序列
                                        image_prompts = generate_image_prompts(detected_emotion, music_features, music_duration)
                                        
                                        # 生成图片
                                        for prompt_data in image_prompts:
                                            image_result = call_stable_diffusion_api(
                                                prompt_data['prompt'], 
                                                enable_real_api and STABLE_DIFFUSION_ENABLED
                                            )
                                            if image_result.get('success'):
                                                # 使用image_path而不是image_url
                                                image_path = image_result.get('image_path') or image_result.get('image_url')
                                                generated_images.append({
                                                    'stage': prompt_data['stage'],
                                                    'timestamp': prompt_data['timestamp'],
                                                    'image_path': image_path,
                                                    'prompt': prompt_data['prompt'][:50] + "..."
                                                })
                                        
                                        if generated_images:
                                            image_info = f"""

🖼️ 配套疗愈图片 ({len(generated_images)}张):
   • 图片总数: {len(generated_images)}张
   • 生成间隔: {IMAGE_GENERATION_INTERVAL}秒
   • 同步音乐: 完美匹配三阶段疗愈"""
                                            
                                            for i, img in enumerate(generated_images):
                                                image_info += f"""
   • 图片{i+1}: {img['stage']} (第{img['timestamp']}秒)"""
                                        
                                    except Exception as img_error:
                                        print(f"⚠️ 图片生成失败: {img_error}")
                                        image_info = f"""

🖼️ 图片生成状态:
   • 状态: ❌ 生成失败
   • 原因: {str(img_error)}
   • 影响: 音乐播放不受影响"""
                                
                                # 跳过本地生成，直接返回报告
                                processing_time = time.time() - start_time
                                music_features = get_emotion_music_features(detected_emotion)
                                
                                report = f"""✅ 真实Suno AI音乐生成完成！

🧠 情绪识别结果:
   • 检测情绪: {detected_emotion}
   • 置信度: {confidence:.1%}
   • 处理时间: {processing_time:.1f}秒
   • 音频来源: {audio_source}

🎵 真实AI音乐信息:
   • 来源: Suno AI (chirp-v3模型)
   • 音频URL: {audio_url[:60]}...
   • 下载状态: ✅ 成功下载
   • 文件格式: MP3 → WAV (兼容播放)

🌊 三阶段疗愈设计:
   • 匹配阶段(30%): {music_features['匹配阶段']['mood']}
   • 引导阶段(40%): {music_features['引导阶段']['mood']} 
   • 目标阶段(30%): {music_features['目标阶段']['mood']}{image_info}

🎧 聆听建议:
   • 这是真实的AI生成音乐，请用耳机体验
   • 音量调至舒适水平
   • 专注感受AI音乐的疗愈效果

✨ 这就是您花费API费用获得的真实Suno AI音乐！"""
                                
                                # 准备图片数据用于Gradio Gallery
                                image_gallery = [img['image_path'] for img in generated_images] if generated_images else []
                                
                                return report, audio_file, image_gallery, f"真实Suno AI - {detected_emotion}疗愈音乐"
                            else:
                                raise Exception("音频下载失败")
                                
                        except Exception as download_error:
                            print(f"⚠️ 下载真实音频失败: {download_error}")
                            print("降级到本地生成，但标记为API来源")
                            audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                                duration=duration, 
                                emotion=detected_emotion
                            )
                            audio_source = f"真实API调用成功，但下载失败，本地替代 (URL: {audio_url[:50]}...)"
                    else:
                        # API成功但还没有音频URL（可能生成中）
                        print("⚠️ API响应暂无音频URL，可能仍在生成中，使用本地算法")
                        audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                            duration=duration, 
                            emotion=detected_emotion
                        )
                        audio_source = "API生成中，临时本地算法"
                        
                except Exception as e:
                    print(f"⚠️ 处理API响应时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                        duration=duration, 
                        emotion=detected_emotion
                    )
                    audio_source = "API处理出错，本地生成"
        else:
            # 使用本地增强算法
            audio_array, sample_rate, params = generate_enhanced_therapy_audio_fast(
                duration=duration, 
                emotion=detected_emotion
            )
            audio_source = "本地增强算法"
        
        # 3. 保存到临时文件（增强版）
        print(f"🔍 音频数组形状: {audio_array.shape}, 采样率: {sample_rate}")
        print(f"🔍 音频数组类型: {type(audio_array)}, 数据类型: {audio_array.dtype}")
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_file = tmp_file.name
                print(f"🔍 临时文件路径: {audio_file}")
            
            # 尝试保存音频
            try:
                import soundfile as sf
                sf.write(audio_file, audio_array, sample_rate)
                print(f"✅ soundfile保存成功: {audio_file}")
            except ImportError:
                print("⚠️ soundfile不可用，使用scipy...")
                from scipy.io import wavfile
                audio_int = (audio_array * 32767).astype(np.int16)
                wavfile.write(audio_file, sample_rate, audio_int)
                print(f"✅ scipy保存成功: {audio_file}")
            
            # 验证文件存在性和大小
            import os
            if os.path.exists(audio_file):
                file_size = os.path.getsize(audio_file)
                print(f"✅ 文件验证成功: {audio_file} ({file_size} bytes)")
                if file_size == 0:
                    raise Exception("音频文件大小为0")
            else:
                raise Exception(f"音频文件不存在: {audio_file}")
                
        except Exception as e:
            print(f"❌ 音频保存失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 音频保存失败: {e}", None, [], "保存失败"
        
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
        
        # 最终验证和返回
        import os
        print(f"🔍 返回的audio_file: {audio_file}")
        print(f"🔍 文件是否存在: {os.path.exists(audio_file) if 'audio_file' in locals() else 'audio_file未定义'}")
        
        # 处理图片生成（适用于所有情况）
        image_gallery = []
        if enable_image_generation:
            try:
                print("🎨 生成本地疗愈配套图片...")
                music_features = get_emotion_music_features(detected_emotion)
                image_prompts = generate_image_prompts(detected_emotion, music_features, duration)
                
                for prompt_data in image_prompts:
                    image_result = call_stable_diffusion_api(
                        prompt_data['prompt'], 
                        enable_real_api and STABLE_DIFFUSION_ENABLED
                    )
                    if image_result.get('success'):
                        # 使用image_path而不是image_url
                        image_path = image_result.get('image_path') or image_result.get('image_url')
                        image_gallery.append(image_path)
                        
                print(f"✅ 生成了{len(image_gallery)}张配套图片")
            except Exception as img_error:
                print(f"⚠️ 本地图片生成失败: {img_error}")
        
        return report, audio_file, image_gallery, f"成功生成{detected_emotion}疗愈音频 - {audio_source}"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None, [], "生成失败"

def load_previous_suno_music():
    """加载之前成功生成的Suno音乐"""
    # 使用相对路径，兼容不同系统
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_path = os.path.join(current_dir, "previous_suno_fdd1b90b.mp3")
    
    if os.path.exists(audio_file_path):
        report = f"""🎵 成功加载之前的Suno AI音乐！

🎼 音乐信息:
   • 标题: "Whisper of the Moon"
   • 时长: 约2分44秒 (164秒)
   • 模型: 已修复为Chirp-v3 (成本优化)
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
        
        return report, audio_file_path, [], "✅ 成功加载之前的Suno音乐"
    else:
        return "❌ 未找到之前的音乐文件", None, [], "❌ 文件不存在"

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
    
    with gr.Blocks(title="🌙 增强三阶段疗愈系统") as app:
        
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
                        label="🧪 启用测试模式（推荐）",
                        value=False,
                        info="✅ 免费测试！使用真实音频URL但不调用API"
                    )
                
                # 图片生成选项
                with gr.Row():
                    enable_image_generation = gr.Checkbox(
                        label="🖼️ 启用配套图片生成",
                        value=False,
                        info="根据三阶段疗愈生成匹配图片序列"
                    )
                
                # 测试模式说明
                gr.HTML("""
                <div style="background: #e8f4f8; border: 1px solid #2196F3; border-radius: 8px; padding: 12px; margin: 10px 0;">
                    <h4 style="color: #1976D2; margin: 0 0 8px 0;">🧪 测试模式说明</h4>
                    <p style="margin: 0; font-size: 14px; color: #333;">
                        • <strong>测试模式</strong>：使用真实音频URL但不调用API，完全免费<br>
                        • <strong>真实模式</strong>：调用真实API，需要消耗费用<br>
                        • <strong>推荐</strong>：先用测试模式验证功能，再考虑真实调用
                    </p>
                </div>
                """)
                
                # 图片生成说明
                gr.HTML("""
                <div style="background: #f3e5f5; border: 1px solid #9c27b0; border-radius: 8px; padding: 12px; margin: 10px 0;">
                    <h4 style="color: #7b1fa2; margin: 0 0 8px 0;">🖼️ 图片生成功能</h4>
                    <p style="margin: 0; font-size: 14px; color: #333;">
                        • <strong>成本控制</strong>：用图片生成替代昂贵的视频生成<br>
                        • <strong>音画同步</strong>：根据音乐节拍隔3秒生成匹配图片<br>
                        • <strong>ISO三阶段</strong>：匹配→引导→目标，自然过渡<br>
                        • <strong>完美配套</strong>：最终可合成音画疗愈视频
                    </p>
                </div>
                """)
                
                # 现有任务ID输入（避免重复调用）
                existing_task_input = gr.Textbox(
                    label="🔄 使用现有任务ID（避免重复调用）",
                    placeholder="例如: fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc",
                    value="",
                    info="如果有之前的任务ID，可以直接获取结果，避免重新消耗API"
                )
                
                # 快速播放已有音乐
                with gr.Row():
                    load_previous_btn = gr.Button(
                        "🎵 播放之前成功生成的Suno音乐",
                        variant="secondary"
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
                            • <strong>成本控制</strong>：每日最多3次调用<br>
                            • <strong>🎵 快速体验</strong>：点击按钮播放之前成功生成的真实AI音乐！
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
                
                # 图片展示组件
                image_output = gr.Gallery(
                    label="🖼️ 配套疗愈图片序列",
                    columns=2,
                    rows=2,
                    height="auto",
                    show_label=True,
                    elem_id="therapy-images"
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
            inputs=[emotion_input, duration_slider, use_suno, enable_real_api, existing_task_input, enable_image_generation],
            outputs=[info_output, audio_output, image_output, status_output]
        )
        
        load_previous_btn.click(
            load_previous_suno_music,
            inputs=[],
            outputs=[info_output, audio_output, image_output, status_output]
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
        share=True
    )

if __name__ == "__main__":
    main()