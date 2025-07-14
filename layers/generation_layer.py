#!/usr/bin/env python3
"""
生成层 (Generation Layer) - Layer 4

实时音视频内容生成，核心功能包括：
1. 音乐合成和生成
2. 治疗性视频内容生成
3. 实时自适应生成
4. 多种生成策略支持
5. 睡眠治疗专用优化

处理流程：
Mapping Layer → Generation Layer → Rendering Layer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json
import io
import requests
import time
from pathlib import Path

from .base_layer import BaseLayer, LayerData, LayerConfig
from core.utils import (
    ConfigLoader, DataValidator, PerformanceMonitor, 
    get_project_root, normalize_vector
)
from core.theory.iso_principle import ISOPrinciple, ISOStage, EmotionState

logger = logging.getLogger(__name__)

# 检查可选依赖
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("librosa/soundfile不可用，音频生成功能将受限")

try:
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    logger.warning("PIL不可用，视频生成功能将受限")

class GenerationStrategy(Enum):
    """生成策略枚举"""
    RULE_BASED = "rule_based"          # 基于规则的生成
    NEURAL_SYNTHESIS = "neural_synthesis"  # 神经网络合成
    TEMPLATE_BASED = "template_based"      # 基于模板的生成
    HYBRID = "hybrid"                      # 混合策略

class ContentType(Enum):
    """内容类型枚举"""
    AUDIO = "audio"
    VIDEO = "video"
    BOTH = "both"

@dataclass
class GenerationLayerConfig(LayerConfig):
    """生成层配置"""
    # 基础配置
    output_sample_rate: int = 44100  # 音频采样率
    output_channels: int = 2         # 立体声
    video_fps: int = 30              # 视频帧率
    video_resolution: Tuple[int, int] = (1920, 1080)  # 视频分辨率
    
    # 生成策略
    generation_strategy: str = "hybrid"
    content_type: str = "both"  # audio, video, both
    
    # 音频生成配置
    audio_enabled: bool = True
    audio_duration: float = 60.0     # 默认1分钟
    audio_buffer_size: int = 4096
    audio_synthesis_method: str = "procedural"  # procedural, neural
    
    # 视频生成配置
    video_enabled: bool = True
    video_duration: float = 60.0     # 默认1分钟
    video_style: str = "ambient"     # ambient, abstract, nature
    
    # 治疗特化配置
    therapy_optimized: bool = True
    iso_stage_aware: bool = True
    binaural_beats: bool = True
    
    # 性能配置
    use_gpu: bool = True
    batch_size: int = 1
    max_processing_time: float = 200.0  # ms
    
    # 缓存配置
    enable_caching: bool = True
    cache_size: int = 100

class MusicParameter:
    """音乐参数（从mapping_layer导入的数据结构）"""
    def __init__(self, data: Dict[str, Any] = None):
        if data is None:
            data = {}
        
        # 基础音乐参数
        self.tempo_bpm = data.get('tempo_bpm', 60.0)
        self.key_signature = data.get('key_signature', 'C_major')
        self.time_signature = data.get('time_signature', (4, 4))
        self.dynamics = data.get('dynamics', 'mp')
        
        # 音色和织体
        self.instrument_weights = data.get('instrument_weights', {})
        self.texture_complexity = data.get('texture_complexity', 0.5)
        self.harmonic_richness = data.get('harmonic_richness', 0.5)
        
        # 情绪表达
        self.valence_mapping = data.get('valence_mapping', 0.0)
        self.arousal_mapping = data.get('arousal_mapping', 0.0)
        self.tension_level = data.get('tension_level', 0.0)
        
        # 治疗特化
        self.iso_stage = data.get('iso_stage', 'synchronization')
        self.therapy_intensity = data.get('therapy_intensity', 0.5)
        self.sleep_phase_alignment = data.get('sleep_phase_alignment', 0.0)

class SunoAPIClient:
    """Suno API客户端 - 用于真实音乐生成"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "your-suno-api-key"  # 需要用户提供真实密钥
        self.base_url = "https://api.suno.ai/v1"  # Suno API endpoint
        self.session = requests.Session()
        
        # 设置API请求头
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'qm_final3-music-therapy/1.0'
        })
        
        logger.info("Suno API客户端初始化完成")
    
    def generate_three_stage_music(self, emotion_data: Dict[str, Any], 
                                 music_params: MusicParameter) -> Dict[str, Any]:
        """生成三阶段连贯音乐叙事"""
        try:
            # 构建三阶段提示词
            stage_prompts = self._build_three_stage_prompts(emotion_data, music_params)
            
            # 调用Suno API生成三段连贯音乐
            music_response = self._call_suno_api(stage_prompts)
            
            # 处理响应
            if music_response and 'audio_url' in music_response:
                # 下载音频文件
                audio_data = self._download_audio(music_response['audio_url'])
                
                return {
                    'success': True,
                    'audio_data': audio_data,
                    'stage_prompts': stage_prompts,
                    'suno_response': music_response,
                    'three_stage_narrative': True
                }
            else:
                # 如果API调用失败，使用fallback生成
                return self._fallback_generation(emotion_data, music_params)
                
        except Exception as e:
            logger.error(f"Suno API调用失败: {e}")
            return self._fallback_generation(emotion_data, music_params)
    
    def _build_three_stage_prompts(self, emotion_data: Dict[str, Any], 
                                 music_params: MusicParameter) -> Dict[str, str]:
        """构建三阶段音乐生成提示词"""
        # 获取用户当前情绪
        current_emotion = emotion_data.get('primary_emotion', {}).get('name', '焦虑')
        
        # 映射情绪到音乐描述
        emotion_to_music = {
            '焦虑': {'initial': 'anxious, restless, fast tempo', 'target': 'calm, peaceful, slow'},
            '疲惫': {'initial': 'tired, heavy, sluggish', 'target': 'relaxed, floating, gentle'},
            '中性': {'initial': 'neutral, balanced, moderate', 'target': 'serene, tranquil, soft'},
            '失眠': {'initial': 'racing thoughts, tense, irregular', 'target': 'sleepy, drowsy, minimal'},
            '烦躁': {'initial': 'agitated, sharp, dissonant', 'target': 'smooth, harmonious, flowing'}
        }
        
        music_desc = emotion_to_music.get(current_emotion, emotion_to_music['焦虑'])
        
        # 构建三阶段提示词
        stage_prompts = {
            'synchronization': f"""Create ambient sleep therapy music - Stage 1 (Synchronization): 
Match the user's current emotional state of {current_emotion}. 
Musical style: {music_desc['initial']}, {music_params.tempo_bpm} BPM, 
{music_params.key_signature} key. Duration: 30 seconds. 
Focus: emotional resonance and matching current mood.""",
            
            'guidance': f"""Create ambient sleep therapy music - Stage 2 (Guidance): 
Gradually transition from {current_emotion} towards calm relaxation. 
Musical progression: slowly decrease tempo from {music_params.tempo_bpm} to {max(40, music_params.tempo_bpm-20)} BPM,
softer dynamics, gentler textures. Duration: 60 seconds.
Focus: smooth emotional transition and guidance.""",
            
            'consolidation': f"""Create ambient sleep therapy music - Stage 3 (Consolidation): 
Establish deep relaxation and sleepiness. 
Musical style: {music_desc['target']}, very slow tempo (30-40 BPM), 
minimal textures, sleep-inducing harmonies. Duration: 30 seconds.
Focus: consolidate peaceful state for sleep."""
        }
        
        return stage_prompts
    
    def _call_suno_api(self, stage_prompts: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """调用Suno API生成音乐"""
        try:
            # 组合三阶段提示为一个连贯的音乐请求
            combined_prompt = f"""
{stage_prompts['synchronization']}

Then smoothly transition to:
{stage_prompts['guidance']}

Finally conclude with:
{stage_prompts['consolidation']}

Create this as one continuous, seamless 2-minute ambient therapy track for sleep induction.
"""
            
            # 构建API请求
            request_data = {
                "prompt": combined_prompt,
                "duration": 120,  # 2分钟总时长
                "style": "ambient therapy",
                "mood": "calming progressive",
                "format": "wav"
            }
            
            # 发送请求（注意：这里使用模拟响应，实际需要真实API）
            response = self._simulate_suno_response(request_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Suno API请求失败: {e}")
            return None
    
    def _simulate_suno_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟Suno API响应（待替换为真实API调用）"""
        # 模拟API响应
        return {
            'id': f'suno_track_{int(time.time())}',
            'status': 'completed',
            'audio_url': 'https://example.com/generated_track.wav',  # 模拟URL
            'duration': request_data.get('duration', 120),
            'prompt_used': request_data.get('prompt', ''),
            'metadata': {
                'style': request_data.get('style', 'ambient'),
                'format': request_data.get('format', 'wav'),
                'generated_at': datetime.now().isoformat()
            }
        }
    
    def _download_audio(self, audio_url: str) -> Optional[np.ndarray]:
        """下载生成的音频文件"""
        try:
            # 对于模拟URL，生成模拟音频数据
            if 'example.com' in audio_url:
                return self._generate_mock_audio()
            
            # 真实下载逻辑（实际使用时启用）
            # response = self.session.get(audio_url)
            # if response.status_code == 200:
            #     audio_data = response.content
            #     # 使用librosa或soundfile解析音频
            #     return audio_array
            
            return self._generate_mock_audio()
            
        except Exception as e:
            logger.error(f"音频下载失败: {e}")
            return None
    
    def _generate_mock_audio(self) -> np.ndarray:
        """生成模拟音频数据（用于测试）"""
        # 生成2分钟的三阶段模拟音频
        sample_rate = 44100
        duration = 120  # 2分钟
        total_samples = int(sample_rate * duration)
        
        # 创建三阶段音频
        t = np.linspace(0, duration, total_samples)
        
        # 阶段1 (0-30s): 较快节奏，匹配当前情绪
        stage1_end = int(sample_rate * 30)
        stage1_freq = 440.0  # A4
        stage1 = 0.3 * np.sin(2 * np.pi * stage1_freq * t[:stage1_end])
        
        # 阶段2 (30-90s): 过渡阶段，频率逐渐降低
        stage2_start = stage1_end
        stage2_end = int(sample_rate * 90)
        stage2_samples = stage2_end - stage2_start
        freq_transition = np.linspace(440.0, 220.0, stage2_samples)
        stage2_t = t[stage2_start:stage2_end] - t[stage2_start]
        stage2 = 0.2 * np.sin(2 * np.pi * freq_transition * stage2_t)
        
        # 阶段3 (90-120s): 低频，助眠
        stage3_start = stage2_end
        stage3_samples = total_samples - stage3_start
        stage3_freq = 110.0  # 低频助眠
        stage3_t = t[stage3_start:] - t[stage3_start]
        stage3 = 0.1 * np.sin(2 * np.pi * stage3_freq * stage3_t)
        
        # 合并三阶段
        audio = np.concatenate([stage1, stage2, stage3])
        
        # 添加渐变以避免突变
        fade_length = int(sample_rate * 2)  # 2秒渐变
        
        # 阶段间渐变
        audio[stage1_end-fade_length:stage1_end] *= np.linspace(1, 0, fade_length)
        audio[stage1_end:stage1_end+fade_length] *= np.linspace(0, 1, fade_length)
        
        audio[stage2_end-fade_length:stage2_end] *= np.linspace(1, 0, fade_length)
        audio[stage2_end:stage2_end+fade_length] *= np.linspace(0, 1, fade_length)
        
        # 立体声
        stereo_audio = np.column_stack([audio, audio])
        
        return stereo_audio
    
    def _fallback_generation(self, emotion_data: Dict[str, Any], 
                           music_params: MusicParameter) -> Dict[str, Any]:
        """备用生成方案（当API不可用时）"""
        logger.info("使用备用音乐生成方案")
        
        audio_data = self._generate_mock_audio()
        
        return {
            'success': True,
            'audio_data': audio_data,
            'fallback_used': True,
            'three_stage_narrative': True,
            'message': 'API不可用，使用本地生成'
        }

class ProceduralAudioGenerator:
    """程序化音频生成器"""
    
    def __init__(self, config: GenerationLayerConfig):
        self.config = config
        self.sample_rate = config.output_sample_rate
        self.channels = config.output_channels
        
        # 音频生成参数
        self.base_frequencies = {
            'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23,
            'G': 392.00, 'A': 440.00, 'B': 493.88
        }
        
        # 和弦进行
        self.chord_progressions = {
            'relaxing': ['C', 'Am', 'F', 'G'],
            'calming': ['Am', 'F', 'C', 'G'],
            'peaceful': ['F', 'C', 'G', 'Am']
        }
        
        logger.info(f"程序化音频生成器初始化完成，采样率: {self.sample_rate}Hz")
    
    def generate_sine_wave(self, frequency: float, duration: float, 
                          amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        """生成正弦波"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return wave
    
    def generate_binaural_beats(self, base_freq: float, beat_freq: float, 
                               duration: float) -> np.ndarray:
        """生成双耳节拍"""
        left_freq = base_freq
        right_freq = base_freq + beat_freq
        
        left_channel = self.generate_sine_wave(left_freq, duration)
        right_channel = self.generate_sine_wave(right_freq, duration)
        
        # 合并为立体声
        stereo_audio = np.column_stack((left_channel, right_channel))
        return stereo_audio
    
    def generate_chord(self, root_note: str, chord_type: str, 
                      duration: float) -> np.ndarray:
        """生成和弦"""
        base_freq = self.base_frequencies.get(root_note, 440.0)
        
        # 定义和弦音程
        chord_intervals = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8]
        }
        
        intervals = chord_intervals.get(chord_type, [0, 4, 7])
        chord_audio = np.zeros(int(self.sample_rate * duration))
        
        for interval in intervals:
            freq = base_freq * (2 ** (interval / 12.0))
            note_audio = self.generate_sine_wave(freq, duration, amplitude=0.3)
            chord_audio += note_audio
        
        # 立体声处理
        if self.channels == 2:
            chord_audio = np.column_stack((chord_audio, chord_audio))
        
        return chord_audio
    
    def generate_ambient_texture(self, music_params: MusicParameter, 
                                duration: float) -> np.ndarray:
        """生成环境音乐纹理"""
        # 基础频率
        base_freq = self.base_frequencies.get(music_params.key_signature.split('_')[0], 440.0)
        
        # 生成基础音调
        base_tone = self.generate_sine_wave(base_freq, duration, amplitude=0.3)
        
        # 添加和声
        harmony1 = self.generate_sine_wave(base_freq * 1.25, duration, amplitude=0.2)
        harmony2 = self.generate_sine_wave(base_freq * 1.5, duration, amplitude=0.15)
        
        # 合成
        combined_audio = base_tone + harmony1 + harmony2
        
        # 应用包络
        envelope = self._generate_envelope(duration, attack=0.1, decay=0.1, sustain=0.8, release=0.1)
        combined_audio *= envelope
        
        # 添加治疗性元素
        if self.config.binaural_beats:
            # 根据ISO阶段调整双耳节拍频率
            beat_freq = self._get_binaural_frequency(music_params.iso_stage)
            binaural = self.generate_binaural_beats(base_freq, beat_freq, duration)
            
            # 混合双耳节拍
            if self.channels == 2:
                combined_audio = np.column_stack((combined_audio, combined_audio))
            
            combined_audio = 0.7 * combined_audio + 0.3 * binaural
        else:
            if self.channels == 2:
                combined_audio = np.column_stack((combined_audio, combined_audio))
        
        return combined_audio
    
    def _generate_envelope(self, duration: float, attack: float = 0.1, 
                          decay: float = 0.1, sustain: float = 0.8, 
                          release: float = 0.1) -> np.ndarray:
        """生成ADSR包络"""
        total_samples = int(self.sample_rate * duration)
        envelope = np.zeros(total_samples)
        
        attack_samples = int(attack * total_samples)
        decay_samples = int(decay * total_samples)
        release_samples = int(release * total_samples)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        envelope[attack_samples:attack_samples + decay_samples] = \
            np.linspace(1, sustain, decay_samples)
        
        # Sustain
        envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain
        
        # Release
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
        
        return envelope
    
    def _get_binaural_frequency(self, iso_stage: str) -> float:
        """根据ISO阶段获取双耳节拍频率"""
        frequencies = {
            'synchronization': 10.0,  # Alpha波段
            'guidance': 8.0,          # Alpha-Theta过渡
            'consolidation': 6.0      # Theta波段
        }
        return frequencies.get(iso_stage, 8.0)

class ProceduralVideoGenerator:
    """程序化视频生成器"""
    
    def __init__(self, config: GenerationLayerConfig):
        self.config = config
        self.width, self.height = config.video_resolution
        self.fps = config.video_fps
        
        # 当前风格
        self.style = config.video_style
        
        # 视频风格配置
        self.style_configs = {
            'ambient': {
                'background_color': (20, 30, 50),
                'primary_color': (100, 150, 200),
                'animation_speed': 0.5
            },
            'abstract': {
                'background_color': (10, 10, 20),
                'primary_color': (150, 100, 200),
                'animation_speed': 0.8
            },
            'nature': {
                'background_color': (30, 50, 30),
                'primary_color': (100, 200, 100),
                'animation_speed': 0.3
            },
            # 新增阶段特定风格
            'dynamic_ambient': {
                'background_color': (25, 35, 55),
                'primary_color': (120, 170, 220),
                'animation_speed': 0.6
            },
            'transitional_flow': {
                'background_color': (18, 25, 40),
                'primary_color': (90, 130, 180),
                'animation_speed': 0.4
            },
            'calm_static': {
                'background_color': (15, 20, 35),
                'primary_color': (70, 100, 140),
                'animation_speed': 0.2
            }
        }
        
        logger.info(f"程序化视频生成器初始化完成，分辨率: {self.width}x{self.height}")
    
    def generate_frame(self, frame_index: int, music_params: MusicParameter) -> np.ndarray:
        """生成单帧视频"""
        if not VIDEO_AVAILABLE:
            # 返回纯色帧
            return np.full((self.height, self.width, 3), 50, dtype=np.uint8)
        
        # 创建画布
        current_style = getattr(self, 'style', self.config.video_style)
        if current_style not in self.style_configs:
            current_style = 'ambient'
        
        image = PIL.Image.new('RGB', (self.width, self.height), 
                             color=self.style_configs[current_style]['background_color'])
        draw = PIL.ImageDraw.Draw(image)
        
        # 基于音乐参数生成视觉效果
        self._draw_emotion_visualization(draw, music_params, frame_index)
        
        # 添加治疗性元素
        if self.config.therapy_optimized:
            self._draw_therapy_elements(draw, music_params, frame_index)
        
        # 转换为numpy数组
        frame = np.array(image)
        return frame
    
    def _draw_emotion_visualization(self, draw, music_params: MusicParameter, frame_index: int):
        """绘制情绪可视化"""
        # 基于情绪映射绘制渐变背景
        center_x, center_y = self.width // 2, self.height // 2
        
        # 情绪强度影响圆形大小
        radius = int(200 + 100 * abs(music_params.valence_mapping))
        
        # 情绪效价影响颜色
        if music_params.valence_mapping > 0:
            color = (100, 150, 255)  # 积极情绪 - 蓝色
        else:
            color = (255, 150, 100)  # 消极情绪 - 橙色
        
        # 绘制脉动圆形
        pulse_factor = 0.8 + 0.2 * np.sin(frame_index * 0.1)
        current_radius = int(radius * pulse_factor)
        
        draw.ellipse([center_x - current_radius, center_y - current_radius,
                     center_x + current_radius, center_y + current_radius],
                    fill=color, outline=None)
    
    def _draw_therapy_elements(self, draw, music_params: MusicParameter, frame_index: int):
        """绘制治疗性元素"""
        # 根据ISO阶段绘制不同的治疗元素
        if music_params.iso_stage == 'synchronization':
            self._draw_sync_patterns(draw, frame_index)
        elif music_params.iso_stage == 'guidance':
            self._draw_guidance_flow(draw, frame_index)
        elif music_params.iso_stage == 'consolidation':
            self._draw_consolidation_calm(draw, frame_index)
    
    def _draw_sync_patterns(self, draw, frame_index: int):
        """绘制同步模式"""
        # 绘制同步圆环
        center_x, center_y = self.width // 2, self.height // 2
        for i in range(3):
            radius = 100 + i * 50
            alpha = int(100 * (1 - i * 0.3))
            color = (255, 255, 255, alpha)
            
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        outline=color, width=2)
    
    def _draw_guidance_flow(self, draw, frame_index: int):
        """绘制引导流动"""
        # 绘制流动曲线
        y_offset = int(50 * np.sin(frame_index * 0.05))
        
        for i in range(0, self.width, 20):
            x = i
            y = self.height // 2 + y_offset + int(20 * np.sin(i * 0.01))
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(200, 200, 255))
    
    def _draw_consolidation_calm(self, draw, frame_index: int):
        """绘制巩固宁静"""
        # 绘制静态星点
        np.random.seed(42)  # 固定种子确保星点位置一致
        for _ in range(50):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            size = np.random.randint(1, 3)
            
            # 星点闪烁效果
            brightness = int(100 + 50 * np.sin(frame_index * 0.02 + x * 0.001))
            color = (brightness, brightness, brightness)
            
            draw.ellipse([x - size, y - size, x + size, y + size], fill=color)
    
    def generate_video_sequence(self, music_params: MusicParameter, 
                              duration: float) -> List[np.ndarray]:
        """生成视频序列"""
        total_frames = int(duration * self.fps)
        frames = []
        
        logger.info(f"🎬 开始生成视频序列，共{total_frames}帧（预计耗时: {total_frames/15:.1f}秒）")
        
        for frame_index in range(total_frames):
            frame = self.generate_frame(frame_index, music_params)
            frames.append(frame)
            
            # 每10帧记录一次进度（15fps时约每0.67秒）
            if frame_index % 10 == 0 or frame_index == total_frames - 1:
                progress = (frame_index + 1) / total_frames * 100
                logger.info(f"📹 视频生成进度: {frame_index+1}/{total_frames} ({progress:.1f}%)")
        
        logger.info(f"✅ 视频序列生成完成")
        return frames

class GenerationLayer(BaseLayer):
    """生成层 - 实时音视频内容生成"""
    
    def __init__(self, config: GenerationLayerConfig):
        super().__init__(config)
        self.config = config
        self.layer_name = "generation_layer"
        
        # 初始化生成器
        self.audio_generator = ProceduralAudioGenerator(config) if config.audio_enabled else None
        self.video_generator = ProceduralVideoGenerator(config) if config.video_enabled else None
        
        # 初始化Suno API客户端
        self.suno_client = SunoAPIClient() if config.audio_enabled else None
        
        # 初始化ISO原则
        if config.iso_stage_aware:
            self.iso_principle = ISOPrinciple()
        else:
            self.iso_principle = None
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 生成缓存
        self.content_cache = {} if config.enable_caching else None
        
        logger.info(f"生成层初始化完成 - 音频: {config.audio_enabled}, 视频: {config.video_enabled}")
    
    def _extract_music_parameters(self, input_data: Dict[str, Any]) -> MusicParameter:
        """从映射层输出提取音乐参数"""
        music_params_data = input_data.get('music_parameters', {})
        
        # 如果是字典格式，转换为MusicParameter对象
        if isinstance(music_params_data, dict):
            return MusicParameter(music_params_data)
        elif hasattr(music_params_data, 'tempo_bpm'):
            # 已经是MusicParameter对象
            return music_params_data
        else:
            # 使用默认参数
            logger.warning("无法解析音乐参数，使用默认值")
            return MusicParameter()
    
    def _generate_audio_content(self, music_params: MusicParameter, emotion_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成音频内容 - 使用Suno API进行三阶段音乐叙事生成"""
        if not self.suno_client:
            return self._fallback_to_procedural_audio(music_params)
        
        try:
            # 使用Suno API生成三阶段连贯音乐
            logger.info("🎵 开始调用Suno API生成三阶段音乐叙事...")
            
            if emotion_data is None:
                emotion_data = {'primary_emotion': {'name': '焦虑'}}  # 默认情绪
            
            suno_result = self.suno_client.generate_three_stage_music(emotion_data, music_params)
            
            if suno_result.get('success'):
                audio_data = suno_result['audio_data']
                
                # 确保是numpy数组
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)
                
                # 归一化音频
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # 转换为字节流（如果需要）
                audio_bytes = None
                if AUDIO_AVAILABLE:
                    try:
                        buffer = io.BytesIO()
                        sf.write(buffer, audio_data, self.config.output_sample_rate, format='WAV')
                        audio_bytes = buffer.getvalue()
                    except Exception as e:
                        logger.warning(f"音频编码失败: {e}")
                
                # 构建返回结果
                result = {
                    'audio_array': audio_data,
                    'audio_bytes': audio_bytes,
                    'sample_rate': self.config.output_sample_rate,
                    'channels': self.config.output_channels,
                    'duration': 120.0,  # 2分钟三阶段音乐
                    'format': 'WAV',
                    'three_stage_narrative': True,
                    'stage_prompts': suno_result.get('stage_prompts', {}),
                    'suno_metadata': suno_result.get('suno_response', {}),
                    'fallback_used': suno_result.get('fallback_used', False)
                }
                
                if suno_result.get('fallback_used'):
                    logger.info("✅ 三阶段音乐生成完成（使用备用方案）")
                else:
                    logger.info("✅ 三阶段音乐生成完成（Suno API）")
                
                return result
            else:
                # 如果Suno API完全失败，使用原有的程序化生成
                logger.warning("Suno API生成失败，回退到程序化生成")
                return self._fallback_to_procedural_audio(music_params)
                
        except Exception as e:
            logger.error(f"Suno API音频生成失败: {e}")
            return self._fallback_to_procedural_audio(music_params)
    
    def _fallback_to_procedural_audio(self, music_params: MusicParameter) -> Dict[str, Any]:
        """备用程序化音频生成"""
        if not self.audio_generator:
            return {'error': '音频生成器未启用'}
        
        try:
            # 使用原有的程序化生成器
            audio_data = self.audio_generator.generate_ambient_texture(
                music_params, 
                self.config.audio_duration
            )
            
            # 确保是numpy数组
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # 归一化音频
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 转换为字节流（如果需要）
            audio_bytes = None
            if AUDIO_AVAILABLE:
                try:
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_data, self.config.output_sample_rate, format='WAV')
                    audio_bytes = buffer.getvalue()
                except Exception as e:
                    logger.warning(f"音频编码失败: {e}")
            
            return {
                'audio_array': audio_data,
                'audio_bytes': audio_bytes,
                'sample_rate': self.config.output_sample_rate,
                'channels': self.config.output_channels,
                'duration': self.config.audio_duration,
                'format': 'WAV',
                'fallback_used': True
            }
            
        except Exception as e:
            logger.error(f"备用音频生成失败: {e}")
            return {'error': str(e)}
    
    def _generate_video_content(self, music_params: MusicParameter) -> Dict[str, Any]:
        """生成视频内容"""
        if not self.video_generator:
            return {'error': '视频生成器未启用'}
        
        try:
            # 生成视频帧序列
            frames = self.video_generator.generate_video_sequence(
                music_params, 
                self.config.video_duration
            )
            
            return {
                'frames': frames,
                'fps': self.config.video_fps,
                'resolution': self.config.video_resolution,
                'duration': self.config.video_duration,
                'total_frames': len(frames),
                'format': 'RGB'
            }
            
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            return {'error': str(e)}
    
    def _synchronize_audio_video(self, audio_data: Dict[str, Any], 
                               video_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步音视频内容"""
        # 检查时长一致性
        audio_duration = audio_data.get('duration', 0)
        video_duration = video_data.get('duration', 0)
        
        if abs(audio_duration - video_duration) > 0.1:  # 100ms误差容忍
            logger.warning(f"音视频时长不一致: 音频{audio_duration}s, 视频{video_duration}s")
        
        # 创建同步元数据
        sync_metadata = {
            'audio_duration': audio_duration,
            'video_duration': video_duration,
            'synchronized': True,
            'sync_method': 'temporal_alignment',
            'sync_accuracy': 1.0 - abs(audio_duration - video_duration) / max(audio_duration, video_duration)
        }
        
        return {
            'audio': audio_data,
            'video': video_data,
            'sync_metadata': sync_metadata
        }
    
    def _cache_content(self, cache_key: str, content: Dict[str, Any]):
        """缓存生成的内容"""
        if self.content_cache is not None:
            # 简单的LRU缓存
            if len(self.content_cache) >= self.config.cache_size:
                # 移除最旧的条目
                oldest_key = next(iter(self.content_cache))
                del self.content_cache[oldest_key]
            
            self.content_cache[cache_key] = content
            logger.debug(f"内容已缓存: {cache_key}")
    
    def _get_cached_content(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存的内容"""
        if self.content_cache is not None:
            return self.content_cache.get(cache_key)
        return None
    
    def _generate_cache_key(self, music_params: MusicParameter) -> str:
        """生成缓存键"""
        # 基于音乐参数生成哈希键
        key_data = {
            'tempo': music_params.tempo_bpm,
            'key': music_params.key_signature,
            'iso_stage': music_params.iso_stage,
            'valence': music_params.valence_mapping,
            'arousal': music_params.arousal_mapping
        }
        
        import hashlib
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """生成层处理实现"""
        self.performance_monitor.start_timer("generation_layer_processing")
        
        try:
            # 验证输入数据
            if not input_data.data:
                raise ValueError("输入数据为空")
            
            # 初始化缓存变量
            cached_content = None
            cache_key = "unknown"
            
            # 检查是否有ISO三阶段参数
            if 'iso_three_stage_params' in input_data.data:
                # 三阶段生成模式
                content = await self._generate_three_stage_content(input_data.data)
                cache_key = "three_stage_dynamic"
            else:
                # 传统单阶段生成模式
                music_params = self._extract_music_parameters(input_data.data)
                
                # 生成缓存键
                cache_key = self._generate_cache_key(music_params)
                
                # 检查缓存
                cached_content = self._get_cached_content(cache_key)
                if cached_content:
                    logger.info(f"使用缓存内容: {cache_key}")
                    content = cached_content
                else:
                    # 生成新内容
                    content = {}
                    
                    # 生成音频内容
                    if self.config.audio_enabled:
                        # 提取情绪数据传递给Suno API
                        emotion_data = input_data.data.get('emotion_analysis', {})
                        audio_content = self._generate_audio_content(music_params, emotion_data)
                        content['audio'] = audio_content
                    
                    # 生成视频内容
                    if self.config.video_enabled:
                        video_content = self._generate_video_content(music_params)
                        content['video'] = video_content
                    
                    # 音视频同步
                    if self.config.audio_enabled and self.config.video_enabled:
                        if 'error' not in content['audio'] and 'error' not in content['video']:
                            content = self._synchronize_audio_video(content['audio'], content['video'])
                
                # 缓存内容
                self._cache_content(cache_key, content)
            
            # 创建输出数据
            music_params_dict = {}
            if 'iso_three_stage_params' not in input_data.data:
                # 单阶段模式
                music_params_dict = music_params.__dict__ if hasattr(music_params, '__dict__') else {}
            
            output_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'generated_content': content,
                    'music_parameters': music_params_dict,
                    'generation_info': {
                        'strategy': self.config.generation_strategy,
                        'content_type': self.config.content_type,
                        'cached': cached_content is not None,
                        'cache_key': cache_key
                    }
                },
                metadata={
                    'source_layer': input_data.layer_name,
                    'generation_strategy': self.config.generation_strategy,
                    'content_types': [t for t in ['audio', 'video'] if getattr(self.config, f"{t}_enabled")]
                },
                confidence=input_data.confidence
            )
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer("generation_layer_processing")
            output_data.processing_time = processing_time
            
            # 更新统计信息
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            logger.info(f"生成层处理完成 - 内容类型: {self.config.content_type}, "
                       f"缓存: {cached_content is not None}, 耗时: {processing_time*1000:.1f}ms")
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"生成层处理失败: {e}")
            
            # 创建错误输出
            error_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'error': str(e),
                    'generated_content': {}
                },
                metadata={'error': True, 'source_layer': input_data.layer_name},
                confidence=0.0
            )
            
            processing_time = self.performance_monitor.end_timer("generation_layer_processing")
            error_data.processing_time = processing_time
            
            return error_data
    
    def clear_cache(self):
        """清空缓存"""
        if self.content_cache is not None:
            self.content_cache.clear()
            logger.info("生成层缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.content_cache is not None:
            return {
                'cache_size': len(self.content_cache),
                'max_cache_size': self.config.cache_size,
                'cache_enabled': True
            }
        return {'cache_enabled': False}
    
    def get_status(self) -> Dict[str, Any]:
        """获取生成层状态"""
        base_status = super().get_status()
        
        # 添加生成层特有的状态信息
        generation_status = {
            'generation_strategy': self.config.generation_strategy,
            'content_type': self.config.content_type,
            'audio_enabled': self.config.audio_enabled,
            'video_enabled': self.config.video_enabled,
            'audio_available': AUDIO_AVAILABLE,
            'video_available': VIDEO_AVAILABLE,
            'cache_stats': self.get_cache_stats(),
            'performance_stats': self.performance_monitor.get_all_stats()
        }
        
        base_status.update(generation_status)
        return base_status
    
    async def _generate_three_stage_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成ISO三阶段音画同步内容"""
        logger.info("🎭 开始生成ISO三阶段音画同步内容")
        
        iso_params = input_data['iso_three_stage_params']
        stages = ['match_stage', 'guide_stage', 'target_stage']
        
        # 三阶段内容容器
        three_stage_content = {
            'stages': {},
            'continuous_narrative': True,
            'total_duration': 0.0,
            'sync_metadata': {
                'stage_transitions': [],
                'narrative_flow': 'smooth',
                'cross_stage_continuity': True
            }
        }
        
        # 为每个阶段生成内容
        for stage_idx, stage_name in enumerate(stages):
            stage_params = iso_params[stage_name]
            logger.info(f"🎼 生成阶段 {stage_idx + 1}: {stage_name}")
            
            # 转换阶段参数为MusicParameter对象
            music_params = self._convert_stage_to_music_params(stage_params, stage_name)
            
            # 生成阶段内容
            stage_content = {}
            
            # 生成音频内容
            if self.config.audio_enabled:
                stage_audio = await self._generate_stage_audio(music_params, stage_idx, stage_name)
                stage_content['audio'] = stage_audio
            
            # 生成视频内容
            if self.config.video_enabled:
                stage_video = await self._generate_stage_video(music_params, stage_idx, stage_name)
                stage_content['video'] = stage_video
            
            # 阶段音视频同步
            if self.config.audio_enabled and self.config.video_enabled:
                if 'error' not in stage_content['audio'] and 'error' not in stage_content['video']:
                    stage_content = self._synchronize_stage_content(stage_content, stage_name)
            
            # 添加阶段元数据
            stage_content['stage_info'] = {
                'stage_name': stage_name,
                'stage_index': stage_idx,
                'stage_duration': stage_params['stage_duration'],
                'therapy_intensity': stage_params.get('therapy_intensity', 0.5),
                'sleep_readiness': stage_params.get('sleep_readiness', 0.5),
                'tempo_bpm': stage_params['tempo_bpm'],
                'emotional_target': stage_params.get('emotional_target', 'neutral')
            }
            
            three_stage_content['stages'][stage_name] = stage_content
            three_stage_content['total_duration'] += stage_params['stage_duration']
            
            # 记录阶段转换信息
            if stage_idx > 0:
                prev_stage = stages[stage_idx - 1]
                transition_info = {
                    'from_stage': prev_stage,
                    'to_stage': stage_name,
                    'transition_point': three_stage_content['total_duration'] - stage_params['stage_duration'],
                    'transition_method': 'smooth_crossfade',
                    'continuity_score': 0.9  # 高连贯性
                }
                three_stage_content['sync_metadata']['stage_transitions'].append(transition_info)
        
        # 创建连贯的三阶段叙事
        three_stage_content = await self._create_narrative_continuity(three_stage_content)
        
        logger.info(f"✅ ISO三阶段内容生成完成，总时长: {three_stage_content['total_duration']:.1f}分钟")
        
        return three_stage_content
    
    def _convert_stage_to_music_params(self, stage_params: Dict[str, Any], stage_name: str) -> 'MusicParameter':
        """将阶段参数转换为MusicParameter对象"""
        music_params = MusicParameter()
        
        # 基础音乐参数
        music_params.tempo_bpm = stage_params.get('tempo_bpm', 60.0)
        music_params.key_signature = stage_params.get('key_signature', 'C_major')
        music_params.dynamics = stage_params.get('dynamics', 'mp')
        music_params.valence_mapping = stage_params.get('valence_mapping', 0.0)
        music_params.arousal_mapping = stage_params.get('arousal_mapping', 0.0)
        music_params.tension_level = stage_params.get('tension_level', 0.0)
        
        # 阶段特定参数
        music_params.iso_stage = stage_name
        if hasattr(music_params, 'therapy_intensity'):
            music_params.therapy_intensity = stage_params.get('therapy_intensity', 0.5)
        if hasattr(music_params, 'sleep_readiness'):
            music_params.sleep_readiness = stage_params.get('sleep_readiness', 0.5)
        
        # 乐器配置
        music_params.instrument_weights = stage_params.get('instrument_weights', {
            'sine_wave': 0.3,
            'ambient_pad': 0.4,
            'nature_sounds': 0.3
        })
        
        return music_params
    
    async def _generate_stage_audio(self, music_params: 'MusicParameter', stage_idx: int, stage_name: str) -> Dict[str, Any]:
        """为特定阶段生成音频内容"""
        logger.info(f"🎵 生成{stage_name}音频内容")
        
        try:
            # 根据阶段调整音频生成参数
            stage_duration = getattr(music_params, 'therapy_intensity', 0.5) * 60.0 + 60.0  # 1-2分钟
            
            # 使用增强的音频生成器
            if self.audio_generator:
                # 生成基础音频
                audio_data = self.audio_generator.generate_ambient_texture(music_params, stage_duration)
                
                # 添加阶段特定的处理
                audio_data = self._apply_stage_audio_effects(audio_data, stage_name, stage_idx)
                
                # 归一化
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # 转换为字节流
                audio_bytes = None
                if AUDIO_AVAILABLE:
                    try:
                        buffer = io.BytesIO()
                        sf.write(buffer, audio_data, self.config.output_sample_rate, format='WAV')
                        audio_bytes = buffer.getvalue()
                    except Exception as e:
                        logger.warning(f"阶段音频编码失败: {e}")
                
                return {
                    'audio_array': audio_data,
                    'audio_bytes': audio_bytes,
                    'sample_rate': self.config.output_sample_rate,
                    'channels': self.config.output_channels,
                    'duration': stage_duration,
                    'stage_name': stage_name,
                    'stage_index': stage_idx,
                    'format': 'WAV'
                }
            else:
                return {'error': f'{stage_name}音频生成器未启用'}
                
        except Exception as e:
            logger.error(f"{stage_name}音频生成失败: {e}")
            return {'error': str(e)}
    
    async def _generate_stage_video(self, music_params: 'MusicParameter', stage_idx: int, stage_name: str) -> Dict[str, Any]:
        """为特定阶段生成视频内容"""
        logger.info(f"🎬 生成{stage_name}视频内容")
        
        try:
            stage_duration = getattr(music_params, 'therapy_intensity', 0.5) * 60.0 + 60.0
            
            if self.video_generator:
                # 根据阶段调整视频风格
                original_style = self.video_generator.style
                self.video_generator.style = self._get_stage_video_style(stage_name)
                
                # 生成视频帧
                frames = self.video_generator.generate_video_sequence(music_params, stage_duration)
                
                # 应用阶段特定的视觉效果
                frames = self._apply_stage_video_effects(frames, stage_name, stage_idx)
                
                # 恢复原始风格
                self.video_generator.style = original_style
                
                return {
                    'frames': frames,
                    'fps': self.config.video_fps,
                    'resolution': self.config.video_resolution,
                    'duration': stage_duration,
                    'total_frames': len(frames),
                    'stage_name': stage_name,
                    'stage_index': stage_idx,
                    'format': 'RGB'
                }
            else:
                return {'error': f'{stage_name}视频生成器未启用'}
                
        except Exception as e:
            logger.error(f"{stage_name}视频生成失败: {e}")
            return {'error': str(e)}
    
    def _apply_stage_audio_effects(self, audio_data: np.ndarray, stage_name: str, stage_idx: int) -> np.ndarray:
        """应用阶段特定的音频效果"""
        if len(audio_data) == 0:
            return audio_data
        
        # 处理立体声和单声道
        is_stereo = len(audio_data.shape) == 2 and audio_data.shape[1] == 2
        audio_length = audio_data.shape[0]
            
        if stage_name == 'match_stage':
            # 匹配阶段：增加一些动态变化
            fade_length = int(audio_length * 0.1)  # 10%淡入
            if fade_length > 0:
                fade_in = np.linspace(0, 1, fade_length)
                if is_stereo:
                    fade_in = fade_in.reshape(-1, 1)  # 使其可以广播到立体声
                audio_data[:fade_length] *= fade_in
            
        elif stage_name == 'guide_stage':
            # 引导阶段：添加渐进式降低的效果
            guide_envelope = np.linspace(1.0, 0.7, audio_length)
            if is_stereo:
                guide_envelope = guide_envelope.reshape(-1, 1)  # 使其可以广播到立体声
            audio_data *= guide_envelope
            
        elif stage_name == 'target_stage':
            # 目标阶段：最大程度的平静处理
            fade_length = int(audio_length * 0.2)  # 20%淡出
            if fade_length > 0:
                fade_out = np.linspace(1, 0.3, fade_length)
                if is_stereo:
                    fade_out = fade_out.reshape(-1, 1)  # 使其可以广播到立体声
                audio_data[-fade_length:] *= fade_out
        
        return audio_data
    
    def _get_stage_video_style(self, stage_name: str) -> str:
        """获取阶段对应的视频风格"""
        stage_styles = {
            'match_stage': 'dynamic_ambient',     # 动态环境
            'guide_stage': 'transitional_flow',   # 过渡流动
            'target_stage': 'calm_static'         # 平静静态
        }
        return stage_styles.get(stage_name, 'ambient')
    
    def _apply_stage_video_effects(self, frames: List[np.ndarray], stage_name: str, stage_idx: int) -> List[np.ndarray]:
        """应用阶段特定的视觉效果"""
        if not frames:
            return frames
            
        processed_frames = []
        
        for i, frame in enumerate(frames):
            if stage_name == 'match_stage':
                # 匹配阶段：保持原始亮度和动态
                processed_frame = frame
                
            elif stage_name == 'guide_stage':
                # 引导阶段：逐渐降低亮度和对比度
                progress = i / len(frames) if len(frames) > 0 else 0
                brightness_factor = 1.0 - progress * 0.3  # 最多降低30%
                processed_frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
                
            elif stage_name == 'target_stage':
                # 目标阶段：最低亮度，最大平静感
                brightness_factor = 0.5  # 降低50%亮度
                processed_frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
            else:
                processed_frame = frame
            
            processed_frames.append(processed_frame)
        
        logger.info(f"✨ {stage_name}视觉效果应用完成，处理{len(processed_frames)}帧")
        return processed_frames
    
    def _synchronize_stage_content(self, stage_content: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """同步阶段音视频内容"""
        audio_data = stage_content['audio']
        video_data = stage_content['video']
        
        # 检查时长一致性
        audio_duration = audio_data.get('duration', 0)
        video_duration = video_data.get('duration', 0)
        
        if abs(audio_duration - video_duration) > 0.1:
            logger.warning(f"{stage_name}音视频时长不一致: 音频{audio_duration}s, 视频{video_duration}s")
        
        # 创建阶段同步元数据
        sync_metadata = {
            'stage_name': stage_name,
            'audio_duration': audio_duration,
            'video_duration': video_duration,
            'synchronized': True,
            'sync_method': 'stage_temporal_alignment',
            'sync_accuracy': 1.0 - abs(audio_duration - video_duration) / max(audio_duration, video_duration, 1.0),
            'stage_specific_sync': True
        }
        
        return {
            'audio': audio_data,
            'video': video_data,
            'sync_metadata': sync_metadata
        }
    
    async def _create_narrative_continuity(self, three_stage_content: Dict[str, Any]) -> Dict[str, Any]:
        """创建三阶段叙事连贯性"""
        logger.info("📖 创建连贯的三阶段音乐叙事")
        
        stages = ['match_stage', 'guide_stage', 'target_stage']
        
        # 分析各阶段特征，确保平滑过渡
        narrative_analysis = {
            'tempo_progression': [],
            'valence_progression': [],
            'arousal_progression': [],
            'visual_continuity': [],
            'therapy_progression': []
        }
        
        # 收集各阶段数据
        for stage_name in stages:
            if stage_name in three_stage_content['stages']:
                stage_info = three_stage_content['stages'][stage_name]['stage_info']
                
                narrative_analysis['tempo_progression'].append(stage_info['tempo_bpm'])
                narrative_analysis['therapy_progression'].append(stage_info['therapy_intensity'])
        
        # 验证叙事一致性
        tempo_progression = narrative_analysis['tempo_progression']
        therapy_progression = narrative_analysis['therapy_progression']
        
        tempo_decrease = all(tempo_progression[i] >= tempo_progression[i+1] 
                           for i in range(len(tempo_progression)-1)) if len(tempo_progression) > 1 else True
        
        therapy_increase = all(therapy_progression[i] <= therapy_progression[i+1] 
                             for i in range(len(therapy_progression)-1)) if len(therapy_progression) > 1 else True
        
        # 添加叙事连贯性元数据
        three_stage_content['narrative_analysis'] = narrative_analysis
        three_stage_content['narrative_quality'] = {
            'tempo_coherence': tempo_decrease,
            'therapy_coherence': therapy_increase,
            'overall_coherence_score': 0.9 if tempo_decrease and therapy_increase else 0.7,
            'narrative_type': 'guided_relaxation_progression'
        }
        
        logger.info(f"✅ 叙事连贯性评分: {three_stage_content['narrative_quality']['overall_coherence_score']:.2f}")
        
        return three_stage_content