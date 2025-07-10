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
            }
        }
        
        logger.info(f"程序化视频生成器初始化完成，分辨率: {self.width}x{self.height}")
    
    def generate_frame(self, frame_index: int, music_params: MusicParameter) -> np.ndarray:
        """生成单帧视频"""
        if not VIDEO_AVAILABLE:
            # 返回纯色帧
            return np.full((self.height, self.width, 3), 50, dtype=np.uint8)
        
        # 创建画布
        image = PIL.Image.new('RGB', (self.width, self.height), 
                             color=self.style_configs[self.config.video_style]['background_color'])
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
        
        logger.info(f"开始生成视频序列，共{total_frames}帧")
        
        for frame_index in range(total_frames):
            frame = self.generate_frame(frame_index, music_params)
            frames.append(frame)
            
            # 每100帧记录一次进度
            if frame_index % 100 == 0:
                logger.debug(f"视频生成进度: {frame_index}/{total_frames}")
        
        logger.info(f"视频序列生成完成")
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
    
    def _generate_audio_content(self, music_params: MusicParameter) -> Dict[str, Any]:
        """生成音频内容"""
        if not self.audio_generator:
            return {'error': '音频生成器未启用'}
        
        try:
            # 生成音频
            audio_data = self.audio_generator.generate_ambient_texture(
                music_params, 
                self.config.audio_duration
            )
            
            # 归一化音频
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
                'format': 'WAV'
            }
            
        except Exception as e:
            logger.error(f"音频生成失败: {e}")
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
            
            # 提取音乐参数
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
                    audio_content = self._generate_audio_content(music_params)
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
            output_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'generated_content': content,
                    'music_parameters': music_params.__dict__,
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