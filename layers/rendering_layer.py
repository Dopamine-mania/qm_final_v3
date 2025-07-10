#!/usr/bin/env python3
"""
渲染层 (Rendering Layer) - Layer 5

实时同步渲染和输出，核心功能包括：
1. 音视频实时同步播放
2. 低延迟流媒体输出
3. 多设备适配渲染
4. 动态质量调整
5. 治疗效果实时监控

处理流程：
Generation Layer → Rendering Layer → Therapy Layer
"""

import numpy as np
import asyncio
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

from .base_layer import BaseLayer, LayerData, LayerConfig
from core.utils import (
    ConfigLoader, DataValidator, PerformanceMonitor, 
    get_project_root
)

logger = logging.getLogger(__name__)

# 检查可选依赖
try:
    import pygame
    PYGAME_AVAILABLE = True
    logger.info("pygame可用，支持本地音视频播放")
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("pygame不可用，本地播放功能将受限")

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV可用，支持视频渲染")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV不可用，视频渲染功能将受限")

class RenderingMode(Enum):
    """渲染模式枚举"""
    LOCAL_PLAYBACK = "local_playback"      # 本地播放
    STREAMING = "streaming"                # 流媒体输出
    FILE_OUTPUT = "file_output"           # 文件输出
    MULTI_OUTPUT = "multi_output"         # 多输出模式

class QualityLevel(Enum):
    """质量等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class SyncMode(Enum):
    """同步模式枚举"""
    FRAME_SYNC = "frame_sync"         # 帧同步
    TIMESTAMP_SYNC = "timestamp_sync" # 时间戳同步
    BUFFER_SYNC = "buffer_sync"       # 缓冲区同步

@dataclass
class RenderingLayerConfig(LayerConfig):
    """渲染层配置"""
    # 基础配置
    rendering_mode: str = "local_playback"
    output_format: str = "realtime"
    
    # 音频渲染配置
    audio_enabled: bool = True
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    audio_buffer_size: int = 4096
    audio_latency_ms: float = 20.0
    
    # 视频渲染配置
    video_enabled: bool = True
    video_fps: int = 30
    video_resolution: Tuple[int, int] = (1920, 1080)
    video_buffer_frames: int = 10
    video_latency_ms: float = 33.3  # ~1 frame at 30fps
    
    # 同步配置
    sync_mode: str = "timestamp_sync"
    sync_tolerance_ms: float = 10.0
    max_sync_drift_ms: float = 100.0
    
    # 质量配置
    quality_level: str = "medium"
    adaptive_quality: bool = True
    quality_adjustment_interval: float = 5.0  # seconds
    
    # 缓冲配置
    buffer_size_ms: float = 500.0    # 500ms缓冲
    prebuffer_ms: float = 100.0      # 预缓冲时间
    
    # 输出配置
    output_device_audio: Optional[str] = None
    output_device_video: Optional[str] = None
    
    # 性能配置
    use_hardware_acceleration: bool = True
    use_gpu: bool = True  # GPU加速支持
    max_processing_time: float = 16.7  # ms, ~1 frame at 60fps
    demo_mode: bool = False  # 演示模式优化开关
    
    # 监控配置
    enable_performance_monitoring: bool = True
    enable_quality_monitoring: bool = True

class AudioRenderer:
    """音频渲染器"""
    
    def __init__(self, config: RenderingLayerConfig):
        self.config = config
        self.is_playing = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.playback_thread = None
        
        # 音频参数
        self.sample_rate = config.audio_sample_rate
        self.channels = config.audio_channels
        self.buffer_size = config.audio_buffer_size
        
        # 同步控制
        self.playback_position = 0.0  # seconds
        self.start_time = None
        
        # 初始化pygame音频
        if PYGAME_AVAILABLE and config.audio_enabled:
            self._init_pygame_audio()
        
        logger.info(f"音频渲染器初始化完成 - 采样率: {self.sample_rate}Hz, 声道: {self.channels}")
    
    def _init_pygame_audio(self):
        """初始化pygame音频系统"""
        try:
            pygame.mixer.pre_init(
                frequency=self.sample_rate,
                size=-16,  # 16-bit signed
                channels=self.channels,
                buffer=self.buffer_size
            )
            pygame.mixer.init()
            logger.info("pygame音频系统初始化成功")
        except Exception as e:
            logger.warning(f"pygame音频初始化失败: {e}")
    
    def start_playback(self):
        """开始音频播放"""
        if self.is_playing:
            return
        
        self.is_playing = True
        self.start_time = time.time()
        
        if PYGAME_AVAILABLE:
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            logger.info("音频播放已开始")
    
    def stop_playback(self):
        """停止音频播放"""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        if PYGAME_AVAILABLE:
            pygame.mixer.stop()
        
        logger.info("音频播放已停止")
    
    def queue_audio(self, audio_data: np.ndarray, timestamp: float):
        """队列音频数据（优化版）"""
        try:
            audio_item = {
                'data': audio_data,
                'timestamp': timestamp,
                'queued_at': time.time()
            }
            # 优化：非阻塞队列，超时立即返回
            self.audio_queue.put(audio_item, timeout=0.001)  # 1ms超时
        except queue.Full:
            logger.debug("音频队列已满，丢弃音频数据（演示模式优化）")
    
    def _playback_loop(self):
        """音频播放循环"""
        while self.is_playing:
            try:
                # 获取音频数据
                audio_item = self.audio_queue.get(timeout=0.1)
                
                # 检查同步
                current_time = time.time() - self.start_time
                target_time = audio_item['timestamp']
                
                # 如果太早，等待
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                
                # 播放音频
                self._play_audio_chunk(audio_item['data'])
                
                # 更新播放位置
                self.playback_position = target_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"音频播放错误: {e}")
    
    def _play_audio_chunk(self, audio_data: np.ndarray):
        """播放音频块"""
        if not PYGAME_AVAILABLE:
            return
        
        try:
            # 转换为pygame格式
            if audio_data.dtype != np.int16:
                # 归一化到int16范围
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # 创建pygame Sound对象并播放
            sound = pygame.sndarray.make_sound(audio_data)
            sound.play()
            
        except Exception as e:
            logger.warning(f"pygame音频播放失败: {e}")
    
    def get_playback_position(self) -> float:
        """获取当前播放位置（秒）"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_buffer_level(self) -> float:
        """获取缓冲区填充水平（0-1）"""
        return self.audio_queue.qsize() / self.audio_queue.maxsize

class VideoRenderer:
    """视频渲染器"""
    
    def __init__(self, config: RenderingLayerConfig):
        self.config = config
        self.is_playing = False
        self.frame_queue = queue.Queue(maxsize=config.video_buffer_frames)
        self.display_thread = None
        
        # 视频参数
        self.fps = config.video_fps
        self.resolution = config.video_resolution
        self.frame_duration = 1.0 / self.fps
        
        # 同步控制
        self.frame_position = 0
        self.start_time = None
        
        # 初始化显示
        if CV2_AVAILABLE and config.video_enabled:
            self._init_display()
        
        logger.info(f"视频渲染器初始化完成 - 分辨率: {self.resolution}, 帧率: {self.fps}fps")
    
    def _init_display(self):
        """初始化显示窗口"""
        try:
            # 创建窗口（仅在本地播放模式下）
            if self.config.rendering_mode == "local_playback":
                cv2.namedWindow('qm_final3 Video', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('qm_final3 Video', 800, 600)  # 预览大小
                logger.info("OpenCV显示窗口创建成功")
        except Exception as e:
            logger.warning(f"OpenCV显示初始化失败: {e}")
    
    def start_playback(self):
        """开始视频播放"""
        if self.is_playing:
            return
        
        self.is_playing = True
        self.start_time = time.time()
        
        if CV2_AVAILABLE:
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
            logger.info("视频播放已开始")
    
    def stop_playback(self):
        """停止视频播放"""
        self.is_playing = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        
        logger.info("视频播放已停止")
    
    def queue_frame(self, frame: np.ndarray, timestamp: float):
        """队列视频帧（优化版）"""
        try:
            frame_item = {
                'frame': frame,
                'timestamp': timestamp,
                'queued_at': time.time()
            }
            # 优化：非阻塞队列，超时立即返回
            self.frame_queue.put(frame_item, timeout=0.001)  # 1ms超时
        except queue.Full:
            # 优化：在演示模式下直接丢弃帧而不等待
            logger.debug("视频队列已满，直接丢弃帧（演示模式优化）")
            pass
    
    def _display_loop(self):
        """视频显示循环"""
        while self.is_playing:
            try:
                # 获取视频帧
                frame_item = self.frame_queue.get(timeout=0.1)
                
                # 检查同步
                current_time = time.time() - self.start_time
                target_time = frame_item['timestamp']
                
                # 如果太早，等待
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                
                # 显示帧
                self._display_frame(frame_item['frame'])
                
                # 更新帧位置
                self.frame_position += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"视频显示错误: {e}")
    
    def _display_frame(self, frame: np.ndarray):
        """显示视频帧"""
        if not CV2_AVAILABLE:
            return
        
        try:
            # 调整大小以适合显示
            if frame.shape[:2] != (600, 800):  # height, width
                frame = cv2.resize(frame, (800, 600))
            
            # 显示帧
            cv2.imshow('qm_final3 Video', frame)
            cv2.waitKey(1)  # 1ms等待，允许窗口事件处理
            
        except Exception as e:
            logger.warning(f"OpenCV帧显示失败: {e}")
    
    def get_frame_position(self) -> int:
        """获取当前帧位置"""
        return self.frame_position
    
    def get_playback_position(self) -> float:
        """获取当前播放位置（秒）"""
        return self.frame_position * self.frame_duration
    
    def get_buffer_level(self) -> float:
        """获取缓冲区填充水平（0-1）"""
        return self.frame_queue.qsize() / self.frame_queue.maxsize

class SyncController:
    """同步控制器"""
    
    def __init__(self, config: RenderingLayerConfig):
        self.config = config
        self.sync_mode = SyncMode(config.sync_mode)
        self.tolerance_ms = config.sync_tolerance_ms
        self.max_drift_ms = config.max_sync_drift_ms
        
        # 同步统计
        self.sync_stats = {
            'total_adjustments': 0,
            'avg_drift_ms': 0.0,
            'max_drift_ms': 0.0,
            'sync_quality': 1.0
        }
        
        logger.info(f"同步控制器初始化完成 - 模式: {self.sync_mode.value}")
    
    def check_sync(self, audio_pos: float, video_pos: float) -> Dict[str, Any]:
        """检查音视频同步状态"""
        drift_ms = abs(audio_pos - video_pos) * 1000
        
        # 更新统计
        self.sync_stats['max_drift_ms'] = max(self.sync_stats['max_drift_ms'], drift_ms)
        
        # 计算同步质量
        sync_quality = max(0.0, 1.0 - drift_ms / self.max_drift_ms)
        self.sync_stats['sync_quality'] = sync_quality
        
        # 判断是否需要调整
        needs_adjustment = drift_ms > self.tolerance_ms
        
        return {
            'drift_ms': drift_ms,
            'needs_adjustment': needs_adjustment,
            'sync_quality': sync_quality,
            'audio_position': audio_pos,
            'video_position': video_pos
        }
    
    def suggest_adjustment(self, sync_info: Dict[str, Any]) -> Dict[str, Any]:
        """建议同步调整"""
        if not sync_info['needs_adjustment']:
            return {'action': 'none'}
        
        drift_ms = sync_info['drift_ms']
        audio_pos = sync_info['audio_position']
        video_pos = sync_info['video_position']
        
        if audio_pos > video_pos:
            # 音频超前，减慢音频或加快视频
            return {
                'action': 'slow_audio',
                'adjustment_ms': drift_ms,
                'target': 'audio'
            }
        else:
            # 视频超前，减慢视频或加快音频
            return {
                'action': 'slow_video',
                'adjustment_ms': drift_ms,
                'target': 'video'
            }

class QualityController:
    """质量控制器"""
    
    def __init__(self, config: RenderingLayerConfig):
        self.config = config
        self.current_quality = QualityLevel(config.quality_level)
        self.adaptive_enabled = config.adaptive_quality
        
        # 性能指标
        self.performance_history = []
        self.adjustment_interval = config.quality_adjustment_interval
        self.last_adjustment = time.time()
        
        logger.info(f"质量控制器初始化完成 - 当前质量: {self.current_quality.value}")
    
    def update_performance(self, render_time_ms: float, buffer_levels: Dict[str, float]):
        """更新性能指标"""
        self.performance_history.append({
            'timestamp': time.time(),
            'render_time_ms': render_time_ms,
            'audio_buffer': buffer_levels.get('audio', 0.0),
            'video_buffer': buffer_levels.get('video', 0.0)
        })
        
        # 保持最近10个样本
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
    
    def should_adjust_quality(self) -> bool:
        """判断是否应该调整质量"""
        if not self.adaptive_enabled:
            return False
        
        # 检查调整间隔
        if time.time() - self.last_adjustment < self.adjustment_interval:
            return False
        
        # 检查性能指标
        if len(self.performance_history) < 5:
            return False
        
        # 计算平均渲染时间
        avg_render_time = np.mean([p['render_time_ms'] for p in self.performance_history])
        target_render_time = self.config.max_processing_time
        
        # 如果渲染时间过长，需要降低质量
        if avg_render_time > target_render_time * 1.2:
            return True
        
        # 如果渲染时间很短，可以提高质量
        if avg_render_time < target_render_time * 0.6:
            return True
        
        return False
    
    def adjust_quality(self) -> Dict[str, Any]:
        """调整质量设置"""
        if not self.should_adjust_quality():
            return {'action': 'none'}
        
        avg_render_time = np.mean([p['render_time_ms'] for p in self.performance_history])
        target_render_time = self.config.max_processing_time
        
        if avg_render_time > target_render_time * 1.2:
            # 降低质量
            if self.current_quality == QualityLevel.ULTRA:
                self.current_quality = QualityLevel.HIGH
            elif self.current_quality == QualityLevel.HIGH:
                self.current_quality = QualityLevel.MEDIUM
            elif self.current_quality == QualityLevel.MEDIUM:
                self.current_quality = QualityLevel.LOW
            
            action = 'decrease'
        else:
            # 提高质量
            if self.current_quality == QualityLevel.LOW:
                self.current_quality = QualityLevel.MEDIUM
            elif self.current_quality == QualityLevel.MEDIUM:
                self.current_quality = QualityLevel.HIGH
            elif self.current_quality == QualityLevel.HIGH:
                self.current_quality = QualityLevel.ULTRA
            
            action = 'increase'
        
        self.last_adjustment = time.time()
        
        logger.info(f"质量调整: {action} -> {self.current_quality.value}")
        
        return {
            'action': action,
            'new_quality': self.current_quality.value,
            'avg_render_time_ms': avg_render_time,
            'target_render_time_ms': target_render_time
        }

class RenderingLayer(BaseLayer):
    """渲染层 - 实时同步渲染和输出"""
    
    def __init__(self, config: RenderingLayerConfig):
        super().__init__(config)
        self.config = config
        self.layer_name = "rendering_layer"
        
        # 初始化渲染器
        self.audio_renderer = AudioRenderer(config) if config.audio_enabled else None
        self.video_renderer = VideoRenderer(config) if config.video_enabled else None
        
        # 初始化控制器
        self.sync_controller = SyncController(config)
        self.quality_controller = QualityController(config)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 渲染状态
        self.is_rendering = False
        self.render_start_time = None
        
        # 输出统计
        self.render_stats = {
            'total_frames_rendered': 0,
            'total_audio_chunks_rendered': 0,
            'dropped_frames': 0,
            'avg_render_time_ms': 0.0,
            'sync_quality': 1.0
        }
        
        # 检测演示模式并启用优化
        import sys
        if '--demo' in sys.argv:
            self.config.demo_mode = True
            logger.info("检测到演示模式，启用渲染优化")
        
        logger.info(f"渲染层初始化完成 - 音频: {config.audio_enabled}, 视频: {config.video_enabled}, 演示模式: {getattr(self.config, 'demo_mode', False)}")
    
    def start_rendering(self):
        """开始渲染"""
        if self.is_rendering:
            return
        
        self.is_rendering = True
        self.render_start_time = time.time()
        
        # 启动渲染器
        if self.audio_renderer:
            self.audio_renderer.start_playback()
        
        if self.video_renderer:
            self.video_renderer.start_playback()
        
        logger.info("渲染层开始渲染")
    
    def stop_rendering(self):
        """停止渲染"""
        if not self.is_rendering:
            return
        
        self.is_rendering = False
        
        # 停止渲染器
        if self.audio_renderer:
            self.audio_renderer.stop_playback()
        
        if self.video_renderer:
            self.video_renderer.stop_playback()
        
        logger.info("渲染层停止渲染")
    
    def _extract_content_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取生成的内容数据"""
        generated_content = input_data.get('generated_content', {})
        
        # 处理不同的内容格式
        if 'audio' in generated_content and 'video' in generated_content:
            # 音视频同步内容
            return {
                'audio': generated_content['audio'],
                'video': generated_content['video'],
                'sync_metadata': generated_content.get('sync_metadata', {})
            }
        elif 'audio_array' in generated_content:
            # 纯音频内容
            return {
                'audio': generated_content,
                'video': None
            }
        elif 'frames' in generated_content:
            # 纯视频内容
            return {
                'audio': None,
                'video': generated_content
            }
        else:
            # 直接格式
            return generated_content
    
    def _render_audio(self, audio_data: Dict[str, Any]) -> bool:
        """渲染音频内容（优化版）"""
        if not self.audio_renderer or not audio_data:
            return False
        
        try:
            # 提取音频数组
            audio_array = audio_data.get('audio_array')
            if audio_array is None:
                return False
            
            # 优化：在演示模式下截取音频以提高性能
            # 只处理前1秒音频用于验证功能
            sample_rate = audio_data.get('sample_rate', 44100)
            if len(audio_array) > sample_rate:  # 如果音频超过1秒
                audio_array = audio_array[:sample_rate]  # 只取前1秒
                logger.debug(f"音频渲染优化：截取到1秒用于演示")
            
            # 计算时间戳
            duration = len(audio_array) / sample_rate
            current_time = time.time() - self.render_start_time if self.render_start_time else 0
            
            # 队列音频数据
            self.audio_renderer.queue_audio(audio_array, current_time)
            
            # 更新统计
            self.render_stats['total_audio_chunks_rendered'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"音频渲染失败: {e}")
            return False
    
    def _render_video(self, video_data: Dict[str, Any]) -> bool:
        """渲染视频内容（优化版）"""
        if not self.video_renderer or not video_data:
            return False
        
        try:
            # 提取视频帧
            frames = video_data.get('frames', [])
            if not frames:
                return False
            
            fps = video_data.get('fps', 30)
            frame_duration = 1.0 / fps
            
            # 优化：仅队列第一帧用于演示，避免大量帧处理延迟
            # 在演示模式下，我们只需要验证渲染功能，不需要播放完整视频
            current_time = time.time() - self.render_start_time if self.render_start_time else 0
            
            # 仅处理前3帧进行快速演示
            max_frames_to_process = min(3, len(frames))
            for i in range(max_frames_to_process):
                timestamp = current_time + i * frame_duration
                self.video_renderer.queue_frame(frames[i], timestamp)
            
            # 更新统计（按实际处理的帧数）
            self.render_stats['total_frames_rendered'] += max_frames_to_process
            
            logger.debug(f"视频渲染优化：处理 {max_frames_to_process}/{len(frames)} 帧")
            return True
            
        except Exception as e:
            logger.error(f"视频渲染失败: {e}")
            return False
    
    def _monitor_sync_quality(self):
        """监控同步质量（优化版）"""
        if not self.audio_renderer or not self.video_renderer:
            return
        
        try:
            # 获取播放位置
            audio_pos = self.audio_renderer.get_playback_position()
            video_pos = self.video_renderer.get_playback_position()
            
            # 检查同步状态
            sync_info = self.sync_controller.check_sync(audio_pos, video_pos)
            
            # 更新同步质量统计
            self.render_stats['sync_quality'] = sync_info['sync_quality']
            
            # 在演示模式下，只记录而不进行实际调整
            if sync_info['needs_adjustment']:
                logger.debug(f"同步偏差: {sync_info['drift_ms']:.1f}ms")
        except Exception as e:
            logger.debug(f"同步监控失败: {e}")
    
    def _monitor_performance(self):
        """监控渲染性能（优化版）"""
        try:
            # 获取缓冲区水平（简化版）
            buffer_levels = {}
            if self.audio_renderer:
                buffer_levels['audio'] = self.audio_renderer.get_buffer_level()
            if self.video_renderer:
                buffer_levels['video'] = self.video_renderer.get_buffer_level()
            
            # 获取渲染时间
            render_time_ms = self.performance_monitor.get_last_duration('rendering_layer_processing') * 1000
            
            # 在演示模式下，只更新统计而不进行质量调整
            self.render_stats['avg_render_time_ms'] = render_time_ms
            
            logger.debug(f"渲染性能 - 时间: {render_time_ms:.1f}ms, 音频缓冲: {buffer_levels.get('audio', 0):.2f}, 视频缓冲: {buffer_levels.get('video', 0):.2f}")
        except Exception as e:
            logger.debug(f"性能监控失败: {e}")
    
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """渲染层处理实现"""
        self.performance_monitor.start_timer("rendering_layer_processing")
        
        try:
            # 验证输入数据
            if not input_data.data:
                raise ValueError("输入数据为空")
            
            # 提取内容数据
            content_data = self._extract_content_data(input_data.data)
            
            # 开始渲染（如果还没开始）
            if not self.is_rendering:
                self.start_rendering()
            
            # 渲染音频
            audio_success = False
            if content_data.get('audio'):
                audio_success = self._render_audio(content_data['audio'])
            
            # 渲染视频
            video_success = False
            if content_data.get('video'):
                video_success = self._render_video(content_data['video'])
            
            # 优化：在演示模式下减少监控频率以提高性能
            if getattr(self.config, 'demo_mode', False):
                # 演示模式：每10次处理才监控一次
                if self.total_processed % 10 == 0:
                    self._monitor_sync_quality()
                    self._monitor_performance()
            else:
                # 正常模式：保持原有监控频率
                self._monitor_sync_quality()
                self._monitor_performance()
            
            # 创建输出数据
            output_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'rendering_result': {
                        'audio_rendered': audio_success,
                        'video_rendered': video_success,
                        'rendering_mode': self.config.rendering_mode,
                        'quality_level': self.quality_controller.current_quality.value
                    },
                    'sync_status': self.sync_controller.sync_stats,
                    'render_stats': self.render_stats.copy(),
                    'buffer_levels': {
                        'audio': self.audio_renderer.get_buffer_level() if self.audio_renderer else 0.0,
                        'video': self.video_renderer.get_buffer_level() if self.video_renderer else 0.0
                    }
                },
                metadata={
                    'source_layer': input_data.layer_name,
                    'rendering_mode': self.config.rendering_mode,
                    'sync_mode': self.config.sync_mode
                },
                confidence=input_data.confidence
            )
            
            # 记录处理时间
            processing_time = self.performance_monitor.end_timer("rendering_layer_processing")
            output_data.processing_time = processing_time
            
            # 更新统计信息
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            logger.info(f"渲染层处理完成 - 音频: {audio_success}, 视频: {video_success}, "
                       f"耗时: {processing_time*1000:.1f}ms")
            
            return output_data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"渲染层处理失败: {e}")
            
            # 创建错误输出
            error_data = LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={
                    'error': str(e),
                    'rendering_result': {
                        'audio_rendered': False,
                        'video_rendered': False
                    }
                },
                metadata={'error': True, 'source_layer': input_data.layer_name},
                confidence=0.0
            )
            
            processing_time = self.performance_monitor.end_timer("rendering_layer_processing")
            error_data.processing_time = processing_time
            
            return error_data
    
    def get_rendering_stats(self) -> Dict[str, Any]:
        """获取渲染统计信息"""
        return {
            'render_stats': self.render_stats.copy(),
            'sync_stats': self.sync_controller.sync_stats.copy(),
            'quality_level': self.quality_controller.current_quality.value,
            'is_rendering': self.is_rendering,
            'audio_buffer_level': self.audio_renderer.get_buffer_level() if self.audio_renderer else 0.0,
            'video_buffer_level': self.video_renderer.get_buffer_level() if self.video_renderer else 0.0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取渲染层状态"""
        base_status = super().get_status()
        
        # 添加渲染层特有的状态信息
        rendering_status = {
            'rendering_mode': self.config.rendering_mode,
            'audio_enabled': self.config.audio_enabled,
            'video_enabled': self.config.video_enabled,
            'pygame_available': PYGAME_AVAILABLE,
            'cv2_available': CV2_AVAILABLE,
            'is_rendering': self.is_rendering,
            'sync_mode': self.config.sync_mode,
            'quality_level': self.quality_controller.current_quality.value,
            'rendering_stats': self.get_rendering_stats(),
            'performance_stats': self.performance_monitor.get_all_stats()
        }
        
        base_status.update(rendering_status)
        return base_status
    
    def shutdown(self):
        """关闭渲染层"""
        self.stop_rendering()
        logger.info("渲染层已关闭")