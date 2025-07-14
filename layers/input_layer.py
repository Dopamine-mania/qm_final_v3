#!/usr/bin/env python3
"""
输入层 (Input Layer) - 六层架构的第一层

负责多模态数据的采集和预处理，支持：
1. 文本输入：用户文字描述的情绪状态
2. 音频输入：实时语音流的采集和预处理
3. 视频输入：摄像头视频流的面部表情捕获

设计特点：
- 异步多模态数据采集
- 实时数据缓冲和同步
- 数据质量检测和过滤
- 硬件自适应配置
"""

import asyncio
import numpy as np
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import yaml

from .base_layer import BaseLayer, LayerData, LayerConfig
from core.utils import DataValidator, PerformanceMonitor

# 条件导入外部依赖
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None
    wave = None

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class InputLayerConfig(LayerConfig):
    """输入层配置"""
    layer_name: str = "InputLayer"
    
    # 多模态配置
    text_enabled: bool = True
    audio_enabled: bool = True
    video_enabled: bool = True
    
    # 文本输入配置
    text_max_length: int = 1000
    text_encoding: str = "utf-8"
    
    # 音频配置
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_size: int = 1024
    audio_format: int = 8 if not PYAUDIO_AVAILABLE else pyaudio.paInt16  # paInt16 = 8
    
    # 视频配置
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 30
    video_device_id: int = 0
    
    # 缓冲配置
    buffer_size: int = 100
    buffer_timeout_ms: int = 1000
    
    # 数据同步配置
    sync_tolerance_ms: int = 100
    sync_enabled: bool = True

class MultimodalDataBuffer:
    """多模态数据缓冲器"""
    
    def __init__(self, config: InputLayerConfig):
        self.config = config
        self.text_buffer = []
        self.audio_buffer = []
        self.video_buffer = []
        self.timestamps = []
        self.buffer_lock = threading.Lock()
        
    def add_text(self, text: str, timestamp: datetime = None):
        """添加文本数据"""
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.buffer_lock:
            self.text_buffer.append({
                'data': text,
                'timestamp': timestamp,
                'type': 'text'
            })
            self._cleanup_buffer()
    
    def add_audio(self, audio_data: np.ndarray, timestamp: datetime = None):
        """添加音频数据"""
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.buffer_lock:
            self.audio_buffer.append({
                'data': audio_data,
                'timestamp': timestamp,
                'type': 'audio'
            })
            self._cleanup_buffer()
    
    def add_video(self, video_frame: np.ndarray, timestamp: datetime = None):
        """添加视频帧数据"""
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.buffer_lock:
            self.video_buffer.append({
                'data': video_frame,
                'timestamp': timestamp,
                'type': 'video'
            })
            self._cleanup_buffer()
    
    def get_synchronized_data(self) -> Dict[str, Any]:
        """获取同步的多模态数据"""
        with self.buffer_lock:
            current_time = datetime.now()
            
            # 查找最近的数据
            recent_text = self._get_recent_data(self.text_buffer, current_time)
            recent_audio = self._get_recent_data(self.audio_buffer, current_time)
            recent_video = self._get_recent_data(self.video_buffer, current_time)
            
            return {
                'text': recent_text,
                'audio': recent_audio,
                'video': recent_video,
                'timestamp': current_time,
                'sync_quality': self._calculate_sync_quality(recent_text, recent_audio, recent_video)
            }
    
    def _get_recent_data(self, buffer: List[Dict], current_time: datetime) -> Optional[Dict]:
        """获取最近的数据"""
        if not buffer:
            return None
            
        # 查找时间窗口内的最新数据
        tolerance_ms = self.config.sync_tolerance_ms
        
        for item in reversed(buffer):
            time_diff = (current_time - item['timestamp']).total_seconds() * 1000
            if time_diff <= tolerance_ms:
                return item
                
        return None
    
    def _calculate_sync_quality(self, text_data, audio_data, video_data) -> float:
        """计算同步质量"""
        available_modalities = sum([
            1 if text_data else 0,
            1 if audio_data else 0,
            1 if video_data else 0
        ])
        
        total_modalities = sum([
            1 if self.config.text_enabled else 0,
            1 if self.config.audio_enabled else 0,
            1 if self.config.video_enabled else 0
        ])
        
        return available_modalities / max(1, total_modalities)
    
    def _cleanup_buffer(self):
        """清理过期的缓冲数据"""
        current_time = datetime.now()
        timeout_ms = self.config.buffer_timeout_ms
        
        def cleanup_single_buffer(buffer):
            return [
                item for item in buffer
                if (current_time - item['timestamp']).total_seconds() * 1000 <= timeout_ms
            ]
        
        self.text_buffer = cleanup_single_buffer(self.text_buffer)
        self.audio_buffer = cleanup_single_buffer(self.audio_buffer)
        self.video_buffer = cleanup_single_buffer(self.video_buffer)
        
        # 保持缓冲区大小
        max_size = self.config.buffer_size
        if len(self.text_buffer) > max_size:
            self.text_buffer = self.text_buffer[-max_size:]
        if len(self.audio_buffer) > max_size:
            self.audio_buffer = self.audio_buffer[-max_size:]
        if len(self.video_buffer) > max_size:
            self.video_buffer = self.video_buffer[-max_size:]

class TextInputProcessor:
    """文本输入处理器"""
    
    def __init__(self, config: InputLayerConfig):
        self.config = config
        self.validator = DataValidator()
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """处理文本输入"""
        # 验证文本长度
        if len(text) > self.config.text_max_length:
            text = text[:self.config.text_max_length]
            
        # 基础清理
        text = self._clean_text(text)
        
        # 提取基础特征
        features = self._extract_text_features(text)
        
        return {
            'text': text,
            'features': features,
            'length': len(text),
            'word_count': len(text.split()),
            'encoding': self.config.text_encoding
        }
    
    def _clean_text(self, text: str) -> str:
        """清理文本数据"""
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        # 移除特殊字符（保留基本标点）
        import re
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:]', '', text)
        
        return text.strip()
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        # 情绪关键词检测
        emotion_keywords = self._detect_emotion_keywords(text)
        
        # 语义特征
        semantic_features = self._extract_semantic_features(text)
        
        return {
            'emotion_keywords': emotion_keywords,
            'semantic_features': semantic_features,
            'text_length': len(text),
            'sentence_count': len([s for s in text.split('.') if s.strip()])
        }
    
    def _detect_emotion_keywords(self, text: str) -> List[str]:
        """检测情绪关键词"""
        # 简化的情绪关键词检测
        emotion_keywords = {
            'anxiety': ['焦虑', '紧张', '担心', '害怕', '不安'],
            'sadness': ['悲伤', '难过', '失望', '沮丧', '伤心'],
            'anger': ['愤怒', '生气', '恼火', '烦躁', '气愤'],
            'fatigue': ['疲惫', '累', '疲劳', '困倦', '疲乏'],
            'stress': ['压力', '紧张', '繁忙', '忙碌', '紧迫']
        }
        
        detected = []
        for category, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected.append(category)
                
        return detected
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """提取语义特征"""
        # 这里可以集成更复杂的NLP模型
        # 目前返回基础特征
        return {
            'sentiment_polarity': 0.0,  # 情感极性
            'subjectivity': 0.0,        # 主观性
            'complexity': len(text.split()) / 20.0  # 复杂度
        }

class AudioInputProcessor:
    """音频输入处理器"""
    
    def __init__(self, config: InputLayerConfig):
        self.config = config
        self.audio_stream = None
        self.is_recording = False
        
    def start_recording(self):
        """开始录音"""
        try:
            if not PYAUDIO_AVAILABLE:
                logger.warning("PyAudio不可用，跳过音频录制")
                return
            self.audio_interface = pyaudio.PyAudio()
            self.audio_stream = self.audio_interface.open(
                format=self.config.audio_format,
                channels=self.config.audio_channels,
                rate=self.config.audio_sample_rate,
                input=True,
                frames_per_buffer=self.config.audio_chunk_size
            )
            self.is_recording = True
            logger.info("音频录制开始")
        except Exception as e:
            logger.error(f"启动音频录制失败: {e}")
            raise
    
    def stop_recording(self):
        """停止录音"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_interface.terminate()
            self.is_recording = False
            logger.info("音频录制停止")
    
    def read_audio_chunk(self) -> Optional[np.ndarray]:
        """读取音频块"""
        if not self.is_recording or not self.audio_stream:
            return None
            
        try:
            audio_data = self.audio_stream.read(
                self.config.audio_chunk_size,
                exception_on_overflow=False
            )
            
            # 转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 规范化
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            return audio_array
        except Exception as e:
            logger.error(f"读取音频数据失败: {e}")
            return None
    
    def process_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """处理音频数据"""
        if audio_data is None or len(audio_data) == 0:
            return None
            
        # 提取音频特征
        features = self._extract_audio_features(audio_data)
        
        return {
            'audio_data': audio_data,
            'features': features,
            'sample_rate': self.config.audio_sample_rate,
            'duration': len(audio_data) / self.config.audio_sample_rate,
            'rms_energy': np.sqrt(np.mean(audio_data**2))
        }
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """提取音频特征"""
        # 基础音频特征
        rms_energy = np.sqrt(np.mean(audio_data**2))
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2
        
        # 频谱特征（简化版）
        fft = np.fft.rfft(audio_data)
        spectrum_magnitude = np.abs(fft)
        spectral_centroid = np.sum(spectrum_magnitude * np.arange(len(spectrum_magnitude))) / np.sum(spectrum_magnitude)
        
        return {
            'rms_energy': float(rms_energy),
            'zero_crossing_rate': float(zero_crossing_rate),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': 0.0,  # 可以进一步实现
            'mfcc_features': []       # 可以集成librosa等库
        }

class VideoInputProcessor:
    """视频输入处理器"""
    
    def __init__(self, config: InputLayerConfig):
        self.config = config
        self.video_capture = None
        self.is_capturing = False
        
    def start_capture(self):
        """开始视频捕获"""
        try:
            if not CV2_AVAILABLE:
                logger.warning("OpenCV不可用，跳过视频捕获")
                return
            self.video_capture = cv2.VideoCapture(self.config.video_device_id)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video_height)
            self.video_capture.set(cv2.CAP_PROP_FPS, self.config.video_fps)
            self.is_capturing = True
            logger.info("视频捕获开始")
        except Exception as e:
            logger.error(f"启动视频捕获失败: {e}")
            raise
    
    def stop_capture(self):
        """停止视频捕获"""
        if self.video_capture:
            self.video_capture.release()
            self.is_capturing = False
            logger.info("视频捕获停止")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取视频帧"""
        if not self.is_capturing or not self.video_capture:
            return None
            
        try:
            ret, frame = self.video_capture.read()
            if ret:
                return frame
            return None
        except Exception as e:
            logger.error(f"读取视频帧失败: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理视频帧"""
        if frame is None:
            return None
            
        # 提取面部特征
        face_features = self._extract_face_features(frame)
        
        return {
            'frame': frame,
            'face_features': face_features,
            'frame_size': frame.shape,
            'timestamp': datetime.now()
        }
    
    def _extract_face_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """提取面部特征"""
        if not CV2_AVAILABLE:
            # 返回默认值，当OpenCV不可用时
            return {
                'faces_detected': 0,
                'face_info': [],
                'frame_quality': 0.5  # 默认质量评分
            }
        
        # 简化的面部特征提取
        # 实际应用中可以集成MediaPipe、OpenCV等库
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 面部检测（使用OpenCV的预训练模型）
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_info = []
        for (x, y, w, h) in faces:
            face_info.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.8,  # 简化的置信度
                'landmarks': []     # 可以进一步实现关键点检测
            })
        
        return {
            'faces_detected': len(faces),
            'face_info': face_info,
            'frame_quality': self._assess_frame_quality(frame)
        }
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """评估帧质量"""
        if not CV2_AVAILABLE:
            # 当OpenCV不可用时返回默认质量评分
            return 0.5
        
        # 简化的帧质量评估
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算拉普拉斯方差（锐度指标）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 规范化到0-1范围
        quality_score = min(1.0, laplacian_var / 1000.0)
        
        return quality_score

class InputLayer(BaseLayer):
    """输入层主类"""
    
    def __init__(self, config: InputLayerConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化处理器
        self.text_processor = TextInputProcessor(config)
        self.audio_processor = AudioInputProcessor(config) if config.audio_enabled else None
        self.video_processor = VideoInputProcessor(config) if config.video_enabled else None
        
        # 初始化缓冲器
        self.data_buffer = MultimodalDataBuffer(config)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 后台任务
        self.background_tasks = []
        self.shutdown_event = threading.Event()
        
        # 启动后台处理
        self._start_background_processing()
    
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """处理输入数据"""
        try:
            # 获取同步的多模态数据
            multimodal_data = self.data_buffer.get_synchronized_data()
            
            # 处理各模态数据
            processed_data = {}
            
            # 处理文本数据
            if self.config.text_enabled and multimodal_data.get('text'):
                text_result = self.text_processor.process_text(
                    multimodal_data['text']['data']
                )
                processed_data['text'] = text_result
            
            # 处理音频数据
            if self.config.audio_enabled and multimodal_data.get('audio'):
                audio_result = self.audio_processor.process_audio(
                    multimodal_data['audio']['data']
                )
                processed_data['audio'] = audio_result
            
            # 处理视频数据
            if self.config.video_enabled and multimodal_data.get('video'):
                video_result = self.video_processor.process_frame(
                    multimodal_data['video']['data']
                )
                processed_data['video'] = video_result
            
            # 计算数据质量
            data_quality = self._calculate_data_quality(processed_data)
            
            # 准备输出数据
            output_data = {
                'multimodal_data': processed_data,
                'data_quality': data_quality,
                'sync_quality': multimodal_data.get('sync_quality', 0.0),
                'timestamp': datetime.now(),
                'enabled_modalities': {
                    'text': self.config.text_enabled,
                    'audio': self.config.audio_enabled,
                    'video': self.config.video_enabled
                }
            }
            
            return LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data=output_data,
                metadata={
                    'processing_stage': 'input_layer_complete',
                    'data_quality': data_quality
                },
                confidence=data_quality
            )
            
        except Exception as e:
            logger.error(f"输入层处理失败: {e}")
            raise
    
    def _calculate_data_quality(self, processed_data: Dict[str, Any]) -> float:
        """计算数据质量"""
        quality_scores = []
        
        # 文本质量
        if 'text' in processed_data:
            text_quality = min(1.0, processed_data['text']['length'] / 50.0)
            quality_scores.append(text_quality)
        
        # 音频质量
        if 'audio' in processed_data:
            audio_quality = min(1.0, processed_data['audio']['rms_energy'] * 10)
            quality_scores.append(audio_quality)
        
        # 视频质量
        if 'video' in processed_data:
            video_quality = processed_data['video']['face_features']['frame_quality']
            quality_scores.append(video_quality)
        
        # 返回平均质量
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _start_background_processing(self):
        """启动后台处理任务"""
        if self.config.audio_enabled and self.audio_processor:
            audio_thread = threading.Thread(
                target=self._background_audio_processing,
                daemon=True
            )
            audio_thread.start()
            self.background_tasks.append(audio_thread)
        
        if self.config.video_enabled and self.video_processor:
            video_thread = threading.Thread(
                target=self._background_video_processing,
                daemon=True
            )
            video_thread.start()
            self.background_tasks.append(video_thread)
    
    def _background_audio_processing(self):
        """后台音频处理"""
        try:
            self.audio_processor.start_recording()
            
            while not self.shutdown_event.is_set():
                audio_chunk = self.audio_processor.read_audio_chunk()
                if audio_chunk is not None:
                    self.data_buffer.add_audio(audio_chunk)
                
                # 避免CPU过度使用
                threading.Event().wait(0.01)
                
        except Exception as e:
            logger.error(f"后台音频处理错误: {e}")
        finally:
            self.audio_processor.stop_recording()
    
    def _background_video_processing(self):
        """后台视频处理"""
        try:
            self.video_processor.start_capture()
            
            while not self.shutdown_event.is_set():
                frame = self.video_processor.read_frame()
                if frame is not None:
                    self.data_buffer.add_video(frame)
                
                # 控制帧率
                threading.Event().wait(1.0 / self.config.video_fps)
                
        except Exception as e:
            logger.error(f"后台视频处理错误: {e}")
        finally:
            self.video_processor.stop_capture()
    
    def add_text_input(self, text: str):
        """添加文本输入"""
        self.data_buffer.add_text(text)
        logger.debug(f"添加文本输入: {text[:50]}...")
    
    # 根据用户规范添加的标准化接口函数
    def capture_video_frame(self) -> np.ndarray:
        """摄像头数据捕获模拟
        
        Returns:
            np.ndarray: 视频帧数据，形状为(height, width, channels)
        """
        try:
            if self.config.video_enabled and self.video_processor and self.video_processor.is_capturing:
                frame = self.video_processor.read_frame()
                if frame is not None:
                    return frame
            
            # 返回模拟视频帧数据
            height, width = self.config.video_height, self.config.video_width
            mock_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            logger.debug("返回模拟视频帧数据")
            return mock_frame
            
        except Exception as e:
            logger.warning(f"视频帧捕获失败，返回模拟数据: {e}")
            # 返回默认模拟帧
            height, width = self.config.video_height, self.config.video_width
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def capture_audio_chunk(self) -> np.ndarray:
        """麦克风数据捕获模拟
        
        Returns:
            np.ndarray: 音频数据块，一维数组
        """
        try:
            if self.config.audio_enabled and self.audio_processor and self.audio_processor.is_recording:
                audio_chunk = self.audio_processor.read_audio_chunk()
                if audio_chunk is not None:
                    return audio_chunk
            
            # 返回模拟音频数据
            chunk_size = self.config.audio_chunk_size
            mock_audio = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
            logger.debug("返回模拟音频数据")
            return mock_audio
            
        except Exception as e:
            logger.warning(f"音频数据捕获失败，返回模拟数据: {e}")
            # 返回静音数据
            chunk_size = self.config.audio_chunk_size
            return np.zeros(chunk_size, dtype=np.float32)
    
    def get_user_text_input(self) -> str:
        """文本输入模拟
        
        Returns:
            str: 用户文本输入
        """
        try:
            # 从缓冲区获取最新的文本输入
            with self.data_buffer.buffer_lock:
                if self.data_buffer.text_buffer:
                    latest_text = self.data_buffer.text_buffer[-1]['data']
                    logger.debug(f"返回缓冲区文本: {latest_text[:50]}...")
                    return latest_text
            
            # 返回模拟文本输入
            mock_texts = [
                "我感到有些焦虑，需要放松一下",
                "今天很累，希望能快点入睡",
                "心情不太好，想要安静的环境",
                "压力很大，需要缓解一下",
                "睡眠质量不好，想改善"
            ]
            import random
            mock_text = random.choice(mock_texts)
            logger.debug(f"返回模拟文本输入: {mock_text}")
            return mock_text
            
        except Exception as e:
            logger.warning(f"文本输入获取失败，返回默认文本: {e}")
            return "用户情绪输入模拟"
    
    def collect_multimodal_data(self) -> Dict[str, Any]:
        """多模态数据收集主函数
        
        整合视频、音频、文本三种模态的数据收集
        
        Returns:
            Dict[str, Any]: 包含所有模态数据的字典
        """
        try:
            collected_data = {}
            collection_timestamp = datetime.now()
            
            # 收集视频数据
            if self.config.video_enabled:
                video_frame = self.capture_video_frame()
                processed_video = self.video_processor.process_frame(video_frame) if self.video_processor else None
                collected_data['video'] = {
                    'raw_frame': video_frame,
                    'processed': processed_video,
                    'enabled': True
                }
            else:
                collected_data['video'] = {
                    'raw_frame': None,
                    'processed': None,
                    'enabled': False
                }
            
            # 收集音频数据
            if self.config.audio_enabled:
                audio_chunk = self.capture_audio_chunk()
                processed_audio = self.audio_processor.process_audio(audio_chunk) if self.audio_processor else None
                collected_data['audio'] = {
                    'raw_chunk': audio_chunk,
                    'processed': processed_audio,
                    'enabled': True
                }
            else:
                collected_data['audio'] = {
                    'raw_chunk': None,
                    'processed': None,
                    'enabled': False
                }
            
            # 收集文本数据
            if self.config.text_enabled:
                text_input = self.get_user_text_input()
                processed_text = self.text_processor.process_text(text_input)
                collected_data['text'] = {
                    'raw_text': text_input,
                    'processed': processed_text,
                    'enabled': True
                }
            else:
                collected_data['text'] = {
                    'raw_text': None,
                    'processed': None,
                    'enabled': False
                }
            
            # 添加元数据
            collected_data['metadata'] = {
                'collection_timestamp': collection_timestamp,
                'enabled_modalities': {
                    'video': self.config.video_enabled,
                    'audio': self.config.audio_enabled,
                    'text': self.config.text_enabled
                },
                'collection_mode': 'multimodal_capture',
                'data_quality': self._calculate_multimodal_quality(collected_data)
            }
            
            logger.debug("多模态数据收集完成")
            return collected_data
            
        except Exception as e:
            logger.error(f"多模态数据收集失败: {e}")
            raise
    
    def _calculate_multimodal_quality(self, collected_data: Dict[str, Any]) -> float:
        """计算多模态数据质量
        
        Args:
            collected_data: 收集到的多模态数据
            
        Returns:
            float: 数据质量评分 (0-1)
        """
        quality_scores = []
        
        # 视频质量评估
        if collected_data['video']['enabled'] and collected_data['video']['processed']:
            video_quality = collected_data['video']['processed']['face_features']['frame_quality']
            quality_scores.append(video_quality)
        
        # 音频质量评估
        if collected_data['audio']['enabled'] and collected_data['audio']['processed']:
            audio_rms = collected_data['audio']['processed']['rms_energy']
            audio_quality = min(1.0, audio_rms * 10)  # 简单的质量评估
            quality_scores.append(audio_quality)
        
        # 文本质量评估
        if collected_data['text']['enabled'] and collected_data['text']['processed']:
            text_length = collected_data['text']['processed']['length']
            text_quality = min(1.0, text_length / 50.0)
            quality_scores.append(text_quality)
        
        # 返回平均质量分数
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def shutdown(self):
        """关闭输入层"""
        self.shutdown_event.set()
        
        # 等待后台任务完成
        for task in self.background_tasks:
            task.join(timeout=1.0)
        
        logger.info("输入层已关闭")

def create_input_layer(config_path: str = None) -> InputLayer:
    """创建输入层实例"""
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        config = InputLayerConfig(**config_data.get('input_layer', {}))
    else:
        config = InputLayerConfig()
    
    return InputLayer(config)