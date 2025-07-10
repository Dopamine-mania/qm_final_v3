#!/usr/bin/env python3
"""
核心工具模块

提供系统所需的基础工具函数，包括：
- 数据验证器
- 性能监控器
- 配置加载器
- 通用工具函数
"""

import time
import yaml
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_text(text: str, max_length: int = 1000) -> bool:
        """验证文本数据"""
        if not isinstance(text, str):
            return False
        if len(text) > max_length:
            return False
        if not text.strip():
            return False
        return True
    
    @staticmethod
    def validate_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """验证音频数据"""
        if not isinstance(audio_data, np.ndarray):
            return False
        if audio_data.ndim != 1:
            return False
        if len(audio_data) == 0:
            return False
        return True
    
    @staticmethod
    def validate_video_frame(frame: np.ndarray) -> bool:
        """验证视频帧数据"""
        if not isinstance(frame, np.ndarray):
            return False
        if frame.ndim != 3:
            return False
        if frame.shape[2] != 3:  # RGB channels
            return False
        return True
    
    @staticmethod
    def validate_emotion_vector(emotion_vector: np.ndarray, dimensions: int = 27) -> bool:
        """验证情绪向量"""
        if not isinstance(emotion_vector, np.ndarray):
            return False
        if emotion_vector.shape[0] != dimensions:
            return False
        if not np.all(np.isfinite(emotion_vector)):
            return False
        return True

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name not in self.start_times:
            logger.warning(f"计时器 {name} 未启动")
            return 0.0
        
        elapsed = time.perf_counter() - self.start_times[name]
        
        # 记录指标
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(elapsed)
        
        # 保持最近1000次记录
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        del self.start_times[name]
        return elapsed
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取性能统计"""
        if name not in self.metrics:
            return {}
        
        times = self.metrics[name]
        return {
            'count': len(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有性能统计"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}

class ConfigLoader:
    """配置加载器"""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载YAML文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """加载JSON配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], file_path: str):
        """保存YAML配置文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"保存YAML文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str):
        """保存JSON配置文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存JSON文件失败: {file_path}, 错误: {e}")
            raise

def setup_logging(
    level: str = "INFO",
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None
):
    """设置日志配置"""
    
    # 设置日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(format_str)
    
    # 设置root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加file handler（如果指定了日志文件）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent

def ensure_directory(path: Union[str, Path]):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def calculate_memory_usage() -> Dict[str, float]:
    """计算内存使用量"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil未安装，无法获取内存使用信息")
        return {}

def calculate_cpu_usage() -> float:
    """计算CPU使用率"""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        logger.warning("psutil未安装，无法获取CPU使用信息")
        return 0.0

def get_gpu_info() -> Dict[str, Any]:
    """获取GPU信息"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,
                'memory_reserved': torch.cuda.memory_reserved() / 1024 / 1024
            }
        else:
            return {'available': False}
    except ImportError:
        return {'available': False, 'error': 'PyTorch未安装'}

class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'memory': calculate_memory_usage(),
            'cpu_usage': calculate_cpu_usage(),
            'gpu_info': get_gpu_info(),
            'timestamp': datetime.now().isoformat()
        }
    
    def log_system_stats(self):
        """记录系统统计信息"""
        stats = self.get_system_stats()
        logger.info(f"系统资源状态: {stats}")

# 工具函数
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """归一化向量"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    return np.dot(vec1_norm, vec2_norm)

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算欧几里得距离"""
    return np.linalg.norm(vec1 - vec2)

def smooth_transition(start_value: float, end_value: float, progress: float) -> float:
    """平滑过渡函数"""
    # 使用平滑的S曲线
    smooth_progress = 3 * progress**2 - 2 * progress**3
    return start_value + (end_value - start_value) * smooth_progress

def interpolate_values(values: List[float], num_steps: int) -> List[float]:
    """值插值"""
    if len(values) < 2:
        return values
    
    result = []
    for i in range(len(values) - 1):
        start_val = values[i]
        end_val = values[i + 1]
        
        for j in range(num_steps):
            progress = j / num_steps
            interpolated = smooth_transition(start_val, end_val, progress)
            result.append(interpolated)
    
    result.append(values[-1])  # 添加最后一个值
    return result