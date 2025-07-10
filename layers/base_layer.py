#!/usr/bin/env python3
"""
六层架构基础接口定义

定义了所有层需要实现的标准接口，确保层间通信的一致性和可扩展性。
基于适配器模式和策略模式，支持不同的实现策略。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import logging

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class LayerData:
    """
    层间数据传输标准格式
    
    所有层之间传递的数据都使用这个标准格式，确保数据的一致性和可追踪性。
    """
    layer_name: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class LayerConfig:
    """
    层配置基类
    
    每个层的配置都继承自这个基类，提供通用的配置参数。
    """
    layer_name: str
    enabled: bool = True
    debug_mode: bool = False
    performance_tracking: bool = True
    max_processing_time: float = 100.0  # ms
    
class LayerInterface(ABC):
    """
    层接口抽象基类
    
    所有六层架构的层都必须实现这个接口，确保一致的行为和可互换性。
    """
    
    @abstractmethod
    def __init__(self, config: LayerConfig):
        """初始化层"""
        pass
    
    @abstractmethod
    async def process(self, input_data: LayerData) -> LayerData:
        """
        处理输入数据并返回处理结果
        
        Args:
            input_data: 上一层传递的数据
            
        Returns:
            LayerData: 处理后的数据，传递给下一层
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取层的当前状态"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置层状态"""
        pass

class BaseLayer(LayerInterface):
    """
    层基类实现
    
    提供了LayerInterface的默认实现，包含通用的功能如性能监控、日志记录等。
    具体的层可以继承这个基类并重写process方法。
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.layer_name = config.layer_name
        self.processing_times = []
        self.error_count = 0
        self.total_processed = 0
        self.logger = logging.getLogger(f"{__name__}.{self.layer_name}")
        
        if config.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            
        self.logger.info(f"初始化 {self.layer_name} 层")
    
    async def process(self, input_data: LayerData) -> LayerData:
        """
        默认处理流程，包含性能监控和错误处理
        """
        start_time = datetime.now()
        
        try:
            # 验证输入数据
            self._validate_input(input_data)
            
            # 执行具体的处理逻辑
            result = await self._process_impl(input_data)
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time = processing_time
            
            # 记录性能数据
            if self.config.performance_tracking:
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 1000:  # 保持最近1000次记录
                    self.processing_times = self.processing_times[-1000:]
            
            # 检查处理时间是否超限
            if processing_time > self.config.max_processing_time:
                self.logger.warning(
                    f"{self.layer_name} 处理时间超限: {processing_time:.2f}ms > {self.config.max_processing_time}ms"
                )
            
            self.total_processed += 1
            
            if self.config.debug_mode:
                self.logger.debug(f"{self.layer_name} 处理完成，耗时: {processing_time:.2f}ms")
                
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"{self.layer_name} 处理错误: {str(e)}")
            
            # 返回错误结果
            return LayerData(
                layer_name=self.layer_name,
                timestamp=datetime.now(),
                data={"error": str(e)},
                metadata={"error_type": type(e).__name__},
                confidence=0.0
            )
    
    @abstractmethod
    async def _process_impl(self, input_data: LayerData) -> LayerData:
        """
        具体的处理逻辑实现
        
        子类必须实现这个方法来定义层的具体行为。
        """
        pass
    
    def _validate_input(self, input_data: LayerData) -> None:
        """
        验证输入数据的有效性
        """
        if not isinstance(input_data, LayerData):
            raise ValueError(f"输入数据必须是 LayerData 类型，实际类型: {type(input_data)}")
        
        if not input_data.data:
            raise ValueError("输入数据不能为空")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取层的详细状态信息
        """
        avg_processing_time = (
            np.mean(self.processing_times) if self.processing_times else 0.0
        )
        
        return {
            "layer_name": self.layer_name,
            "enabled": self.config.enabled,
            "total_processed": self.total_processed,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.total_processed),
            "avg_processing_time_ms": avg_processing_time,
            "max_processing_time_ms": max(self.processing_times) if self.processing_times else 0.0,
            "min_processing_time_ms": min(self.processing_times) if self.processing_times else 0.0,
            "recent_processing_times": self.processing_times[-10:] if self.processing_times else []
        }
    
    def reset(self) -> None:
        """
        重置层状态
        """
        self.processing_times = []
        self.error_count = 0
        self.total_processed = 0
        self.logger.info(f"{self.layer_name} 状态已重置")

class LayerPipeline:
    """
    层管道 - 管理六层架构的数据流
    
    负责协调各层之间的数据传递，监控整体性能，处理异常情况。
    """
    
    def __init__(self, layers: List[LayerInterface]):
        self.layers = layers
        self.layer_count = len(layers)
        self.pipeline_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_total_time": 0.0
        }
        self.logger = logging.getLogger(f"{__name__}.LayerPipeline")
        
        self.logger.info(f"初始化层管道，包含 {self.layer_count} 层")
    
    async def process(self, initial_data: LayerData) -> LayerData:
        """
        通过整个管道处理数据
        """
        start_time = datetime.now()
        self.pipeline_stats["total_requests"] += 1
        
        try:
            current_data = initial_data
            
            for i, layer in enumerate(self.layers):
                self.logger.debug(f"处理第 {i+1}/{self.layer_count} 层: {layer.layer_name}")
                
                current_data = await layer.process(current_data)
                
                # 检查是否有错误
                if "error" in current_data.data:
                    self.logger.error(f"层 {layer.layer_name} 处理失败: {current_data.data['error']}")
                    self.pipeline_stats["failed_requests"] += 1
                    return current_data
            
            # 计算总处理时间
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            current_data.metadata["total_pipeline_time"] = total_time
            
            # 更新统计信息
            self.pipeline_stats["successful_requests"] += 1
            self._update_avg_time(total_time)
            
            self.logger.info(f"管道处理完成，总耗时: {total_time:.2f}ms")
            
            return current_data
            
        except Exception as e:
            self.pipeline_stats["failed_requests"] += 1
            self.logger.error(f"管道处理异常: {str(e)}")
            
            return LayerData(
                layer_name="pipeline",
                timestamp=datetime.now(),
                data={"error": str(e)},
                metadata={"error_type": type(e).__name__},
                confidence=0.0
            )
    
    def _update_avg_time(self, new_time: float) -> None:
        """
        更新平均处理时间
        """
        total_successful = self.pipeline_stats["successful_requests"]
        current_avg = self.pipeline_stats["avg_total_time"]
        
        # 使用增量平均算法
        self.pipeline_stats["avg_total_time"] = (
            (current_avg * (total_successful - 1) + new_time) / total_successful
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        获取整个管道的状态
        """
        layer_statuses = [layer.get_status() for layer in self.layers]
        
        return {
            "pipeline_stats": self.pipeline_stats,
            "layer_count": self.layer_count,
            "layer_statuses": layer_statuses,
            "total_avg_time_ms": self.pipeline_stats["avg_total_time"],
            "success_rate": (
                self.pipeline_stats["successful_requests"] / 
                max(1, self.pipeline_stats["total_requests"])
            )
        }
    
    def reset_all(self) -> None:
        """
        重置所有层和管道统计
        """
        for layer in self.layers:
            layer.reset()
        
        self.pipeline_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_total_time": 0.0
        }
        
        self.logger.info("管道和所有层状态已重置")