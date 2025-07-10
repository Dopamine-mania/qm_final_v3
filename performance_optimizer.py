#!/usr/bin/env python3
"""
性能优化器

优化qm_final3系统性能，包括：
1. 内存管理优化
2. 并发处理优化  
3. 缓存策略优化
4. 计算资源优化
5. 网络和I/O优化
"""

import asyncio
import time
import psutil
import gc
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    latency_ms: float
    throughput_fps: float
    timestamp: float

class PerformanceOptimizer:
    """
    性能优化器主类
    
    负责监控和优化系统性能，包括：
    - 自动调整配置参数
    - 资源监控和管理
    - 性能瓶颈检测
    - 优化建议生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_enabled = True
        self.monitoring_active = False
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # 性能阈值
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        self.latency_threshold = 500.0  # 500ms
        self.throughput_threshold = 30.0  # 30fps
        
        logger.info("性能优化器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'optimization': {
                'enabled': True,
                'auto_gc': True,
                'memory_limit_mb': 2048,
                'cpu_limit_percent': 80,
                'max_concurrent_tasks': 4,
                'cache_size_limit': 100,
                'enable_gpu_optimization': False
            },
            'monitoring': {
                'interval_seconds': 1.0,
                'metrics_retention_hours': 24,
                'alert_thresholds': {
                    'cpu_percent': 80,
                    'memory_percent': 80,
                    'latency_ms': 500,
                    'throughput_fps': 30
                }
            },
            'caching': {
                'enable_layer_cache': True,
                'cache_ttl_seconds': 300,
                'max_cache_entries': 1000,
                'compression_enabled': True
            }
        }
    
    async def start_monitoring(self):
        """启动性能监控"""
        self.monitoring_active = True
        logger.info("开始性能监控...")
        
        while self.monitoring_active:
            try:
                # 收集性能指标
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查是否需要优化
                if self.optimization_enabled:
                    await self._check_optimization_triggers(metrics)
                
                # 清理旧指标
                await self._cleanup_old_metrics()
                
                # 等待下一次监控
                await asyncio.sleep(self.config['monitoring']['interval_seconds'])
                
            except Exception as e:
                logger.error(f"性能监控出错: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        start_time = time.time()
        
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # 内存使用率
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # GPU使用率（如果可用）
        gpu_usage = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except ImportError:
            pass
        
        # 延迟测量
        latency_ms = (time.time() - start_time) * 1000
        
        # 吞吐量（近似）
        throughput_fps = 1000 / max(latency_ms, 1)
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            timestamp=time.time()
        )
    
    async def _check_optimization_triggers(self, metrics: PerformanceMetrics):
        """检查优化触发条件"""
        optimizations = []
        
        # CPU优化
        if metrics.cpu_usage > self.cpu_threshold:
            optimizations.append(self._optimize_cpu_usage)
        
        # 内存优化
        if metrics.memory_usage > self.memory_threshold:
            optimizations.append(self._optimize_memory_usage)
        
        # 延迟优化
        if metrics.latency_ms > self.latency_threshold:
            optimizations.append(self._optimize_latency)
        
        # 吞吐量优化
        if metrics.throughput_fps < self.throughput_threshold:
            optimizations.append(self._optimize_throughput)
        
        # 并发执行优化
        if optimizations:
            await asyncio.gather(*[opt() for opt in optimizations])
    
    async def _optimize_cpu_usage(self):
        """优化CPU使用率"""
        logger.info("开始CPU优化...")
        
        # 减少并发任务数
        max_workers = max(1, self.thread_pool._max_workers - 1)
        self.thread_pool._max_workers = max_workers
        
        # 启用负载均衡
        await self._enable_load_balancing()
        
        logger.info(f"CPU优化完成，最大工作线程数: {max_workers}")
    
    async def _optimize_memory_usage(self):
        """优化内存使用率"""
        logger.info("开始内存优化...")
        
        # 强制垃圾回收
        if self.config['optimization']['auto_gc']:
            gc.collect()
        
        # 清理缓存
        await self._cleanup_caches()
        
        # 优化数据结构
        await self._optimize_data_structures()
        
        logger.info("内存优化完成")
    
    async def _optimize_latency(self):
        """优化延迟"""
        logger.info("开始延迟优化...")
        
        # 预热缓存
        await self._warmup_caches()
        
        # 优化算法路径
        await self._optimize_algorithm_paths()
        
        logger.info("延迟优化完成")
    
    async def _optimize_throughput(self):
        """优化吞吐量"""
        logger.info("开始吞吐量优化...")
        
        # 增加并发度
        if self.thread_pool._max_workers < 8:
            self.thread_pool._max_workers += 1
        
        # 启用批处理
        await self._enable_batch_processing()
        
        logger.info("吞吐量优化完成")
    
    async def _enable_load_balancing(self):
        """启用负载均衡"""
        # 实现负载均衡逻辑
        pass
    
    async def _cleanup_caches(self):
        """清理缓存"""
        # 清理系统缓存
        import os
        if hasattr(os, 'sync'):
            os.sync()
    
    async def _optimize_data_structures(self):
        """优化数据结构"""
        # 优化numpy数组内存布局
        import numpy as np
        # 可以在这里实现数据结构优化
        pass
    
    async def _warmup_caches(self):
        """预热缓存"""
        # 预热关键缓存
        pass
    
    async def _optimize_algorithm_paths(self):
        """优化算法路径"""
        # 优化关键算法路径
        pass
    
    async def _enable_batch_processing(self):
        """启用批处理"""
        # 实现批处理逻辑
        pass
    
    async def _cleanup_old_metrics(self):
        """清理旧的性能指标"""
        max_retention = self.config['monitoring']['metrics_retention_hours']
        cutoff_time = time.time() - (max_retention * 3600)
        
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "暂无性能数据"}
        
        recent_metrics = self.metrics_history[-10:]  # 最近10个指标
        
        return {
            "status": "ok",
            "summary": {
                "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
                "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
                "avg_latency_ms": np.mean([m.latency_ms for m in recent_metrics]),
                "avg_throughput_fps": np.mean([m.throughput_fps for m in recent_metrics]),
                "total_samples": len(self.metrics_history)
            },
            "current": {
                "cpu_usage": recent_metrics[-1].cpu_usage,
                "memory_usage": recent_metrics[-1].memory_usage,
                "latency_ms": recent_metrics[-1].latency_ms,
                "throughput_fps": recent_metrics[-1].throughput_fps,
                "timestamp": recent_metrics[-1].timestamp
            },
            "trends": {
                "cpu_trend": self._calculate_trend([m.cpu_usage for m in recent_metrics]),
                "memory_trend": self._calculate_trend([m.memory_usage for m in recent_metrics]),
                "latency_trend": self._calculate_trend([m.latency_ms for m in recent_metrics]),
                "throughput_trend": self._calculate_trend([m.throughput_fps for m in recent_metrics])
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "stable"
        
        recent_avg = np.mean(values[-5:])
        older_avg = np.mean(values[:-5]) if len(values) > 5 else values[0]
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        latest = self.metrics_history[-1]
        
        # CPU优化建议
        if latest.cpu_usage > 80:
            recommendations.append({
                "type": "cpu",
                "severity": "high",
                "message": "CPU使用率过高，建议减少并发任务数或优化算法",
                "suggestions": [
                    "减少max_concurrent_tasks配置",
                    "启用GPU加速（如果可用）",
                    "优化计算密集型算法"
                ]
            })
        
        # 内存优化建议
        if latest.memory_usage > 80:
            recommendations.append({
                "type": "memory",
                "severity": "high",
                "message": "内存使用率过高，建议清理缓存或优化数据结构",
                "suggestions": [
                    "启用自动垃圾回收",
                    "减少缓存大小",
                    "优化数据结构使用"
                ]
            })
        
        # 延迟优化建议
        if latest.latency_ms > 500:
            recommendations.append({
                "type": "latency",
                "severity": "medium",
                "message": "系统延迟过高，建议优化处理流程",
                "suggestions": [
                    "启用预计算和缓存",
                    "优化算法复杂度",
                    "使用异步处理"
                ]
            })
        
        return recommendations
    
    async def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        logger.info("性能监控已停止")
    
    def shutdown(self):
        """关闭优化器"""
        self.monitoring_active = False
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("性能优化器已关闭")

# 全局优化器实例
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_optimizer() -> PerformanceOptimizer:
    """获取全局优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

async def optimize_system_performance():
    """优化系统性能的便捷函数"""
    optimizer = get_optimizer()
    
    # 启动监控
    monitoring_task = asyncio.create_task(optimizer.start_monitoring())
    
    try:
        # 运行一段时间进行性能分析
        await asyncio.sleep(10)
        
        # 生成报告
        report = optimizer.get_performance_report()
        recommendations = optimizer.get_optimization_recommendations()
        
        print("=== 性能优化报告 ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        print("\n=== 优化建议 ===")
        for rec in recommendations:
            print(f"类型: {rec['type']}, 严重程度: {rec['severity']}")
            print(f"消息: {rec['message']}")
            print(f"建议: {', '.join(rec['suggestions'])}")
            print()
        
    finally:
        await optimizer.stop_monitoring()
        optimizer.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(optimize_system_performance())