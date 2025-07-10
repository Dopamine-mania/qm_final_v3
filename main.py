#!/usr/bin/env python3
"""
qm_final3 主程序

六层架构的心境流转系统主入口，提供：
1. 系统初始化和配置加载
2. 六层架构的创建和管理
3. Web界面和API服务
4. 性能监控和日志记录

用法：
    python main.py                    # 启动完整系统
    python main.py --config custom.yaml  # 使用自定义配置
    python main.py --demo             # 演示模式
    python main.py --test             # 测试模式
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from core.utils import setup_logging, ConfigLoader, get_project_root
from layers.base_layer import LayerPipeline
from layers.input_layer import InputLayer, InputLayerConfig
from layers.fusion_layer import FusionLayer, FusionLayerConfig
from layers.mapping_layer import MappingLayer, MappingLayerConfig
from layers.generation_layer import GenerationLayer, GenerationLayerConfig
from layers.rendering_layer import RenderingLayer, RenderingLayerConfig
from layers.therapy_layer import TherapyLayer, TherapyLayerConfig

import logging

# 设置日志
logger = logging.getLogger(__name__)

class QMFinal3System:
    """
    qm_final3 系统主类
    
    负责整个六层架构系统的初始化、管理和协调。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path or "configs/six_layer_architecture.yaml"
        self.config = None
        self.layers = []
        self.pipeline = None
        self.is_running = False
        
        # 加载配置
        self._load_config()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化层
        self._initialize_layers()
        
        # 创建管道
        self._create_pipeline()
        
        logger.info("qm_final3 系统初始化完成")
    
    def _load_config(self):
        """加载配置文件"""
        try:
            config_full_path = get_project_root() / self.config_path
            logger.info(f"尝试加载配置文件: {config_full_path}")
            if not config_full_path.exists():
                logger.error(f"配置文件不存在: {config_full_path}")
                raise FileNotFoundError(f"配置文件不存在: {config_full_path}")
            
            self.config = ConfigLoader.load_yaml(str(config_full_path))
            logger.info(f"✅ 配置加载成功: {config_full_path}")
            logger.info(f"配置文件包含层数: {len(self.config.get('layers', {}))}")
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            logger.warning("使用默认配置（注意：默认配置可能不包含GPU优化）")
            # 使用默认配置
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'system': {
                'name': '心境流转 qm_final3',
                'version': '3.0.0',
                'debug_mode': False
            },
            'layers': {
                'input_layer': {
                    'enabled': True,
                    'layer_name': 'InputLayer',
                    'text_enabled': True,
                    'audio_enabled': False,  # 暂时禁用以简化测试
                    'video_enabled': False,  # 暂时禁用以简化测试
                    'max_processing_time': 50.0
                },
                'fusion_layer': {
                    'enabled': True,
                    'layer_name': 'FusionLayer',
                    'total_emotions': 27,
                    'base_emotions': 9,
                    'extended_emotions': 18,
                    'fusion_strategy': 'confidence_weighted',
                    'enable_emotion_relationships': True,
                    'use_gpu': True,  # 启用GPU加速
                    'max_processing_time': 150.0
                },
                'mapping_layer': {
                    'enabled': True,
                    'layer_name': 'MappingLayer',
                    'mapping_strategy': 'hybrid_fusion',
                    'kg_enabled': True,
                    'mlp_enabled': True,
                    'kg_weight': 0.6,
                    'mlp_weight': 0.4,
                    'sleep_therapy_mode': True,
                    'use_gpu': True,  # 启用GPU加速
                    'max_processing_time': 100.0
                },
                'generation_layer': {
                    'enabled': True,
                    'layer_name': 'GenerationLayer',
                    'generation_strategy': 'hybrid',
                    'content_type': 'both',
                    'audio_enabled': True,
                    'video_enabled': True,
                    'audio_duration': 5.0,   # 5秒实时生成
                    'video_duration': 5.0,   # 5秒实时生成
                    'video_fps': 15,          # 降低帧率
                    'video_resolution': [480, 270],  # 降低分辨率
                    'therapy_optimized': True,
                    'iso_stage_aware': True,
                    'binaural_beats': True,
                    'use_gpu': True,  # 启用GPU加速
                    'max_processing_time': 200.0
                },
                'rendering_layer': {
                    'enabled': True,
                    'layer_name': 'RenderingLayer',
                    'rendering_mode': 'local_playback',
                    'audio_enabled': True,
                    'video_enabled': True,
                    'sync_mode': 'timestamp_sync',
                    'quality_level': 'medium',
                    'adaptive_quality': True,
                    'audio_latency_ms': 50.0,  # 适中延迟
                    'video_latency_ms': 50.0,  # 适中延迟
                    'buffer_size_ms': 500.0,   # 500ms缓冲
                    'use_gpu': True,  # 启用GPU加速
                    'max_processing_time': 16.7  # ~60fps
                },
                'therapy_layer': {
                    'enabled': True,
                    'layer_name': 'TherapyLayer',
                    'default_session_duration': 600.0,  # 10分钟测试
                    'synchronization_duration_ratio': 0.25,
                    'guidance_duration_ratio': 0.50,
                    'consolidation_duration_ratio': 0.25,
                    'effectiveness_check_interval': 30.0,  # 每30秒检查
                    'enable_adaptive_adjustment': True,
                    'enable_session_recording': True,
                    'max_processing_time': 50.0
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        logging_config = self.config.get('logging', {})
        setup_logging(
            level=logging_config.get('level', 'INFO'),
            format_str=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            log_file=logging_config.get('file')
        )
    
    def _initialize_layers(self):
        """初始化所有层"""
        layers_config = self.config.get('layers', {})
        
        # 初始化输入层
        input_config = layers_config.get('input_layer', {})
        if input_config.get('enabled', True):
            # 处理参数兼容性和映射
            processed_config = {
                'layer_name': input_config.get('layer_name', 'InputLayer'),
                'enabled': input_config.get('enabled', True),
                'debug_mode': input_config.get('debug_mode', False),
                'max_processing_time': input_config.get('max_processing_time_ms', input_config.get('max_processing_time', 50.0)),
                # 模态配置
                'text_enabled': input_config.get('modalities', {}).get('text', {}).get('enabled', True),
                'audio_enabled': input_config.get('modalities', {}).get('audio', {}).get('enabled', False),
                'video_enabled': input_config.get('modalities', {}).get('video', {}).get('enabled', False),
            }
            input_layer_config = InputLayerConfig(**processed_config)
            input_layer = InputLayer(input_layer_config)
            self.layers.append(input_layer)
            logger.info("输入层初始化完成")
        
        # 初始化融合层
        fusion_config = layers_config.get('fusion_layer', {})
        if fusion_config.get('enabled', True):
            # 处理参数兼容性和映射
            processed_fusion_config = {
                'layer_name': fusion_config.get('layer_name', 'FusionLayer'),
                'enabled': fusion_config.get('enabled', True),
                'debug_mode': fusion_config.get('debug_mode', False),
                'max_processing_time': fusion_config.get('max_processing_time_ms', fusion_config.get('max_processing_time', 150.0)),
                # 融合层特有配置
                'total_emotions': fusion_config.get('emotion_dimensions', fusion_config.get('total_emotions', 27)),
                'fusion_strategy': fusion_config.get('fusion_strategy', 'confidence_weighted'),
                'enable_emotion_relationships': fusion_config.get('enable_emotion_relationships', True),
                'use_gpu': fusion_config.get('use_gpu', True),
            }
            fusion_layer_config = FusionLayerConfig(**processed_fusion_config)
            fusion_layer = FusionLayer(fusion_layer_config)
            self.layers.append(fusion_layer)
            logger.info("融合层初始化完成")
        
        # 初始化映射层
        mapping_config = layers_config.get('mapping_layer', {})
        if mapping_config.get('enabled', True):
            # 处理参数兼容性和映射
            processed_mapping_config = {
                'layer_name': mapping_config.get('layer_name', 'MappingLayer'),
                'enabled': mapping_config.get('enabled', True),
                'debug_mode': mapping_config.get('debug_mode', False),
                'max_processing_time': mapping_config.get('max_processing_time_ms', mapping_config.get('max_processing_time', 100.0)),
                # 映射层特有配置
                'mapping_strategy': mapping_config.get('mapping_strategy', 'hybrid_fusion'),
                'kg_enabled': mapping_config.get('kg_enabled', True),
                'mlp_enabled': mapping_config.get('mlp_enabled', True),
                'kg_weight': mapping_config.get('kg_weight', 0.6),
                'mlp_weight': mapping_config.get('mlp_weight', 0.4),
                'sleep_therapy_mode': mapping_config.get('sleep_therapy_mode', True),
                'use_gpu': mapping_config.get('use_gpu', True),
            }
            mapping_layer_config = MappingLayerConfig(**processed_mapping_config)
            mapping_layer = MappingLayer(mapping_layer_config)
            self.layers.append(mapping_layer)
            logger.info("映射层初始化完成")
        
        # 初始化生成层
        generation_config = layers_config.get('generation_layer', {})
        if generation_config.get('enabled', True):
            # 读取嵌套的video_generation配置
            video_gen_config = generation_config.get('video_generation', {})
            
            # 处理参数兼容性和映射
            processed_generation_config = {
                'layer_name': generation_config.get('layer_name', 'GenerationLayer'),
                'enabled': generation_config.get('enabled', True),
                'debug_mode': generation_config.get('debug_mode', False),
                'max_processing_time': generation_config.get('max_processing_time_ms', generation_config.get('max_processing_time', 200.0)),
                # 生成层特有配置
                'generation_strategy': generation_config.get('generation_strategy', 'hybrid'),
                'content_type': generation_config.get('content_type', 'both'),
                'audio_enabled': generation_config.get('audio_enabled', True),
                'video_enabled': generation_config.get('video_enabled', True),
                'audio_duration': video_gen_config.get('duration', 30.0),  # 与视频时长保持一致
                'video_duration': video_gen_config.get('duration', 30.0),  # 从video_generation读取
                'video_fps': video_gen_config.get('fps', 30),              # 从video_generation读取
                'video_resolution': video_gen_config.get('resolution', [1920, 1080]),  # 从video_generation读取
                'therapy_optimized': generation_config.get('therapy_optimized', True),
                'iso_stage_aware': generation_config.get('iso_stage_aware', True),
                'binaural_beats': generation_config.get('binaural_beats', True),
                'use_gpu': generation_config.get('use_gpu', True),
            }
            generation_layer_config = GenerationLayerConfig(**processed_generation_config)
            generation_layer = GenerationLayer(generation_layer_config)
            self.layers.append(generation_layer)
            logger.info("生成层初始化完成")
        
        # 初始化渲染层
        rendering_config = layers_config.get('rendering_layer', {})
        if rendering_config.get('enabled', True):
            # 处理参数兼容性和映射
            processed_rendering_config = {
                'layer_name': rendering_config.get('layer_name', 'RenderingLayer'),
                'enabled': rendering_config.get('enabled', True),
                'debug_mode': rendering_config.get('debug_mode', False),
                'max_processing_time': rendering_config.get('max_processing_time_ms', rendering_config.get('max_processing_time', 16.7)),
                # 渲染层特有配置
                'rendering_mode': rendering_config.get('rendering_mode', 'local_playback'),
                'audio_enabled': rendering_config.get('audio_enabled', True),
                'video_enabled': rendering_config.get('video_enabled', True),
                'sync_mode': rendering_config.get('sync_mode', 'timestamp_sync'),
                'quality_level': rendering_config.get('quality_level', 'medium'),
                'adaptive_quality': rendering_config.get('adaptive_quality', True),
                'audio_latency_ms': rendering_config.get('audio_latency_ms', 50.0),
                'video_latency_ms': rendering_config.get('video_latency_ms', 50.0),
                'buffer_size_ms': rendering_config.get('buffer_size_ms', 500.0),
                'use_gpu': rendering_config.get('use_gpu', True),
            }
            rendering_layer_config = RenderingLayerConfig(**processed_rendering_config)
            rendering_layer = RenderingLayer(rendering_layer_config)
            self.layers.append(rendering_layer)
            logger.info("渲染层初始化完成")
        
        # 初始化治疗层
        therapy_config = layers_config.get('therapy_layer', {})
        if therapy_config.get('enabled', True):
            # 处理参数兼容性和映射
            processed_therapy_config = {
                'layer_name': therapy_config.get('layer_name', 'TherapyLayer'),
                'enabled': therapy_config.get('enabled', True),
                'debug_mode': therapy_config.get('debug_mode', False),
                'max_processing_time': therapy_config.get('max_processing_time_ms', therapy_config.get('max_processing_time', 50.0)),
                # 治疗层特有配置
                'default_session_duration': therapy_config.get('default_session_duration', 600.0),
                'synchronization_duration_ratio': therapy_config.get('synchronization_duration_ratio', 0.25),
                'guidance_duration_ratio': therapy_config.get('guidance_duration_ratio', 0.50),
                'consolidation_duration_ratio': therapy_config.get('consolidation_duration_ratio', 0.25),
                'effectiveness_check_interval': therapy_config.get('effectiveness_check_interval', 30.0),
                'enable_adaptive_adjustment': therapy_config.get('enable_adaptive_adjustment', True),
                'enable_session_recording': therapy_config.get('enable_session_recording', True),
            }
            therapy_layer_config = TherapyLayerConfig(**processed_therapy_config)
            therapy_layer = TherapyLayer(therapy_layer_config)
            self.layers.append(therapy_layer)
            logger.info("治疗层初始化完成")
        
        logger.info(f"共初始化 {len(self.layers)} 层")
    
    def _create_pipeline(self):
        """创建处理管道"""
        if self.layers:
            self.pipeline = LayerPipeline(self.layers)
            logger.info("处理管道创建完成")
        else:
            logger.warning("没有可用的层，无法创建管道")
    
    async def start(self):
        """启动系统"""
        try:
            self.is_running = True
            logger.info("系统启动中...")
            
            # 系统状态检查
            await self._system_health_check()
            
            # 启动主循环
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            raise
    
    async def _system_health_check(self):
        """系统健康检查"""
        logger.info("开始系统健康检查...")
        
        # 检查配置
        if not self.config:
            raise RuntimeError("配置未加载")
        
        # 检查层状态
        if not self.layers:
            raise RuntimeError("没有可用的层")
        
        # 检查管道
        if not self.pipeline:
            raise RuntimeError("处理管道未创建")
        
        # 检查各层状态
        for layer in self.layers:
            status = layer.get_status()
            if not status.get('enabled', True):
                logger.warning(f"层 {layer.layer_name} 未启用")
        
        logger.info("系统健康检查通过")
    
    async def _main_loop(self):
        """主循环"""
        logger.info("🚀 进入演示主循环（按Ctrl+C停止）...")
        
        cycle_count = 0
        try:
            while self.is_running:
                cycle_count += 1
                logger.info(f"🔄 ===== 第 {cycle_count} 轮处理开始 =====")
                
                # 处理输入（这里简化为测试用例）
                await self._process_test_input()
                
                logger.info(f"✅ ===== 第 {cycle_count} 轮处理完成 =====")
                logger.info(f"⏱️  等待1秒后开始第 {cycle_count + 1} 轮...")
                
                # 等待一段时间
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭...")
            await self.stop()
    
    async def _process_test_input(self):
        """处理测试输入"""
        # 创建测试输入数据
        from layers.base_layer import LayerData
        from datetime import datetime
        
        # 添加文本输入到输入层
        if self.layers:
            input_layer = self.layers[0]  # 假设第一层是输入层
            if hasattr(input_layer, 'add_text_input'):
                input_layer.add_text_input("我今天感到很焦虑，躺在床上睡不着")
        
        # 创建测试数据
        test_data = LayerData(
            layer_name="system_test",
            timestamp=datetime.now(),
            data={"test_input": "测试数据"},
            metadata={"source": "main_loop"}
        )
        
        # 通过管道处理
        if self.pipeline:
            logger.info("🔗 开始6层管道处理...")
            result = await self.pipeline.process(test_data)
            logger.info(f"🎯 6层管道处理结果: {result.layer_name}, 置信度: {result.confidence:.2f}")
    
    async def stop(self):
        """停止系统"""
        logger.info("系统停止中...")
        self.is_running = False
        
        # 关闭各层
        for layer in self.layers:
            if hasattr(layer, 'shutdown'):
                layer.shutdown()
        
        logger.info("系统已停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_info': {
                'name': self.config.get('system', {}).get('name', 'qm_final3'),
                'version': self.config.get('system', {}).get('version', '3.0.0'),
                'is_running': self.is_running,
                'layer_count': len(self.layers)
            },
            'pipeline_status': self.pipeline.get_pipeline_status() if self.pipeline else None,
            'layer_statuses': [layer.get_status() for layer in self.layers]
        }

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="qm_final3 - 六层架构心境流转系统")
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    parser.add_argument('--demo', action='store_true', help='演示模式')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建系统实例
    system = QMFinal3System(config_path=args.config)
    
    # 设置信号处理
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在关闭系统...")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.test:
            # 测试模式
            logger.info("运行测试模式...")
            status = system.get_system_status()
            print(f"系统状态: {status}")
            
            # 简单测试
            await system._system_health_check()
            logger.info("测试模式完成")
            
        elif args.demo:
            # 演示模式
            logger.info("运行演示模式...")
            await system.start()
            
        else:
            # 正常启动
            await system.start()
            
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        sys.exit(1)
    
    finally:
        await system.stop()

if __name__ == "__main__":
    asyncio.run(main())