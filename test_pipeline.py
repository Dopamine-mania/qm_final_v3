#!/usr/bin/env python3
"""
完整数据流测试脚本

测试六层架构的完整数据处理流程，包括：
1. 输入层数据处理
2. 融合层情绪识别
3. 映射层KG-MLP处理
4. 生成层内容生成
5. 渲染层同步渲染
6. 治疗层FSM流程
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from main import QMFinal3System
from layers.base_layer import LayerData
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_full_pipeline():
    """测试完整数据处理管道"""
    print("🚀 开始完整数据流测试...")
    print("=" * 50)
    
    try:
        # 初始化系统
        print("📋 初始化系统...")
        system = QMFinal3System()
        
        # 创建测试数据
        test_cases = [
            {
                'name': '焦虑失眠测试',
                'text': '我今天感到很焦虑，躺在床上睡不着觉',
                'expected_emotions': ['anxiety', 'insomnia', 'restlessness']
            },
            {
                'name': '平静放松测试', 
                'text': '我感到很平静和放松，准备入睡',
                'expected_emotions': ['calm', 'relaxation', 'drowsiness']
            },
            {
                'name': '抑郁情绪测试',
                'text': '我感到很沮丧和悲伤，整夜难眠',
                'expected_emotions': ['sadness', 'depression', 'sleep_depression']
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 测试 {i}: {test_case['name']}")
            print(f"输入文本: {test_case['text']}")
            
            # 创建测试数据
            test_data = LayerData(
                layer_name='test_input',
                timestamp=datetime.now(),
                data={'text': test_case['text']},
                metadata={
                    'source': 'pipeline_test',
                    'test_case': test_case['name'],
                    'test_id': i
                }
            )
            
            start_time = time.time()
            
            try:
                # 通过管道处理
                result = await system.pipeline.process(test_data)
                
                processing_time = (time.time() - start_time) * 1000
                
                print(f"✅ 处理成功!")
                print(f"   层级: {result.layer_name}")
                print(f"   置信度: {result.confidence:.3f}")
                print(f"   处理时间: {processing_time:.2f}ms")
                
                # 检查结果数据
                if hasattr(result, 'data') and result.data:
                    print(f"   数据类型: {type(result.data)}")
                    if isinstance(result.data, dict):
                        print(f"   数据键: {list(result.data.keys())}")
                
                results.append({
                    'test_case': test_case['name'],
                    'success': True,
                    'confidence': result.confidence,
                    'processing_time_ms': processing_time,
                    'result': result
                })
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                results.append({
                    'test_case': test_case['name'],
                    'success': False,
                    'error': str(e),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
        
        # 输出总结
        print("\n" + "=" * 50)
        print("📊 测试总结")
        print("=" * 50)
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        print(f"总测试数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"失败数: {total_count - success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        if success_count > 0:
            avg_time = sum(r['processing_time_ms'] for r in results if r['success']) / success_count
            avg_confidence = sum(r['confidence'] for r in results if r['success']) / success_count
            print(f"平均处理时间: {avg_time:.2f}ms")
            print(f"平均置信度: {avg_confidence:.3f}")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['test_case']}: {result['processing_time_ms']:.2f}ms")
            if not result['success']:
                print(f"     错误: {result['error']}")
        
        # 停止系统
        await system.stop()
        
        print(f"\n🎉 测试完成!")
        return results
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_layer_by_layer():
    """逐层测试功能"""
    print("\n🔍 开始逐层功能测试...")
    print("=" * 50)
    
    try:
        system = QMFinal3System()
        
        # 测试输入层
        print("🧪 测试输入层...")
        input_layer = system.layers[0]  # 输入层
        input_status = input_layer.get_status()
        print(f"   状态: {input_status}")
        
        # 测试融合层
        print("🧪 测试融合层...")
        fusion_layer = system.layers[1]  # 融合层
        fusion_status = fusion_layer.get_status()
        print(f"   情绪维度: {fusion_status.get('emotion_dimensions', 'N/A')}")
        print(f"   GPU可用: {fusion_status.get('gpu_available', False)}")
        
        # 测试映射层
        print("🧪 测试映射层...")
        mapping_layer = system.layers[2]  # 映射层
        mapping_status = mapping_layer.get_status()
        print(f"   映射策略: {mapping_status.get('mapping_strategy', 'N/A')}")
        print(f"   KG权重: {mapping_status.get('kg_weight', 'N/A')}")
        print(f"   MLP权重: {mapping_status.get('mlp_weight', 'N/A')}")
        
        # 测试生成层
        print("🧪 测试生成层...")
        generation_layer = system.layers[3]  # 生成层
        generation_status = generation_layer.get_status()
        print(f"   音频启用: {generation_status.get('audio_enabled', False)}")
        print(f"   视频启用: {generation_status.get('video_enabled', False)}")
        
        # 测试渲染层
        print("🧪 测试渲染层...")
        rendering_layer = system.layers[4]  # 渲染层
        rendering_status = rendering_layer.get_status()
        print(f"   同步模式: {rendering_status.get('sync_mode', 'N/A')}")
        print(f"   质量级别: {rendering_status.get('quality_level', 'N/A')}")
        
        # 测试治疗层
        print("🧪 测试治疗层...")
        therapy_layer = system.layers[5]  # 治疗层
        therapy_status = therapy_layer.get_status()
        print(f"   FSM状态: {therapy_status.get('fsm_state', 'N/A')}")
        print(f"   活跃会话: {therapy_status.get('active_sessions', 0)}")
        
        await system.stop()
        
        print("✅ 逐层测试完成!")
        
    except Exception as e:
        print(f"❌ 逐层测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主测试函数"""
    print("🎯 qm_final3 完整功能测试")
    print("时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # 完整管道测试
    pipeline_results = await test_full_pipeline()
    
    # 逐层测试
    await test_layer_by_layer()
    
    print("\n" + "=" * 60)
    print("🏁 所有测试完成!")

if __name__ == "__main__":
    asyncio.run(main())