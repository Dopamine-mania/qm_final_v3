#!/usr/bin/env python3
"""
GPU加速测试脚本
检测GPU可用性并优化系统配置以充分利用GPU算力
"""

import torch
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_availability():
    """测试GPU可用性"""
    print("🚀 GPU可用性测试")
    print("=" * 50)
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # GPU信息
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 设置默认GPU
        if gpu_count > 0:
            torch.cuda.set_device(0)
            print(f"默认GPU: {torch.cuda.current_device()}")
    
    return cuda_available

def test_tensor_operations():
    """测试GPU张量操作性能"""
    print("\n🧮 GPU张量操作性能测试")
    print("=" * 50)
    
    # 测试数据
    size = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建张量
    start_time = time.time()
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    creation_time = time.time() - start_time
    print(f"张量创建时间: {creation_time*1000:.2f}ms")
    
    # 矩阵乘法
    start_time = time.time()
    c = torch.matmul(a, b)
    matmul_time = time.time() - start_time
    print(f"矩阵乘法时间: {matmul_time*1000:.2f}ms")
    
    # 神经网络层
    start_time = time.time()
    linear = torch.nn.Linear(size, size).to(device)
    output = linear(a)
    nn_time = time.time() - start_time
    print(f"神经网络层时间: {nn_time*1000:.2f}ms")
    
    # 内存使用
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU内存使用: {memory_allocated:.1f}MB / {memory_cached:.1f}MB")
    
    return {
        'device': device,
        'creation_time_ms': creation_time * 1000,
        'matmul_time_ms': matmul_time * 1000,
        'nn_time_ms': nn_time * 1000
    }

def test_emotion_classifier_performance():
    """测试27维情绪分类器的GPU性能"""
    print("\n🎭 27维情绪分类器性能测试")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模拟情绪分类器
    class EmotionClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Linear(768, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU()
            )
            self.emotion_head = torch.nn.Linear(128, 27)
            self.confidence_head = torch.nn.Linear(128, 1)
            self.intensity_head = torch.nn.Linear(128, 27)
        
        def forward(self, x):
            features = self.features(x)
            emotions = torch.softmax(self.emotion_head(features), dim=-1)
            confidence = torch.sigmoid(self.confidence_head(features))
            intensity = torch.sigmoid(self.intensity_head(features))
            return emotions, confidence, intensity
    
    # 创建模型
    model = EmotionClassifier().to(device)
    model.eval()
    
    # 测试数据
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        # 输入数据 (batch_size, 768) - 模拟BERT特征
        input_data = torch.randn(batch_size, 768, device=device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # 性能测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                emotions, confidence, intensity = model(input_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        throughput = batch_size * 100 / (end_time - start_time)  # samples/sec
        
        print(f"批大小 {batch_size:2d}: {avg_time:.2f}ms/batch, {throughput:.1f} samples/sec")

def optimize_gpu_settings():
    """优化GPU设置"""
    print("\n⚙️ GPU优化设置")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # 启用CUDNN基准测试
        torch.backends.cudnn.benchmark = True
        print("✅ 启用CUDNN基准测试")
        
        # 设置内存分配策略
        torch.cuda.empty_cache()
        print("✅ 清空GPU缓存")
        
        # 显示优化后的GPU状态
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器: {props.multi_processor_count}")
            print(f"  最大线程/块: {props.max_threads_per_block}")
            print(f"  最大块维度: {props.max_block_dims}")
            print(f"  最大网格维度: {props.max_grid_dims}")
    else:
        print("❌ 未检测到GPU，将使用CPU")
        print("建议:")
        print("- 检查CUDA安装")
        print("- 检查PyTorch GPU版本")
        print("- 检查GPU驱动")

def main():
    """主函数"""
    print("🎯 qm_final3 GPU加速测试")
    print("时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # 1. 测试GPU可用性
    gpu_available = test_gpu_availability()
    
    # 2. 测试张量操作性能
    tensor_performance = test_tensor_operations()
    
    # 3. 测试情绪分类器性能
    test_emotion_classifier_performance()
    
    # 4. 优化GPU设置
    optimize_gpu_settings()
    
    # 5. 性能建议
    print("\n💡 性能建议")
    print("=" * 50)
    
    if gpu_available:
        print("✅ GPU可用，建议:")
        print("- 使用批处理 (batch_size >= 4)")
        print("- 启用混合精度训练 (fp16)")
        print("- 使用GPU加速的视频处理")
        print("- 优化数据传输 (pin_memory=True)")
        
        if tensor_performance['matmul_time_ms'] < 10:
            print("- GPU性能优秀，可以处理实时任务")
        elif tensor_performance['matmul_time_ms'] < 50:
            print("- GPU性能良好，适合批处理任务")
        else:
            print("- GPU性能一般，考虑降低模型复杂度")
    else:
        print("❌ GPU不可用，建议:")
        print("- 检查CUDA环境")
        print("- 使用CPU优化版本")
        print("- 降低批大小和模型复杂度")
        print("- 启用多线程处理")
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()