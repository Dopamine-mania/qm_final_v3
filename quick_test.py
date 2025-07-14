#!/usr/bin/env python3
"""
🧪 快速测试脚本 - 验证音画同步修复版本
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查系统依赖...")
    
    # 检查Python模块
    required_modules = [
        'gradio', 'numpy', 'cv2', 'soundfile', 'PIL'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - 可用")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} - 缺失")
    
    # 检查系统命令
    system_commands = ['ffmpeg']
    missing_commands = []
    
    for cmd in system_commands:
        try:
            result = subprocess.run([cmd, '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd} - 可用")
            else:
                missing_commands.append(cmd)
                print(f"❌ {cmd} - 不可用")
        except FileNotFoundError:
            missing_commands.append(cmd)
            print(f"❌ {cmd} - 未找到")
    
    return missing_modules, missing_commands

def test_system_import():
    """测试系统导入"""
    print("\n🔧 测试系统导入...")
    
    try:
        from main import QMFinal3System
        print("✅ QMFinal3System导入成功")
        
        # 尝试初始化系统
        system = QMFinal3System()
        print("✅ 系统初始化成功")
        
        # 检查层级
        if hasattr(system, 'layers') and system.layers:
            print(f"✅ 找到 {len(system.layers)} 个层级")
            for i, layer in enumerate(system.layers):
                print(f"  - 层级 {i}: {layer.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 系统测试失败: {e}")
        return False

def test_gradio_fixed():
    """测试修复版本是否存在"""
    print("\n📱 检查修复版本...")
    
    fixed_path = Path("gradio_fixed.py")
    if fixed_path.exists():
        print("✅ gradio_fixed.py 存在")
        
        # 检查关键功能
        with open(fixed_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        key_functions = [
            'create_synchronized_video',
            'process_emotion_debug',
            'init_system'
        ]
        
        for func in key_functions:
            if func in content:
                print(f"✅ 找到关键函数: {func}")
            else:
                print(f"❌ 缺少关键函数: {func}")
        
        return True
    else:
        print("❌ gradio_fixed.py 不存在")
        return False

def run_quick_test():
    """运行快速测试"""
    print("🚀 开始快速测试...")
    
    # 检查依赖
    missing_modules, missing_commands = check_dependencies()
    
    if missing_modules:
        print(f"\n⚠️ 缺少Python模块: {', '.join(missing_modules)}")
        print("请运行: pip install gradio numpy opencv-python soundfile pillow")
    
    if missing_commands:
        print(f"\n⚠️ 缺少系统命令: {', '.join(missing_commands)}")
        print("请安装ffmpeg")
    
    # 测试系统导入
    system_ok = test_system_import()
    
    # 测试修复版本
    fixed_ok = test_gradio_fixed()
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print(f"依赖检查: {'✅ 通过' if not missing_modules and not missing_commands else '❌ 有缺失'}")
    print(f"系统导入: {'✅ 通过' if system_ok else '❌ 失败'}")
    print(f"修复版本: {'✅ 通过' if fixed_ok else '❌ 失败'}")
    
    if system_ok and fixed_ok and not missing_modules and not missing_commands:
        print("\n🎉 所有测试通过！可以运行修复版本:")
        print("python gradio_fixed.py")
    else:
        print("\n⚠️ 需要解决上述问题后再运行修复版本")
    
    print("="*50)

if __name__ == "__main__":
    run_quick_test()