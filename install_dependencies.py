#!/usr/bin/env python3
"""
📦 依赖安装脚本
一键安装所有必要的Python包
"""

import subprocess
import sys
import os

def install_package(package_name):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False

def main():
    """主函数"""
    print("🚀 开始安装睡眠疗愈AI系统依赖...")
    print("="*60)
    
    # 必要的Python包
    required_packages = [
        "gradio",
        "numpy",
        "opencv-python",
        "soundfile",
        "pillow",
        "torch",
        "torchvision",
        "torchaudio",
        "requests",
        "pydub",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "pyyaml",
        "librosa",
        "transformers",
        "sentence-transformers"
    ]
    
    success_count = 0
    failed_packages = []
    
    for package in required_packages:
        print(f"📦 正在安装 {package}...")
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print("\n" + "="*60)
    print(f"📊 安装结果统计:")
    print(f"✅ 成功安装: {success_count}/{len(required_packages)} 个包")
    
    if failed_packages:
        print(f"❌ 安装失败: {', '.join(failed_packages)}")
        print("\n💡 对于失败的包，请尝试手动安装：")
        for package in failed_packages:
            print(f"pip install {package}")
    else:
        print("🎉 所有依赖安装成功！")
    
    print("\n🔧 系统命令检查:")
    
    # 检查系统命令
    system_commands = {
        "ffmpeg": "音视频处理工具",
        "git": "版本控制工具"
    }
    
    for cmd, description in system_commands.items():
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd} - {description} 可用")
            else:
                print(f"❌ {cmd} - {description} 不可用")
        except FileNotFoundError:
            print(f"❌ {cmd} - {description} 未安装")
    
    print("\n📋 安装完成后请运行:")
    print("python quick_test.py  # 验证系统状态")
    print("python gradio_demo_mode.py  # 运行演示模式")
    print("="*60)

if __name__ == "__main__":
    main()