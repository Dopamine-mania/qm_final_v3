#!/usr/bin/env python3
"""
最简单的公共链接方案
使用多种免费隧道服务
"""

import subprocess
import threading
import time
import sys
import os

def start_streamlit():
    """启动Streamlit应用"""
    subprocess.run(['streamlit', 'run', 'app.py', '--server.port=8502', '--server.headless=true'])

def try_serveo():
    """尝试serveo隧道"""
    try:
        print("🔄 尝试serveo隧道...")
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', 
            '-R', '80:localhost:8502', 'serveo.net'
        ], timeout=15, capture_output=True, text=True)
        return True
    except:
        return False

def try_ngrok_simple():
    """尝试简单的ngrok连接（可能有免费额度）"""
    try:
        print("🔄 尝试ngrok...")
        result = subprocess.run(['ngrok', 'http', '8502'], timeout=10)
        return True
    except:
        return False

def main():
    print("🚀 正在启动睡眠疗愈AI系统...")
    
    # 启动Streamlit线程
    streamlit_thread = threading.Thread(target=start_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # 等待Streamlit启动
    print("⏳ 等待Streamlit启动...")
    time.sleep(8)
    
    print("\n" + "="*60)
    print("🎉 Streamlit已启动在端口8502！")
    print("🌐 本地访问: http://localhost:8502")
    print("\n💡 创建公共链接的几种方法：")
    print("\n方法1 - 手动serveo (推荐):")
    print("ssh -o StrictHostKeyChecking=no -R 80:localhost:8502 serveo.net")
    print("\n方法2 - 手动bore:")
    print("wget https://github.com/ekzhang/bore/releases/download/v0.5.1/bore-v0.5.1-x86_64-unknown-linux-musl.tar.gz")
    print("tar -xzf bore-v0.5.1-x86_64-unknown-linux-musl.tar.gz")
    print("./bore local 8502 --to bore.pub")
    print("\n方法3 - localtunnel (需要npm):")
    print("npx localtunnel --port 8502")
    print("\n💡 任选一种方法在新终端中执行即可获得公共链接！")
    print("⚠️  按 Ctrl+C 停止服务")
    print("="*60 + "\n")
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        sys.exit(0)

if __name__ == "__main__":
    main()