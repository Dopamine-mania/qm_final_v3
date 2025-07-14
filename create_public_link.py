#!/usr/bin/env python3
"""
免费公共链接创建脚本
使用SSH隧道服务，无需注册账户
"""

import subprocess
import threading
import time
import sys
import signal

def start_streamlit():
    """启动Streamlit应用"""
    subprocess.run(['streamlit', 'run', 'app.py', '--server.port=8502', '--server.headless=true'])

def create_tunnel():
    """创建SSH隧道"""
    try:
        print("🌐 正在创建公共访问隧道...")
        # 使用localhost.run服务创建隧道
        result = subprocess.run([
            'ssh', '-R', '80:localhost:8502', 
            'ssh.localhost.run'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ 隧道创建成功！")
        else:
            print(f"❌ 隧道创建失败: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏳ SSH隧道连接中，这是正常的...")
    except Exception as e:
        print(f"❌ 错误: {e}")

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
    print("🎉 Streamlit已启动！")
    print("🌐 本地链接: http://localhost:8502")
    print("📱 现在请打开另一个终端窗口，运行以下命令创建公共链接：")
    print()
    print("ssh -R 80:localhost:8502 ssh.localhost.run")
    print()
    print("💡 运行后会显示类似: https://abc123.localhost.run 的公共链接")
    print("🔗 此链接任何人都能访问！")
    print("⚠️  按 Ctrl+C 可以停止服务")
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