#!/usr/bin/env python3
"""
公共隧道创建脚本
自动启动Streamlit并创建ngrok公共访问链接
"""

from pyngrok import ngrok
import time
import subprocess
import threading
import sys

def start_streamlit():
    """启动Streamlit应用"""
    subprocess.run(['streamlit', 'run', 'app.py', '--server.port=8502', '--server.headless=true'])

def main():
    print("🚀 正在启动睡眠疗愈AI系统...")
    
    # 启动Streamlit线程
    streamlit_thread = threading.Thread(target=start_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # 等待Streamlit启动
    print("⏳ 等待Streamlit启动...")
    time.sleep(8)
    
    # 创建公共隧道
    try:
        print("🌐 正在创建公共访问链接...")
        public_url = ngrok.connect(8502)
        
        print("\n" + "="*60)
        print("🎉 成功！您的睡眠疗愈AI已启动！")
        print(f"🌐 公共访问链接: {public_url}")
        print("💡 此链接任何人都能访问！")
        print("🎵 您可以在界面中测试情绪识别和音乐生成")
        print("⚠️  按 Ctrl+C 可以停止服务")
        print("="*60 + "\n")
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("💡 提示: 如果遇到authtoken错误，ngrok免费版本有一些限制")
        print("但通常第一次使用应该可以正常工作")
        sys.exit(1)

if __name__ == "__main__":
    main()