#!/usr/bin/env python3
"""
🎵 获取之前的Suno音乐
使用已有的任务ID获取音乐，避免重新调用API
"""

import http.client
import json
import time
import tempfile
import os

# API配置
API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"

# 刚才的任务ID
PREVIOUS_TASK_ID = "fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc"

def fetch_suno_result(task_id):
    """查询Suno API任务结果"""
    print(f"🔍 查询任务结果: {task_id}")
    
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        
        # 添加请求头
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        # 查询任务状态
        conn.request("GET", f"/suno/fetch?task_id={task_id}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        
        print(f"🔍 API响应状态: {res.status}")
        print(f"🔍 API响应数据: {data.decode('utf-8')}")
        
        if res.status == 200:
            # 检查响应是否为空
            if not data or len(data.strip()) == 0:
                print("⚠️ API返回空响应")
                return None
            
            try:
                result = json.loads(data.decode("utf-8"))
                return result
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                print(f"   原始数据: {data.decode('utf-8')}")
                return None
        else:
            print(f"❌ 查询失败，状态码: {res.status}")
            print(f"   响应内容: {data.decode('utf-8')}")
            return None
            
    except Exception as e:
        print(f"⚠️ 查询任务状态出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_audio_url(result):
    """从API结果中提取音频URL"""
    if not result or not isinstance(result, dict):
        return None
    
    print(f"🔍 分析API响应结构...")
    
    # 尝试多种可能的结构
    audio_url = None
    
    # 方式1: data直接包含音频信息
    if 'data' in result:
        data = result['data']
        if isinstance(data, dict):
            if 'audio_url' in data:
                audio_url = data['audio_url']
            elif 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                # 有数组的情况
                first_item = data['data'][0]
                if isinstance(first_item, dict) and 'audio_url' in first_item:
                    audio_url = first_item['audio_url']
    
    return audio_url

def download_audio(url, filename):
    """下载音频文件"""
    try:
        import urllib.request
        print(f"🔄 开始下载: {url}")
        urllib.request.urlretrieve(url, filename)
        
        # 验证文件
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            file_size = os.path.getsize(filename)
            print(f"✅ 下载成功: {filename} ({file_size} bytes)")
            return True
        else:
            print(f"❌ 下载失败: 文件为空或不存在")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    print("🎵 获取之前的Suno音乐")
    print(f"📋 任务ID: {PREVIOUS_TASK_ID}")
    print("=" * 50)
    
    # 1. 查询任务状态
    result = fetch_suno_result(PREVIOUS_TASK_ID)
    
    if not result:
        print("❌ 无法获取任务信息")
        return
    
    # 2. 提取音频URL
    audio_url = extract_audio_url(result)
    
    if not audio_url:
        print("❌ 未找到音频URL")
        print("💡 可能原因:")
        print("   - 音乐还在生成中")
        print("   - 任务失败了")
        print("   - API响应格式变化")
        return
    
    print(f"🎵 找到音频URL: {audio_url}")
    
    # 3. 下载音频
    output_file = f"previous_suno_{PREVIOUS_TASK_ID[:8]}.mp3"
    
    if download_audio(audio_url, output_file):
        print(f"🎉 成功！")
        print(f"🎧 音频文件: {output_file}")
        print(f"📂 完整路径: {os.path.abspath(output_file)}")
        print("\n💡 现在你可以:")
        print("   1. 直接播放这个文件听效果")
        print("   2. 把这个文件路径用在Gradio界面测试")
        
        return os.path.abspath(output_file)
    else:
        print("❌ 下载失败")
        return None

if __name__ == "__main__":
    try:
        result = main()
        if result:
            print(f"\n🔗 音频文件路径: {result}")
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc()