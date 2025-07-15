#!/usr/bin/env python3
"""
测试现有任务ID获取音频 - 不花钱验证逻辑
"""

import http.client
import json

# 使用你之前提供的任务ID
TASK_ID = "fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc"

def test_fetch_existing_task():
    """测试获取现有任务的音频"""
    print(f"🔍 测试查询现有任务: {TASK_ID}")
    
    try:
        conn = http.client.HTTPSConnection("feiai.chat")
        payload = ''
        headers = {
           'Accept': 'application/json',
           'Content-Type': 'application/json',
           'Authorization': 'Bearer sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1'
        }
        conn.request("GET", f"/suno/fetch/{TASK_ID}", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        print(f"📊 响应状态: {res.status}")
        response_text = data.decode("utf-8")
        print(f"📄 响应内容: {response_text}")
        
        if res.status == 200:
            try:
                result = json.loads(response_text)
                print(f"✅ JSON解析成功")
                print(f"🔍 响应结构: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                # 检查是否有音频URL
                audio_url = None
                if result.get('code') == 'success' and result.get('data'):
                    task_data = result.get('data')
                    if isinstance(task_data, dict):
                        if 'data' in task_data and isinstance(task_data['data'], list) and len(task_data['data']) > 0:
                            for audio_item in task_data['data']:
                                if audio_item.get('audio_url'):
                                    audio_url = audio_item.get('audio_url')
                                    print(f"🎵 找到音频URL: {audio_url}")
                                    print(f"🎼 音频标题: {audio_item.get('title', 'Unknown')}")
                                    print(f"⏱️ 音频时长: {audio_item.get('duration', 'Unknown')}")
                                    break
                
                if audio_url:
                    print(f"✅ 成功找到音频，可以下载播放！")
                    return audio_url
                else:
                    print(f"❌ 没有找到可用的音频URL")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                return None
        else:
            print(f"❌ API请求失败: {res.status}")
            return None
            
    except Exception as e:
        print(f"❌ 请求出错: {e}")
        return None

if __name__ == "__main__":
    audio_url = test_fetch_existing_task()
    if audio_url:
        print(f"\n🎉 验证成功！现有逻辑应该能找到这个音频URL并下载播放")
        print(f"🔗 音频URL: {audio_url}")
    else:
        print(f"\n😞 没有找到音频，可能任务还在处理中或已过期")