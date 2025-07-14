#!/usr/bin/env python3
"""
🔍 深度分析第二轮：测试正确的认证方法
发现submit需要"令牌"，测试Bearer token和其他可能的端点
"""

import http.client
import json
import time

API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"
TEST_TASK_ID = "fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc"

def test_submit_bearer_token():
    """测试: submit端点使用Bearer token"""
    print("🧪 测试: submit端点 - Bearer token")
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        payload = json.dumps({
            "gpt_description_prompt": "calm sleep",
            "make_instrumental": True,
            "mv": "chirp-v3-0",
            "prompt": "calm sleep"
        })
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        conn.request("POST", "/suno/submit/music", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        response_text = data.decode('utf-8')
        print(f"   响应: {response_text[:200]}...")
        
        if res.status == 200:
            try:
                result = json.loads(response_text)
                print(f"   ✅ 成功！任务ID: {result.get('data')}")
                return True, result.get('data')
            except:
                print(f"   ❌ JSON解析失败")
                return False, None
        return False, None
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False, None

def test_fetch_with_correct_task(task_id):
    """测试: 使用正确任务ID的fetch"""
    print(f"🧪 测试: fetch端点 - 使用新任务ID {task_id}")
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        conn.request("GET", f"/suno/fetch?task_id={task_id}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        response_text = data.decode('utf-8')
        is_json = response_text.strip().startswith('{')
        print(f"   响应类型: {'JSON' if is_json else 'HTML'}")
        print(f"   响应: {response_text[:200]}...")
        
        if is_json:
            try:
                result = json.loads(response_text)
                print(f"   ✅ 成功解析JSON!")
                return True, result
            except:
                print(f"   ❌ JSON解析失败")
        return False, None
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False, None

def test_alternative_fetch_endpoints(task_id):
    """测试: 其他可能的fetch端点"""
    endpoints = [
        f"/suno/fetch/{task_id}",  # REST风格
        f"/suno/status?id={task_id}",  # 可能的状态端点
        f"/suno/query?task_id={task_id}",  # 可能的查询端点
        f"/v1/suno/fetch?task_id={task_id}",  # 可能有版本前缀
    ]
    
    print(f"🧪 测试: 其他可能的fetch端点")
    
    for endpoint in endpoints:
        try:
            print(f"   尝试: {endpoint}")
            conn = http.client.HTTPSConnection(BASE_URL)
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {API_KEY}'
            }
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            response_text = data.decode('utf-8')
            is_json = response_text.strip().startswith('{')
            
            print(f"     状态码: {res.status}, 类型: {'JSON' if is_json else 'HTML'}")
            if is_json:
                print(f"     ✅ 找到JSON响应: {response_text[:100]}...")
                try:
                    result = json.loads(response_text)
                    return True, result, endpoint
                except:
                    pass
        except Exception as e:
            print(f"     ❌ 错误: {e}")
    
    return False, None, None

def main():
    print("🔍 深度分析第二轮：正确的认证方法")
    print("=" * 60)
    
    # 第一步：用正确方法测试submit
    print("🔍 第一步：使用Bearer token测试submit")
    print("-" * 40)
    
    success, new_task_id = test_submit_bearer_token()
    print()
    
    if success and new_task_id:
        print(f"🎉 submit成功！新任务ID: {new_task_id}")
        print()
        
        # 第二步：用新任务ID测试fetch
        print("🔍 第二步：使用新任务ID测试fetch")
        print("-" * 40)
        
        # 等待一下让任务开始处理
        print("⏳ 等待3秒让任务开始...")
        time.sleep(3)
        
        fetch_success, fetch_result = test_fetch_with_correct_task(new_task_id)
        print()
        
        if not fetch_success:
            print("🔍 第三步：尝试其他fetch端点")
            print("-" * 40)
            alt_success, alt_result, alt_endpoint = test_alternative_fetch_endpoints(new_task_id)
            if alt_success:
                print(f"🎉 找到正确的fetch端点: {alt_endpoint}")
                fetch_success, fetch_result = alt_success, alt_result
        
        # 总结
        print("📊 最终结果")
        print("=" * 60)
        if fetch_success:
            print("✅ 成功找到完整的API调用流程！")
            print(f"   submit: Bearer token认证")
            print(f"   fetch: 相同的Bearer token认证")
            print(f"   新任务ID: {new_task_id}")
            if fetch_result:
                print(f"   任务状态: {fetch_result.get('status', 'unknown')}")
        else:
            print("⚠️ submit成功，但fetch仍有问题")
            print("💡 可能需要等待更长时间或使用不同的fetch方法")
    else:
        print("❌ submit仍然失败，需要进一步研究认证方法")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()