#!/usr/bin/env python3
"""
🔍 深度分析：测试不同的API认证方法
根据用户要求进行深度思考，找到Suno API的正确调用方式
"""

import http.client
import json
import time

API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"
TEST_TASK_ID = "fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc"

def test_fetch_no_auth():
    """测试1: fetch端点不使用任何认证"""
    print("🧪 测试1: fetch端点 - 无认证")
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        headers = {'Accept': 'application/json'}
        conn.request("GET", f"/suno/fetch?task_id={TEST_TASK_ID}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        print(f"   响应类型: {'JSON' if data.decode('utf-8').strip().startswith('{') else 'HTML'}")
        print(f"   前100字符: {data.decode('utf-8')[:100]}...")
        return res.status == 200 and data.decode('utf-8').strip().startswith('{')
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

def test_fetch_api_key_param():
    """测试2: fetch端点 - API key作为URL参数"""
    print("🧪 测试2: fetch端点 - API key作为URL参数")
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        headers = {'Accept': 'application/json'}
        url = f"/suno/fetch?task_id={TEST_TASK_ID}&api_key={API_KEY}"
        conn.request("GET", url, headers=headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        print(f"   响应类型: {'JSON' if data.decode('utf-8').strip().startswith('{') else 'HTML'}")
        print(f"   前100字符: {data.decode('utf-8')[:100]}...")
        return res.status == 200 and data.decode('utf-8').strip().startswith('{')
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

def test_fetch_custom_header():
    """测试3: fetch端点 - 自定义header"""
    print("🧪 测试3: fetch端点 - X-API-Key header")
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        headers = {
            'Accept': 'application/json',
            'X-API-Key': API_KEY
        }
        conn.request("GET", f"/suno/fetch?task_id={TEST_TASK_ID}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        print(f"   响应类型: {'JSON' if data.decode('utf-8').strip().startswith('{') else 'HTML'}")
        print(f"   前100字符: {data.decode('utf-8')[:100]}...")
        return res.status == 200 and data.decode('utf-8').strip().startswith('{')
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

def test_submit_no_auth():
    """测试4: submit端点 - 按照API文档不使用认证"""
    print("🧪 测试4: submit端点 - 无认证（按API文档）")
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
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/suno/submit/music", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        print(f"   响应类型: {'JSON' if data.decode('utf-8').strip().startswith('{') else 'HTML'}")
        print(f"   前100字符: {data.decode('utf-8')[:100]}...")
        
        if res.status == 200:
            try:
                result = json.loads(data.decode('utf-8'))
                print(f"   ✅ JSON解析成功: {result}")
                return True, result
            except:
                print(f"   ❌ JSON解析失败")
                return False, None
        return False, None
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False, None

def test_submit_with_auth():
    """测试5: submit端点 - 使用API key"""
    print("🧪 测试5: submit端点 - 使用API key参数")
    try:
        conn = http.client.HTTPSConnection(BASE_URL)
        payload = json.dumps({
            "gpt_description_prompt": "calm sleep",
            "make_instrumental": True,
            "mv": "chirp-v3-0",
            "prompt": "calm sleep",
            "api_key": API_KEY  # 尝试在payload中包含API key
        })
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/suno/submit/music", payload, headers)
        res = conn.getresponse()
        data = res.read()
        print(f"   状态码: {res.status}")
        print(f"   响应类型: {'JSON' if data.decode('utf-8').strip().startswith('{') else 'HTML'}")
        print(f"   前100字符: {data.decode('utf-8')[:100]}...")
        
        if res.status == 200:
            try:
                result = json.loads(data.decode('utf-8'))
                print(f"   ✅ JSON解析成功: {result}")
                return True, result
            except:
                print(f"   ❌ JSON解析失败")
                return False, None
        return False, None
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False, None

def main():
    print("🔍 深度分析：Suno API认证方法测试")
    print("=" * 60)
    print("📋 目标：找到正确的API调用方式")
    print("🎯 任务ID:", TEST_TASK_ID)
    print()

    # 测试fetch端点的不同认证方法
    print("🔍 第一阶段：测试fetch端点")
    print("-" * 40)
    
    methods = [
        ("无认证", test_fetch_no_auth),
        ("URL参数认证", test_fetch_api_key_param),
        ("自定义header", test_fetch_custom_header)
    ]
    
    successful_fetch_methods = []
    for method_name, test_func in methods:
        success = test_func()
        if success:
            successful_fetch_methods.append(method_name)
        print()
    
    # 测试submit端点的不同认证方法
    print("🔍 第二阶段：测试submit端点")
    print("-" * 40)
    
    submit_methods = [
        ("按API文档无认证", test_submit_no_auth),
        ("payload中包含API key", test_submit_with_auth)
    ]
    
    successful_submit_methods = []
    for method_name, test_func in submit_methods:
        success, result = test_func()
        if success:
            successful_submit_methods.append((method_name, result))
        print()
    
    # 总结
    print("📊 测试结果总结")
    print("=" * 60)
    print(f"✅ 成功的fetch方法: {successful_fetch_methods or '无'}")
    print(f"✅ 成功的submit方法: {[m[0] for m in successful_submit_methods] or '无'}")
    
    if successful_fetch_methods and successful_submit_methods:
        print("\n🎉 找到了有效的API调用方法！")
        print("💡 建议：使用这些方法更新现有代码")
    else:
        print("\n❌ 需要进一步分析API文档或联系API提供商")
        
    return successful_fetch_methods, successful_submit_methods

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback
        traceback.print_exc()