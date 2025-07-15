#!/usr/bin/env python3
"""
🎵 Suno音频管理器 - 找回所有付费生成的音乐
解决问题：用户付费生成了多首音乐，但只能播放一首的问题
"""

import os
import json
import http.client
import urllib.request
import tempfile
from pathlib import Path

# API配置
API_KEY = "sk-sSxgx9y9kFOdio1I63qm8aSG1XhhHIOk9Yy2chKNnEvq0jq1"
BASE_URL = "feiai.chat"

# 已知的任务ID列表（需要根据实际情况添加）
KNOWN_TASK_IDS = [
    "fdd1b90b-47e2-44ca-a3b9-8b7ff83554dc",  # 现有的
    # 可以继续添加其他任务ID
]

class SunoAudioManager:
    """Suno音频管理器"""
    
    def __init__(self):
        self.audio_cache = {}
        self.cache_file = "suno_audio_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """加载音频缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.audio_cache = json.load(f)
                print(f"📂 加载音频缓存: {len(self.audio_cache)} 条记录")
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
            self.audio_cache = {}
    
    def save_cache(self):
        """保存音频缓存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.audio_cache, f, ensure_ascii=False, indent=2)
            print(f"💾 保存音频缓存: {len(self.audio_cache)} 条记录")
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
    
    def fetch_task_result(self, task_id):
        """查询单个任务结果"""
        print(f"🔍 查询任务: {task_id}")
        
        try:
            conn = http.client.HTTPSConnection(BASE_URL)
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {API_KEY}'
            }
            
            conn.request("GET", f"/suno/fetch/{task_id}", headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            if res.status == 200 and data:
                try:
                    result = json.loads(data.decode("utf-8"))
                    return result
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析失败: {e}")
                    return None
            else:
                print(f"❌ 查询失败: {res.status}")
                return None
                
        except Exception as e:
            print(f"⚠️ 查询出错: {e}")
            return None
    
    def extract_audio_info(self, result, task_id):
        """从API结果提取音频信息"""
        if not result or not isinstance(result, dict):
            return None
        
        # 检查任务状态
        status = result.get('status', 'unknown')
        task_data = result.get('data', {})
        
        # 🔥 关键修复：检查是否已有可用音频，不用等SUCCESS
        audio_items = []
        
        if 'data' in task_data and isinstance(task_data['data'], list):
            for audio_item in task_data['data']:
                if isinstance(audio_item, dict) and audio_item.get('audio_url'):
                    audio_info = {
                        'task_id': task_id,
                        'audio_url': audio_item.get('audio_url'),
                        'title': audio_item.get('title', f'Suno_{task_id[:8]}'),
                        'duration': audio_item.get('duration', 'Unknown'),
                        'status': status,
                        'model': audio_item.get('model_name', 'chirp-v3'),
                        'prompt': audio_item.get('metadata', {}).get('gpt_description_prompt', 'Unknown')
                    }
                    audio_items.append(audio_info)
        
        return audio_items if audio_items else None
    
    def download_audio(self, audio_info):
        """下载音频文件"""
        audio_url = audio_info['audio_url']
        task_id = audio_info['task_id']
        title = audio_info['title'].replace('/', '_').replace('\\', '_')  # 安全文件名
        
        filename = f"suno_{task_id[:8]}_{title}.mp3"
        
        try:
            print(f"📥 下载: {title}")
            
            # 使用临时文件避免下载中断
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
            
            # 下载文件
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            request = urllib.request.Request(audio_url, headers=headers)
            
            with urllib.request.urlopen(request, timeout=30) as response:
                with open(tmp_path, 'wb') as f:
                    f.write(response.read())
            
            # 验证下载
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                # 移动到最终位置
                if os.path.exists(filename):
                    os.remove(filename)
                os.rename(tmp_path, filename)
                
                file_size = os.path.getsize(filename)
                print(f"✅ 下载成功: {filename} ({file_size:,} bytes)")
                
                # 更新缓存
                audio_info['local_file'] = os.path.abspath(filename)
                audio_info['file_size'] = file_size
                self.audio_cache[task_id] = audio_info
                
                return filename
            else:
                print(f"❌ 下载失败: 文件为空")
                return None
                
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def scan_all_tasks(self):
        """扫描所有已知任务ID"""
        print("🔍 扫描所有Suno任务...")
        print("=" * 50)
        
        found_audios = []
        
        for task_id in KNOWN_TASK_IDS:
            result = self.fetch_task_result(task_id)
            if result:
                audio_items = self.extract_audio_info(result, task_id)
                if audio_items:
                    found_audios.extend(audio_items)
                    print(f"✅ 任务 {task_id[:8]}: 找到 {len(audio_items)} 个音频")
                else:
                    print(f"⚠️ 任务 {task_id[:8]}: 无音频数据")
            else:
                print(f"❌ 任务 {task_id[:8]}: 查询失败")
        
        return found_audios
    
    def download_all_missing(self, audio_list):
        """下载所有缺失的音频文件"""
        print(f"\n📥 开始下载 {len(audio_list)} 个音频文件...")
        print("=" * 50)
        
        downloaded = []
        for audio_info in audio_list:
            # 检查是否已下载
            task_id = audio_info['task_id']
            if task_id in self.audio_cache and 'local_file' in self.audio_cache[task_id]:
                local_file = self.audio_cache[task_id]['local_file']
                if os.path.exists(local_file):
                    print(f"⏭️ 跳过已存在: {os.path.basename(local_file)}")
                    downloaded.append(local_file)
                    continue
            
            # 下载音频
            filename = self.download_audio(audio_info)
            if filename:
                downloaded.append(filename)
        
        return downloaded
    
    def list_local_audios(self):
        """列出本地所有Suno音频文件"""
        print("\n🎵 本地Suno音频文件:")
        print("=" * 50)
        
        local_files = []
        
        # 扫描当前目录的音频文件
        for pattern in ['suno_*.mp3', 'previous_suno_*.mp3']:
            for file_path in Path('.').glob(pattern):
                if file_path.is_file():
                    local_files.append(str(file_path))
        
        # 从缓存中获取信息
        for i, file_path in enumerate(local_files, 1):
            file_size = os.path.getsize(file_path)
            
            # 尝试从缓存获取详细信息
            task_info = None
            for task_id, cached_info in self.audio_cache.items():
                if cached_info.get('local_file') == os.path.abspath(file_path):
                    task_info = cached_info
                    break
            
            if task_info:
                print(f"{i:2d}. {os.path.basename(file_path)}")
                print(f"    📝 标题: {task_info.get('title', 'Unknown')}")
                print(f"    🆔 任务: {task_info.get('task_id', 'Unknown')[:8]}...")
                print(f"    📊 大小: {file_size:,} bytes")
                print(f"    ⏱️ 时长: {task_info.get('duration', 'Unknown')}")
                print(f"    🎯 提示: {task_info.get('prompt', 'Unknown')}")
            else:
                print(f"{i:2d}. {os.path.basename(file_path)} ({file_size:,} bytes)")
            print()
        
        return local_files

def main():
    """主函数"""
    print("🎵 Suno音频管理器")
    print("🎯 目标: 找回所有付费生成的音乐")
    print("=" * 50)
    
    manager = SunoAudioManager()
    
    # 1. 扫描所有任务
    audio_list = manager.scan_all_tasks()
    
    if not audio_list:
        print("😞 没有找到可用的音频")
        print("💡 可能原因:")
        print("   - 任务还在生成中")
        print("   - 需要添加更多任务ID到 KNOWN_TASK_IDS")
        return
    
    print(f"\n🎉 找到 {len(audio_list)} 个音频:")
    for i, audio in enumerate(audio_list, 1):
        print(f"{i}. {audio['title']} (任务: {audio['task_id'][:8]}...)")
    
    # 2. 下载所有音频
    downloaded = manager.download_all_missing(audio_list)
    manager.save_cache()
    
    # 3. 显示本地文件列表
    local_files = manager.list_local_audios()
    
    print(f"\n🎊 完成！")
    print(f"📥 下载: {len(downloaded)} 个文件")
    print(f"📂 本地: {len(local_files)} 个音频文件")
    print("\n💡 现在你可以:")
    print("   1. 直接播放任何一个文件")
    print("   2. 修改Gradio界面支持选择不同音频")
    print("   3. 在界面中添加音频列表功能")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc()