# 🔑 API配置指南

## 问题说明

你说得对！我确实忽略了一个关键问题：**没有配置真实的API密钥，系统根本无法生成真实的音乐和视频**。

## 🎯 解决方案

### 1. 立即可用的演示模式

```bash
# 运行演示模式，使用模拟数据展示完整功能
python gradio_demo_mode.py
```

**演示模式特点：**
- ✅ 不需要任何API密钥
- ✅ 完整的音画同步视频展示
- ✅ 三阶段音乐叙事演示
- ✅ 情绪识别功能演示
- ✅ 立即可用，无需配置

### 2. 真实API配置（生产环境）

#### 2.1 创建API配置文件

```bash
# 创建API配置文件
cp configs/api_config_template.yaml configs/api_config.yaml
```

#### 2.2 配置Suno API

```yaml
# configs/api_config.yaml
suno_api:
  base_url: "https://api.suno.ai/v1"
  api_key: "YOUR_SUNO_API_KEY"  # 需要购买
  model: "chirp-v3-5"
  
runway_api:
  base_url: "https://api.runwayml.com/v1"
  api_key: "YOUR_RUNWAY_API_KEY"  # 需要购买
  model: "gen-2"

openai_api:
  base_url: "https://api.openai.com/v1"
  api_key: "YOUR_OPENAI_API_KEY"  # 需要购买
  model: "gpt-4"
```

#### 2.3 获取API密钥

**Suno API (音乐生成):**
- 网站：https://suno.ai/
- 价格：约$10/月基础版
- 功能：AI音乐生成

**Runway API (视频生成):**
- 网站：https://runwayml.com/
- 价格：约$15/月基础版
- 功能：AI视频生成

**OpenAI API (文本处理):**
- 网站：https://openai.com/api/
- 价格：按使用量计费
- 功能：文本理解和生成

## 🚀 快速上手步骤

### 步骤1：安装依赖
```bash
python install_dependencies.py
```

### 步骤2：运行演示模式
```bash
python gradio_demo_mode.py
```

### 步骤3：体验完整功能
1. 打开浏览器访问生成的链接
2. 输入你的情绪描述
3. 点击"演示：生成音画同步疗愈视频"
4. 观看15秒的三阶段音画同步疗愈视频

## 🎬 演示模式 vs 真实模式

| 功能 | 演示模式 | 真实模式 |
|------|----------|----------|
| 音乐生成 | 模拟三阶段音频 | Suno AI真实音乐 |
| 视频生成 | 模拟疗愈视觉 | Runway AI真实视频 |
| 情绪识别 | 关键词匹配 | 27维细粒度识别 |
| 成本 | 免费 | 需要API费用 |
| 效果 | 演示概念 | 真实疗愈效果 |

## 🔧 如何配置真实API

1. **购买API服务**（如果预算允许）
2. **修改配置文件**：`configs/api_config.yaml`
3. **运行真实版本**：`python gradio_app.py`

## 💡 推荐方案

**对于演示和概念验证：**
- 使用 `gradio_demo_mode.py`
- 无需任何费用
- 完整功能展示

**对于真实部署：**
- 配置API密钥
- 使用 `gradio_app.py`
- 真实AI生成内容

## 🎯 总结

你的反馈非常中肯！我创建了：

1. **演示模式** - 立即可用，无需API
2. **依赖安装脚本** - 一键安装所有依赖
3. **API配置指南** - 详细的真实API配置说明

现在你可以：
- 立即运行演示模式看到完整效果
- 有预算时再配置真实API

这样既解决了当前的演示需求，又为未来的真实部署做好了准备！