# Tavily 远程 MCP 服务器

本服务通过远程 MCP 协议连接到 Tavily 的官方 MCP 服务器，提供强大的网页搜索、数据提取、网站映射和爬取功能。

## 功能列表

### 1. tavily_search - 实时网页搜索
使用 Tavily 搜索引擎进行实时网页搜索，获取最新、最准确的信息。

**特点：**
- 实时搜索最新内容
- 智能结果排序
- 支持多种搜索参数

### 2. tavily_extract - 智能数据提取
从网页中智能提取结构化数据，无需手动解析 HTML。

**特点：**
- 自动识别页面结构
- 提取关键信息
- 清洗和格式化数据

### 3. tavily_map - 网站结构映射
创建网站的结构化映射，了解网站的层次结构和页面关系。

**特点：**
- 自动发现页面链接
- 生成网站拓扑图
- 识别重要页面

### 4. tavily_crawl - 系统化网站爬取
系统化地探索和爬取网站内容，适合大规模数据采集。

**特点：**
- 智能爬取策略
- 遵守 robots.txt
- 可配置爬取深度

## 配置说明

服务配置文件位于 [`config.json`](./config.json)：

```json
{
    "name": "tavily_remote",
    "description": "Tavily 远程 MCP 服务器",
    "version": "1.0.0",
    "type": "streamableHttp",
    "url": "https://mcp.tavily.com/mcp/",
    "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
    }
}
```

### 更换 API Key

1. 访问 [Tavily 官网](https://www.tavily.com/) 注册并获取 API Key
2. 编辑 [`config.json`](./config.json) 文件
3. 将 `YOUR_API_KEY` 替换为你的实际 API Key
4. 重启代理服务器以使配置生效

### 启用/禁用服务

编辑 `mcp_servers/enabled.txt` 文件：

```
# 启用 Tavily 服务
tavily_remote

# 禁用 Tavily 服务（注释或删除）
# tavily_remote
```

## 使用示例

### 在代理服务器中使用

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="http://127.0.0.1:8002/v1"
)

# Tavily 工具会自动加载
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": "搜索最新的人工智能新闻"}
    ]
)

print(response.choices[0].message.content)
```

### 工具名称格式

在代理服务器中，Tavily 工具的名称格式为：
- `tavily_remote_search` - 网页搜索
- `tavily_remote_extract` - 数据提取
- `tavily_remote_map` - 网站映射
- `tavily_remote_crawl` - 网站爬取

## 相关链接

- [Tavily 官网](https://www.tavily.com/)
- [Tavily MCP 文档](https://github.com/tavily-ai/tavily-mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [获取 API Key](https://app.tavily.com/home)

## 注意事项

1. **API Key 安全**：不要将包含 API Key 的配置文件提交到版本控制系统
2. **使用限制**：请遵守 Tavily 的 API 使用限制和条款
3. **网络连接**：确保服务器能够访问 `https://mcp.tavily.com`
4. **费用**：Tavily 提供免费额度，超出后可能需要付费

## 故障排除

### 连接失败
- 检查网络连接
- 验证 API Key 是否正确
- 确认 Tavily 服务是否正常

### 工具不可用
- 检查 `enabled.txt` 是否包含 `tavily_remote`
- 重启代理服务器
- 查看服务器日志了解详细错误信息

### API 配额超限
- 检查 Tavily 账户的使用情况
- 考虑升级到付费计划
- 优化工具调用频率