# MCP 服务管理目录

此目录用于管理所有 MCP (Model Context Protocol) 服务。

## 目录结构

```
mcp_servers/
├── __init__.py          # 管理模块
├── README.md            # 本说明文件
├── enabled.txt          # 启用的服务列表
├── baidu_search/        # 百度搜索服务 (stdio)
│   ├── config.json      # 服务配置
│   ├── server.py        # 服务主程序
│   └── README.md        # 服务说明
├── baidu_ai_search/     # 百度 AI 搜索服务 (HTTP)
│   └── config.json      # 服务配置
└── <其他服务>/           # 其他 MCP 服务
    ├── config.json
    └── server.py (仅 stdio 类型需要)
```

## 启用/禁用服务

服务的启用状态由 `enabled.txt` 文件控制：

```
# MCP 服务启用列表
# 每行一个服务文件夹名，以 # 开头的行为注释
# 不在列表中的服务将不会被启用

baidu_search
baidu_ai_search
```

- **启用服务**：在 `enabled.txt` 中添加服务文件夹名
- **禁用服务**：从 `enabled.txt` 中删除该行或注释掉

## 添加新服务

### STDIO 类型服务

1. 创建服务目录：
   ```bash
   mkdir mcp_servers/my_service
   ```

2. 创建配置文件 `config.json`：
   ```json
   {
       "name": "my_service",
       "description": "服务描述",
       "version": "1.0.0",
       "type": "stdio",
       "command": "python",
       "args": [],
       "env": {
           "API_KEY": "your-api-key"
       }
   }
   ```

3. 创建服务主程序 `server.py`，实现 MCP 协议。

4. 启用服务：在 `enabled.txt` 中添加 `my_service`

### HTTP/SSE 类型服务

1. 创建服务目录：
   ```bash
   mkdir mcp_servers/my_http_service
   ```

2. 创建配置文件 `config.json`：
   ```json
   {
       "name": "my_http_service",
       "description": "HTTP 服务描述",
       "version": "1.0.0",
       "type": "streamableHttp",
       "url": "https://api.example.com/mcp",
       "headers": {
           "Authorization": "Bearer your-api-key"
       }
   }
   ```

3. 启用服务：在 `enabled.txt` 中添加 `my_http_service`

## 配置文件格式

### config.json

| 字段 | 类型 | 说明 |
|------|------|------|
| name | string | 服务名称 |
| description | string | 服务描述 |
| version | string | 服务版本 |
| type | string | 服务类型：`stdio`、`streamableHttp`、`sse` |
| command | string | (stdio) 启动命令 |
| args | array | (stdio) 命令参数 |
| env | object | (stdio) 环境变量 |
| url | string | (http/sse) 服务 URL |
| headers | object | (http/sse) 请求头 |

### 服务类型

| 类型 | 说明 | 需要 server.py |
|------|------|----------------|
| stdio | 通过子进程和标准 IO 通信 | 是 |
| streamableHttp | 通过 HTTP 请求通信 | 否 |
| sse | 通过 Server-Sent Events 通信 | 否 |

## Python API

```python
from mcp_servers import (
    get_available_servers,
    get_enabled_servers,
    enable_server,
    disable_server,
    generate_mcp_config
)

# 获取所有可用服务
servers = get_available_servers()
for name, info in servers.items():
    print(f"{name}: {'启用' if info['enabled'] else '禁用'}")

# 获取启用的服务列表
enabled = get_enabled_servers()
print(f"已启用: {enabled}")

# 启用/禁用服务
enable_server("baidu_search")
disable_server("baidu_search")

# 生成 MCP 配置
config = generate_mcp_config()
```

## 内置服务

### baidu_search (stdio)

百度网页搜索服务，使用本地 Python 脚本调用百度搜索 API。

**工具**:
- `baidu_web_search`: 执行百度网页搜索

**环境变量**:
- `BAIDU_APPBUILDER_API_KEY`: 百度 AppBuilder API Key

### baidu_ai_search (HTTP)

百度 AI 搜索服务，通过 HTTP 直接调用百度 MCP 服务端点。

**特点**:
- 无需本地脚本
- 直接连接百度云服务
- 支持 AI 总结功能