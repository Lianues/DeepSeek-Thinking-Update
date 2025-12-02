"""
MCP (Model Context Protocol) 客户端实现
支持连接 MCP 服务器并将工具转换为 OpenAI 兼容格式

支持的服务器类型：
- stdio: 通过子进程和标准输入输出通信
- streamableHttp: 通过 HTTP 请求通信
- sse: 通过 Server-Sent Events 通信

配置方式：通过 mcp_servers/ 目录自动发现服务
启用的服务列表存储在 mcp_servers/enabled.txt 文件中
"""

import json
import subprocess
import threading
import queue
import os
import uuid
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class MCPServerType(Enum):
    """MCP 服务器类型"""
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamableHttp"
    SSE = "sse"


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPServerConfig:
    """MCP 服务器配置"""
    name: str
    server_type: MCPServerType
    description: str
    enabled: bool
    # stdio 类型配置
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # streamableHttp 类型配置
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class MCPConnectionBase(ABC):
    """MCP 连接基类"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.tools: List[MCPTool] = []
        self.running = False
    
    @abstractmethod
    def start(self) -> bool:
        """启动连接"""
        pass
    
    @abstractmethod
    def stop(self):
        """停止连接"""
        pass
    
    @abstractmethod
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """调用工具"""
        pass
    
    def get_tools(self) -> List[MCPTool]:
        """获取工具列表"""
        return self.tools


class MCPHttpConnection(MCPConnectionBase):
    """HTTP 类型的 MCP 连接"""
    
    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self.session_id: Optional[str] = None
    
    def start(self) -> bool:
        """启动 HTTP 连接"""
        try:
            if not self.config.url:
                print(f"HTTP MCP 服务器 {self.config.name} 缺少 URL 配置")
                return False
            
            # 初始化连接
            self._initialize()
            
            # 获取工具列表
            self._list_tools()
            
            self.running = True
            print(f"HTTP MCP 服务器 {self.config.name} 已连接，获取到 {len(self.tools)} 个工具")
            return True
            
        except Exception as e:
            print(f"连接 HTTP MCP 服务器 {self.config.name} 失败: {e}")
            return False
    
    def stop(self):
        """停止 HTTP 连接"""
        self.running = False
        self.session_id = None
        self.tools = []
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        if self.config.headers:
            headers.update(self.config.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers
    
    def _send_request(self, method: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送 JSON-RPC 请求"""
        if not self.config.url:
            return None
        
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            payload["params"] = params
        
        try:
            response = requests.post(
                self.config.url,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            
            # 检查响应头中的 session ID
            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
            
            # 处理不同的响应类型
            content_type = response.headers.get("Content-Type", "")
            
            if "text/event-stream" in content_type:
                # SSE 响应，解析事件流
                return self._parse_sse_response(response.text)
            else:
                # JSON 响应
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 202:
                    # 异步响应，可能需要轮询
                    return {"status": "accepted"}
                else:
                    print(f"HTTP 请求失败: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"发送 HTTP 请求失败: {e}")
            return None
    
    def _parse_sse_response(self, sse_text: str) -> Optional[Dict]:
        """解析 SSE 响应"""
        result = None
        for line in sse_text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        parsed = json.loads(data)
                        # 保留最后一个有 result 的响应
                        if "result" in parsed:
                            result = parsed
                        elif "error" in parsed:
                            result = parsed
                    except json.JSONDecodeError:
                        continue
        return result
    
    def _initialize(self):
        """初始化 MCP 连接"""
        response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "DeepSeek-MCP-Client",
                "version": "1.0.0"
            }
        })
        
        if response and "result" in response:
            # 发送 initialized 通知
            self._send_request("notifications/initialized", {})
    
    def _list_tools(self):
        """获取工具列表"""
        response = self._send_request("tools/list", {})
        
        if response and "result" in response:
            tools_data = response["result"].get("tools", [])
            self.tools = []
            for tool in tools_data:
                self.tools.append(MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.config.name
                ))
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """调用工具"""
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if response and "result" in response:
            content = response["result"].get("content", [])
            # 合并所有文本内容
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts) if text_parts else json.dumps(content)
        elif response and "error" in response:
            return f"错误: {response['error'].get('message', '未知错误')}"
        
        return None


class MCPStdioConnection(MCPConnectionBase):
    """标准 IO 类型的 MCP 连接（通过子进程）"""
    
    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.pending_requests: Dict[int, queue.Queue] = {}
        self.reader_thread: Optional[threading.Thread] = None
    
    def start(self) -> bool:
        """启动 MCP 服务器进程"""
        try:
            if not self.config.command:
                print(f"STDIO MCP 服务器 {self.config.name} 缺少 command 配置")
                return False
            
            env = os.environ.copy()
            if self.config.env:
                env.update(self.config.env)
            
            args = self.config.args or []
            self.process = subprocess.Popen(
                [self.config.command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0
            )
            
            self.running = True
            self.reader_thread = threading.Thread(target=self._read_responses, daemon=True)
            self.reader_thread.start()
            
            # 初始化连接
            self._initialize()
            
            # 获取工具列表
            self._list_tools()
            
            print(f"STDIO MCP 服务器 {self.config.name} 已启动，注册了 {len(self.tools)} 个工具")
            return True
            
        except Exception as e:
            print(f"启动 STDIO MCP 服务器 {self.config.name} 失败: {e}")
            return False
    
    def stop(self):
        """停止 MCP 服务器进程"""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.tools = []
    
    def _read_responses(self):
        """读取服务器响应的后台线程"""
        while self.running and self.process:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    message = json.loads(line.decode('utf-8').strip())
                    self._handle_message(message)
                except json.JSONDecodeError:
                    continue
                    
            except Exception as e:
                if self.running:
                    print(f"读取 MCP 响应错误: {e}")
                break
    
    def _handle_message(self, message: Dict[str, Any]):
        """处理收到的消息"""
        if "id" in message and message["id"] in self.pending_requests:
            self.pending_requests[message["id"]].put(message)
    
    def _send_request(self, method: str, params: Optional[Dict] = None, timeout: float = 30) -> Optional[Dict]:
        """发送请求并等待响应"""
        if not self.process or not self.process.stdin:
            return None
        
        self.request_id += 1
        request_id = self.request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            request["params"] = params
        
        response_queue = queue.Queue()
        self.pending_requests[request_id] = response_queue
        
        try:
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode('utf-8'))
            self.process.stdin.flush()
            
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except queue.Empty:
                print(f"请求超时: {method}")
                return None
        finally:
            del self.pending_requests[request_id]
    
    def _initialize(self):
        """初始化 MCP 连接"""
        response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "DeepSeek-MCP-Client",
                "version": "1.0.0"
            }
        })
        
        if response and "result" in response:
            self._send_notification("notifications/initialized", {})
    
    def _send_notification(self, method: str, params: Optional[Dict] = None):
        """发送通知（不等待响应）"""
        if not self.process or not self.process.stdin:
            return
        
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params:
            notification["params"] = params
        
        try:
            notification_json = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_json.encode('utf-8'))
            self.process.stdin.flush()
        except Exception as e:
            print(f"发送通知失败: {e}")
    
    def _list_tools(self):
        """获取工具列表"""
        response = self._send_request("tools/list", {})
        
        if response and "result" in response:
            tools_data = response["result"].get("tools", [])
            self.tools = []
            for tool in tools_data:
                self.tools.append(MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.config.name
                ))
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """调用工具"""
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if response and "result" in response:
            content = response["result"].get("content", [])
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts) if text_parts else json.dumps(content)
        elif response and "error" in response:
            return f"错误: {response['error'].get('message', '未知错误')}"
        
        return None


class MCPSSEConnection(MCPConnectionBase):
    """SSE (Server-Sent Events) 类型的 MCP 连接"""
    
    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self.session_id: Optional[str] = None
        self.sse_endpoint: Optional[str] = None  # SSE 端点 URL
    
    def start(self) -> bool:
        """启动 SSE 连接"""
        try:
            if not self.config.url:
                print(f"SSE MCP 服务器 {self.config.name} 缺少 URL 配置")
                return False
            
            # 初始化连接
            self._initialize()
            
            # 获取工具列表
            self._list_tools()
            
            self.running = True
            print(f"SSE MCP 服务器 {self.config.name} 已连接，获取到 {len(self.tools)} 个工具")
            return True
            
        except Exception as e:
            print(f"连接 SSE MCP 服务器 {self.config.name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop(self):
        """停止 SSE 连接"""
        self.running = False
        self.session_id = None
        self.sse_endpoint = None
        self.tools = []
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        if self.config.headers:
            headers.update(self.config.headers)
        return headers
    
    def _send_request(self, method: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送 JSON-RPC 请求并处理 SSE 响应"""
        if not self.config.url:
            return None
        
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            payload["params"] = params
        
        try:
            # SSE 通常需要使用流式请求
            response = requests.post(
                self.config.url,
                json=payload,
                headers=self._get_headers(),
                timeout=60,
                stream=True  # 启用流式响应
            )
            
            # 检查响应头中的 session ID
            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
            
            # 解析 SSE 响应
            return self._parse_sse_stream(response)
                    
        except Exception as e:
            print(f"发送 SSE 请求失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_sse_stream(self, response) -> Optional[Dict]:
        """解析 SSE 流式响应"""
        result = None
        event_type = None
        data_buffer = []
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                
                line = line.strip() if isinstance(line, str) else line.decode('utf-8').strip()
                
                if not line:
                    # 空行表示事件结束，处理缓冲的数据
                    if data_buffer:
                        data = "\n".join(data_buffer)
                        try:
                            parsed = json.loads(data)
                            if "result" in parsed:
                                result = parsed
                            elif "error" in parsed:
                                result = parsed
                        except json.JSONDecodeError:
                            pass
                        data_buffer = []
                        event_type = None
                    continue
                
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        data_buffer.append(data)
                elif line.startswith("id:"):
                    # 事件 ID，可用于重连
                    pass
                elif line.startswith("retry:"):
                    # 重连时间
                    pass
            
            # 处理最后的数据
            if data_buffer:
                data = "\n".join(data_buffer)
                try:
                    parsed = json.loads(data)
                    if "result" in parsed:
                        result = parsed
                    elif "error" in parsed:
                        result = parsed
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"解析 SSE 流失败: {e}")
        
        return result
    
    def _initialize(self):
        """初始化 MCP 连接"""
        response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "DeepSeek-MCP-Client",
                "version": "1.0.0"
            }
        })
        
        if response and "result" in response:
            # 发送 initialized 通知
            self._send_request("notifications/initialized", {})
    
    def _list_tools(self):
        """获取工具列表"""
        response = self._send_request("tools/list", {})
        
        if response and "result" in response:
            tools_data = response["result"].get("tools", [])
            self.tools = []
            for tool in tools_data:
                self.tools.append(MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.config.name
                ))
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """调用工具"""
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        if response and "result" in response:
            content = response["result"].get("content", [])
            # 合并所有文本内容
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts) if text_parts else json.dumps(content)
        elif response and "error" in response:
            return f"错误: {response['error'].get('message', '未知错误')}"
        
        return None


def create_connection(config: MCPServerConfig) -> Optional[MCPConnectionBase]:
    """根据配置创建合适的连接"""
    if config.server_type == MCPServerType.STREAMABLE_HTTP:
        return MCPHttpConnection(config)
    elif config.server_type == MCPServerType.STDIO:
        return MCPStdioConnection(config)
    elif config.server_type == MCPServerType.SSE:
        return MCPSSEConnection(config)
    else:
        print(f"不支持的服务器类型: {config.server_type}")
        return None


class MCPManager:
    """MCP 管理器 - 管理多个 MCP 服务器连接"""
    
    def __init__(self):
        """初始化 MCP 管理器"""
        self.servers: Dict[str, MCPServerConfig] = {}
        self.connections: Dict[str, MCPConnectionBase] = {}
        self.tools: Dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self._load_from_directory()
    
    def _load_from_directory(self):
        """从 mcp_servers 目录加载服务配置"""
        try:
            from mcp_servers import get_available_servers
            
            servers = get_available_servers()
            loaded_count = 0
            
            for name, info in servers.items():
                if not info["enabled"]:
                    continue
                
                config = info.get("config", {})
                server_type_str = config.get("type", "stdio")
                
                try:
                    server_type = MCPServerType(server_type_str)
                except ValueError:
                    print(f"未知的服务器类型 '{server_type_str}'，使用默认 stdio")
                    server_type = MCPServerType.STDIO
                
                # 根据服务器类型设置配置
                if server_type == MCPServerType.STDIO:
                    self.servers[name] = MCPServerConfig(
                        name=name,
                        server_type=server_type,
                        description=config.get("description", ""),
                        enabled=True,
                        command=config.get("command", "python"),
                        args=[info["server_file"]] + config.get("args", []),
                        env=config.get("env")
                    )
                else:
                    # HTTP/SSE 类型
                    self.servers[name] = MCPServerConfig(
                        name=name,
                        server_type=server_type,
                        description=config.get("description", ""),
                        enabled=True,
                        url=config.get("url"),
                        headers=config.get("headers")
                    )
                
                loaded_count += 1
            
            if loaded_count > 0:
                print(f"从 mcp_servers 目录加载了 {loaded_count} 个服务配置")
                self.start_enabled_servers()
            else:
                print("mcp_servers 目录中没有启用的服务")
            
            return True
        except ImportError as e:
            print(f"无法导入 mcp_servers 模块: {e}")
            return False
        except Exception as e:
            print(f"加载 MCP 配置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reload_config(self):
        """重新加载配置"""
        self.stop_all_servers()
        self.servers.clear()
        self.connections.clear()
        self.tools.clear()
        self._load_from_directory()
    
    def start_enabled_servers(self):
        """启动所有启用的服务器"""
        for name, server in self.servers.items():
            if server.enabled:
                self.start_server(name)
    
    def start_server(self, name: str) -> bool:
        """启动指定服务器"""
        if name not in self.servers:
            print(f"服务器不存在: {name}")
            return False
        
        if name in self.connections:
            print(f"服务器已在运行: {name}")
            return True
        
        config = self.servers[name]
        connection = create_connection(config)
        
        if connection and connection.start():
            self.connections[name] = connection
            # 注册工具
            for tool in connection.get_tools():
                full_name = f"{name}_{tool.name}"
                tool.name = full_name
                self.tools[full_name] = tool
            return True
        
        return False
    
    def stop_server(self, name: str):
        """停止指定服务器"""
        if name in self.connections:
            self.connections[name].stop()
            # 移除相关工具
            self.tools = {k: v for k, v in self.tools.items() if v.server_name != name}
            del self.connections[name]
            print(f"MCP 服务器 {name} 已停止")
    
    def stop_all_servers(self):
        """停止所有服务器"""
        for name in list(self.connections.keys()):
            self.stop_server(name)
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """获取 OpenAI 格式的工具列表"""
        openai_tools = []
        for tool in self.tools.values():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        return openai_tools
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """调用工具"""
        if tool_name not in self.tools:
            return f"工具不存在: {tool_name}"
        
        tool = self.tools[tool_name]
        server_name = tool.server_name
        
        if server_name not in self.connections:
            return f"服务器未连接: {server_name}"
        
        connection = self.connections[server_name]
        # 使用原始工具名称（去掉服务器前缀）
        original_name = tool_name.replace(f"{server_name}_", "", 1)
        return connection.call_tool(original_name, arguments)
    
    def enable_server(self, name: str) -> bool:
        """
        启用服务器
        注意：这只在运行时启用，要永久启用需要修改 enabled.txt
        """
        try:
            from mcp_servers import enable_server as enable_server_file
            if enable_server_file(name):
                self.reload_config()
                return True
        except ImportError:
            pass
        return False
    
    def disable_server(self, name: str) -> bool:
        """
        禁用服务器
        注意：这只在运行时禁用，要永久禁用需要修改 enabled.txt
        """
        try:
            from mcp_servers import disable_server as disable_server_file
            if disable_server_file(name):
                self.stop_server(name)
                return True
        except ImportError:
            pass
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取所有服务器状态"""
        status = {}
        for name, server in self.servers.items():
            is_running = name in self.connections
            tools_count = len([t for t in self.tools.values() if t.server_name == name])
            status[name] = {
                "type": server.server_type.value,
                "enabled": server.enabled,
                "running": is_running,
                "description": server.description,
                "tools_count": tools_count
            }
        return status


# 全局 MCP 管理器实例
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """
    获取全局 MCP 管理器实例
    
    Returns:
        MCP 管理器实例
    """
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager


def create_tool_executor(mcp_manager: MCPManager) -> Callable[[str, Dict[str, Any]], Optional[str]]:
    """创建工具执行器函数"""
    def executor(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        return mcp_manager.call_tool(tool_name, arguments)
    return executor


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("MCP 客户端测试")
    print("=" * 60)
    
    # 创建管理器
    manager = get_mcp_manager()
    
    # 显示状态
    print("\n服务器状态:")
    status = manager.get_status()
    for name, info in status.items():
        print(f"  {name}:")
        print(f"    类型: {info['type']}")
        print(f"    启用: {info['enabled']}")
        print(f"    运行中: {info['running']}")
        print(f"    工具数: {info['tools_count']}")
    
    # 获取 OpenAI 格式的工具
    print("\nOpenAI 格式工具:")
    tools = manager.get_openai_tools()
    for tool in tools:
        desc = tool['function'].get('description', '')[:50]
        print(f"  - {tool['function']['name']}: {desc}...")
    
    print("\n" + "=" * 60)