"""
DeepSeek OpenAI 兼容代理服务器
监听本地端口，接收 OpenAI 格式的请求，转发到 DeepSeek API 并进行工具调用优化
支持 MCP (Model Context Protocol) 工具集成
支持流式响应 (streaming)
"""

import json
import re
import os
import requests
from typing import List, Dict, Any, Optional, Generator, Iterator
from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI
import time
import uuid

# MCP 支持
try:
    from mcp_servers.mcp_client import get_mcp_manager, MCPManager
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("警告: MCP 客户端未找到，MCP 功能将被禁用")

app = Flask(__name__, static_folder='static', static_url_path='/static')


# ==================== 配置加载 ====================

def load_config(config_path: str = "config.jsonc") -> Dict[str, Any]:
    """加载配置文件（支持 JSONC 格式，带注释）"""
    default_config = {
        "chat_completions_url": "https://api.deepseek.com/v1/chat/completions",
        "models_url": "https://api.deepseek.com/v1/models",
        "api_key": "",
        "access_keys": [],
        "allow_user_api_key": True,
        "host": "127.0.0.1",
        "port": 8002,
        "debug": False,
        "mcp_enabled": True,
        "auto_execute_mcp_tools": True,
        "system_prompt_enabled": False,
        "system_prompt": "## 工具调用注意事项\n\n当你使用工具获取信息时，请注意以下几点：\n\n1. **工具调用结果不会保存在对话历史中**：每次工具调用的原始结果只会在当前回合可见，后续对话中将无法再访问这些原始数据。\n\n2. **主动提取和整理信息**：在收到工具返回的结果后，请在你的思考过程中提取所有有用的信息，包括：\n   - 关键数据和数值\n   - 重要的名称、日期、地点等\n   - 相关的上下文信息\n   - 可能在后续对话中需要引用的内容\n\n3. **在回复中复述关键信息**：将提取的重要信息融入你的回复中，这样用户和你都能在后续对话中参考这些信息。\n\n4. **结构化输出**：当工具返回大量信息时，请以清晰、结构化的方式呈现，便于理解和后续引用。"
    }
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除 JSONC 注释（更完善的处理）
        # 1. 移除多行注释 /* ... */
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # 2. 移除单行注释 // ...（但保留字符串中的 //）
        lines = []
        for line in content.split('\n'):
            # 简单处理：查找不在字符串中的 // 注释
            # 如果行中有 //, 只保留 // 之前的部分（简化处理）
            if '//' in line:
                # 检查 // 是否在字符串中
                in_string = False
                quote_char = None
                comment_pos = -1
                for i, char in enumerate(line):
                    if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == '/' and i < len(line) - 1 and line[i+1] == '/' and not in_string:
                        comment_pos = i
                        break
                
                if comment_pos >= 0:
                    line = line[:comment_pos]
            lines.append(line)
        content = '\n'.join(lines)
        
        config = json.loads(content)
        
        # 调试：显示从配置文件加载的关键配置
        print(f"✓ 配置文件加载成功: {config_path}")
        if 'port' in config:
            print(f"  - 配置文件中的端口: {config['port']}")
        if 'host' in config:
            print(f"  - 配置文件中的主机: {config['host']}")
        
        # 合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}，使用默认配置")
        import traceback
        traceback.print_exc()
        return default_config


def get_base_url_from_chat_url(chat_url: str) -> str:
    """从聊天补全 URL 中提取基础 URL（用于 OpenAI SDK）"""
    # 移除 /chat/completions 部分，保留到 /v1
    if '/chat/completions' in chat_url:
        return chat_url.rsplit('/chat/completions', 1)[0]
    return chat_url


def validate_access_key(auth_header: str) -> tuple:
    """
    验证访问密钥
    返回: (是否有效, API Key, 错误消息)
    """
    global CONFIG
    
    if not auth_header.startswith('Bearer '):
        return False, None, "Missing or invalid Authorization header"
    
    user_key = auth_header[7:]
    access_keys = CONFIG.get("access_keys", [])
    allow_user_api_key = CONFIG.get("allow_user_api_key", True)
    config_api_key = CONFIG.get("api_key", "")
    
    # 如果没有配置访问密钥，允许所有请求
    if not access_keys:
        # 如果允许用户使用自己的 API Key
        if allow_user_api_key and user_key:
            return True, user_key, None
        # 否则使用配置的 API Key
        if config_api_key:
            return True, config_api_key, None
        # 没有配置 API Key，返回用户提供的
        return True, user_key, None
    
    # 检查是否是有效的访问密钥
    if user_key in access_keys:
        # 使用配置的 API Key
        if config_api_key:
            return True, config_api_key, None
        return False, None, "Server API key not configured"
    
    # 如果允许用户使用自己的 API Key
    if allow_user_api_key:
        return True, user_key, None
    
    return False, None, "Invalid access key"


# 全局配置
CONFIG: Dict[str, Any] = {}

# 全局 MCP 管理器
mcp_manager: Optional['MCPManager'] = None


class DeepSeekProxy:
    """DeepSeek 代理处理器"""
    
    def __init__(self, api_key: str, mcp_mgr: Optional['MCPManager'] = None):
        """初始化客户端"""
        global CONFIG
        base_url = get_base_url_from_chat_url(CONFIG.get("chat_completions_url", "https://api.deepseek.com/v1/chat/completions"))
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.mcp_manager = mcp_mgr
    
    def _message_to_dict(self, message) -> Dict[str, Any]:
        """将消息对象转换为字典格式"""
        result = {
            "role": message.role,
            "content": message.content or "",
        }
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            result["reasoning_content"] = message.reasoning_content
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "arguments": tc.function.arguments,
                        "name": tc.function.name
                    },
                    "type": tc.type,
                    "index": tc.index if hasattr(tc, 'index') else i
                }
                for i, tc in enumerate(message.tool_calls)
            ]
        
        return result
    
    def _flatten_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压扁 tool_calls，去掉 id，保留 function、type、index"""
        if not tool_calls:
            return []
        
        flattened = []
        for i, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                flattened.append({
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    },
                    "type": tc.get("type", "function"),
                    "index": tc.get("index", i)
                })
        
        return flattened
    
    def _merge_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        assistant_msg_index: int,
        new_reasoning: str,
        new_content: str,
        new_tool_calls: Optional[List]
    ):
        """合并助手消息"""
        # 删除所有 tool 消息
        while messages and isinstance(messages[-1], dict) and messages[-1].get('role') == 'tool':
            messages.pop()
        
        if assistant_msg_index is not None and assistant_msg_index < len(messages):
            prev_assistant = messages[assistant_msg_index]
            
            # 将旧的 tool_calls 压扁并添加到 reasoning_content
            old_reasoning = prev_assistant.get('reasoning_content', '') or ''
            old_tool_calls = prev_assistant.get('tool_calls', [])
            
            if old_tool_calls:
                flattened_tools = self._flatten_tool_calls(old_tool_calls)
                tools_obj = {"tool_calls": flattened_tools}
                tools_json = json.dumps(tools_obj, ensure_ascii=False)
                
                if old_reasoning:
                    old_reasoning = old_reasoning + "\n\n" + tools_json
                else:
                    old_reasoning = tools_json
            
            # 追加新的思维链
            if new_reasoning:
                combined_reasoning = old_reasoning + "\n\n" + new_reasoning if old_reasoning else new_reasoning
                prev_assistant['reasoning_content'] = combined_reasoning
            
            # 更新工具调用
            if new_tool_calls:
                prev_assistant['tool_calls'] = [
                    {
                        "id": tc.id,
                        "function": {
                            "arguments": tc.function.arguments,
                            "name": tc.function.name
                        },
                        "type": tc.type,
                        "index": tc.index if hasattr(tc, 'index') else i
                    }
                    for i, tc in enumerate(new_tool_calls)
                ]
            else:
                if 'tool_calls' in prev_assistant:
                    del prev_assistant['tool_calls']
                prev_assistant['content'] = new_content
    
    def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """执行 MCP 工具"""
        if not self.mcp_manager:
            return None
        return self.mcp_manager.call_tool(tool_name, arguments)
    
    def _is_mcp_tool(self, tool_name: str) -> bool:
        """检查是否是 MCP 工具"""
        if not self.mcp_manager:
            return False
        return tool_name in self.mcp_manager.tools
    
    def process_request_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-reasoner",
        tools: Optional[List[Dict[str, Any]]] = None,
        execute_mcp_tools: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        处理流式聊天补全请求
        
        Args:
            messages: 消息列表
            model: 模型名称
            tools: 工具列表（可选，如果启用 MCP 会自动合并 MCP 工具）
            execute_mcp_tools: 是否自动执行 MCP 工具调用
            **kwargs: 其他参数
        
        Yields:
            SSE 格式的流式响应数据
        """
        # 合并 MCP 工具到工具列表
        combined_tools = list(tools) if tools else []
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_openai_tools()
            combined_tools.extend(mcp_tools)
        
        # 如果没有任何工具，设置为 None
        if not combined_tools:
            combined_tools = None
        
        messages_copy = [msg.copy() for msg in messages]
        iteration = 0
        max_iterations = 10
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_time = int(time.time())
        
        while iteration < max_iterations:
            # 调用 DeepSeek API（流式）
            stream_response = self.client.chat.completions.create(
                model=model,
                messages=messages_copy,
                tools=combined_tools,
                stream=True,
                **kwargs
            )
            
            # 收集流式响应
            reasoning_content = ""
            content = ""
            tool_calls_data = {}  # id -> {function: {name, arguments}, type, index}
            finish_reason = None
            
            for chunk in stream_response:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                chunk_finish_reason = chunk.choices[0].finish_reason
                
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
                
                # 处理 reasoning_content 增量
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    # 发送 reasoning_content chunk
                    chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "reasoning_content": delta.reasoning_content
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 处理 content 增量
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                    # 发送 content chunk
                    chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": delta.content
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 处理 tool_calls 增量
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_index = tc.index if hasattr(tc, 'index') else 0
                        if tc_index not in tool_calls_data:
                            tool_calls_data[tc_index] = {
                                "id": tc.id if hasattr(tc, 'id') and tc.id else f"call_{tc_index}",
                                "type": tc.type if hasattr(tc, 'type') else "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if hasattr(tc, 'id') and tc.id:
                            tool_calls_data[tc_index]["id"] = tc.id
                        
                        if hasattr(tc, 'function'):
                            if hasattr(tc.function, 'name') and tc.function.name:
                                tool_calls_data[tc_index]["function"]["name"] += tc.function.name
                            if hasattr(tc.function, 'arguments') and tc.function.arguments:
                                tool_calls_data[tc_index]["function"]["arguments"] += tc.function.arguments
            
            # 流式响应结束后，检查是否有工具调用
            tool_calls_list = [tool_calls_data[i] for i in sorted(tool_calls_data.keys())] if tool_calls_data else None
            
            # 如果没有工具调用，结束流式响应
            if not tool_calls_list or finish_reason == "stop":
                # 发送结束 chunk
                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": finish_reason or "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 检查是否有 MCP 工具调用需要执行
            if tool_calls_list and execute_mcp_tools and self.mcp_manager:
                mcp_tool_calls = []
                non_mcp_tool_calls = []
                
                for tc in tool_calls_list:
                    if self._is_mcp_tool(tc["function"]["name"]):
                        mcp_tool_calls.append(tc)
                    else:
                        non_mcp_tool_calls.append(tc)
                
                # 执行 MCP 工具调用
                if mcp_tool_calls:
                    # 将工具调用压扁并追加到 reasoning_content
                    flattened_tools = []
                    for tc in tool_calls_list:
                        flattened_tools.append({
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            },
                            "type": tc.get("type", "function"),
                            "index": tc.get("index", 0)
                        })
                    
                    tools_obj = {"tool_calls": flattened_tools}
                    tools_json = json.dumps(tools_obj, ensure_ascii=False)
                    
                    # 发送压扁的工具调用作为 reasoning_content（后面添加两个换行符）
                    tool_calls_reasoning = ("\n\n" if reasoning_content else "") + tools_json + "\n\n"
                    chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "reasoning_content": tool_calls_reasoning
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
                    # 更新累积的 reasoning_content
                    if reasoning_content:
                        reasoning_content += tool_calls_reasoning
                    else:
                        reasoning_content = tools_json
                    
                    # 构建助手消息（包含合并后的 reasoning_content）
                    assistant_msg = {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": tool_calls_list
                    }
                    if reasoning_content:
                        assistant_msg["reasoning_content"] = reasoning_content
                    
                    messages_copy.append(assistant_msg)
                    
                    for tc in mcp_tool_calls:
                        args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                        result = self._execute_mcp_tool(tc["function"]["name"], args)
                        
                        # 添加工具结果到消息
                        messages_copy.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result or "工具执行失败"
                        })
                    
                    iteration += 1
                    continue  # 继续下一轮对话
                
                # 如果只有非 MCP 工具调用，结束并返回工具调用请求
                if non_mcp_tool_calls:
                    final_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "tool_calls"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            
            # 如果有工具调用但不自动执行，结束并返回
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }]
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # 达到最大迭代次数
        error_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "[达到最大工具调用迭代次数]"
                },
                "logprobs": None,
                "finish_reason": "length"
            }]
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    def process_request(
        self,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-reasoner",
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        execute_mcp_tools: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理聊天补全请求（非流式）
        
        Args:
            messages: 消息列表
            model: 模型名称
            tools: 工具列表（可选，如果启用 MCP 会自动合并 MCP 工具）
            stream: 是否流式（此方法仅处理非流式，流式请使用 process_request_stream）
            execute_mcp_tools: 是否自动执行 MCP 工具调用
            **kwargs: 其他参数
        """
        
        # 合并 MCP 工具到工具列表
        combined_tools = list(tools) if tools else []
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_openai_tools()
            combined_tools.extend(mcp_tools)
        
        # 如果没有任何工具，设置为 None
        if not combined_tools:
            combined_tools = None
        
        messages_copy = [msg.copy() for msg in messages]
        iteration = 0
        max_iterations = 10
        assistant_msg_index = None
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        while iteration < max_iterations:
            # 调用 DeepSeek API（使用合并后的工具列表）
            response = self.client.chat.completions.create(
                model=model,
                messages=messages_copy,
                tools=combined_tools,
                **kwargs
            )
            
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            # 累计 token 使用（保留详细信息）
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                total_usage["prompt_tokens"] += getattr(usage, 'prompt_tokens', 0)
                total_usage["completion_tokens"] += getattr(usage, 'completion_tokens', 0)
                total_usage["total_tokens"] += getattr(usage, 'total_tokens', 0)
                
                # 保留详细的 usage 信息（如果存在）
                if hasattr(usage, 'prompt_tokens_details'):
                    if "prompt_tokens_details" not in total_usage:
                        total_usage["prompt_tokens_details"] = {"cached_tokens": 0}
                    total_usage["prompt_tokens_details"]["cached_tokens"] += getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                
                if hasattr(usage, 'completion_tokens_details'):
                    if "completion_tokens_details" not in total_usage:
                        total_usage["completion_tokens_details"] = {"reasoning_tokens": 0}
                    total_usage["completion_tokens_details"]["reasoning_tokens"] += getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
                
                if hasattr(usage, 'prompt_cache_hit_tokens'):
                    if "prompt_cache_hit_tokens" not in total_usage:
                        total_usage["prompt_cache_hit_tokens"] = 0
                    total_usage["prompt_cache_hit_tokens"] += getattr(usage, 'prompt_cache_hit_tokens', 0)
                
                if hasattr(usage, 'prompt_cache_miss_tokens'):
                    if "prompt_cache_miss_tokens" not in total_usage:
                        total_usage["prompt_cache_miss_tokens"] = 0
                    total_usage["prompt_cache_miss_tokens"] += getattr(usage, 'prompt_cache_miss_tokens', 0)
            
            # 提取响应内容
            new_reasoning = getattr(message, 'reasoning_content', None) or ""
            new_content = message.content or ""
            new_tool_calls = message.tool_calls
            
            if iteration == 0:
                # 首次调用：添加助手消息
                new_msg_dict = self._message_to_dict(message)
                messages_copy.append(new_msg_dict)
                assistant_msg_index = len(messages_copy) - 1
            else:
                # 后续调用：合并到之前的助手消息
                self._merge_assistant_message(
                    messages_copy,
                    assistant_msg_index,
                    new_reasoning,
                    new_content,
                    new_tool_calls
                )
            
            # 如果没有工具调用，返回结果
            if new_tool_calls is None or finish_reason == "stop":
                final_msg = messages_copy[assistant_msg_index]
                
                # 构建消息对象（按照 DeepSeek 官方顺序）
                message_obj = {
                    "role": "assistant",
                    "content": final_msg.get("content", ""),
                    "reasoning_content": final_msg.get("reasoning_content", "")
                }
                
                # 构建响应（匹配 DeepSeek 官方格式）
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message_obj,
                            "logprobs": None,
                            "finish_reason": finish_reason
                        }
                    ],
                    "usage": total_usage
                }
                
                # 添加 system_fingerprint（如果响应中有）
                if hasattr(response, 'system_fingerprint'):
                    result["system_fingerprint"] = response.system_fingerprint
                
                return result
            
            # 检查是否有 MCP 工具调用需要执行
            if new_tool_calls and execute_mcp_tools and self.mcp_manager:
                mcp_tool_calls = []
                non_mcp_tool_calls = []
                
                for tc in new_tool_calls:
                    if self._is_mcp_tool(tc.function.name):
                        mcp_tool_calls.append(tc)
                    else:
                        non_mcp_tool_calls.append(tc)
                
                # 执行 MCP 工具调用
                if mcp_tool_calls:
                    for tc in mcp_tool_calls:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        result = self._execute_mcp_tool(tc.function.name, args)
                        
                        # 添加工具结果到消息
                        messages_copy.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result or "工具执行失败"
                        })
                    
                    iteration += 1
                    continue  # 继续下一轮对话
                
                # 如果只有非 MCP 工具调用，返回给客户端处理
                if non_mcp_tool_calls:
                    final_msg = messages_copy[assistant_msg_index]
                    
                    message_obj = {
                        "role": "assistant",
                        "content": final_msg.get("content", ""),
                        "reasoning_content": final_msg.get("reasoning_content", ""),
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "arguments": tc.function.arguments,
                                    "name": tc.function.name
                                },
                                "type": tc.type,
                                "index": tc.index if hasattr(tc, 'index') else i
                            }
                            for i, tc in enumerate(non_mcp_tool_calls)
                        ]
                    }
                    
                    result = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message_obj,
                                "logprobs": None,
                                "finish_reason": "tool_calls"
                            }
                        ],
                        "usage": total_usage
                    }
                    
                    if hasattr(response, 'system_fingerprint'):
                        result["system_fingerprint"] = response.system_fingerprint
                    
                    return result
            
            # 如果有工具调用但没有提供工具函数且没有 MCP，返回工具调用请求
            # 让客户端自己执行工具
            if new_tool_calls and not combined_tools:
                final_msg = messages_copy[assistant_msg_index]
                
                message_obj = {
                    "role": "assistant",
                    "content": final_msg.get("content", ""),
                    "reasoning_content": final_msg.get("reasoning_content", ""),
                    "tool_calls": final_msg.get("tool_calls", [])
                }
                
                result = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message_obj,
                            "logprobs": None,
                            "finish_reason": "tool_calls"
                        }
                    ],
                    "usage": total_usage
                }
                
                if hasattr(response, 'system_fingerprint'):
                    result["system_fingerprint"] = response.system_fingerprint
                
                return result
            
            # 如果有工具调用，但这是代理服务器，我们不执行工具
            # 返回工具调用请求给客户端
            final_msg = messages_copy[assistant_msg_index]
            
            message_obj = {
                "role": "assistant",
                "content": final_msg.get("content", ""),
                "reasoning_content": final_msg.get("reasoning_content", ""),
                "tool_calls": final_msg.get("tool_calls", [])
            }
            
            result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message_obj,
                        "logprobs": None,
                        "finish_reason": "tool_calls"
                    }
                ],
                "usage": total_usage
            }
            
            if hasattr(response, 'system_fingerprint'):
                result["system_fingerprint"] = response.system_fingerprint
            
            return result
        
        # 达到最大迭代次数
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "达到最大迭代次数"
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": total_usage
        }


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理聊天补全请求"""
    global mcp_manager, CONFIG
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        # 验证访问密钥并获取 API Key
        auth_header = request.headers.get('Authorization', '')
        is_valid, api_key, error_msg = validate_access_key(auth_header)
        if not is_valid:
            return jsonify({"error": {"message": error_msg, "type": "auth_error"}}), 401
        
        # 提取参数
        messages = data.get('messages', [])
        model = data.get('model')
        
        # 模型参数是必须的
        if not model:
            return jsonify({"error": {"message": "Missing required parameter: model", "type": "invalid_request_error"}}), 400
        
        # 添加系统提示词（如果启用）
        if CONFIG.get('system_prompt_enabled', False):
            system_prompt = CONFIG.get('system_prompt', '')
            if system_prompt:
                # 检查消息数组开头是否已有 system 消息
                if messages and messages[0].get('role') == 'system':
                    # 将系统提示词追加到现有 system 消息
                    messages = messages.copy()
                    messages[0] = messages[0].copy()
                    messages[0]['content'] = system_prompt + "\n\n" + messages[0]['content']
                else:
                    # 在开头添加新的 system 消息
                    messages = [{"role": "system", "content": system_prompt}] + messages
        
        tools = data.get('tools')
        stream = data.get('stream', False)
        execute_mcp_tools = data.get('execute_mcp_tools', CONFIG.get('auto_execute_mcp_tools', True))
        
        # 其他参数
        kwargs = {}
        for key in ['temperature', 'top_p', 'max_tokens', 'presence_penalty', 'frequency_penalty']:
            if key in data:
                kwargs[key] = data[key]
        
        # 创建代理处理器（传入 MCP 管理器）
        proxy = DeepSeekProxy(api_key, mcp_manager)
        
        # 流式响应
        if stream:
            def generate():
                try:
                    for chunk in proxy.process_request_stream(
                        messages=messages,
                        model=model,
                        tools=tools,
                        execute_mcp_tools=execute_mcp_tools,
                        **kwargs
                    ):
                        yield chunk
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "code": "internal_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        # 非流式响应
        response = proxy.process_request(
            messages=messages,
            model=model,
            tools=tools,
            stream=False,
            execute_mcp_tools=execute_mcp_tools,
            **kwargs
        )
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型（从后端 API 获取）"""
    global CONFIG
    
    models_url = CONFIG.get('models_url', 'https://api.deepseek.com/v1/models')
    api_key = CONFIG.get('api_key', '')
    
    try:
        # 优先使用配置的 API Key
        if api_key:
            headers = {'Authorization': f'Bearer {api_key}'}
            response = requests.get(models_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({
                    "error": {
                        "message": f"Failed to fetch models from API: {response.status_code}",
                        "type": "api_error"
                    }
                }), response.status_code
        
        # 如果请求中有 API Key，用请求中的
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            user_key = auth_header[7:]
            headers = {'Authorization': f'Bearer {user_key}'}
            response = requests.get(models_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({
                    "error": {
                        "message": f"Failed to fetch models from API: {response.status_code}",
                        "type": "api_error"
                    }
                }), response.status_code
        
        # 没有可用的 API Key
        return jsonify({
            "error": {
                "message": "No API key available to fetch models",
                "type": "auth_error"
            }
        }), 401
        
    except requests.exceptions.Timeout:
        return jsonify({
            "error": {
                "message": "Request to models API timed out",
                "type": "timeout_error"
            }
        }), 504
    except Exception as e:
        return jsonify({
            "error": {
                "message": f"Failed to fetch models: {str(e)}",
                "type": "server_error"
            }
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    global mcp_manager
    
    status = {"status": "ok"}
    
    if mcp_manager:
        status["mcp"] = {
            "available": True,
            "servers": mcp_manager.get_status(),
            "tools_count": len(mcp_manager.tools)
        }
    else:
        status["mcp"] = {"available": False}
    
    return jsonify(status)


# ==================== MCP 管理 API ====================

@app.route('/v1/mcp/status', methods=['GET'])
def mcp_status():
    """获取 MCP 状态"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    if not mcp_manager:
        return jsonify({"error": "MCP 管理器未初始化"}), 503
    
    return jsonify({
        "servers": mcp_manager.get_status(),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "server": tool.server_name
            }
            for tool in mcp_manager.tools.values()
        ]
    })


@app.route('/v1/mcp/tools', methods=['GET'])
def mcp_tools():
    """获取 MCP 工具列表（OpenAI 格式）"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"tools": []})
    
    return jsonify({
        "tools": mcp_manager.get_openai_tools()
    })


@app.route('/v1/mcp/servers', methods=['GET'])
def mcp_list_servers():
    """列出所有 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"servers": {}})
    
    return jsonify({
        "servers": mcp_manager.get_status()
    })


@app.route('/v1/mcp/servers', methods=['POST'])
def mcp_add_server():
    """添加 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    if not mcp_manager:
        return jsonify({"error": "MCP 管理器未初始化"}), 503
    
    data = request.get_json()
    name = data.get('name')
    command = data.get('command')
    args = data.get('args', [])
    description = data.get('description', '')
    enabled = data.get('enabled', True)
    env = data.get('env')
    
    if not name or not command:
        return jsonify({"error": "缺少必要参数: name, command"}), 400
    
    success = mcp_manager.add_server(name, command, args, description, enabled, env)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"服务器 {name} 已添加",
            "status": mcp_manager.get_status().get(name)
        })
    else:
        return jsonify({"error": f"添加服务器 {name} 失败"}), 500


@app.route('/v1/mcp/servers/<name>', methods=['DELETE'])
def mcp_remove_server(name: str):
    """移除 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    success = mcp_manager.remove_server(name)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"服务器 {name} 已移除"
        })
    else:
        return jsonify({"error": f"服务器 {name} 不存在"}), 404


@app.route('/v1/mcp/servers/<name>/start', methods=['POST'])
def mcp_start_server(name: str):
    """启动 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    success = mcp_manager.start_server(name)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"服务器 {name} 已启动",
            "status": mcp_manager.get_status().get(name)
        })
    else:
        return jsonify({"error": f"启动服务器 {name} 失败"}), 500


@app.route('/v1/mcp/servers/<name>/stop', methods=['POST'])
def mcp_stop_server(name: str):
    """停止 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    mcp_manager.stop_server(name)
    
    return jsonify({
        "success": True,
        "message": f"服务器 {name} 已停止"
    })


@app.route('/v1/mcp/reload', methods=['POST'])
def mcp_reload():
    """重新加载 MCP 配置"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    if mcp_manager:
        mcp_manager.reload_config()
        return jsonify({
            "success": True,
            "message": "MCP 配置已重新加载",
            "servers": mcp_manager.get_status()
        })
    else:
        return jsonify({"error": "MCP 管理器未初始化"}), 503


@app.route('/v1/mcp/servers/all', methods=['GET'])
def mcp_list_all_servers():
    """列出所有可用的 MCP 服务器（包括禁用的）"""
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import get_available_servers
        servers = get_available_servers()
        
        result = {}
        for name, info in servers.items():
            result[name] = {
                "name": name,
                "type": info.get("type", "stdio"),
                "description": info.get("description", ""),
                "enabled": info.get("enabled", False),
                "config": info.get("config", {}),
                "path": info.get("path", ""),
                "server_file": info.get("server_file"),
                "running": mcp_manager and name in mcp_manager.connections if mcp_manager else False,
                "tools_count": len([t for t in mcp_manager.tools.values() if t.server_name == name]) if mcp_manager else 0
            }
        
        return jsonify({"servers": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/mcp/servers/<name>/enable', methods=['POST'])
def mcp_enable_server(name: str):
    """启用 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import enable_server
        success = enable_server(name)
        
        if success:
            # 重新加载配置
            if mcp_manager:
                mcp_manager.reload_config()
            
            return jsonify({
                "success": True,
                "message": f"服务器 {name} 已启用"
            })
        else:
            return jsonify({"error": f"服务器 {name} 不存在"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/mcp/servers/<name>/disable', methods=['POST'])
def mcp_disable_server(name: str):
    """禁用 MCP 服务器"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import disable_server
        success = disable_server(name)
        
        if success:
            # 停止运行中的服务器
            if mcp_manager and name in mcp_manager.connections:
                mcp_manager.stop_server(name)
            
            return jsonify({
                "success": True,
                "message": f"服务器 {name} 已禁用"
            })
        else:
            return jsonify({"error": f"禁用服务器 {name} 失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/mcp/servers/<name>/details', methods=['GET'])
def mcp_server_details(name: str):
    """获取 MCP 服务器详情"""
    global mcp_manager
    
    if not MCP_AVAILABLE:
        return jsonify({"error": "MCP 功能不可用"}), 503
    
    try:
        from mcp_servers import get_available_servers
        servers = get_available_servers()
        
        if name not in servers:
            return jsonify({"error": f"服务器 {name} 不存在"}), 404
        
        info = servers[name]
        config = info.get("config", {})
        
        # 获取工具列表
        tools = []
        if mcp_manager and name in mcp_manager.connections:
            for tool in mcp_manager.tools.values():
                if tool.server_name == name:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    })
        
        result = {
            "name": name,
            "type": info.get("type", "stdio"),
            "description": info.get("description", ""),
            "enabled": info.get("enabled", False),
            "running": mcp_manager and name in mcp_manager.connections if mcp_manager else False,
            "path": info.get("path", ""),
            "server_file": info.get("server_file"),
            "config": config,
            "tools": tools
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 静态文件服务 ====================

@app.route('/')
def index():
    """返回首页"""
    return app.send_static_file('index.html')


@app.route('/admin')
def admin():
    """MCP 管理界面"""
    return app.send_static_file('mcp_admin.html')


@app.route('/tools')
def tools_page():
    """MCP 工具列表页面"""
    return app.send_static_file('tools.html')


@app.route('/status')
def status_page():
    """健康状态页面"""
    return app.send_static_file('health.html')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek OpenAI 兼容代理服务器')
    parser.add_argument('--config', type=str, default='config.jsonc', help='配置文件路径')
    parser.add_argument('--host', type=str, help='监听地址（覆盖配置文件）')
    parser.add_argument('--port', type=int, help='监听端口（覆盖配置文件）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-mcp', action='store_true', help='禁用 MCP 功能')
    
    args = parser.parse_args()
    
    # 加载配置文件
    CONFIG = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    host = args.host or CONFIG.get('host', '127.0.0.1')
    port = args.port or CONFIG.get('port', 8002)
    debug = args.debug or CONFIG.get('debug', False)
    mcp_enabled = CONFIG.get('mcp_enabled', True) and not args.no_mcp
    
    print("=" * 60)
    print("DeepSeek OpenAI 兼容代理服务器")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"监听地址: http://{host}:{port}")
    print(f"API 端点: http://{host}:{port}/v1/chat/completions")
    print(f"后端 API: {CONFIG.get('chat_completions_url', 'N/A')}")
    
    # 显示访问控制状态
    access_keys = CONFIG.get('access_keys', [])
    if access_keys:
        print(f"访问控制: 已启用 ({len(access_keys)} 个密钥)")
    else:
        print("访问控制: 已禁用（开放访问）")
    
    if CONFIG.get('api_key'):
        print("转发 Key: 已配置")
    else:
        print("转发 Key: 未配置（使用用户 Key）")
    
    # 初始化 MCP（从 mcp_servers 目录自动发现服务）
    if MCP_AVAILABLE and mcp_enabled:
        try:
            mcp_manager = get_mcp_manager()
            print(f"MCP 配置: mcp_servers/ 目录")
            print(f"MCP 服务器: {len(mcp_manager.servers)} 个配置, {len(mcp_manager.connections)} 个运行中")
            print(f"MCP 工具: {len(mcp_manager.tools)} 个可用")
        except Exception as e:
            print(f"MCP 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            mcp_manager = None
    else:
        print("MCP: 已禁用")
    
    print("=" * 60)
    
    app.run(host=host, port=port, debug=debug)