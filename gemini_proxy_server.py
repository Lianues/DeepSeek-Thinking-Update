"""
Gemini API 代理服务器
监听本地端口，接收 Gemini 格式的请求，转发到 Gemini API 并进行工具调用优化
支持 MCP (Model Context Protocol) 工具集成
支持流式响应 (streaming)
支持思考签名 (thought_signature) 的处理
"""

import json
import re
import os
import copy
import requests
from typing import List, Dict, Any, Optional, Generator, Iterator
from flask import Flask, request, jsonify, Response, stream_with_context
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

def load_config(config_path: str = "gemini_config.jsonc") -> Dict[str, Any]:
    """加载配置文件（支持 JSONC 格式，带注释）"""
    default_config = {
        "gemini_api_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "gemini_models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "api_key": "",
        "access_keys": [],
        "allow_user_api_key": True,
        "host": "127.0.0.1",
        "port": 8003,
        "debug": False,
        "mcp_enabled": True,
        "auto_execute_mcp_tools": True,
        "max_iterations": 100,
        "api_retry_count": 2,
        "api_retry_delay": 5,
        "api_timeout": 300,
        "system_prompt": "## 工具调用注意事项\n\n当你使用工具获取信息时，请注意以下几点：\n\n1. **工具调用结果不会保存在对话历史中**：每次工具调用的原始结果只会在当前回合可见，后续对话中将无法再访问这些原始数据。\n\n2. **主动提取和整理信息**：在收到工具返回的结果后，请在你的思考过程中提取所有有用的信息。\n\n3. **在回复中复述关键信息**：将提取的重要信息融入你的回复中，这样用户和你都能在后续对话中参考这些信息。\n\n4. **结构化输出**：当工具返回大量信息时，请以清晰、结构化的方式呈现，便于理解和后续引用。"
    }
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除 JSONC 注释
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        lines = []
        for line in content.split('\n'):
            if '//' in line:
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
        
        print(f"✓ 配置文件加载成功: {config_path}")
        if 'port' in config:
            print(f"  - 配置文件中的端口: {config['port']}")
        if 'host' in config:
            print(f"  - 配置文件中的主机: {config['host']}")
        
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}，使用默认配置")
        import traceback
        traceback.print_exc()
        return default_config


def validate_access_key(auth_header: str) -> tuple:
    """
    验证访问密钥
    返回: (是否有效, API Key, 错误消息)
    """
    global CONFIG
    
    # Gemini API 使用 x-goog-api-key 头或 Bearer token
    user_key = auth_header
    
    access_keys = CONFIG.get("access_keys", [])
    allow_user_api_key = CONFIG.get("allow_user_api_key", True)
    config_api_key = CONFIG.get("api_key", "")
    
    # 如果没有配置访问密钥，允许所有请求
    if not access_keys:
        if allow_user_api_key and user_key:
            return True, user_key, None
        if config_api_key:
            return True, config_api_key, None
        return True, user_key, None
    
    # 检查是否是有效的访问密钥
    if user_key in access_keys:
        if config_api_key:
            return True, config_api_key, None
        return False, None, "Server API key not configured"
    
    if allow_user_api_key:
        return True, user_key, None
    
    return False, None, "Invalid access key"


# 全局配置
CONFIG: Dict[str, Any] = {}

# 全局 MCP 管理器
mcp_manager: Optional['MCPManager'] = None


class GeminiProxy:
    """Gemini 代理处理器"""
    
    def __init__(self, api_key: str, mcp_mgr: Optional['MCPManager'] = None):
        """初始化客户端"""
        global CONFIG
        self.api_key = api_key
        self.base_url = CONFIG.get("gemini_api_url", "https://generativelanguage.googleapis.com/v1beta/models")
        self.mcp_manager = mcp_mgr
    
    def _format_tool_call_text(self, tool_name: str, arguments: str) -> str:
        """将工具调用格式化为简单文本格式，前后添加两个换行符"""
        return f"\n\n「调用工具：{tool_name}|内容：{arguments}」\n\n"
    
    def _replace_old_tool_results(self, contents: List[Dict[str, Any]], original_contents_length: int):
        """
        将代理服务器添加的旧工具调用结果替换为占位符「调用完毕」
        只处理代理服务器添加的工具结果（索引 >= original_contents_length），不处理用户原始传入的
        
        Args:
            contents: 消息列表
            original_contents_length: 用户原始请求中 contents 的长度
        """
        # 找到代理服务器添加的最后一个 functionResponse 的位置
        last_function_response_index = -1
        for i in range(len(contents) - 1, original_contents_length - 1, -1):
            if isinstance(contents[i], dict) and contents[i].get('role') == 'user':
                parts = contents[i].get('parts', [])
                has_function_response = any(
                    isinstance(p, dict) and 'functionResponse' in p
                    for p in parts
                )
                if has_function_response:
                    last_function_response_index = i
                    break
        
        # 只替换代理服务器添加的工具结果（除了最后一个）
        for i in range(original_contents_length, len(contents)):
            if i == last_function_response_index:
                continue  # 跳过最后一个 functionResponse
            content = contents[i]
            if isinstance(content, dict) and content.get('role') == 'user':
                parts = content.get('parts', [])
                for part in parts:
                    if isinstance(part, dict) and 'functionResponse' in part:
                        # 替换为占位符
                        part['functionResponse']['response'] = {'result': '调用完毕'}
    
    def _find_current_turn_start(self, contents: List[Dict[str, Any]]) -> int:
        """
        从后往前查找当前轮次的开始位置
        当前轮次是从最后一个包含普通文本（非 functionResponse）的 user 消息开始
        """
        for i in range(len(contents) - 1, -1, -1):
            content = contents[i]
            if isinstance(content, dict) and content.get('role') == 'user':
                parts = content.get('parts', [])
                has_text = any(
                    isinstance(p, dict) and 'text' in p 
                    for p in parts
                )
                has_only_function_response = all(
                    isinstance(p, dict) and 'functionResponse' in p 
                    for p in parts
                )
                if has_text and not has_only_function_response:
                    return i
        return 0
    
    def _extract_thought_signatures(self, content: Dict[str, Any]) -> Dict[int, str]:
        """
        从 model content 中提取思考签名
        返回: {part_index: signature}
        """
        signatures = {}
        parts = content.get('parts', [])
        for i, part in enumerate(parts):
            if isinstance(part, dict) and 'thoughtSignature' in part:
                signatures[i] = part['thoughtSignature']
        return signatures
    
    def _has_function_call(self, content: Dict[str, Any]) -> bool:
        """检查 content 是否包含函数调用"""
        parts = content.get('parts', [])
        return any(
            isinstance(p, dict) and 'functionCall' in p 
            for p in parts
        )
    
    def _get_function_calls(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 content 中提取所有函数调用"""
        function_calls = []
        parts = content.get('parts', [])
        for i, part in enumerate(parts):
            if isinstance(part, dict) and 'functionCall' in part:
                fc = part['functionCall'].copy()
                fc['_part_index'] = i
                if 'thoughtSignature' in part:
                    fc['_thought_signature'] = part['thoughtSignature']
                function_calls.append(fc)
        return function_calls
    
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
    
    def _get_mcp_tools_as_gemini_format(self) -> List[Dict[str, Any]]:
        """将 MCP 工具转换为 Gemini 格式"""
        if not self.mcp_manager:
            return []
        
        function_declarations = []
        for tool in self.mcp_manager.tools.values():
            decl = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.input_schema:
                decl["parameters"] = tool.input_schema
            function_declarations.append(decl)
        
        if function_declarations:
            return [{"functionDeclarations": function_declarations}]
        return []
    
    def _make_gemini_request(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        stream: bool = False,
        retry_count: int = 0,
        max_retries: int = 2,
        retry_delay: float = 5.0
    ) -> requests.Response:
        """
        发送请求到 Gemini API
        支持自动重试（当非首次请求失败时）
        
        Args:
            retry_count: 当前重试次数（内部使用）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        
        endpoint = "streamGenerateContent" if stream else "generateContent"
        url = f"{self.base_url}/{model}:{endpoint}"
        
        # 流式请求需要添加 alt=sse 参数
        if stream:
            url += "?alt=sse"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": contents
        }
        
        if tools:
            payload["tools"] = tools
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        # 添加系统指令
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        global CONFIG
        timeout = CONFIG.get('api_timeout', 300)
        
        if stream:
            return requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
        else:
            return requests.post(url, headers=headers, json=payload, timeout=timeout)
    
    def _make_gemini_request_with_retry(
        self,
        model: str,
        contents: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        stream: bool = False,
        is_internal_call: bool = False
    ) -> requests.Response:
        """
        发送请求到 Gemini API，支持自动重试
        
        Args:
            is_internal_call: 是否是内部调用（工具调用后的后续请求）
        """
        global CONFIG
        max_retries = CONFIG.get('api_retry_count', 2)
        retry_delay = CONFIG.get('api_retry_delay', 5)
        
        response = self._make_gemini_request(
            model=model,
            contents=contents,
            tools=tools,
            generation_config=generation_config,
            system_instruction=system_instruction,
            stream=stream
        )
        
        # 如果是内部调用且请求失败，进行重试
        if is_internal_call and response.status_code != 200:
            for retry in range(max_retries):
                print(f"[重试] API 请求失败 (状态码: {response.status_code})，{retry_delay}秒后重试 ({retry + 1}/{max_retries})...")
                time.sleep(retry_delay)
                
                response = self._make_gemini_request(
                    model=model,
                    contents=contents,
                    tools=tools,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                    stream=stream
                )
                
                if response.status_code == 200:
                    print(f"[重试] 第 {retry + 1} 次重试成功")
                    break
                else:
                    print(f"[重试] 第 {retry + 1} 次重试仍然失败 (状态码: {response.status_code})")
        
        return response
    
    def _should_return_signature(self, request_contents: List[Dict[str, Any]], part_index: int, part: Dict[str, Any]) -> bool:
        """
        判断是否应该在响应中返回思考签名
        检查用户是否在请求中回传了之前回合的签名
        """
        # 检查请求内容中是否有 thoughtSignature
        for content in request_contents:
            if content.get('role') == 'model':
                parts = content.get('parts', [])
                for p in parts:
                    if isinstance(p, dict) and 'thoughtSignature' in p:
                        return True
        return True  # 默认返回签名
    
    def _build_final_response_with_accumulated_thoughts(
        self,
        last_result: Dict[str, Any],
        accumulated_thought_texts: List[str],
        accumulated_tool_texts: List[str]
    ) -> Dict[str, Any]:
        """
        构建包含所有累积思考内容的最终响应
        
        Args:
            last_result: 最后一次 API 返回的结果
            accumulated_thought_texts: 累积的思考文本列表
            accumulated_tool_texts: 累积的工具调用占位符文本列表
        
        Returns:
            包含所有累积思考的响应
        """
        # 构建新的 parts 列表
        final_parts = []
        
        # 1. 拼接所有思考文本和工具调用占位符为一个字符串
        thought_sections = []
        
        # 添加累积的思考文本
        for text in accumulated_thought_texts:
            if text.strip():
                thought_sections.append(text)
        
        # 添加工具调用占位符（多个工具用单换行分隔）
        if accumulated_tool_texts:
            tools_text = "\n".join(accumulated_tool_texts)
            thought_sections.append(tools_text)
        
        # 所有思考部分用两个换行符连接
        if thought_sections:
            combined_thought_text = "\n\n".join(thought_sections)
            final_parts.append({"text": combined_thought_text, "thought": True})
        
        # 2. 从最后一次结果中提取非思考的 parts
        candidates = last_result.get('candidates', [])
        if candidates:
            candidate = candidates[0]
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            for part in parts:
                if isinstance(part, dict):
                    # 跳过已经收集的思考 parts 和 functionCall
                    if part.get('thought'):
                        continue
                    if 'functionCall' in part:
                        continue
                    # 添加非思考内容（如最终的文本回复）
                    final_parts.append(copy.deepcopy(part))
        
        # 构建最终响应
        final_result = {
            "candidates": [{
                "content": {
                    "parts": final_parts,
                    "role": "model"
                },
                "index": 0
            }]
        }
        
        # 复制候选的 finishReason
        if candidates:
            final_result["candidates"][0]["finishReason"] = candidates[0].get('finishReason', 'STOP')
        
        # 复制元数据字段
        if 'usageMetadata' in last_result:
            final_result['usageMetadata'] = last_result['usageMetadata']
        if 'modelVersion' in last_result:
            final_result['modelVersion'] = last_result['modelVersion']
        if 'responseId' in last_result:
            final_result['responseId'] = last_result['responseId']
        
        return final_result
    
    def process_request_stream(
        self,
        contents: List[Dict[str, Any]],
        model: str = "gemini-2.5-flash",
        tools: Optional[List[Dict[str, Any]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        execute_mcp_tools: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        处理流式请求
        """
        # 合并 MCP 工具
        combined_tools = list(tools) if tools else []
        if self.mcp_manager:
            mcp_tools = self._get_mcp_tools_as_gemini_format()
            combined_tools.extend(mcp_tools)
        
        if not combined_tools:
            combined_tools = None
        
        contents_copy = [c.copy() if isinstance(c, dict) else c for c in contents]
        iteration = 0
        max_iterations = CONFIG.get('max_iterations', 100)
        
        # 记录用户原始请求中 contents 的长度
        original_contents_length = len(contents_copy)
        
        while iteration < max_iterations:
            # 替换代理服务器添加的旧工具结果
            self._replace_old_tool_results(contents_copy, original_contents_length)
            
            # 发送流式请求（内部调用时启用重试）
            try:
                response = self._make_gemini_request_with_retry(
                    model=model,
                    contents=contents_copy,
                    tools=combined_tools,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                    stream=True,
                    is_internal_call=(iteration > 0)
                )
            except requests.exceptions.Timeout as e:
                # 超时错误
                error_data = {"error": {"code": 504, "message": f"Request timeout: {str(e)}", "status": "DEADLINE_EXCEEDED"}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            except requests.exceptions.RequestException as e:
                # 其他请求错误
                error_data = {"error": {"code": 503, "message": f"Request failed: {str(e)}", "status": "UNAVAILABLE"}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            if response.status_code != 200:
                # 传递原始的错误响应
                try:
                    error_data = response.json()
                except:
                    error_data = {"error": {"message": response.text}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 用于累积完整的响应内容
            accumulated_content_parts = []
            function_calls_parts = []
            has_function_call = False
            last_chunk_data = None
            received_any_content = False
            stream_error = None
            
            # 处理流式响应
            # 使用 iter_lines 并手动 UTF-8 解码以避免编码问题
            for line in response.iter_lines(decode_unicode=False):
                if line:
                    # 手动使用 UTF-8 解码
                    try:
                        line_str = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        # 如果解码失败，跳过这一行
                        continue
                    
                    if not line_str:
                        continue
                    
                    # 移除可能的 "data: " 前缀
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                    
                    if line_str == '[DONE]':
                        continue
                    
                    try:
                        data = json.loads(line_str)
                        last_chunk_data = data
                        
                        # 检查是否是错误响应
                        if 'error' in data:
                            stream_error = data
                            # 如果是内部调用且还没有发送任何内容，可以尝试重试
                            if iteration > 0 and not received_any_content:
                                print(f"[流错误] 检测到流内错误: {data.get('error', {}).get('message', 'Unknown error')}")
                                # 跳出循环，尝试重试
                                break
                            else:
                                # 非内部调用或已发送内容，直接返回错误
                                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                        
                        received_any_content = True
                        
                        # 收集内容用于后续处理
                        candidates = data.get('candidates', [])
                        for candidate in candidates:
                            content = candidate.get('content', {})
                            parts = content.get('parts', [])
                            
                            for part in parts:
                                accumulated_content_parts.append(part)
                                
                                if 'functionCall' in part:
                                    function_calls_parts.append(part)
                                    has_function_call = True
                        
                        # 如果有 functionCall，不转发原始块
                        # 而是等待收集完成后发送"调用工具"占位符
                        if not has_function_call:
                            # 没有 functionCall，直接转发（保持所有字段）
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                        
                    except json.JSONDecodeError:
                        # 如果不是有效的 JSON，跳过
                        continue
            
            # 如果检测到流内错误且是内部调用，尝试重试
            if stream_error and iteration > 0 and not received_any_content:
                max_retries = CONFIG.get('api_retry_count', 2)
                retry_delay = CONFIG.get('api_retry_delay', 5)
                
                retry_success = False
                for retry in range(max_retries):
                    print(f"[重试] 流请求失败，{retry_delay}秒后重试 ({retry + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    
                    # 重新发送请求
                    response = self._make_gemini_request(
                        model=model,
                        contents=contents_copy,
                        tools=combined_tools,
                        generation_config=generation_config,
                        system_instruction=system_instruction,
                        stream=True
                    )
                    
                    if response.status_code != 200:
                        print(f"[重试] 第 {retry + 1} 次重试失败 (状态码: {response.status_code})")
                        continue
                    
                    # 检查流的第一块是否是错误
                    first_line_error = False
                    for line in response.iter_lines(decode_unicode=False):
                        if line:
                            try:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    line_str = line_str[6:]
                                if line_str and line_str != '[DONE]':
                                    first_data = json.loads(line_str)
                                    if 'error' in first_data:
                                        print(f"[重试] 第 {retry + 1} 次重试仍然失败 (流内错误)")
                                        first_line_error = True
                                        stream_error = first_data
                                        break
                                    else:
                                        # 成功！重新处理整个流
                                        print(f"[重试] 第 {retry + 1} 次重试成功")
                                        retry_success = True
                                        # 发送第一块数据
                                        if 'candidates' in first_data:
                                            for candidate in first_data.get('candidates', []):
                                                content = candidate.get('content', {})
                                                for part in content.get('parts', []):
                                                    accumulated_content_parts.append(part)
                                                    if 'functionCall' in part:
                                                        function_calls_parts.append(part)
                                                        has_function_call = True
                                            if not has_function_call:
                                                yield f"data: {json.dumps(first_data, ensure_ascii=False)}\n\n"
                                        break
                            except:
                                continue
                        break
                    
                    if retry_success:
                        # 继续处理剩余的流
                        for line in response.iter_lines(decode_unicode=False):
                            if line:
                                try:
                                    line_str = line.decode('utf-8').strip()
                                    if line_str.startswith('data: '):
                                        line_str = line_str[6:]
                                    if line_str == '[DONE]':
                                        continue
                                    if line_str:
                                        data = json.loads(line_str)
                                        if 'error' in data:
                                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                                            yield "data: [DONE]\n\n"
                                            return
                                        
                                        for candidate in data.get('candidates', []):
                                            content = candidate.get('content', {})
                                            for part in content.get('parts', []):
                                                accumulated_content_parts.append(part)
                                                if 'functionCall' in part:
                                                    function_calls_parts.append(part)
                                                    has_function_call = True
                                        
                                        if not has_function_call:
                                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                                except:
                                    continue
                        break
                    
                    if first_line_error:
                        continue
                
                if not retry_success:
                    # 重试全部失败，返回最后的错误
                    yield f"data: {json.dumps(stream_error, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            
            # 如果没有收到任何内容且没有函数调用，可能是出错了
            if not received_any_content and not function_calls_parts:
                if stream_error:
                    yield f"data: {json.dumps(stream_error, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # 如果没有函数调用，正常结束
            if not function_calls_parts:
                yield "data: [DONE]\n\n"
                return
            
            # 有函数调用，发送"调用工具"占位符给客户端
            if function_calls_parts:
                tool_call_texts = []
                for fc_part in function_calls_parts:
                    fc = fc_part.get('functionCall', {})
                    name = fc.get('name', '')
                    args = json.dumps(fc.get('args', {}), ensure_ascii=False)
                    tool_call_texts.append(self._format_tool_call_text(name, args))
                
                tools_text = "\n".join(tool_call_texts)
                # 构建与标准响应格式一致的占位块
                chunk_data = {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": tools_text, "thought": True}],
                            "role": "model"
                        },
                        "index": 0
                    }]
                }
                # 如果有 last_chunk_data，复制其中的元数据字段
                if last_chunk_data:
                    if 'modelVersion' in last_chunk_data:
                        chunk_data['modelVersion'] = last_chunk_data['modelVersion']
                    if 'responseId' in last_chunk_data:
                        chunk_data['responseId'] = last_chunk_data['responseId']
                
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            # 处理 MCP 工具调用
            if function_calls_parts and execute_mcp_tools and self.mcp_manager:
                mcp_tool_calls = []
                non_mcp_tool_calls = []
                
                for fc_part in function_calls_parts:
                    fc = fc_part.get('functionCall', {})
                    tool_name = fc.get('name', '')
                    if self._is_mcp_tool(tool_name):
                        mcp_tool_calls.append(fc_part)
                    else:
                        non_mcp_tool_calls.append(fc_part)
                
                # 执行 MCP 工具调用
                if mcp_tool_calls:
                    # 构建 model 响应消息（保留所有 parts 包括 functionCall 和思考签名）
                    model_content = {
                        "role": "model",
                        "parts": accumulated_content_parts
                    }
                    contents_copy.append(model_content)
                    
                    # 执行工具并添加响应
                    function_responses = []
                    for fc_part in mcp_tool_calls:
                        fc = fc_part.get('functionCall', {})
                        name = fc.get('name', '')
                        args = fc.get('args', {})
                        
                        result = self._execute_mcp_tool(name, args)
                        
                        function_responses.append({
                            "functionResponse": {
                                "name": name,
                                "response": {"result": result or "工具执行失败"}
                            }
                        })
                    
                    # 添加函数响应
                    contents_copy.append({
                        "role": "user",
                        "parts": function_responses
                    })
                    
                    iteration += 1
                    continue
                
                # 如果只有非 MCP 工具调用，已经转发完毕，直接结束
                if non_mcp_tool_calls:
                    yield "data: [DONE]\n\n"
                    return
            
            # 有函数调用但不自动执行，已经转发完毕，直接结束
            yield "data: [DONE]\n\n"
            return
        
        # 达到最大迭代次数
        error_data = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "[达到最大工具调用迭代次数]"}]
                },
                "finishReason": "MAX_TOKENS"
            }]
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    def process_request(
        self,
        contents: List[Dict[str, Any]],
        model: str = "gemini-2.5-flash",
        tools: Optional[List[Dict[str, Any]]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        stream: bool = False,
        execute_mcp_tools: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理非流式请求
        """
        # 合并 MCP 工具
        combined_tools = list(tools) if tools else []
        if self.mcp_manager:
            mcp_tools = self._get_mcp_tools_as_gemini_format()
            combined_tools.extend(mcp_tools)
        
        if not combined_tools:
            combined_tools = None
        
        contents_copy = [c.copy() if isinstance(c, dict) else c for c in contents]
        iteration = 0
        max_iterations = CONFIG.get('max_iterations', 100)
        
        # 记录用户原始请求中 contents 的长度
        original_contents_length = len(contents_copy)
        
        # 用于累积所有迭代的思考内容和工具调用占位符
        accumulated_thought_texts = []  # 累积的思考文本
        accumulated_tool_texts = []  # 累积的工具调用占位符文本
        last_result = None  # 保存最后一次的结果用于复制元数据
        
        while iteration < max_iterations:
            # 替换代理服务器添加的旧工具结果
            self._replace_old_tool_results(contents_copy, original_contents_length)
            
            # 发送请求（内部调用时启用重试）
            try:
                response = self._make_gemini_request_with_retry(
                    model=model,
                    contents=contents_copy,
                    tools=combined_tools,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                    stream=False,
                    is_internal_call=(iteration > 0)
                )
            except requests.exceptions.Timeout as e:
                # 超时错误
                return {"_status_code": 504, "_response": {"error": {"code": 504, "message": f"Request timeout: {str(e)}", "status": "DEADLINE_EXCEEDED"}}}
            except requests.exceptions.RequestException as e:
                # 其他请求错误
                return {"_status_code": 503, "_response": {"error": {"code": 503, "message": f"Request failed: {str(e)}", "status": "UNAVAILABLE"}}}
            
            if response.status_code != 200:
                # 传递原始的错误响应和状态码
                try:
                    error_response = response.json()
                except:
                    error_response = {"error": {"message": response.text}}
                return {"_status_code": response.status_code, "_response": error_response}
            
            result = response.json()
            last_result = result  # 保存最后一次结果
            
            # 提取候选响应
            candidates = result.get('candidates', [])
            if not candidates:
                # 如果有累积的思考内容，需要添加到结果中
                if accumulated_thought_texts or accumulated_tool_texts:
                    return self._build_final_response_with_accumulated_thoughts(
                        result, accumulated_thought_texts, accumulated_tool_texts
                    )
                return result
            
            candidate = candidates[0]
            content = candidate.get('content', {})
            finish_reason = candidate.get('finishReason', '')
            
            # 收集本次响应中的思考内容（提取文本）
            parts = content.get('parts', [])
            for part in parts:
                if isinstance(part, dict) and part.get('thought') and 'text' in part:
                    accumulated_thought_texts.append(part['text'])
            
            # 检查是否有函数调用
            function_calls = self._get_function_calls(content)
            
            # 如果没有函数调用，构建包含所有累积思考的最终响应
            if not function_calls:
                if accumulated_thought_texts or accumulated_tool_texts:
                    return self._build_final_response_with_accumulated_thoughts(
                        result, accumulated_thought_texts, accumulated_tool_texts
                    )
                return result
            
            # 有函数调用，记录工具调用占位符
            if function_calls:
                # 构建"调用工具"占位符文本
                for fc in function_calls:
                    name = fc.get('name', '')
                    args = json.dumps(fc.get('args', {}), ensure_ascii=False)
                    accumulated_tool_texts.append(self._format_tool_call_text(name, args))
                
                # 处理 MCP 工具调用
                if execute_mcp_tools and self.mcp_manager:
                    mcp_tool_calls = []
                    non_mcp_tool_calls = []
                    
                    for fc in function_calls:
                        tool_name = fc.get('name', '')
                        if self._is_mcp_tool(tool_name):
                            mcp_tool_calls.append(fc)
                        else:
                            non_mcp_tool_calls.append(fc)
                    
                    # 执行 MCP 工具调用
                    if mcp_tool_calls:
                        # 构建 model 响应消息（保留完整的 parts 包括 functionCall 和思考签名）
                        # 使用深拷贝确保所有字段（包括 thoughtSignature）都被保留
                        model_content = {
                            "role": content.get("role", "model"),
                            "parts": copy.deepcopy(content.get("parts", []))
                        }
                        contents_copy.append(model_content)
                        
                        # 执行工具并添加响应
                        function_responses = []
                        for fc in mcp_tool_calls:
                            name = fc.get('name', '')
                            args = fc.get('args', {})
                            
                            result_data = self._execute_mcp_tool(name, args)
                            
                            function_responses.append({
                                "functionResponse": {
                                    "name": name,
                                    "response": {"result": result_data or "工具执行失败"}
                                }
                            })
                        
                        # 添加函数响应
                        contents_copy.append({
                            "role": "user",
                            "parts": function_responses
                        })
                        
                        iteration += 1
                        continue
                
                # 如果有非 MCP 工具调用或不自动执行，返回包含累积思考的响应
                return self._build_final_response_with_accumulated_thoughts(
                    result, accumulated_thought_texts, accumulated_tool_texts
                )
        
        # 达到最大迭代次数
        return {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "[达到最大工具调用迭代次数]"}]
                },
                "finishReason": "MAX_TOKENS"
            }]
        }


@app.route('/v1beta/models/<model>:generateContent', methods=['POST'])
def generate_content(model: str):
    """处理 generateContent 请求"""
    global mcp_manager, CONFIG
    
    try:
        data = request.get_json()
        
        # 获取 API Key
        api_key = request.headers.get('x-goog-api-key', '')
        if not api_key:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                api_key = auth_header[7:]
        
        is_valid, actual_api_key, error_msg = validate_access_key(api_key)
        if not is_valid:
            return jsonify({"error": {"message": error_msg, "type": "auth_error"}}), 401
        
        # 提取参数
        contents = data.get('contents', [])
        tools = data.get('tools')
        generation_config = data.get('generationConfig')
        execute_mcp_tools = data.get('execute_mcp_tools', CONFIG.get('auto_execute_mcp_tools', True))
        
        # 处理系统指令：如果用户提供了 systemInstruction，将配置的系统提示词添加到前面
        user_system_instruction = data.get('systemInstruction')
        config_prompt = CONFIG.get('system_prompt', '')
        system_instruction = None
        
        if config_prompt:
            # 配置中有系统提示词
            if user_system_instruction:
                # 用户也提供了 systemInstruction，提取其文本内容
                if isinstance(user_system_instruction, dict):
                    user_text = ""
                    parts = user_system_instruction.get('parts', [])
                    for part in parts:
                        if isinstance(part, dict) and 'text' in part:
                            user_text += part['text']
                    # 将配置的提示词添加到前面
                    system_instruction = config_prompt + "\n\n" + user_text if user_text else config_prompt
                elif isinstance(user_system_instruction, str):
                    system_instruction = config_prompt + "\n\n" + user_system_instruction
                else:
                    system_instruction = config_prompt
            else:
                # 用户没有提供，只使用配置的
                system_instruction = config_prompt
        elif user_system_instruction:
            # 配置中没有，但用户提供了，直接使用用户的
            if isinstance(user_system_instruction, dict):
                user_text = ""
                parts = user_system_instruction.get('parts', [])
                for part in parts:
                    if isinstance(part, dict) and 'text' in part:
                        user_text += part['text']
                system_instruction = user_text
            elif isinstance(user_system_instruction, str):
                system_instruction = user_system_instruction
        
        # 创建代理处理器
        proxy = GeminiProxy(actual_api_key, mcp_manager)
        
        # 处理请求
        response = proxy.process_request(
            contents=contents,
            model=model,
            tools=tools,
            generation_config=generation_config,
            system_instruction=system_instruction,
            stream=False,
            execute_mcp_tools=execute_mcp_tools
        )
        
        # 检查是否是错误响应（带有原始状态码）
        if isinstance(response, dict) and '_status_code' in response:
            status_code = response['_status_code']
            error_response = response['_response']
            return jsonify(error_response), status_code
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }), 500


@app.route('/v1beta/models/<model>:streamGenerateContent', methods=['POST'])
def stream_generate_content(model: str):
    """处理 streamGenerateContent 请求"""
    global mcp_manager, CONFIG
    
    try:
        data = request.get_json()
        
        # 获取 API Key
        api_key = request.headers.get('x-goog-api-key', '')
        if not api_key:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                api_key = auth_header[7:]
        
        is_valid, actual_api_key, error_msg = validate_access_key(api_key)
        if not is_valid:
            return jsonify({"error": {"message": error_msg, "type": "auth_error"}}), 401
        
        # 提取参数
        contents = data.get('contents', [])
        tools = data.get('tools')
        generation_config = data.get('generationConfig')
        execute_mcp_tools = data.get('execute_mcp_tools', CONFIG.get('auto_execute_mcp_tools', True))
        
        # 处理系统指令：如果用户提供了 systemInstruction，将配置的系统提示词添加到前面
        user_system_instruction = data.get('systemInstruction')
        config_prompt = CONFIG.get('system_prompt', '')
        system_instruction = None
        
        if config_prompt:
            # 配置中有系统提示词
            if user_system_instruction:
                # 用户也提供了 systemInstruction，提取其文本内容
                if isinstance(user_system_instruction, dict):
                    user_text = ""
                    parts = user_system_instruction.get('parts', [])
                    for part in parts:
                        if isinstance(part, dict) and 'text' in part:
                            user_text += part['text']
                    # 将配置的提示词添加到前面
                    system_instruction = config_prompt + "\n\n" + user_text if user_text else config_prompt
                elif isinstance(user_system_instruction, str):
                    system_instruction = config_prompt + "\n\n" + user_system_instruction
                else:
                    system_instruction = config_prompt
            else:
                # 用户没有提供，只使用配置的
                system_instruction = config_prompt
        elif user_system_instruction:
            # 配置中没有，但用户提供了，直接使用用户的
            if isinstance(user_system_instruction, dict):
                user_text = ""
                parts = user_system_instruction.get('parts', [])
                for part in parts:
                    if isinstance(part, dict) and 'text' in part:
                        user_text += part['text']
                system_instruction = user_text
            elif isinstance(user_system_instruction, str):
                system_instruction = user_system_instruction
        
        # 创建代理处理器
        proxy = GeminiProxy(actual_api_key, mcp_manager)
        
        def generate():
            try:
                for chunk in proxy.process_request_stream(
                    contents=contents,
                    model=model,
                    tools=tools,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                    execute_mcp_tools=execute_mcp_tools
                ):
                    yield chunk
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
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
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }), 500


@app.route('/v1beta/models', methods=['GET'])
def list_models():
    """列出可用模型（自动获取所有页面）"""
    global CONFIG
    
    api_key = request.headers.get('x-goog-api-key', '')
    if not api_key:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            api_key = auth_header[7:]
    
    is_valid, actual_api_key, error_msg = validate_access_key(api_key)
    if not is_valid:
        return jsonify({"error": {"message": error_msg}}), 401
    
    try:
        # 使用配置的模型列表 URL
        models_url = CONFIG.get("gemini_models_url", "https://generativelanguage.googleapis.com/v1beta/models")
        
        headers = {
            "x-goog-api-key": actual_api_key
        }
        
        # 获取所有模型（自动处理分页）
        all_models = []
        page_token = None
        
        while True:
            # 构建请求参数，每页获取 1000 个（足够大）
            params = {"pageSize": 10000}
            if page_token:
                params["pageToken"] = page_token
            
            response = requests.get(models_url, headers=headers, params=params, timeout=30)
            
            if response.status_code != 200:
                return jsonify({
                    "error": {
                        "message": f"Failed to fetch models: {response.status_code}",
                        "details": response.text
                    }
                }), response.status_code
            
            data = response.json()
            models = data.get("models", [])
            all_models.extend(models)
            
            # 检查是否有下一页
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        
        # 返回合并后的结果（不包含 nextPageToken）
        return jsonify({"models": all_models})
    
    except Exception as e:
        return jsonify({
            "error": {
                "message": f"Failed to fetch models: {str(e)}"
            }
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    global mcp_manager
    
    status = {"status": "ok", "service": "gemini-proxy"}
    
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
    """获取 MCP 工具列表（Gemini 格式）"""
    global mcp_manager
    
    if not MCP_AVAILABLE or not mcp_manager:
        return jsonify({"tools": []})
    
    proxy = GeminiProxy("", mcp_manager)
    tools = proxy._get_mcp_tools_as_gemini_format()
    
    return jsonify({
        "tools": tools
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
    
    parser = argparse.ArgumentParser(description='Gemini API 代理服务器')
    parser.add_argument('--config', type=str, default='gemini_config.jsonc', help='配置文件路径')
    parser.add_argument('--host', type=str, help='监听地址（覆盖配置文件）')
    parser.add_argument('--port', type=int, help='监听端口（覆盖配置文件）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-mcp', action='store_true', help='禁用 MCP 功能')
    
    args = parser.parse_args()
    
    # 加载配置文件
    CONFIG = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    host = args.host or CONFIG.get('host', '127.0.0.1')
    port = args.port or CONFIG.get('port', 8003)
    debug = args.debug or CONFIG.get('debug', False)
    mcp_enabled = CONFIG.get('mcp_enabled', True) and not args.no_mcp
    
    print("=" * 60)
    print("Gemini API 代理服务器")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"监听地址: http://{host}:{port}")
    print(f"API 端点: http://{host}:{port}/v1beta/models/<model>:generateContent")
    print(f"流式端点: http://{host}:{port}/v1beta/models/<model>:streamGenerateContent")
    print(f"后端 API: {CONFIG.get('gemini_api_url', 'N/A')}")
    
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
    
    # 初始化 MCP
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