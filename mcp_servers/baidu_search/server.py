"""
百度搜索 MCP 服务器
提供百度网页搜索功能，支持 MCP 协议
"""

import json
import sys
import os
import requests
from typing import Optional, List, Dict, Any


# 百度搜索 API 配置
BAIDU_SEARCH_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"


class BaiduSearchMCPServer:
    """百度搜索 MCP 服务器"""
    
    def __init__(self, api_key: str):
        """
        初始化服务器
        
        Args:
            api_key: 百度 AppBuilder API Key
        """
        self.api_key = api_key
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        search_recency_filter: Optional[str] = None,
        sites: Optional[List[str]] = None,
        block_websites: Optional[List[str]] = None,
        edition: str = "standard"
    ) -> Dict[str, Any]:
        """
        执行百度网页搜索
        
        Args:
            query: 搜索查询内容
            top_k: 返回结果数量 (最大50)
            search_recency_filter: 时间过滤 (week/month/semiyear/year)
            sites: 限定搜索的站点列表
            block_websites: 需要屏蔽的站点列表
            edition: 搜索版本 (standard/lite)
        
        Returns:
            搜索结果
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求体
        body = {
            "messages": [
                {
                    "content": query,
                    "role": "user"
                }
            ],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [
                {"type": "web", "top_k": min(top_k, 50)}
            ],
            "edition": edition
        }
        
        # 添加时间过滤
        if search_recency_filter and search_recency_filter in ["week", "month", "semiyear", "year"]:
            body["search_recency_filter"] = search_recency_filter
        
        # 添加站点过滤
        if sites:
            body["search_filter"] = {
                "match": {
                    "site": sites[:20]  # 最多20个站点
                }
            }
        
        # 添加屏蔽站点
        if block_websites:
            body["block_websites"] = block_websites
        
        try:
            response = requests.post(
                BAIDU_SEARCH_URL,
                headers=headers,
                json=body,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "references": []}
    
    def format_search_results(self, results: Dict[str, Any]) -> str:
        """
        格式化搜索结果为JSON格式
        
        Args:
            results: 搜索结果
        
        Returns:
            格式化后的JSON字符串
        """
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理 MCP 请求
        
        Args:
            request: MCP 请求对象
        
        Returns:
            MCP 响应对象
        """
        method = request.get("method", "")
        request_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "baidu-search-mcp-server",
                        "version": "1.0.0"
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "baidu_web_search",
                            "description": "使用百度搜索引擎搜索网页内容，返回完整的JSON格式搜索结果，包含标题、链接、详细内容等信息。适合搜索新闻、资讯、知识问答等内容。",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "搜索查询内容"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "description": "返回结果数量，默认10，最大50",
                                        "default": 10
                                    },
                                    "search_recency_filter": {
                                        "type": "string",
                                        "description": "时间过滤: week(最近7天), month(最近30天), semiyear(最近180天), year(最近365天)",
                                        "enum": ["week", "month", "semiyear", "year"]
                                    },
                                    "sites": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "限定搜索的站点列表，最多20个"
                                    },
                                    "block_websites": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "需要屏蔽的站点列表"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "baidu_web_search":
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 10)
                search_recency_filter = arguments.get("search_recency_filter")
                sites = arguments.get("sites")
                block_websites = arguments.get("block_websites")
                
                if not query:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Missing required parameter: query"
                        }
                    }
                
                results = self.search(
                    query=query,
                    top_k=top_k,
                    search_recency_filter=search_recency_filter,
                    sites=sites,
                    block_websites=block_websites
                )
                
                # 直接返回JSON格式结果
                formatted = self.format_search_results(results)
                
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": formatted
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        elif method == "notifications/initialized":
            # 这是一个通知，不需要响应
            return None
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }
    
    def run_stdio(self):
        """运行 STDIO 模式的 MCP 服务器"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    continue
                
                response = self.handle_request(request)
                
                if response is not None:
                    print(json.dumps(response), flush=True)
                    
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                print(json.dumps(error_response), flush=True)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='百度搜索 MCP 服务器')
    parser.add_argument('--api-key', type=str, help='百度 AppBuilder API Key')
    parser.add_argument('--mode', type=str, default='stdio', choices=['stdio', 'http'],
                        help='运行模式: stdio 或 http')
    parser.add_argument('--port', type=int, default=8080, help='HTTP 模式端口')
    
    args = parser.parse_args()
    
    # 获取 API Key（优先从环境变量获取）
    api_key = args.api_key or os.environ.get('BAIDU_APPBUILDER_API_KEY')
    
    if not api_key:
        print("错误: 请提供百度 AppBuilder API Key", file=sys.stderr)
        print("可以通过 --api-key 参数或 BAIDU_APPBUILDER_API_KEY 环境变量设置", file=sys.stderr)
        sys.exit(1)
    
    server = BaiduSearchMCPServer(api_key)
    
    if args.mode == 'stdio':
        server.run_stdio()
    else:
        # HTTP 模式（可以后续实现）
        print(f"HTTP 模式暂未实现", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()