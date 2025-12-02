"""
MCP 服务管理模块
每个 MCP 服务放在独立的子目录中
启用的服务列表存储在 enabled.txt 文件中
"""

import os
import json
from typing import Dict, List, Any, Optional, Set

MCP_SERVERS_DIR = os.path.dirname(os.path.abspath(__file__))
ENABLED_FILE = os.path.join(MCP_SERVERS_DIR, "enabled.txt")


def get_enabled_servers() -> Set[str]:
    """
    读取启用的服务列表
    
    Returns:
        启用的服务名称集合
    """
    enabled = set()
    
    if not os.path.exists(ENABLED_FILE):
        return enabled
    
    try:
        with open(ENABLED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                enabled.add(line)
    except Exception as e:
        print(f"读取 enabled.txt 失败: {e}")
    
    return enabled


def save_enabled_servers(enabled: Set[str]):
    """
    保存启用的服务列表
    
    Args:
        enabled: 启用的服务名称集合
    """
    try:
        with open(ENABLED_FILE, 'w', encoding='utf-8') as f:
            f.write("# MCP 服务启用列表\n")
            f.write("# 每行一个服务文件夹名，以 # 开头的行为注释\n")
            f.write("# 不在列表中的服务将不会被启用\n\n")
            for name in sorted(enabled):
                f.write(f"{name}\n")
    except Exception as e:
        print(f"保存 enabled.txt 失败: {e}")


def get_available_servers() -> Dict[str, Dict[str, Any]]:
    """
    扫描所有可用的 MCP 服务
    
    Returns:
        服务名称 -> 服务配置
    """
    servers = {}
    enabled = get_enabled_servers()
    
    for name in os.listdir(MCP_SERVERS_DIR):
        server_dir = os.path.join(MCP_SERVERS_DIR, name)
        
        # 跳过非目录和特殊目录
        if not os.path.isdir(server_dir) or name.startswith('_') or name.startswith('.'):
            continue
        
        # 读取服务配置
        config_file = os.path.join(server_dir, 'config.json')
        if not os.path.exists(config_file):
            continue
        
        config = {}
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            continue
        
        server_type = config.get("type", "stdio")
        
        # 检查服务主文件（仅 stdio 类型需要）
        server_file = os.path.join(server_dir, 'server.py')
        if server_type == "stdio" and not os.path.exists(server_file):
            continue
        
        # 检查是否启用（在 enabled.txt 中）
        is_enabled = name in enabled
        
        servers[name] = {
            "path": server_dir,
            "server_file": server_file if server_type == "stdio" else None,
            "enabled": is_enabled,
            "config": config,
            "description": config.get("description", ""),
            "type": server_type
        }
    
    return servers


def enable_server(name: str) -> bool:
    """启用指定服务"""
    server_dir = os.path.join(MCP_SERVERS_DIR, name)
    if not os.path.isdir(server_dir):
        return False
    
    enabled = get_enabled_servers()
    enabled.add(name)
    save_enabled_servers(enabled)
    return True


def disable_server(name: str) -> bool:
    """禁用指定服务"""
    enabled = get_enabled_servers()
    if name in enabled:
        enabled.remove(name)
        save_enabled_servers(enabled)
    return True


def get_server_config(name: str) -> Optional[Dict[str, Any]]:
    """获取指定服务的配置"""
    servers = get_available_servers()
    return servers.get(name)


def generate_mcp_config() -> Dict[str, Any]:
    """
    根据目录结构生成 MCP 配置
    
    Returns:
        MCP 配置字典
    """
    servers = get_available_servers()
    mcp_servers = {}
    
    for name, info in servers.items():
        if not info["enabled"]:
            continue
        
        config = info.get("config", {})
        server_type = config.get("type", "stdio")
        
        server_config = {
            "type": server_type,
            "description": config.get("description", ""),
            "enabled": True
        }
        
        if server_type == "stdio":
            server_config["command"] = config.get("command", "python")
            server_config["args"] = [info["server_file"]] + config.get("args", [])
            if config.get("env"):
                server_config["env"] = config["env"]
        elif server_type in ("streamableHttp", "sse"):
            if config.get("url"):
                server_config["url"] = config["url"]
            if config.get("headers"):
                server_config["headers"] = config["headers"]
        
        mcp_servers[name] = server_config
    
    return {
        "mcpServers": mcp_servers,
        "settings": {
            "autoStartServers": True,
            "connectionTimeout": 30,
            "retryAttempts": 3
        }
    }


if __name__ == "__main__":
    # 测试代码
    print("可用的 MCP 服务:")
    servers = get_available_servers()
    for name, info in servers.items():
        status = "✓ 已启用" if info["enabled"] else "✗ 已禁用"
        print(f"  [{status}] {name}: {info['description']}")
    
    print("\n启用的服务:")
    enabled = get_enabled_servers()
    for name in enabled:
        print(f"  - {name}")
    
    print("\n生成的 MCP 配置:")
    config = generate_mcp_config()
    print(json.dumps(config, ensure_ascii=False, indent=2))