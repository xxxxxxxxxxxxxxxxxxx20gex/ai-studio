# MCP 使用方式速查表

> 💡 **使用指南**：本文件是 MCP 的「实战速查 + 代码模版」，默认你已经对 MCP 的概念有基本了解。  
> 理论部分请参考：`MCP基础知识.md`；与 Agent / Agno 的整体关系可以看：`MCP_Agent_Agno对比.md`。

## 📋 三种使用方式对比

| 特性 | Stdio（本地进程） | HTTP/SSE（远程） | 自定义传输 |
|------|------------------|----------------|-----------|
| **通信方式** | 标准输入输出 | HTTP协议 | 自定义实现 |
| **适用场景** | 本地开发、个人工具 | 企业应用、云服务 | 特殊需求 |
| **难度** | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 困难 |
| **远程访问** | ❌ 不支持 | ✅ 支持 | 看实现 |
| **多客户端** | ❌ 单一客户端 | ✅ 多客户端 | 看实现 |
| **安全性** | ✅ 高（本地） | ⚠️ 需配置 | 看实现 |
| **性能** | ✅ 快 | ⚠️ 网络延迟 | 看实现 |
| **配置复杂度** | 简单 | 中等 | 复杂 |

---

## 🎯 选择建议

```
个人使用、本地开发
    ↓
  Stdio 方式 ✅

多人共享、远程访问
    ↓
  HTTP/SSE 方式 ✅

特殊网络环境、定制需求
    ↓
  自定义传输 ✅
```

---

## 💻 快速上手代码

### 1️⃣ Stdio 方式（推荐初学者）

**Server 端**：
```python
# weather_server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("weather")

@server.tool()
def get_weather(city: str) -> str:
    return f"{city}: 晴天, 25℃"

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    asyncio.run(main())
```

**Client 配置**（Claude Desktop）：
```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_server.py"]
    }
  }
}
```

**运行**：
```bash
# 直接运行（会等待 stdio 输入）
python weather_server.py

# 或者在 Claude Desktop 中自动启动
```

---

### 2️⃣ HTTP/SSE 方式（适合团队）

**Server 端**：
```python
# weather_http_server.py
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

server = Server("weather")

@server.tool()
def get_weather(city: str) -> str:
    return f"{city}: 晴天, 25℃"

async def handle_sse(request):
    async with SseServerTransport("/messages") as transport:
        await server.run(transport.read_stream, transport.write_stream)

app = Starlette(routes=[Route("/sse", endpoint=handle_sse)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Client 配置**：
```json
{
  "mcpServers": {
    "weather": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**运行**：
```bash
# 启动服务器
python weather_http_server.py

# 服务器会监听 8000 端口
# 访问 http://localhost:8000/sse
```

---

### 3️⃣ 在 Python 代码中调用 MCP（作为 Client）

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp():
    # 配置 Server 参数
    params = StdioServerParameters(
        command="python",
        args=["weather_server.py"]
    )
    
    # 连接 Server
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 查看可用工具
            tools = await session.list_tools()
            print("可用工具:", [t.name for t in tools.tools])
            
            # 调用工具
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "北京"}
            )
            print("结果:", result.content)

# 运行
asyncio.run(use_mcp())
```

---

## 🏢 在不同应用中使用

### Claude Desktop
```json
// 配置文件：%APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "API_KEY": "your-key"
      }
    }
  }
}
```

### VS Code / Cursor
```json
// .vscode/settings.json
{
  "mcp.servers": {
    "my-server": {
      "command": "python",
      "args": ["server.py"]
    }
  }
}
```

### 自定义 Agent（如 Agno）
```python
from agno.agent import Agent
from mcp.client import ClientSession

# 1. 连接 MCP Server
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as mcp:
        await mcp.initialize()
        
        # 2. 获取工具
        tools = await mcp.list_tools()
        
        # 3. 创建 Agent
        agent = Agent(
            model=YourModel(),
            tools=convert_mcp_tools(tools),  # 转换工具格式
        )
        
        agent.cli_app()
```

---

## 📁 典型项目结构

### Stdio 方式
```
my_mcp_project/
├── server.py           # MCP Server
├── config.json         # Claude Desktop 配置
└── requirements.txt
```

### HTTP 方式
```
my_mcp_project/
├── server.py           # MCP Server (HTTP)
├── Dockerfile          # 容器部署
├── requirements.txt
└── .env               # 环境变量
```

---

## 🚀 部署方式

### 本地部署（Stdio）
```bash
# 1. 安装依赖
pip install mcp

# 2. 编辑配置
code %APPDATA%\Claude\claude_desktop_config.json

# 3. 重启 Claude Desktop
```

### 云端部署（HTTP）
```bash
# 使用 Docker
docker build -t mcp-server .
docker run -p 8000:8000 mcp-server

# 或使用云平台（如 Railway, Render）
# 1. 推送代码到 Git
# 2. 连接仓库
# 3. 自动部署
```

---

## 🔧 调试技巧

### Stdio 方式调试
```python
# 添加日志
import logging
logging.basicConfig(
    level=logging.DEBUG,
    filename='mcp_server.log'  # 输出到文件，不影响 stdio
)

@server.tool()
def get_weather(city: str) -> str:
    logging.info(f"查询城市: {city}")
    return f"{city}: 晴天"
```

### HTTP 方式调试
```python
# 可以用浏览器测试
# 访问 http://localhost:8000/sse

# 或用 curl
curl http://localhost:8000/sse
```

---

## 📊 性能对比

| 指标 | Stdio | HTTP |
|------|-------|------|
| **启动时间** | < 100ms | ~500ms |
| **调用延迟** | ~10ms | ~50ms (局域网) |
| **内存占用** | ~50MB | ~100MB |
| **并发能力** | 1 客户端 | 多客户端 |

---

## 💡 常见问题

### Q: Stdio 方式如何传递环境变量？
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "API_KEY": "your-key",
        "DEBUG": "true"
      }
    }
  }
}
```

### Q: HTTP 方式如何处理认证？
```python
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware

# 添加认证中间件
app = Starlette(
    routes=[...],
    middleware=[
        Middleware(AuthenticationMiddleware, backend=...)
    ]
)
```

### Q: 如何在同一个应用中使用多个 MCP Server？
```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_server.py"]
    },
    "database": {
      "command": "python",
      "args": ["db_server.py"]
    },
    "api": {
      "url": "http://api.example.com/sse"
    }
  }
}
```

---

## 🎓 学习路线

```
第1步: Stdio 方式（1天）
  ├─ 创建简单 Server
  ├─ 在 Claude Desktop 中测试
  └─ 添加多个工具

第2步: 理解协议（1天）
  ├─ Resources vs Tools vs Prompts
  ├─ 查看通信日志
  └─ 理解 JSON-RPC

第3步: HTTP 方式（2天）
  ├─ 创建 HTTP Server
  ├─ 本地测试
  └─ 部署到云端

第4步: 实际项目（按需）
  ├─ 集成数据库
  ├─ 调用第三方 API
  └─ 构建完整应用
```

---

*最后更新：2025年*

