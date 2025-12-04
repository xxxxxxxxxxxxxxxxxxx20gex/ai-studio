"""
MCP (Model Context Protocol) 简单示例
展示如何创建一个基础的 MCP Server
"""

# 注意：需要安装 mcp 包
# pip install mcp

from mcp.server import Server
from mcp.server.models import InitializationOptions
import asyncio


# ============================================================
# 创建 MCP Server
# ============================================================
server = Server("demo-server")


# ============================================================
# 1️⃣ 定义 Tools (工具) - 可执行操作
# ============================================================
@server.tool()
def calculate_sum(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b


@server.tool()
def get_weather(city: str) -> str:
    """获取城市天气（模拟）"""
    weather_data = {
        "北京": "晴天 25℃",
        "上海": "多云 22℃",
        "深圳": "小雨 28℃"
    }
    return weather_data.get(city, f"{city}天气未知")


# ============================================================
# 2️⃣ 定义 Resources (资源) - 只读数据
# ============================================================
@server.resource("demo://greeting")
def greeting_resource() -> str:
    """问候语资源"""
    return "欢迎使用 MCP Demo Server！"


@server.resource("demo://status")
def status_resource() -> str:
    """服务器状态"""
    return "运行中 ✅"


# ============================================================
# 3️⃣ 定义 Prompts (提示词模板) - 可重用模板
# ============================================================
@server.prompt()
def translate_prompt(text: str, target_lang: str = "英文") -> str:
    """翻译提示词模板"""
    return f"请将以下文本翻译成{target_lang}：\n{text}"


@server.prompt()
def code_review_prompt(code: str) -> str:
    """代码审查提示词模板"""
    return f"""请审查以下代码：

```python
{code}
```

请从以下方面评审：
1. 代码规范性
2. 潜在问题
3. 优化建议
"""


# ============================================================
# 4️⃣ 启动服务器
# ============================================================
async def main():
    """主函数"""
    print("🚀 MCP Server 启动中...")
    print(f"📦 已注册 {len(server.list_tools())} 个工具")
    print(f"📄 已注册 {len(server.list_resources())} 个资源")
    print(f"💬 已注册 {len(server.list_prompts())} 个提示词模板")
    print()
    
    # 展示已注册的功能
    print("=" * 60)
    print("已注册的工具:")
    for tool in server.list_tools():
        print(f"  • {tool.name}: {tool.description}")
    
    print("\n已注册的资源:")
    for resource in server.list_resources():
        print(f"  • {resource.uri}")
    
    print("\n已注册的提示词:")
    for prompt in server.list_prompts():
        print(f"  • {prompt.name}: {prompt.description}")
    print("=" * 60)
    
    # 运行服务器
    async with server.run():
        print("\n✅ MCP Server 正在运行...")
        print("按 Ctrl+C 停止\n")
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 MCP Server 已停止")

