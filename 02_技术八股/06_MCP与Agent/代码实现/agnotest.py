"""
Agno 框架学习示例 - 最简版本
展示核心功能：Agent、工具、交互
"""

import random
from agno.agent import Agent
from agno.models.deepseek import DeepSeek


def get_weather(city: str) -> str:
    """获取城市天气（模拟数据）"""
    conditions = ["晴天", "多云", "小雨", "阴天"]
    temp = random.randint(15, 30)
    condition = random.choice(conditions)
    return f"{city}：{condition}，{temp}℃"



agent = Agent(
    name="天气助手",
    model=DeepSeek(
        id="deepseek-chat",
        api_key="sk-c34d633720d04d94a73a3b74d817e52b",  # 替换为你的密钥
    ),
    tools=[get_weather],  # 把函数传给Agent
    instructions="你是天气助手，用get_weather工具查询天气，用中文回答",
)



if __name__ == "__main__":
    print("\n" + "="*60)
    print("Agno 框架学习示例")
    print("="*60)
    
    # 方式1：单次查询
    print("\n【方式1：单次查询】")
    response = agent.run("北京天气怎么样？")
    print(response.content)
    
    print("\n" + "="*60)
    
    # 方式2：交互式CLI（推荐）
    print("\n【方式2：交互式CLI】")
    print("输入 'exit' 退出\n")
    agent.cli_app(stream=True)
