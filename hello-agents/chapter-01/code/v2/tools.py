import os

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from tavily import TavilyClient


def get_weather(city: str) -> str:
    """Query real-time weather via wttr.in."""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """Search attraction recommendations via Tavily."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]

        formatted_results = [
            f"- {result['title']}: {result['content']}"
            for result in response.get("results", [])
        ]
        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attraction",
            "description": "根据城市和天气搜索推荐的旅游景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                    },
                    "weather": {
                        "type": "string",
                        "description": "当前天气描述，用于匹配适合的景点",
                    },
                },
                "required": ["city", "weather"],
            },
        },
    },
]


def tool_descriptions() -> str:
    """Generate tool list text for the system prompt."""
    lines = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        params = fn["parameters"]["properties"]
        param_str = ", ".join(f"{k}: str" for k in params)
        lines.append(f"- `{fn['name']}({param_str})`: {fn['description']}")
    return "\n".join(lines)
