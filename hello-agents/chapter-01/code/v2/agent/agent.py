import json
from typing import Any

from agent.client import LLMClient
from agent.parser import ParsedAction, has_react_action, parse_react_output, truncate_extra_pairs
from config import Settings
from tools import AVAILABLE_TOOLS, TOOL_SCHEMAS, tool_descriptions


def _build_system_prompt() -> str:
    return f"""你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
{tool_descriptions()}

# 工作方式:
- 优先通过工具调用（function calling）获取信息
- 若无法使用工具调用，则按以下格式输出：
  Thought: [你的思考过程和下一步计划]
  Action: [具体行动，单行不换行]

Action 格式：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只采取一个行动
- 信息足够回答用户问题时，给出最终答案或 Finish[最终答案]
- 不要编造天气或景点信息，务必先调用工具
"""


class TravelAgent:
    """Hybrid agent: Tool Calling primary, text ReAct fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = LLMClient(
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": settings.prompt},
        ]

    def run(self) -> str | None:
        print(f"用户输入: {self.settings.prompt}\n" + "=" * 40)

        for step in range(1, self.settings.max_steps + 1):
            print(f"--- 循环 {step} ---\n")
            try:
                message = self.client.chat(self.messages, tools=TOOL_SCHEMAS)
            except Exception as e:
                print(f"调用 LLM 失败: {e}")
                return None

            result = self._handle_response(message)
            if result is not None:
                return result

        print(f"已达到最大循环次数 ({self.settings.max_steps})，任务未结束。")
        return None

    def _handle_response(self, message) -> str | None:
        content = (message.content or "").strip()
        tool_calls = message.tool_calls or []

        if tool_calls:
            return self._handle_tool_calls(message, content, tool_calls)

        if content and has_react_action(content):
            return self._handle_react_fallback(content)

        if content:
            print(f"模型输出:\n{content}\n")
            print(f"任务完成，最终答案: {content}")
            return content

        observation = "错误: 模型返回为空，请继续任务。"
        print(f"Observation: {observation}\n" + "=" * 40)
        self.messages.append({"role": "assistant", "content": ""})
        self.messages.append({"role": "user", "content": f"Observation: {observation}"})
        return None

    def _handle_tool_calls(self, message, content: str, tool_calls) -> str | None:
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        self.messages.append(assistant_msg)

        if content:
            print(f"模型输出:\n{content}\n")
        else:
            print("模型请求工具调用\n")

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                kwargs = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                kwargs = {}

            print(f"Action: {tool_name}({kwargs})")
            observation = self._execute_tool(tool_name, kwargs)
            print(f"Observation: {observation}\n" + "=" * 40)

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                }
            )

        return None

    def _handle_react_fallback(self, content: str) -> str | None:
        truncated = truncate_extra_pairs(content)
        if truncated != content:
            print("已截断多余的 Thought-Action 对")
            content = truncated

        print(f"模型输出 (ReAct 回退):\n{content}\n")
        self.messages.append({"role": "assistant", "content": content})

        parsed = parse_react_output(content)
        return self._dispatch_parsed_action(parsed)

    def _dispatch_parsed_action(self, parsed: ParsedAction) -> str | None:
        if parsed.kind == "finish":
            print(f"任务完成，最终答案: {parsed.final_answer}")
            return parsed.final_answer

        if parsed.kind == "error":
            observation = f"错误: {parsed.error}"
            print(f"Observation: {observation}\n" + "=" * 40)
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})
            return None

        assert parsed.tool_name is not None
        assert parsed.kwargs is not None
        print(f"Action: {parsed.tool_name}({parsed.kwargs})")
        observation = self._execute_tool(parsed.tool_name, parsed.kwargs)
        print(f"Observation: {observation}\n" + "=" * 40)
        self.messages.append({"role": "user", "content": f"Observation: {observation}"})
        return None

    def _execute_tool(self, tool_name: str, kwargs: dict) -> str:
        fn = AVAILABLE_TOOLS.get(tool_name)
        if not fn:
            return f"错误:未定义的工具 '{tool_name}'"
        try:
            return fn(**kwargs)
        except TypeError as e:
            return f"错误:工具参数不匹配 - {e}"
