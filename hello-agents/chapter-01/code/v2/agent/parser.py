import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class ParsedAction:
    kind: Literal["tool", "finish", "error"]
    tool_name: str | None = None
    kwargs: dict[str, str] | None = None
    final_answer: str | None = None
    error: str | None = None


def truncate_extra_pairs(text: str) -> str:
    """Keep only the first Thought-Action pair when the model repeats itself."""
    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def has_react_action(text: str) -> bool:
    return bool(re.search(r"Action:\s*\S", text, re.DOTALL))


def parse_react_output(text: str) -> ParsedAction:
    """Parse Thought/Action/Finish from model text (fallback path only)."""
    text = truncate_extra_pairs(text)

    action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL)
    if not action_match:
        return ParsedAction(
            kind="error",
            error="未能解析到 Action 字段。请严格遵循 'Thought: ... Action: ...' 格式。",
        )

    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
        if not finish_match:
            return ParsedAction(
                kind="error",
                error="Finish 格式无效，请使用 Finish[最终答案] 格式。",
            )
        return ParsedAction(kind="finish", final_answer=finish_match.group(1).strip())

    tool_match = re.search(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
    if not tool_match or not args_match:
        return ParsedAction(
            kind="error",
            error=f"无法解析工具调用: {action_str}",
        )

    tool_name = tool_match.group(1)
    args_str = args_match.group(1)
    kwargs = dict(re.findall(r"""(\w+)=["']([^"']*)["']""", args_str))

    return ParsedAction(kind="tool", tool_name=tool_name, kwargs=kwargs)
