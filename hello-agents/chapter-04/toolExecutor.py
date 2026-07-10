from ast import Call
import re
from typing import Dict, Any

class ToolExecutor:

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def registerTool(self, tool_name: str,description: str, tool_function: callable):

        if tool_name in self.tools:
            print(f"Tool '{tool_name}' 已存在, 覆盖原有工具")
        self.tools[tool_name] = {
            "description": description,
            "function": tool_function
        }

        print(f"Tool '{tool_name}' 注册成功")
    
    def getTool(self,tool_name: str) -> callable:
        return self.tools.get(tool_name, {}).get("function", None)

    def getToolDescription(self) -> str:
        return "\n\n".join(
            f"{name} 工具描述:\n{info.get('description', '没有描述')}"
            for name, info in self.tools.items()
        )

