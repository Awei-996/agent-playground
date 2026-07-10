from email import message
import re
from helloAgentsLLM import HelloAgentsLLM
from toolExecutor import ToolExecutor

REACT_PROMPT_TEMPLATE = """
 请注意，你是一个又能力调用外部工具的智能助手。
 
 可用工具:
  {tools}

 请严格按照一下格式进行回应：
 Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
 Action: 你决定采取的行动，必须是以下格式一下之一：
 - `{{tool_name}}[{{tool_input}}]`: 调用一个可用工具
 - `Finish[最终答案]`: 当你认为已经获得最终答案时。
 - 当你收集足够的信息，能够回答用户最终答案时，你必须在Action：字段后面使用Finish[最终答案]来输出最终答案

 现在开始解决一下问题
 Question: {question}
 History: {history}
"""


class ReActAgent:
    
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):

        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1

            print(f"\n--- 第 {current_step} 步 ---")
            # 格式化提示词
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.getToolDescription(),
                question=question,
                history="/n".join(self.history)
            )
            # 调用LLM思考
            messages = [{"role": "user", "content": prompt}]
            response_text  = self.llm_client.think(messages = messages)

            if not response_text:
                print("LLM调用失败")
                break
            # 解析LLM的输出
            thought, action= self._parse_output(response_text)
            if thought:
                print(f"Thought: {thought}")
            if not action:
                print("Action: 字段为空，继续思考")
                break
            # 执行action
            if action.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)

            if not tool_name or not tool_input:
                continue

            print(f"🔧 调用工具: {tool_name} 输入: {tool_input}")
            tool_function = self.tool_executor.getTool(tool_name=tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        print("已达到最大步数，流程终止。")
        return None
    
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
