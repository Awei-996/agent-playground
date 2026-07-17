from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.tools import tool 
from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
import operator
from typing import Literal

# 初始化LLM
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

# 定义消息状态
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int

# 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """
    multiply: 两个数字相乘
    Args:
        a: 第一个数字
        b: 第二个数字
    Returns:
        int: 两个数字相乘的结果
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    add: 两个数字相加
    Args:
        a: 第一个数字
        b: 第二个数字
    Returns:
        int: 两个数字相加的结果
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    subtract: 两个数字相减
    Args:
        a: 第一个数字
        b: 第二个数字
    Returns:
        int: 两个数字相减的结果
    """
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """
    divide: 两个数字相除
    Args:
        a: 第一个数字
        b: 第二个数字
    Returns:
        float: 两个数字相除的结果
    """
    return a / b

# 绑定工具
tools = [multiply, add, subtract, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# 定义模型节点
def llm_call(state: dict):
    """ 决定llm是否使用工具 """
    messages = [SystemMessage(content="你是一个数学助手，请使用工具计算结果")] + state["messages"]
    print("\nllm_call_messages--------------------------------: ", messages)
    response = llm_with_tools.invoke(messages)
    print("\nllm_call--------------------------------: ", response)
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# 定义工具节点
def tool_node(state: dict):
    """ 执行工具调用 """
    result = []
    print("\ntool_node------------: ", state)
    for tool_call in state["messages"][-1].tool_calls:
        print("\ntool_call--------------------------------: ", tool_call)
        tool = tools_by_name[tool_call["name"]]
        print("\ntool--------------------------------: ", tool)
        observation = tool.invoke(tool_call["args"])
        print("\nobservation--------------------------------: ", observation)
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    print("\nresult--------------------------------: ", result)
    return { "messages": result }

# 定义结束逻辑
def should_continue(state: MessageState) -> Literal["tool_node", END]:
    """ 根据LLM是否进行了工具调用来决定是继续循环还是停止循环 """
    print("\nshould_continue--------------------------------: ", state)
    messages = state["messages"]
    last_message = messages[-1]
    print("\nlast_message--------------------------------: ", last_message)
    if last_message.tool_calls:
        return "tool_node"
    
    return END


# 定义流程图
agent = StateGraph(MessageState)

# 添加节点
agent.add_node("llm_call", llm_call)
agent.add_node("tool_node", tool_node)

# 添加边
agent.add_edge(START, "llm_call")
agent.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent.add_edge("tool_node", "llm_call")

agent_compiled = agent.compile()

from IPython.display import Image, display
agent_compiled.get_graph(xray=True).draw_mermaid_png(
    output_file_path="agent_graph.png"
)
print("\n图已保存到 agent_graph.png")

# messages = [HumanMessage(content="1 + 1 = ?")]
messages = [HumanMessage(content="先计算 1 + 1，再把结果乘以 2")]

messages = agent_compiled.invoke({"messages": messages})

for message in messages["messages"]:
    message.pretty_print()
    # print(message.content)