from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class StateAgent(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber: int

graph = StateGraph(StateAgent)

# 定义节点

def add(state: StateAgent) -> StateAgent:
    """ 加法运算 """
    state['finalNumber'] = state['number1'] + state['number2']
    return state

def subtract(state: StateAgent) -> StateAgent:
    """ 减法运算 """
    state['finalNumber'] = state['number1'] - state['number2']
    return state

def decide_next_node(state: StateAgent) -> StateAgent:
    """ 决定下一个节点 """
    if state['operation'] == '+':
        return "add"
    elif state['operation'] == '-':
        return "subtract"
    else:
        return "error"

def error(state: StateAgent) -> StateAgent:
    """ 错误处理 """
    print("错误：操作符不正确")
    return state

graph.add_node("add", add)
graph.add_node("subtract", subtract)
graph.add_node("error", error)
graph.add_node("router_next_node", lambda state: state)

# 设置边
graph.add_edge(START, "router_next_node")
graph.add_conditional_edges(
    "router_next_node",
    decide_next_node,
    {
        "add": "add",
        "subtract": "subtract",
        "error": "error",
    }
)
graph.add_edge("add", END)
graph.add_edge("subtract", END)
graph.add_edge("error", END)


agent = graph.compile()

# from IPython.display import Image, display
# display(Image(agent.get_graph().draw_mermaid_png()))

result = agent.invoke({"number1": 10, "operation": "+", "number2": 20})
print(result['finalNumber'])