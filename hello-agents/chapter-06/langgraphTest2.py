from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List,Union
import random


class StateAgent(TypedDict):
    input: int
    current_stage: int
    process_output: List[int]
    final_output: int | str
    counter: int
    judgment_result: str
    limit_stage: int


def setup_node(state: StateAgent) -> dict:
    """设置初始状态"""
    return {
        "current_stage": 0,
        "process_output": [],
        "counter": 0,
        "judgment_result": "未开始",
        "limit_stage": 10,
    }


def random_number_node(state: StateAgent) -> dict:
    """根据上次判断结果，在范围内生成随机猜测数"""
    if state["judgment_result"] == "未开始":
        low, high = 1, 100
    elif state["judgment_result"] == "大于":
        low, high = 1, state["current_stage"]
    elif state["judgment_result"] == "小于":
        low, high = state["current_stage"], 100
    else:
        low, high = 1, 100

    current = random.randint(low, high)
    return {
        "current_stage": current,
        "process_output": state["process_output"] + [current],
        "counter": state["counter"] + 1,
    }


def check_number_node(state: StateAgent) -> dict:
    """比较猜测值与目标值，更新判断结果"""
    if state["current_stage"] > state["input"]:
        judgment = "大于"
    elif state["current_stage"] < state["input"]:
        judgment = "小于"
    else:
        return {
            "judgment_result": "等于",
            "final_output": state["current_stage"],
        }

    # 用完次数仍未猜中
    if state["counter"] >= state["limit_stage"]:
        return {
            "judgment_result": "超过次数",
            "final_output": "超过次数",
        }

    return {"judgment_result": judgment}


def route_after_check(state: StateAgent) -> str:
    """路由函数：只读 state，返回下一个节点名"""
    return state["judgment_result"]


graph = StateGraph(StateAgent)

graph.add_node("setup", setup_node)
graph.add_node("random_number", random_number_node)
graph.add_node("check", check_number_node)

graph.add_edge(START, "setup")
graph.add_edge("setup", "random_number")
graph.add_edge("random_number", "check")
graph.add_conditional_edges(
    "check",
    route_after_check,
    {
        "大于": "random_number",
        "小于": "random_number",
        "等于": END,
        "超过次数": END,
    },
)

agent = graph.compile()
result = agent.invoke({"input": 50})
print(result)
