from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] | None =  Field(None,description="路由过程的下一步")

# 把Route转化为JSON Schema,这样LLM返回就是Route对象
router = llm.with_structured_output(Route)


class State(TypedDict):
    input: str
    decision: str
    output: str

def write_poem(state: State) -> State:
    """ 生成一首诗 """
    llm_response = llm.invoke(f"生成一首关于 {state['input']} 的诗")
    return {"output": llm_response.content}

def write_story(state: State) -> State:
    """ 生成一个故事 """
    llm_response = llm.invoke(f"生成一个关于 {state['input']} 的故事")
    return {"output": llm_response.content}

def write_joke(state: State) -> State:
    """ 生成一个笑话 """
    llm_response = llm.invoke(f"生成一个关于 {state['input']} 的笑话")
    return {"output": llm_response.content}

def route_next_node(state: State) -> State:
    """ 将输入路由到相应的节点 """
    decision = router.invoke([
        SystemMessage(content="你是一个路由器，根据用户输入的路径，决定下一步应该路由到哪个节点"),
        HumanMessage(content=state['input'])
    ])

    return {"decision": decision.step}

def router_decision(state: State) -> State:
    """ 根据决策决定下一步应该路由到哪个节点 """
    if state['decision'] == "poem":
        return "write_poem"
    elif state['decision'] == "story":
        return "write_story"
    elif state['decision'] == "joke":
        return "write_joke"

graph = StateGraph(State)
graph.add_node("route_next_node", route_next_node)
graph.add_node("write_poem", write_poem)
graph.add_node("write_story", write_story)
graph.add_node("write_joke", write_joke)

graph.add_edge(START, "route_next_node")
graph.add_conditional_edges(
    "route_next_node",
    router_decision,
    {
        "write_poem": "write_poem",
        "write_story": "write_story",
        "write_joke": "write_joke"
    }
)
graph.add_edge("write_poem", END)
graph.add_edge("write_story", END)
graph.add_edge("write_joke", END)

workflow = graph.compile()


from IPython.display import Image, display
workflow.get_graph(xray=True).draw_mermaid_png(
    output_file_path="router_graph.png"
)
print("\n图已保存到 router_graph.png")

result = workflow.invoke({"input": "写一首关于爱情的诗"})
print(result)