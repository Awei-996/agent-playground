from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import TypedDict


load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)


class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def generate_joke(state: State) -> dict:
    """ 生成一个笑话 """
    llm_response = llm.invoke(f"生成一个关于 {state['topic']} 的笑话")
    return {"joke": llm_response.content}

def generate_story(state: State) -> dict:
    """ 生成一个故事 """
    llm_response = llm.invoke(f"生成一个关于 {state['topic']} 的故事")
    return {"story": llm_response.content}

def generate_poem(state: State) -> dict:
    """ 生成一首诗 """
    llm_response = llm.invoke(f"生成一首关于 {state['topic']} 的诗")
    return {"poem": llm_response.content}


def combine_output(state: State) -> dict:
    """ 组合输出 """
    combined_output = f"笑话: {state['joke']}\n故事: {state['story']}\n诗: {state['poem']}"
    return {"combined_output": combined_output}

graph = StateGraph(State)
graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_story", generate_story)
graph.add_node("generate_poem", generate_poem)
graph.add_node("combine_output", combine_output)

graph.add_edge(START, "generate_joke")
graph.add_edge(START, "generate_story")
graph.add_edge(START, "generate_poem")
graph.add_edge("generate_joke", "combine_output")
graph.add_edge("generate_story", "combine_output")
graph.add_edge("generate_poem", "combine_output")
graph.add_edge("combine_output", END)

workflow = graph.compile()

from IPython.display import Image, display
workflow.get_graph(xray=True).draw_mermaid_png(
    output_file_path="story_generator_graph.png"
)
print("\n图已保存到 story_generator_graph.png")


result = workflow.invoke({"topic": "编程"})
print(result)
