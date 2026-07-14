from langgraph.graph import StateGraph, add_messages, END, START
from typing import Annotated, TypedDict
import os
import re
import sys
from datetime import date
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


#--- 定义全局状态
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str

# --- 配置环境变量
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# ---创建理解需求节点

def understand_query_node(state: SearchState) -> SearchState:
    """理解用户查询并生成搜索关键词"""
    user_message = state["messages"][-1].content

    understand_prompt = f"""今天是 {date.today()}，分析用户的查询："{user_message}"
        请完成两个任务：
        1. 简洁总结用户想要了解什么
        2. 生成最适合搜索引擎的关键词（中英文均可，要精准，注意使用当前年份）

        格式：
        理解：[用户需求总结]
        搜索词：[最佳搜索关键词]"""

    response = llm.invoke([HumanMessage(content=understand_prompt)])
    response_text = response.content
    search_query = user_message
    match = re.search(r"搜索词[：:]\s*(.+)", response_text)
    if match:
        search_query = match.group(1).strip().split("\n")[0]
    return {
        "user_query": user_message,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")]
    }

# ---创建搜索节点
def tavily_search_node(state: SearchState) -> SearchState:
    """使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:
        tavily_results = tavily_client.search(
            query=search_query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
        parts = []
        if tavily_results.get("answer"):
            parts.append(f"摘要：{tavily_results['answer']}")
        for item in tavily_results.get("results", []):
            content = item.get("content") or item.get("snippet", "")
            parts.append(f"{item.get('title', '')}\n{content}")
        results_text = "\n\n".join(parts) or "未找到相关结果"
        return {
            "search_results": results_text,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成，正在整理答案...")]
        }
    except Exception as e:
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题...")]
        }

# --- 回答节点
def generate_answer_node(state: SearchState) -> dict:
    if state["step"] == "searched":
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
            用户问题：{state['user_query']}
            搜索结果：\n{state['search_results']}
            请综合搜索结果，提供准确、有用的回答。"""
        response = llm.invoke([SystemMessage(content=answer_prompt)])
    else:
        fallback_prompt = f"""搜索暂时不可用（{state['search_results']}）。
            请基于您的知识回答用户的问题：{state['user_query']}"""
        response = llm.invoke([SystemMessage(content=fallback_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

def create_search_assistant():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 编译图
    memory = InMemorySaver()
    agent = workflow.compile(checkpointer=memory)
    return agent

# --- 创建助手实例
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    search_assistant = create_search_assistant()
    user_query = input("请输入您的问题：")
    response = search_assistant.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config={"configurable": {"thread_id": "1"}},
    )
    print("\n" + response["final_answer"])
