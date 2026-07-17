from typing import Literal, List, TypedDict
import os

from langgraph.graph import END, StateGraph, START
from langgraph.types import Command, interrupt, RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage



# 定义邮件分类
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str

    classification: EmailClassification | None

    search_results: List[str] | None
    customer_history: dict | None

    draft_response: str | None
    messages: List[str] | None


# 初始化LLM
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

# 定义节点

def read_email(state: EmailAgentState) -> dict:
    """ 读取邮件内容 """
    print("\nread_email: ", state)
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation","human_review","bug_tracking","draft_response"]]:
    """ 使用LLM对电子邮件的意图和紧急程度进行分类，然后相应地进行路由 """
    # 创建返回EmailClassification字典的结构化LLM
    structured_llm = llm.with_structured_output(EmailClassification)
    # 创建提示词
    classification_prompt = f"""
        分析此客户电子邮件并对其进行分类:

        Email: {state['email_content']}
        From: {state['sender_email']}

        提供分类，包括意图、紧迫性、主题和总结
    """

    # 直接以字典形式获取结构化响应
    classification = structured_llm.invoke(classification_prompt)
    print("\nclassify_intent: ", classification)
    # 根据分类确定下一个节点
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"
    print("\ngoto: ", goto)
    # 将分类存储为状态中的单个字典
    return Command(
        update={"classification": classification},
        goto=goto
    )

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """ 在知识库中搜索相关信息 """

    # 根据分类构建搜索查询
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"
    try:
        # 存储原始搜索结果，而不是格式化文本
        search_result = [
            "通过设置>安全>更改密码重置密码",
            "密码必须至少包含12个字符",
            "包括大写、小写、数字和符号"
        ]
    except Exception as e:
        search_result = [f"搜索暂时不可用: {str(e)}"]
    
    return Command(
        update={"search_results": search_result},
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """ 创建或更新错误跟踪单 """

    # 在错误跟踪系统中创建或更新错误跟踪单
    ticket_id = "BUG-12345"  # 这里通过API创建

    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )

def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """ 根据上下文生成响应，并根据质量路由 """

    classification = state.get('classification', {})

    # 根据需要从原始状态数据格式化上下文
    context_sections = []

    if state.get('search_results'):
        # 格式化搜索结果用于提示
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"相关文件:\n{formatted_docs}")

    if state.get('customer_history'):
        # 格式化客户数据用于提示
        context_sections.append(f"客户层: {state['customer_history'].get('tier', 'standard')}")

    # 构建带有格式化上下文的提示
    draft_prompt = f"""
        为这封客户电子邮件起草一个响应:
        {state['email_content']}

        Email intent: {classification.get('intent', 'unknown')}
        Urgency level: {classification.get('urgency', 'medium')}

        {chr(10).join(context_sections)}

        准则:
        - 专业且有帮助
        - 解决他们的具体问题
        - 在相关时使用提供的文档
    """

    response = llm.invoke(draft_prompt)

    # 根据紧急程度和意图确定是否需要人工审查
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    # 根据需要路由到适当的下一个节点
    goto = "human_review" if needs_review else "send_reply"

    return Command(
        update={"draft_response": response.content},  # 只存储原始响应
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """ 使用中断暂停人工审查，并根据决定路由 """
    print("\nhuman_review: ", state)
    classification = state.get('classification', {})

    # interrupt() 必须首先出现 - 任何在它之前运行的代码将在恢复时重新运行
    human_decision = interrupt({
        "email_id": state.get('email_id',''),
        "original_email": state.get('email_content',''),
        "draft_response": state.get('draft_response',''),
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "请审查并批准/编辑此响应"
    })
    print("\nhuman_decision: ", human_decision)
    # 现在处理人类的决定
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response',''))},
            goto="send_reply"
        )
    else:
        # 拒绝意味着人类将直接处理
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """ 发送电子邮件响应 """
    # 集成电子邮件服务
    print(f"\nsend_reply: {state['draft_response'][:100]}...")
    return {}


# 定义图
workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# 为可能出现暂时故障的节点添加重试策略
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)


# 设置图的边
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# 使用检查指针编译以实现持久性
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

from IPython.display import Image, display
app.get_graph(xray=True).draw_mermaid_png(
    output_file_path="email_agent_graph.png"
)
print("\n图已保存到 email_agent_graph.png")


def prompt_human_review(interrupt_items) -> dict:
    """从中断信息读取控制台输入，构造 resume 数据。"""
    payload = interrupt_items[0].value if hasattr(interrupt_items[0], "value") else interrupt_items[0]

    print("\n========== 人工审核 ==========")
    print(f"邮件 ID: {payload.get('email_id', '')}")
    print(f"意图: {payload.get('intent', '')} | 紧急程度: {payload.get('urgency', '')}")
    print(f"\n原始邮件:\n{payload.get('original_email', '')}")
    print(f"\nAI 草稿回复:\n{payload.get('draft_response', '')}")
    print("==============================")
    print("操作: [y] 直接发送 AI 草稿  [e] 手动输入后发送  [n] 拒绝")

    while True:
        choice = input("\n请选择 (y/e/n): ").strip().lower()
        if choice in ("y", "yes", "是"):
            return {
                "approved": True,
                "edited_response": payload.get("draft_response", ""),
            }
        if choice in ("e", "edit", "编辑", "手动"):
            print("请输入自定义回复（单独一行输入 END 结束）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            custom_response = "\n".join(lines).strip()
            if not custom_response:
                print("回复不能为空，请重新选择。")
                continue
            return {
                "approved": True,
                "edited_response": custom_response,
            }
        if choice in ("n", "no", "否"):
            return {"approved": False}
        print("请输入 y、e 或 n")


if __name__ == "__main__":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    config = {"configurable": {"thread_id": "email-001"}}

    # 场景1：billing + critical（紧急账单）
    # 路由：classify_intent → human_review → interrupt 人工审核 → send_reply
    # initial_state: EmailAgentState = {
    #     "email_content": "我的订阅被收取了两次费用！这很紧急！",
    #     "sender_email": "customer@example.com",
    #     "email_id": "email-billing-001",
    # }

    # 场景2：question（普通咨询，低优先级）
    # 路由：classify_intent → search_documentation → draft_response → send_reply（不触发 interrupt）
    # initial_state: EmailAgentState = {
    #     "email_content": "你好，我忘记了登录密码，请问如何重置？",
    #     "sender_email": "user@example.com",
    #     "email_id": "email-question-001",
    # }

    # 场景3：bug（缺陷报告）
    # 路由：classify_intent → bug_tracking → draft_response → send_reply
    initial_state: EmailAgentState = {
        "email_content": "更新到 v2.3 后，每次点击导出按钮应用就会崩溃，请帮忙看看。",
        "sender_email": "dev@example.com",
        "email_id": "email-bug-001",
    }

    # 场景5：complex（复杂问题，经 draft_response 触发人工审核）
    # 路由：classify_intent → draft_response → human_review → interrupt → send_reply
    # initial_state: EmailAgentState = {
    #     "email_content": "我们团队 50 人需要迁移数据、保留权限并对接 SSO，请给完整方案和时间表。",
    #     "sender_email": "it-admin@example.com",
    #     "email_id": "email-complex-001",
    # }



    result = app.invoke(initial_state, config=config)

    if result.get("__interrupt__"):
        print("\n等待人工审核...")
        resume_data = prompt_human_review(result["__interrupt__"])
        result = app.invoke(Command(resume=resume_data), config=config)

    print("最终结果:", result)
