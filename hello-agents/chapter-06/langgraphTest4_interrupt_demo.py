from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class EmailState(TypedDict):
    email_content: str
    response_text: str | None


def human_review_node(state: EmailState):
    interrupt(
        {
            "approved": False,
            "edited_response": state.get("response_text") or "",
        }
    )
    return {"response_text": "placeholder"}


app = (
    StateGraph(EmailState)
    .add_node("human_review", human_review_node)
    .add_edge(START, "human_review")
    .add_edge("human_review", END)
    .compile(checkpointer=InMemorySaver())
)

initial_state = {
    "email_content": "我的订阅被收取了两次费用！这很紧急！",
    "response_text": "草稿响应",
}

# 运行并使用thread_id实现持久性
config = {"configurable": {"thread_id": "customer_123"}}
stream = app.stream_events(initial_state, config, version="v3")
_ = stream.output  # 驱动流完成
# 图将在human_review处暂停
print(f"human review interrupt:{stream.interrupts}")

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "我们真诚地道歉，因为收取了两次费用。我已立即发起退款...",
    }
)

# 恢复执行
resumed = app.stream_events(human_response, config, version="v3")
final_state = resumed.output
print("电子邮件发送成功！")