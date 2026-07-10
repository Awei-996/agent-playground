from planner import Planner

from planExecutor import PlanExecutor
from helloAgentsLLM import HelloAgentsLLM

class PlanAndSolveAgent:

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = PlanExecutor(self.llm_client)

    def run(self, question: str) -> str:
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)
        
        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
         # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)
        
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")