from toolExecutor import ToolExecutor
from tools import search
from helloAgentsLLM import HelloAgentsLLM
from reactAgentsLLM import ReActAgent
from planAndSolveAgent import PlanAndSolveAgent
from reflectionAgent import ReflectionAgent

def toolsUsage():
    # 初始化工具
    toolExecutor = ToolExecutor()
    # 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    # 查看可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getToolDescription())
    # 智能体调用
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"
    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")

def reactAgentsUsage():
    # 初始化工具
    toolExecutor = ToolExecutor()
    # 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    # 初始化LLM
    llm_client = HelloAgentsLLM()
    # 初始化智能体
    reactAgent = ReActAgent(llm_client=llm_client, tool_executor=toolExecutor)
    # 运行智能体
    question = "华为最新款手机是什么型号"
    final_answer = reactAgent.run(question=question)
    print(f"🎉 最终答案: {final_answer}")

def planAndSolveAgentsUsage():
    # 初始化LLM
    llm_client = HelloAgentsLLM()
    # 初始化智能体
    planAndSolveAgent = PlanAndSolveAgent(llm_client=llm_client)
    # 运行智能体
    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    final_answer = planAndSolveAgent.run(question=question)
    print(f"🎉 最终答案: {final_answer}")


def reflectionAgentsUsage():
    # 初始化LLM
    llm_client = HelloAgentsLLM()
    # 初始化智能体
    reflectionAgent = ReflectionAgent(llm_client=llm_client)
    # 运行智能体
    question = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    final_answer = reflectionAgent.run(task=question)
    print(f"🎉 最终答案: {final_answer}")

if __name__ == "__main__":
    # toolsUsage()
    # reactAgentsUsage()
    # planAndSolveAgentsUsage()
    reflectionAgentsUsage()
