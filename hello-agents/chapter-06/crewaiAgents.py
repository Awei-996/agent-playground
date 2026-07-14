from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
import os

load_dotenv()

llm = LLM(
    model=os.getenv("MODEL_NAME"),
    provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7,
)

# ---------定义四个角色
# 1. 产品经理
product_manager = Agent(
    role="产品经理",
    goal="将用户的需求转化为清晰、可执行的开发计划",
    backstory="""
    你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。
    你的核心职责包括：
    1. 需求分析：深入理解用户需求，识别核心功能和边界条件
    2. 技术规划：基于需求制定清晰的技术实现路径
    3. 风险评估：识别潜在的技术风险和用户体验问题
    4. 协调沟通：与工程师和其他团队成员进行有效沟通
    当接到开发任务时，请按以下结构进行分析：
    1. 需求理解与分析
    2. 功能模块划分
    3. 技术选型建议
    4. 实现优先级排序
    5. 验收标准定义
    请简洁明了地回应，并在分析完成后说"请工程师开始实现
    """,
    llm=llm,
    verbose=True,
    allow_delegation=True,

)
# 2. 软件工程师
engineer = Agent(
    role="软件工程师",
    goal="根据产品需求编写完整、可运行的应用程序代码",
    backstory="""你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。
你的技术专长包括：
1. Python 编程：熟练掌握 Python 语法和最佳实践
2. Web 开发：精通 Streamlit、Flask、Django 等框架
3. API 集成：有丰富的第三方 API 集成经验
4. 错误处理：注重代码的健壮性和异常处理
当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理
请提供完整的可运行代码，并在完成后说"请代码审查员检查"。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)
# 3. 代码审查员
code_reviewer = Agent(
    role="代码审查员",
    goal="审查代码质量、安全性和最佳实践，提出具体改进建议",
    backstory="""你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。
你的审查重点包括：
1. 代码质量：检查可读性、可维护性和性能
2. 安全性：识别潜在的安全漏洞和风险点
3. 最佳实践：确保代码遵循行业标准和最佳实践
4. 错误处理：验证异常处理的完整性和合理性
审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量
请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)
# 4. 用户代理
user_proxy = Agent(
    role="用户代理",
    goal="代表最终用户验证应用功能是否符合需求，并给出测试结论",
    backstory="""你是用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 验证最终代码实现是否满足需求
3. 从用户角度评估功能完整性和用户体验
4. 提供用户反馈和建议
完成测试后请回复 TERMINATE。""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# -------------定义任务
dev_task_description = """我们需要开发一个比特币价格显示应用，具体要求如下：
核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能
技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态
请团队协作完成这个任务，从需求分析到最终实现。"""

# Task 1: 产品经理分析需求

task_analysis = Task(
    description=dev_task_description,
    expected_output="包含需求分析、功能模块划分、技术选型、优先级和验收标准的开发计划",
    agent=product_manager,
)



# Task 2: 工程师 — 编码（接收产品经理输出）
task_coding = Task(
    description="根据产品经理的开发计划，编写完整的 Streamlit 比特币价格应用代码。",
    expected_output="完整的可运行 Python/Streamlit 代码，含注释和错误处理",
    agent=engineer,
    context=[task_analysis],  # 关键：传递上游任务结果
)
# Task 3: 审查员 — 代码审查（接收工程师输出）
task_review = Task(
    description="审查工程师提交的代码，检查质量、安全性和最佳实践。",
    expected_output="详细的代码审查报告，包含问题点和优化建议",
    agent=code_reviewer,
    context=[task_coding],
)
# Task 4: 用户代理 — 测试验收（接收全部上游结果）
task_test = Task(
    description="从用户角度验证应用是否满足原始需求，给出测试结论。",
    expected_output="用户测试报告；若功能满足需求，最后一行必须包含 TERMINATE",
    agent=user_proxy,
    context=[task_analysis, task_coding, task_review],
)
# -----组件团队

def run_software_development_team():
    print("正在创建团队...")
    crew = Crew(
        agents=[product_manager, engineer, code_reviewer, user_proxy],
        tasks=[task_analysis, task_coding, task_review, task_test],
        verbose=True,
    )

    print("🚀 启动 CrewAI 软件开发团队协作...")
    print("=" * 60)


    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")
    print(f"\n📋 最终结果：\n{result}")
    return result

if __name__ == "__main__":
    try:
        result = run_software_development_team()
        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()