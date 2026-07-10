import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端优先使用传入的参数如果没有则使用环境变量中的
        """
        self.model = model or os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_NAME")
        apiKey = apiKey or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("model, apiKey, baseUrl 不能为空")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        print(f"正在使用{self.model}大模型")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            print("大模型响应成功")
            collected_content = []

            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print("\n------")

            return "".join(collected_content)
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return None

if __name__ == "__main__":
    try:
        llClient = HelloAgentsLLM()
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code"},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("----调用llm---")
        responseText = llClient.think(exampleMessages)
        if responseText:
            print("---完整模型响应---")
            print(responseText)
    except ValueError as e:
        print(e)
