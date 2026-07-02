import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

V2_DIR = Path(__file__).resolve().parent

DEFAULT_PROMPT = (
    "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
)


def load_env() -> None:
    """Load v2/.env, falling back to parent code/.env."""
    v2_env = V2_DIR / ".env"
    parent_env = V2_DIR.parent / ".env"
    if v2_env.exists():
        load_dotenv(v2_env)
    elif parent_env.exists():
        load_dotenv(parent_env)


@dataclass
class Settings:
    api_key: str
    base_url: str
    model_name: str
    prompt: str
    max_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 01 Travel Agent (v2)")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="User question for the agent",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum agent loop iterations (default: 5)",
    )
    return parser.parse_args()


def load_settings() -> Settings:
    load_env()
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    missing = [
        name
        for name, value in [
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url),
            ("MODEL_NAME", model_name),
        ]
        if not value
    ]
    if missing:
        print(f"错误: 缺少环境变量: {', '.join(missing)}", file=sys.stderr)
        print("请在 v2/.env 中配置，或执行: cp ../.env .env", file=sys.stderr)
        sys.exit(1)

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        prompt=args.prompt,
        max_steps=args.max_steps,
    )
