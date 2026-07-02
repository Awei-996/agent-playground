from agent.agent import TravelAgent
from config import load_settings


def main() -> None:
    settings = load_settings()
    agent = TravelAgent(settings)
    agent.run()


if __name__ == "__main__":
    main()
