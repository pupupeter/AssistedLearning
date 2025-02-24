import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

async def main() -> None:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    agent = AssistantAgent("assistant", OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=gemini_api_key,
    ))
    print(await agent.run(task="Say 'Hello World!'"))

asyncio.run(main())