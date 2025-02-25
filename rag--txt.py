import os
from dotenv import load_dotenv
import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# 載入 .env 檔案中的環境變數
load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")
file_path = "test.txt"  # 替換為你的本機 txt 檔案路徑

# 讀取 txt 檔案內容
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
else:
    file_content = "找不到指定的檔案，請確認路徑是否正確。"

# 設定 LLM，包含 retrieval 工具
model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key=gemini_api_key,
    tools=[{"type": "retrieval"}]
)

# 建立代理人
assistant = AssistantAgent("assistant", model_client)
web_surfer = MultimodalWebSurfer("web_surfer", model_client)
user_proxy = UserProxyAgent("user_proxy")

# 終止條件：當使用者輸入 "exit" 時
termination_condition = TextMentionTermination("exit")

# 代理人循環對話
team = RoundRobinGroupChat([
    web_surfer, assistant, user_proxy
], termination_condition=termination_condition)

# 啟動對話，讓 AI 分析 txt 檔案內容並總結
async def main():
    task = f"請根據以下內容撰寫摘要：\n\n{file_content}"
    await Console(team.run_stream(task=task))

if __name__ == '__main__':
    asyncio.run(main())