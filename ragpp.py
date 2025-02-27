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
)
embedding_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key=gemini_api_key,
)

# 建立代理人
assistant = AssistantAgent("assistant", model_client)
web_surfer = MultimodalWebSurfer("web_surfer", embedding_client)
user_proxy = UserProxyAgent("user_proxy")

# 終止條件：當使用者輸入 "exit" 時
termination_condition = TextMentionTermination("exit")

# 代理人循環對話
team = RoundRobinGroupChat([
    web_surfer, assistant, user_proxy
], termination_condition=termination_condition)

# 檢索步驟：基於文件內容進行檢索
async def retrieve_info(file_content):
    # 模擬一個檢索步驟，這裡你可以用 embedding_client 或其他檢索工具進行資料檢索
    # 假設檢索到與文件內容相關的資料，這裡簡單模擬為同一內容（可以根據需求進行修改）
    query = f"與以下內容相關的資料：\n\n{file_content}"
    retrieved_data = await embedding_client.chat_complete(
        query
    )  # 使用適當的 API 方法進行檢索，根據需要調整
    return retrieved_data['choices'][0]['text']

# 啟動對話，讓 AI 進行檢索和生成摘要
async def main():
    # 進行檢索操作
    retrieved_data = await retrieve_info(file_content)

    # 結合檢索到的資料和原始文件內容，作為生成的輸入
    task = f"請根據以下內容和檢索結果撰寫摘要：\n\n{file_content}\n\n{retrieved_data}"
    await Console(team.run_stream(task=task))

if __name__ == '__main__':
    asyncio.run(main())
