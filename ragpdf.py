import os
import fitz  # PyMuPDF
import asyncio
import numpy as np
from dotenv import load_dotenv

# 引入 autogen 相關模組
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# 載入環境變數
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
file_path = "test.pdf"  # 替換為你的本機 PDF 檔案路徑

# 1. 讀取 PDF 內容
def extract_text_from_pdf(file_path):
    if not os.path.exists(file_path):
        return "找不到指定的檔案，請確認路徑是否正確。"
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        return f"無法讀取 PDF：{e}"
    return text.strip()

# 2. 將文件切分成較小的片段 (以固定單字數為例)
def split_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# 3. 計算兩向量間的餘弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

# 4. 使用 embedding_client 為文字片段生成向量
async def get_embedding(text, client):
    try:
        # 假設 embedding_client 可透過 create 方法取得向量結果
        response = await client.create(
            model="gemini-1.5-flash-8b",
            messages=[
                {"role": "system", "content": "請為下列文字生成向量嵌入表示。"},
                {"role": "user", "content": text}
            ]
        )
        # 假設返回結果在 choices[0]['embedding'] (依照實際 API 格式調整)
        embedding = response['choices'][0].get('embedding')
    except Exception as e:
        print(f"Embedding error: {e}")
        embedding = None
    # 若沒有返回 embedding，可暫時以隨機向量示意（實際請使用正確向量）
    if embedding is None:
        embedding = np.random.rand(768).tolist()
    return np.array(embedding)

# 5. 建立向量索引：為每個片段生成向量，並存入 index 清單
async def build_vector_index(chunks, client):
    index = []
    for chunk in chunks:
        embedding = await get_embedding(chunk, client)
        index.append({"chunk": chunk, "embedding": embedding})
    return index

# 6. 根據查詢生成查詢向量，計算與各片段的相似度，返回 top k 相關片段
async def retrieve_relevant_chunks(query, index, client, k=3):
    query_embedding = await get_embedding(query, client)
    similarities = []
    for item in index:
        sim = cosine_similarity(query_embedding, item["embedding"])
        similarities.append((sim, item["chunk"]))
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for sim, chunk in similarities[:k]]
    return top_chunks

# 7. 整合流程：從 PDF 讀取、切分、索引、檢索，再結合 web_surfer 進行網路檢索，最後交由生成模型生成摘要
async def main():
    # 讀取 PDF 內容
    pdf_text = extract_text_from_pdf(file_path)
    if "找不到" in pdf_text or "無法" in pdf_text:
        print(pdf_text)
        return

    # 將 PDF 內容切分成片段
    chunks = split_text(pdf_text, max_words=200)

    # 初始化模型客戶端
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=gemini_api_key,
    )
    # embedding_client 用於生成向量
    embedding_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=gemini_api_key,
    )

    # 建立本地向量索引
    print("正在建立向量索引...")
    vector_index = await build_vector_index(chunks, embedding_client)

    # 定義查詢，例如「請根據文件生成摘要」
    query = "請根據上述文件內容生成摘要。"
    relevant_chunks = await retrieve_relevant_chunks(query, vector_index, embedding_client, k=3)
    local_context = "\n\n".join(relevant_chunks)

    # 同時，利用 web_surfer 執行網路檢索（例如查詢相似主題的最新資料）
    web_query = "與文件主題相關的最新資訊或背景資料"
    
    # 建立 web_surfer 代理人，這裡改用 model_client
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    
    # 優先嘗試調用 web_surfer.search 方法（若有提供）
    try:
        if hasattr(web_surfer, "search"):
            web_response = await web_surfer.search(query=web_query)
        else:
            # 若沒有 search 方法，則調用 client.create 並調整提示
            web_response = await web_surfer.client.create(
                model="gemini-1.5-flash-8b",
                messages=[
                    {"role": "system", "content": "請搜尋與下列內容相關的最新網路資訊。"},
                    {"role": "user", "content": web_query}
                ]
            )
        web_context = web_response['choices'][0].get('text', "無法取得網路資訊。")
    except Exception as e:
        print(f"Web search error: {e}")
        web_context = "無法取得網路資訊。"

    # 將本地檢索與網路檢索結果結合
    full_context = f"本地文件相關資訊：\n{local_context}\n\n網路檢索結果：\n{web_context}"

    # 建構生成摘要的提示
    task_prompt = f"請根據以下資訊生成摘要：\n\n{full_context}\n\n摘要："

    # 建立對話代理人，這裡包含 web_surfer、assistant 與 user_proxy
    assistant = AssistantAgent("assistant", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    termination_condition = TextMentionTermination("exit")
    team = RoundRobinGroupChat([web_surfer, assistant, user_proxy], termination_condition=termination_condition)

    # 啟動對話代理人進行摘要生成
    await Console(team.run_stream(task=task_prompt))

if __name__ == '__main__':
    asyncio.run(main())

