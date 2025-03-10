import os
import fitz  # PyMuPDF
import asyncio
import numpy as np
from dotenv import load_dotenv
import gradio as gr
from PyPDF2 import PdfReader

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

# 2. 將文件切分成較小的片段
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
        response = await client.create(
            model="gemini-2.0-flash",
            messages=[{"role": "system", "content": "請為下列文字生成向量嵌入表示。"}, {"role": "user", "content": text}]
        )
        embedding = response['choices'][0].get('embedding')
    except Exception as e:
        print(f"Embedding error: {e}")
        embedding = None
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
async def process_pdf(file_obj, query):
    file_path = file_obj.name
    # 讀取 PDF 內容
    pdf_text = extract_text_from_pdf(file_path)
    if "找不到" in pdf_text or "無法" in pdf_text:
        return pdf_text

    # 將 PDF 內容切分成片段
    chunks = split_text(pdf_text, max_words=200)

    # 初始化模型客戶端
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=gemini_api_key,
    )
    embedding_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )

    # 建立本地向量索引
    vector_index = await build_vector_index(chunks, embedding_client)

    # 使用查詢檢索相關片段
    relevant_chunks = await retrieve_relevant_chunks(query, vector_index, embedding_client, k=3)
    local_context = "\n\n".join(relevant_chunks)

    # 建立 web_surfer 代理人
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    try:
        web_response = await web_surfer.client.create(
            model="gemini-1.5-flash-8b",
            messages=[{"role": "system", "content": "請搜尋與下列內容相關的最新網路資訊。"}, {"role": "user", "content": query}]
        )
        web_context = web_response['choices'][0].get('text', "無法取得網路資訊。")
    except Exception as e:
        print(f"Web search error: {e}")
        web_context = "無法取得網路資訊。"

    # 組合本地資料和網路資料
    full_context = f"本地文件相關資訊：\n{local_context}\n\n網路檢索結果：\n{web_context}"

    # 輸出最終摘要
    task_prompt = f"請根據以下資訊生成摘要：\n\n{full_context}\n\n摘要："
    assistant = AssistantAgent("assistant", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    termination_condition = TextMentionTermination("exit")
    team = RoundRobinGroupChat([web_surfer, assistant, user_proxy], termination_condition=termination_condition)

    return await Console(team.run_stream(task=task_prompt))

# Gradio UI
def gradio_interface(file_obj, query):
    response = asyncio.run(process_pdf(file_obj, query))
    return response

with gr.Blocks() as demo:
    gr.Markdown("### PDF 內容提取與摘要生成系統")

    file_input = gr.File(label="上傳 PDF")
    query_input = gr.Textbox(label="請輸入查詢")
    output_display = gr.Textbox(label="摘要")

    start_btn = gr.Button("生成摘要")

    start_btn.click(fn=gradio_interface, inputs=[file_input, query_input], outputs=output_display)

demo.queue().launch()
