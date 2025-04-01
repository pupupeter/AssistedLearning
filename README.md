## Lesson: DATA STRAUTRE
 [HW1](https://github.com/pupupeter/AssistedLearning/blob/main/ragpp.py)
 [HW2](https://github.com/pupupeter/AssistedLearning/blob/main/%E6%88%91%E4%B8%8D%E7%9F%A5%E9%81%93.py)

HW2
![image](https://github.com/user-attachments/assets/0e4356ab-7e86-4e1f-8cfe-d81b19525297)


 [HW3](https://github.com/pupupeter/naverplaywright)






























# AssistedLearning 
forAssistedLearning


### 請注意 這些code 皆是在vscode裡執行，且是使用虛擬環境來製作的

要有的資料夾
![image](https://github.com/user-attachments/assets/e4e3ae79-b009-4dd3-88ad-1b97e1822989) 裡面有


![image](https://github.com/user-attachments/assets/0d74700d-5b5a-42fb-b48a-ccb994d680ec)

## 要有的資料夾
- `new-venv`

## .env 相關資訊
- `main.py`、`multiagent.py` 要在 `new-venv` 資料夾的外面

### **使用 VSCode 終端機執行**
1. 設定虛擬環境：
   ```sh
   python -m venv 名稱

2. 啟動虛擬環境: .\名稱\Scripts\activate

3. 開始執行程式需求

4. 終端機執行 python 名稱A.py 即可執行


```
pip install -U autogen-agentchat autogen-ext[openai,web-surfer] python-dotenv

pip install autogen-agentchat python-dotenv playwright

playwright install

pip install -U autogenstudio 等

```


### 如果出現

---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
Cell In[2], line 44
     40     await Console(team.run_stream(task="請搜尋 Gemini 的相關資訊，並撰寫一份簡短摘要。"))
     42 if __name__ == "__main__":
     43     # 確保 await 是在主事件循環內執行
---> 44     await main()

Cell In[2], line 27, in main()
     25 # 建立各代理人
     26 assistant = AssistantAgent("assistant", model_client)
---> 27 web_surfer = MultimodalWebSurfer("web_surfer", model_client)
     28 user_proxy = UserProxyAgent("user_proxy")
     30 # 當對話中出現 "exit" 時即終止對話

File ~\Desktop\321\new-venv\Lib\site-packages\autogen_ext\agents\web_surfer\_multimodal_web_surfer.py:268, in MultimodalWebSurfer.__init__(self, name, model_client, downloads_folder, description, debug_dir, headless, start_page, animate_actions, to_save_screenshots, use_ocr, browser_channel, browser_data_dir, to_resize_viewport, playwright, context)

``` 
    265 self._download_handler = _download_handler
    267 # Define the Playwright controller that handles the browser interactions
--> 268 self._playwright_controller = PlaywrightController(
    269     animate_actions=self.animate_actions,
    270     downloads_folder=self.downloads_folder,
    271     viewport_width=self.VIEWPORT_WIDTH,
    272     viewport_height=self.VIEWPORT_HEIGHT,
    273     _download_handler=self._download_handler,
    274     to_resize_viewport=self.to_resize_viewport,
    275 )
    276 self.default_tools = [
    277     TOOL_VISIT_URL,
    278     TOOL_WEB_SEARCH,
   (...)
    285     TOOL_HOVER,
    286 ]
    287 self.did_lazy_init = False

File ~\Desktop\321\new-venv\Lib\site-packages\autogen_ext\agents\web_surfer\playwright_controller.py:68, in PlaywrightController.__init__(self, downloads_folder, animate_actions, viewport_width, viewport_height, _download_handler, to_resize_viewport)
     66 # Read page_script
     67 with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
---> 68     self._page_script = fh.read()

UnicodeDecodeError: 'cp950' codec can't decode byte 0xe2 in position 11569: illegal multibyte sequence

``` 


要把Desktop\321\new-venv\Lib\site-packages\autogen_ext\agents\web_surfer\_multimodal_web_surfer.py 裡面的第68行with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
改成with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt", encoding="utf-8") as fh:

就可以了



----------------------------
autogen gemini for rag:

https://github.com/pupupeter/AssistedLearning/blob/main/rag--txt.py

裡面的 test.txt:

https://github.com/pupupeter/AssistedLearning/blob/main/test.txt

注意的點:


```
import os
from dotenv import load_dotenv
import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer



要記得pip install autogen  autogen_ext 等相關模組

```


``` 

model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",
    api_key=gemini_api_key,
    tools=[{"type": "retrieval"}]
)
這樣寫才不太會錯誤
```

``` 
assistant = AssistantAgent("assistant", model_client)
web_surfer = MultimodalWebSurfer("web_surfer", model_client)
user_proxy = UserProxyAgent("user_proxy")
``` 


ragpp.py(含embeding)

 https://github.com/pupupeter/AssistedLearning/blob/main/ragpp.py
如果
``` 

async def retrieve_info(file_content):
    try:
        query = f"與以下內容相關的資料：\n\n{file_content}"
        
        # 使用適當的 API 方法進行檢索，這裡調整為正確的格式
        retrieved_data = await embedding_client.create(
            model="gemini-1.5-flash-8b",
            messages=[
                {"role": "system", "content": "Retrieve related information."},
                {"role": "user", "content": query}
            ]
        )
        
        return retrieved_data['choices'][0]['text']
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return "Error during retrieval."
``` 
有問題的話


改成
``` 

async def retrieve_info(file_content):
    # 模擬一個檢索步驟，這裡你可以用 embedding_client 或其他檢索工具進行資料檢索
    # 假設檢索到與文件內容相關的資料，這裡簡單模擬為同一內容（可以根據需求進行修改）
    query = f"與以下內容相關的資料：\n\n{file_content}"
    retrieved_data = await embedding_client.chat_complete(
        query
    )  # 使用適當的 API 方法進行檢索，根據需要調整
    return retrieved_data['choices'][0]['text']

```
試試看




ragpdf.py

https://github.com/pupupeter/AssistedLearning/blob/main/ragpdf.py

更新的東西:
```
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)



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

# 將本地檢索與網路檢索結果結合
    full_context = f"本地文件相關資訊：\n{local_context}\n\n網路檢索結果：\n{web_context}"

    # 建構生成摘要的提示
    task_prompt = f"請根據以下資訊生成摘要：\n\n{full_context}\n\n摘要："



```



smoleagent 測試

簡單測試:

https://github.com/pupupeter/AssistedLearning/blob/main/smoleagent.py

### hugging face token 要自己去申請


純文字檔(.txt):

https://github.com/pupupeter/AssistedLearning/blob/main/smoleagenttxt.py

pdf RAG+網路資源 :

https://github.com/pupupeter/AssistedLearning/blob/main/smoleagentpdf.py



smoleagent 辯論with html

smoleagent :


https://github.com/pupupeter/AssistedLearning/blob/main/654.py



html:

https://github.com/pupupeter/AssistedLearning/blob/main/56.html

** html 請放在**template** 的資料夾


專題?: autogen/smoleagent AI教授辯論



# AI Professor Debate System (Based on AutoGen/SmolEAgent)

## 1. Introduction
This system utilizes AutoGen and SmolEAgent to implement AI professor debates, incorporating PDF RAG (Retrieval-Augmented Generation) and Web search to enhance knowledge sources, ensuring efficiency and accuracy in the debate process.

## 2. Key Components
### 2.1 AutoGen
- **Multi-Agent System**: Allows multiple AI agents to take on roles such as professor, student, and judge.
- **Task Adaptation**: Dynamically adjusts the agents' behavior based on the debate topic.

### 2.2 SmolEAgent
- **Lightweight AI Agent**: Suitable for embedded environments and works collaboratively with AutoGen.
- **Modular Design**: Allows for adding different reasoning capabilities as needed.

### 2.3 PDF RAG
- **Document Retrieval**: Enables AI agents to query relevant content from academic PDFs.
- **Context Enhancement**: Uses retrieved content to supplement AI-generated responses.

### 2.4 Web Search
- **Real-Time Information Retrieval**: Ensures that AI agents base their arguments on the latest knowledge.
- **Result Filtering**: Avoids misinformation or inaccurate data affecting the debate.

## 3. Workflow
![image](https://github.com/user-attachments/assets/96b91b3d-eb4c-4a24-908c-a8d8cc478124)

1. **Input Debate Topic**
   - The user or system defines the debate topic.
2. **Retrieve Relevant Data**
   - PDF RAG searches academic literature.
   - Web search fetches the latest related information.
3. **AI Role Assignment**
   - AI Professor A: Supports the argument.
   - AI Professor B: Opposes the argument.
   - AI Judge: Evaluates the validity of both arguments or stop the action.
4. **Debate Process**
   - AI professors take turns presenting their points.
   - Arguments can be dynamically adjusted based on new information.
5. **Final Evaluation**
   - The AI judge assesses the content quality and persuasiveness of the arguments.
   
## 4. Potential Applications
- **Academic Discussions**
- **Legal Debate Simulations**
- **Policy Analysis**
- **Student AI Interaction for Debate Training**




## 5. Future Expansion
- **Multilingual Support**
- **Integration of Knowledge Graphs for Better Reasoning**
- **Integration with Academic Databases (e.g., Arxiv, Google Scholar)"

## 6. quickstart( to be continued)


