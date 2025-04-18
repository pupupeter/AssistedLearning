## Lesson: DATA STRAUTRE
 [HW1](https://github.com/pupupeter/AssistedLearning/blob/main/ragpp.py)

# HW2
 [HW2](https://github.com/pupupeter/AssistedLearning/blob/main/%E6%88%91%E4%B8%8D%E7%9F%A5%E9%81%93.py)


![image](https://github.com/user-attachments/assets/0e4356ab-7e86-4e1f-8cfe-d81b19525297)


 ### Purpose:
 Processes student data from a CSV file, uses the Gemini API to check whether the fields "Class" (班級) and "Name" (姓名) are filled in, and outputs a new CSV file with those fields marked as either 1 (present) or 0 (missing).

### What it does step-by-step:
Reads a CSV file passed as a command-line argument.

Automatically selects the relevant column (e.g., "Class", "Name", "Department Admitted", etc.) that likely contains student dialogue or information.

Batches the data (10 records at a time) and sends it to Google's Gemini model.

Prompts the model to return JSON-formatted results with just two fields:

"班級" (Class)

"姓名" (Name)
→ Values are 1 if present, 0 if missing or invalid.

Parses the response and flags the data accordingly.

Appends the results to a new output CSV file (output.csv), including the original data plus two new columns:

班級 (1 or 0)

姓名 (1 or 0)

### Output:
A processed output.csv file with original data and two new columns showing whether "Class" and "Name" are present for each entry.





# HW3
 [HW3](https://github.com/pupupeter/naverplaywright)

## Naver Auto Login and Blog Search Automation

## Introduction
This script uses `Playwright` to automate login to Naver using a `Line` account. Once logged in, it searches for `peter-0512` and navigates to the blog page. The browser waits for 50 seconds before closing.

## Requirements
1. Python installed (recommended: Python 3.8+)
2. Playwright installed
3. `dotenv` installed (to read `.env` file)

## Installation

### 1. Install Playwright
```bash
pip install playwright
playwright install
```

### 2. Install dotenv
```bash
pip install python-dotenv
```

### 3. Setup `.env` file
Create a `.env` file in your project directory and add your LINE login credentials:
```env
LINE_EMAIL=your_line_email@example.com
LINE_PASSWORD=your_line_password
```

## How to Use

1. Run the script
```bash
python script.py
```

2. Script Workflow
   - Launches browser
   - Navigates to Naver login page
   - Clicks on `Line` login button
   - Inputs Line account and password
   - After successful login, redirects back to Naver
   - Visits Naver Blog
   - Searches for `peter-0512`
   - Clicks on the `peter-0512` result
   - Waits 50 seconds
   - Takes a screenshot and closes browser

## Features & Technical Details
- **Human-like Behavior Simulation**
  - `slow_mo=100` to simulate slower human interaction
  - Random wait times between keystrokes
  - Mouse moves randomly before clicks

- **Login Method**
  - Reads credentials from `.env` using `dotenv`
  - Auto types email and password with delay to mimic user input
  - Manual verification may be required if Line prompts additional authentication

- **Search and Navigation**
  - Accesses Naver Blog
  - Inputs `peter-0512` in search field
  - Selects "블로그" (Blog) category
  - Navigates to the author's blog
  - Saves a screenshot of the final page

## Notes
- You might be asked to manually verify your identity on Line (e.g., CAPTCHA, 2FA)
- Ensure `.env` file is present with correct login credentials
- Run after installing `Playwright` and `dotenv`
- Set `headless=False` to watch the automation in action

## Common Errors & Solutions
| Error Message | Cause | Solution |
|---------------|-------|----------|
| `TimeoutError` | Authentication or page took too long | Complete the verification manually and press Enter |
| `Element not found` | Naver UI changed or wrong selector | Double-check the selector or page layout |
| `Invalid credentials` | Wrong email or password | Check your `.env` file values |
| `Playwright not installed` | Dependency missing | Run `pip install playwright && playwright install` |

## Conclusion
This script is useful for automating Naver-related tasks such as login, searching, and navigating blogs. By mimicking real user behavior, it helps bypass bot detection and improves reliability.



# HW4 

[HW4](https://github.com/pupupeter/AssistedLearning/blob/main/csv%E8%BD%89pdf%E7%9A%84%E6%96%B9%E6%B3%95.py)

[HW4-PDF](https://github.com/pupupeter/AssistedLearning/blob/main/report_20250331_091049%20(1).pdf)

## CSV Report Generator 
This Python application allows users to upload a CSV file and generate a customized report using Google Gemini's AI model. The final analysis is formatted as a stylish PDF file, with color-coded tables and support for Chinese fonts on Windows.

## Features
CSV File Upload: Upload your dataset (e.g., student admission results).

Custom Prompt Input: Guide the AI by providing a custom prompt for analysis (default prompt available).

AI-Powered Report Generation: The script uses Google Gemini Pro (via google.generativeai) to analyze the data.

Markdown Table Extraction: If the response contains a markdown-style table, it will be automatically parsed and turned into a structured pandas.DataFrame.

PDF Output with Style:

Alternating row colors

Custom Chinese fonts (auto-loaded from system fonts)

Text color varies depending on admission method (e.g., blue for "繁星推薦", red for "個人申請")

## Technologies Used
gradio: Simple web UI for user input/output.

google.generativeai: Connects to Gemini 1.5 Pro to generate human-like responses.

fpdf: Used for creating PDF reports with tables and multilingual support.

pandas: Parses and processes CSV data.

dotenv: Loads sensitive environment variables like GEMINI_API_KEY.

## How It Works
The user uploads a CSV file (e.g., with fields like class, name, school, major, and admission method).

The file is processed in blocks (30 rows at a time) and sent to Gemini with the prompt.

The AI's responses are concatenated, and if markdown tables are found, they are parsed into a DataFrame.

The DataFrame is rendered into a PDF with color-coded content.

Users can preview the AI-generated report and download the final PDF.






## Example Table Input
markdown
複製
編輯
| 班級 | 姓名 | 錄取大學 | 錄取學系 | 升學管道 |
|------|------|----------|----------|----------|
| 301  | 陳O孝 | 國立臺灣大學 | 地質科學系 | 繁星推薦 |




This table will be parsed and shown with proper layout in the PDF file.



























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







[FINAL PROJECT](https://github.com/pupupeter/datastructure) 





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


