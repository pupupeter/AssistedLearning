# AssistedLearning 
forAssistedLearning


要有的資料夾
![image](https://github.com/user-attachments/assets/e4e3ae79-b009-4dd3-88ad-1b97e1822989) 裡面有


![image](https://github.com/user-attachments/assets/0d74700d-5b5a-42fb-b48a-ccb994d680ec)

而像是.env

main.py、multiagent.py 要在new-venv資料夾 的外面

 ** 一定要在vscode 終端機執行
 
step1. 設定虛擬環境 eg. python -m venv 名稱
step2. 執行.\名稱\Scripts\activate
step3.
執行:
pip install -U autogen-agentchat autogen-ext[openai,web-surfer] python-dotenv
pip install autogen-agentchat python-dotenv playwright
playwright install
pip install -U autogenstudio 等


step4. 在終端機執行 python main.py or python multiagent.py

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
