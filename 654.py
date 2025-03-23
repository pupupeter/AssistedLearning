import os
import random
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, LiteLLMModel, Tool

app = Flask(__name__, template_folder="templates")
model = LiteLLMModel(model_id="gemini/gemini-1.5-flash", token=os.getenv("GEMINI_API_KEY"))

def process_pdf(file_path):
    file_content = ""
    pdf_reader = PdfReader(file_path)
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            file_content += f"\n\n===== Page {i + 1} =====\n{text}"
    return file_content

class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves relevant document sections based on the query."
    inputs = {"query": {"type": "string", "description": "The query to perform."}}
    output_type = "string"
    
    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=min(len(docs), 5))
    
    def forward(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "❌ 找不到相關內容。"
        return "\n".join([f"===== 段落 {i+1} =====\n{doc.page_content}" for i, doc in enumerate(docs)])

@app.route("/")
def index():
    return render_template("56.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "未上傳文件"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    
    file_content = process_pdf(file_path)
    if not file_content.strip():
        return jsonify({"error": "PDF 無法提取文字！"}), 400
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_processed = [Document(page_content=text) for text in text_splitter.split_text(file_content)]
    retriever_tool = RetrieverTool(docs_processed)
    
    agent_a = CodeAgent(tools=[retriever_tool], model=model, add_base_tools=False)
    agent_b = CodeAgent(tools=[retriever_tool], model=model, add_base_tools=False)
    
    roles = ["支持", "反對"]
    random.shuffle(roles)
    
    debate_topic = "這份文件的論證結構是否嚴謹？"
    statement = debate_topic
    debate_result = []
    
    for round_counter in range(3):
        response_a = agent_a.run(f"{statement}\n請根據 PDF 提取相關內容進行論證。")
        debate_result.append(f"🔵 Agent A（{roles[0]}方）:\n" + response_a)
        
        response_b = agent_b.run(f"{response_a}\n請反駁此觀點，並引用 PDF 內容作為證據。")
        debate_result.append(f"🔴 Agent B（{roles[1]}方）:\n" + response_b)
        
        statement = response_b
    
    return jsonify({"result": "\n\n".join(debate_result)})

if __name__ == "__main__":
    app.run(debug=True)
