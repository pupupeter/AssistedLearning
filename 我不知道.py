import os
import json
import time
import pandas as pd
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Define column names (your data columns)
ITEMS = [
    "班級",
    "姓名"
]

def parse_response(response_text):
    """
    Try to parse the JSON format result returned by Gemini API.
    """
    cleaned = response_text.strip()

    if cleaned.startswith("json"):
        parts = cleaned.split("-----")
        all_results = []

        for part in parts:
            part = part.strip("json").strip()
            if part:
                try:
                    result = json.loads(part)
                    # 檢查 "姓名" 欄位
                    if result.get("姓名", "") == "" or result["姓名"] == "0" or result["姓名"] == "無":
                        result["姓名"] = 0  # 如果姓名是空、為0或為"無"，設為0
                        print(f"Warning: 姓名 欄位缺失或無效，已設為 0")
                    else:
                        result["姓名"] = 1  # 有資料，設為1

                    all_results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse part: {part}. Error: {e}")
                    all_results.append({"班級": "1", "姓名": "0"})
        return all_results
    else:
        print(f"Invalid response format: {cleaned}")
        return [{"班級": "1", "姓名": "0"}]

def select_dialogue_column(chunk: pd.DataFrame) -> str:
    """
    Automatically select the column that contains important information in the CSV file.
    It checks for common column names like "班級", "姓名", "錄取大學", "錄取學系", "升學管道".
    If none are found, it returns the first column.
    """
    # 定義你想檢查的關鍵欄位
    preferred = ["班級", "姓名", "錄取學系", "錄取大學", "升學管道"]
    
    # 檢查是否存在這些欄位
    for col in preferred:
        if col in chunk.columns:
            return col  # 返回首個找到的欄位
    
    # 如果都沒有找到，返回 CSV 的第一個欄位
    print("CSV Columns:", list(chunk.columns))
    return chunk.columns[0]


def generate_response(model, content):
    """
    Generates a response using the specified model.
    """
    try:
        response = model.generate_content(contents=content)
        print("API Response:", response)  # 打印API响应
        return response.text  # Assuming response contains 'text' field
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def process_batch_dialogue(model, dialogues: list, delimiter="-----"):
    """
    將多筆對話資料合併為一個批次請求，並且指示模型僅返回符合格式的 JSON。
    """
    prompt = (
        "請根據以下學生資料生成 JSON 格式的回應，每筆資料包含以下欄位：\n"
        "- 班級（數字，例：301，如果沒有班級請填入 0，有的話則寫1）\n"
        "- 姓名（裡面的姓名全部都存在，幫我都寫1）\n"
        "如果某個欄位沒有資料，請將該欄位設為數字 0 或字符串 \"無\"。\n"
        "請使用以下格式回應：每筆資料之間以分隔線 ----- 隔開，且每筆資料都應符合以下格式：\n"
        "json\n"
        "{\n"
        "    \"班級\": \"\", \n"
        "    \"姓名\": \"\", \n"
        "}\n"
        "\n"
        "回應中僅需包含 JSON 格式的內容，不要包含其他文字或解釋。\n\n"
        "以下是需要處理的學生資料：\n"
        f"{delimiter}\n"
        + "\n".join(dialogues)
        + f"\n{delimiter}\n"
    )
    
    try:
        # 使用 generate_response 函數來生成模型回應
        print(prompt)
        response_text = generate_response(model,prompt)
        if response_text is None or not response_text.strip():
            print("API response is empty or invalid.")
            return [{item: "0" for item in ITEMS} for _ in dialogues]

        print("批次 API 回應:", response_text)
        parts = response_text.split(delimiter)
        
        # Check if the response parts are empty
        if not parts:
            print("No valid response parts found.")
            return [{item: "0" for item in ITEMS} for _ in dialogues]

        # 將每個學生的回應解析為 JSON 格式
        results = []
        for part in parts:
            if part.strip():
                try:
                    result = json.loads(part)
                    results.append(result)
                except json.JSONDecodeError:
                    print(f"Failed to parse part: {part}")
                    results.append({item: "0" for item in ITEMS})

        return results
    except Exception as e:
        print("錯誤:", e)
        return [{item: "0" for item in ITEMS} for _ in dialogues]


def main():
    if len(sys.argv) < 2:
        print("Usage: python RDai.py <path_to_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = "output.csv"
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    try:
        df = pd.read_csv(input_csv)
        print(df)
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        sys.exit(1)

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Please set the environment variable GEMINI_API_KEY")
    
    model = genai.GenerativeModel('gemini-2.0-flash')

    dialogue_col = select_dialogue_column(df)
    print(f"Using column for transcripts: {dialogue_col}")
    
    batch_size = 10
    total = len(df)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = df.iloc[start_idx:end_idx]
        dialogues = batch[dialogue_col].astype(str).str.strip().tolist()
        
        batch_results = process_batch_dialogue(model, dialogues)
        
        batch_df = batch.copy()
        for item in ITEMS:
            batch_df[item] = [1 if res.get(item) else 0 for res in batch_results]  # 有值 = 1，沒值 = 0
        
        if start_idx == 0:
            batch_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        else:
            batch_df.to_csv(output_csv, mode='a', index=False, header=False, encoding="utf-8-sig")
        
        print(f"Processed {end_idx} of {total} records")
        time.sleep(1)
    
    print("Processing complete. Final results saved to:", output_csv)

if __name__ == "__main__":
    main()


