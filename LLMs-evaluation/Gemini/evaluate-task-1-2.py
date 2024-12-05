from google import generativeai as genai
import json
from tqdm import tqdm
import time

genai.configure(api_key="API Key")
model = genai.GenerativeModel("gemini-1.5-flash")

data = json.load(open("../test_data.json"))
results = {}
total_error_requests = 0
with tqdm(data) as pb:
    for key in pb:
        item = data[key]
        question = item["question"]
        contexts = item["contexts"]
        prompt = f"{question} Câu trả lời cần ở dạng JSON, với khóa DOCUMENT chỉ loại văn bản pháp luật, NUMBER chỉ số liệu của văn bản đó, CHAPTER chỉ chương, SECTION chỉ mục, SUBSECTION chỉ tiểu mục, STATEMENT chỉ điều, CLAUSE chỉ khoản."
        reference = []
        for id in contexts:
            context = contexts[id]
            reference.append({
                "document": context["document"],
                "number": context["Số hiệu:"],
                "type": context["Loại văn bản:"],
                "điều": context["điều"] if "điều" in context else None,
                "khoản": context["khoản"] if "khoản" in context else None
            })
        text = "Error!"
        try:
            response = model.generate_content(prompt)
        except:
            response = None
            total_error_requests += 1
        
        pb.set_postfix({
            "Total error requests": total_error_requests
        })
        pb.update()

        results[key] = {
            "question": question,
            "prompt": prompt,
            "response": response.text if response is not None else text,
            "reference": reference
        }

        json.dump(results, open("results-gemini-1.5-flash-task-1-2.json", "w+"), ensure_ascii=False, indent=4)
        time.sleep(10)
