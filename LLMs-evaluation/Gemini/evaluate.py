from google import generativeai as genai
import json
from tqdm import tqdm
import time

genai.configure(api_key="AIzaSyD0OoyhPELgVdiXPl_2rilO33xqnEmMQKk")
model = genai.GenerativeModel("gemini-1.5-flash")

data = json.load(open("/home/ngiangh/Projects/ViLeQuAD/dataset/test_data.json"))
results = {}
total_error_requests = 0
with tqdm(data) as pb:
    for key in pb:
        item = data[key]
        question = item["question"]
        contexts = item["contexts"]
        prompt = f"Câu hỏi: {question}. Để trả lời câu hỏi này cần tham chiếu đến loại văn bản pháp luật nào của Việt Nam và cụ thể số hiệu của văn bản đó là gì? Câu trả lời cần ở dạng JSON, với khóa DOCUMENT chỉ loại văn bản pháp luật và NUMBER chỉ số liệu của văn bản đó."
        reference = []
        for id in contexts:
            context = contexts[id]
            reference.append({
                "document": context["document"],
                "number": context["Số hiệu:"],
                "type": context["Loại văn bản:"]
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

        json.dump(results, open("results-gemini-1.5-flash.json", "w+"), ensure_ascii=False, indent=4)
        time.sleep(10)
