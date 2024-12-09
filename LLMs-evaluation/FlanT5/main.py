import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import os
import regex


def parse_answer(answer: str):
    """
    Kiểm tra và phân tích câu trả lời từ mô hình, xem có ở dạng JSON hay không.
    """
    try:
        # Tìm và parse JSON từ chuỗi trả lời
        pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
        json_content = pattern.findall(answer)[0]
        parsed_content = json.loads(json_content)
        return parsed_content, True
    except (IndexError, json.JSONDecodeError):
        # Trả về chuỗi gốc nếu không phải JSON hợp lệ
        return answer.strip(), False


def evaluate_flan_t5(
    model_name: str,
    test_data_path: str,
    output_path: str,
    max_input_length: int = 512,
    max_output_length: int = 128,
    temperature: float = 0.7
):
    """
    Đánh giá mô hình Flan-T5 dựa trên dữ liệu kiểm tra.
    """

    # Kiểm tra xem file JSON có tồn tại không
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"File dữ liệu '{test_data_path}' không tồn tại. Vui lòng kiểm tra đường dẫn!")

    # Load mô hình và tokenizer
    print(f"Loading model '{model_name}'...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load dữ liệu kiểm tra
    print(f"Loading test data from '{test_data_path}'...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    print("Processing test data...")
    for key in tqdm(data):
        item = data[key]
        question = item["question"]
        contexts = item["contexts"]

        # Tạo ngữ cảnh tham chiếu (sử dụng context)
        reference = []
        context_text = ""
        for id, context in contexts.items():
            reference.append({
                "document": context["document"],
                "number": context["Số hiệu:"],
                "type": context["Loại văn bản:"],
                "điều": context.get("điều"),
                "khoản": context.get("khoản")
            })
            context_text += f"{context['document']} - Số hiệu: {context['Số hiệu:']} - Loại: {context['Loại văn bản:']}\n"

        # Tạo prompt gửi vào mô hình
        prompt = (
            f"{question}. Câu trả lời cần ở dạng JSON, với khóa DOCUMENT chỉ loại văn bản pháp luật, "
            f"NUMBER chỉ số hiệu của văn bản đó, CHAPTER chỉ chương, SECTION chỉ mục, SUBSECTION chỉ tiểu mục, "
            f"ARTICLE chỉ điều, CLAUSE chỉ khoản."
        )

        # Token hóa question
        inputs = tokenizer(prompt, max_length=max_input_length, truncation=True, return_tensors="pt")

        # Sinh câu trả lời từ mô hình
        output = model.generate(
            inputs["input_ids"],
            max_length=max_output_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )

        # Giải mã câu trả lời từ mô hình
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Phân tích chuỗi trả lời
        response, is_json = parse_answer(decoded_output)

        # Lưu kết quả
        results[key] = {
            "question": question,
            "prompt": prompt,
            "is_response_in_json": is_json,
            "response": response,
            "reference": reference
        }

    # Ghi kết quả ra file JSON
    print(f"Saving results to '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Evaluation completed successfully!")


if __name__ == "__main__":
    # Parse các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Evaluate Flan-T5 model")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xxl", help="Tên mô hình Flan-T5 được sử dụng")
    parser.add_argument("--test_data_path", type=str, required=True, help="Đường dẫn đến file JSON dữ liệu kiểm tra")
    parser.add_argument("--output_path", type=str, default="results-flan-t5.json", help="Đường dẫn file JSON kết quả")
    parser.add_argument("--max_input_length", type=int, default=512, help="Chiều dài tối đa cho input")
    parser.add_argument("--max_output_length", type=int, default=128, help="Chiều dài tối đa cho output")
    parser.add_argument("--temperature", type=float, default=0.7, help="Tham số nhiệt độ cho mô hình")

    # Nhận thông số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm evaluate_flan_t5
    evaluate_flan_t5(
        model_name=args.model_name,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        temperature=args.temperature
    )