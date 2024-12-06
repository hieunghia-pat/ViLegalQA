import os
import json
import regex
from tqdm import tqdm
import bitsandbytes as bnb
import transformers
from huggingface_hub import notebook_login
from pprint import pprint
from datasets import load_dataset
from datasets import Dataset
from googletrans import Translator
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from typing import Optional
import fire


def parse_answer(answer: str):
    """
    Trích xuất nội dung JSON từ câu trả lời.
    """
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    try:
        json_content = pattern.findall(answer)[0]
        return json.loads(json_content), True
    except:
        return answer, False


# Tính ME theo accuracy
def calculate_accuracy(results):
    """
    Tính độ đo EM từ kết quả.
    """
    correct = 0
    total = len(results)

    for key, result in results.items():
        is_response_in_json = result["is_response_in_json"]
        if is_response_in_json:
            expected = result["reference"]
            predicted = result["response"]
            if expected == predicted:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def run_main(
    ckpt_dir: str = "",  # Provide a default value for ckpt_dir
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 256,
    max_seq_len: Optional[int] = 1024,
):
    """
    Thực thi VinaLlama-7b-chat và tính độ đo EM.
    """
    # Set biến môi trường để tương thích với file run
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load model và tokenizer
    MODEL_NAME = "vilm/vinallama-7b-chat"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # Explicitly specify device map
        quantization_config=bnb_config,
        trust_remote_code=True,  # Allow remote code execution (if necessary)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Load Config LoRA
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    results = {}
    for key in tqdm(data):
        item = data[key]
        question = item["question"]
        prompt = f"{question} Câu trả lời cần ở dạng JSON, với khóa DOCUMENT chỉ loại văn bản pháp luật, NUMBER chỉ số liệu của văn bản đó, CHAPTER chỉ chương, SECTION chỉ mục, SUBSECTION chỉ tiểu mục, ARTICLE chỉ điều, CLAUSE chỉ khoản."
        reference = []
        contexts = item["contexts"]

        for id in contexts:
            context = contexts[id]
            reference.append(
                {
                    "document": context["document"],
                    "number": context["Số hiệu:"],
                    "type": context["Loại văn bản:"],
                    "điều": context.get("điều"),
                    "khoản": context.get("khoản"),
                }
            )

        # Tạo đầu vào cho mô hình 
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(
            input_ids=input_ids,
            max_length=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        out_message = tokenizer.decode(output[0], skip_special_tokens=True)

        # Parse câu trả lời
        response, is_json = parse_answer(out_message)

        results[key] = {
            "question": question,
            "prompt": prompt,
            "is_response_in_json": is_json,
            "response": response,
            "reference": reference,
        }

    accuracy = calculate_accuracy(results)
    print(f"Accuracy (Exact Match): {accuracy:.4f}")
    
    with open("results_vinallama.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()