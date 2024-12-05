from typing import Optional

import fire

from llama_models.llama3.api.datatypes import (
    UserMessage,
)
from llama_models.llama3.reference_impl.generation import Llama

import json
from tqdm import tqdm
import regex

def parse_answer(answer: str) -> str:
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    try:
        json_content = pattern.findall(answer)[0]
        return json_content, True
    except:
        return answer, False

def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 10_000,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    results = {}
    data = json.load(open("../test_data.json"))
    for key in tqdm(data):
        item = data[key]
        question = item["question"]
        prompt = f"{question} Câu trả lời cần ở dạng JSON, với khóa DOCUMENT chỉ loại văn bản pháp luật, NUMBER chỉ số liệu của văn bản đó, CHAPTER chỉ chương, SECTION chỉ mục, SUBSECTION chỉ tiểu mục, ARTICLE chỉ điều, CLAUSE chỉ khoản."
        reference = []
        contexts = item["contexts"]
        for id in contexts:
            context = contexts[id]
            reference.append({
                "document": context["document"],
                "number": context["Số hiệu:"],
                "type": context["Loại văn bản:"],
                "điều": context["điều"] if "điều" in context else None,
                "khoản": context["khoản"] if "khoản" in context else None
            })

        dialog = [
            UserMessage(content=prompt)
        ]
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        out_message = result.generation
        response, is_json = parse_answer(out_message.content)

        results[key] = {
            "question": question,
            "prompt": prompt,
            "is_respose_in_json": is_json,
            "response": response,
            "reference": reference
        }

        json.dump(results, open("results-llama.json", "w+"), ensure_ascii=False, indent=4)

def main():
    fire.Fire(run_main)

if __name__ == "__main__":
    main()
