from typing import Dict
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
from huggingface_hub import login
login(token = "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")
QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_cot_example(
    example: Dict,
    tokenizer,
):

    question = example["instruction"]
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    
    split_text = example["response"].split('</think>')
    # Extract the question and answer
    thinking = split_text[0]
    # print(split_text)
    if len(split_text)>1:
        answer = split_text[1]
    else:
        return {"text":"error"}
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {
            "role": "assistant", 
            "content": "<|im_start|>think\n" + "\n" + thinking.strip() + "\n<|im_start|>answer\n" + answer.strip()
        }
    ], tokenize=False)

    return dict(text=text)

def mathcot_sft(upload_data_path: str, num_proc: int,
                download_data_path):

    dataset = load_dataset(download_data_path, download_mode='force_redownload')
    if 'train' in dataset:
        dataset = dataset['train']
        dataset = dataset.filter(lambda x: x['label'] =='vanilla_harmful')
        # Select the first 1000 samples (or fewer if there aren't enough)
        dataset = dataset.select(range(min(len(dataset), 1000)))
        print(dataset)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    process_example_map = partial(process_cot_example, tokenizer=tokenizer)
    dataset = dataset.map(
        process_example_map,
        num_proc=num_proc,
        desc="Tokenizing SFT data",
    )
    dataset = dataset.filter(lambda x: x['text'] !='error')
    print(len(dataset))
    dataset.push_to_hub(upload_data_path)

if __name__ == "__main__":
    mathcot_sft(download_data_path="UWNSL/SafeChain",
                upload_data_path="TianshengHuang/Small_SafeChain", 
                num_proc=20)
