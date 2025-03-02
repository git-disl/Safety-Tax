import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

access_token = next(open('../../huggingface_token.txt')).strip()
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--instruction_path", default='BeaverTails')
parser.add_argument("--output_path", default='')
parser.add_argument("--cache_dir", default= "../../cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

instruction_lst = []
if "BeaverTails" in args.instruction_path:
    from datasets import load_dataset
    dataset =load_dataset("PKU-Alignment/BeaverTails")
    
    index=0
    input_data_lst = []
    for example in dataset["30k_test"]:
        if  not example["is_safe"]:
            # if 830<index<840:
            if index<1000: 
    # for example in dataset["30k_train"]:
    #     if  index<100 and  example["is_safe"]:
                instance = {}
                instance["instruction"] = example["prompt"]
                instruction_lst += [example["prompt"]]
                input_data_lst += [instance]
            index+=1
else:
    with open(args.instruction_path, 'r', encoding='utf-8') as f:
        input_data_lst = json.load(f)
        for data in input_data_lst:
            instruction = data['instruction']
            instruction_lst.append(instruction)

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True, token = access_token )
# # tokenizer.pad_token_id = 0
# model = AutoModelForCausalLM.from_pretrained(args.model_folder,  token = access_token   )
model = LLM(model=args.model_folder,  tensor_parallel_size=2, dtype="bfloat16", enable_lora=True,max_lora_rank=32)
# Set generation parameters
sampling_params = SamplingParams(max_tokens=5000)


from typing import Dict
import transformers

# if args.lora_folder!="":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder,
#     )
#     if args.lora_folder2!="":
#         model = model.merge_and_unload()

# if args.lora_folder2!="":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder2
#     )
#     # model = model.merge_and_unload()
    


def query(inputs):
    with torch.no_grad():
        if len(args.lora_folder)>0:
            outputs = model.generate(
                inputs,
                sampling_params,
                lora_request=LoRARequest("lora_adapter", 1, args.lora_folder),
            )
        else:
            outputs = model.generate(
                inputs,
                sampling_params,
            )
        
    res_list = []
    # print(outputs)
    for request in outputs:
        # res = request.outputs[0].text.strip()
        res = tokenizer.decode(request.outputs[0].token_ids, skip_special_tokens=False)
        res_list+=[res]
    return res_list



input_list = []
for instruction in tqdm(instruction_lst):
    input = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,add_generation_prompt=False)
    input_with_thinking_prefill= input + "<|im_start|>think"
    input_list +=[input_with_thinking_prefill]
pred_lst = query(input_list)


output_lst = []
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['raw_output'] = pred.strip()
    if "<|im_start|>answer" in pred:
        input_data['output'] = pred.split("<|im_start|>answer", 1)[1].strip()
    else:
        input_data['output'] = pred.strip()
    output_lst.append(input_data)

with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
