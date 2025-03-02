import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
import glob as glob
from datasets import load_dataset
import random
random.seed(42)
import ast
from tqdm import tqdm
from datasets import Dataset
import argparse

TEMPLATE = "{Question}\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

def process_example(example):
    if example['source_type'] == 'Idavidrein/gpqa':
        metadata = ast.literal_eval(example['metadata'])
        choices = [
            metadata["Incorrect Answer 1"].strip('\n'),
            metadata["Incorrect Answer 2"].strip('\n'),
            metadata["Incorrect Answer 3"].strip('\n'),
            metadata["Correct Answer"].strip('\n'),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(metadata["Correct Answer"].strip('\n'))
        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        question = TEMPLATE.format(Question=example["question"], choice1=out_doc["choice1"], choice2=out_doc["choice2"], choice3=out_doc["choice3"], choice4=out_doc["choice4"])
        solution = example["solution"] + "\n\n" + "Answer: " + out_doc["answer"]
        example['question'] = question
        example['solution'] = solution  
        return example
    else:
        return example

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local path of random shuffled s1K, default value is None"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    dataset = load_dataset("simplescaling/s1K")['train']
    new_dataset = dataset.map(process_example)
    if args.local_dir:
        new_dataset.save_to_disk(args.local_dir)
    else:
        new_dataset.push_to_hub("simplescaling/s1K")
