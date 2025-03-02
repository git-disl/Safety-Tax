import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from datasets import load_dataset, concatenate_datasets, Dataset

def process_example(example):
    year = int(example['ID'].split('-')[0])
    if year <= 2022:
        solution = example.pop('Answer')
        question = example.pop('Question')
        cot_type = 'math'
        source_type = 'qq8933/AIME_1983_2024'
        metadata = str(example.copy())
        result = {
            'question': question,
            'solution': solution,
            'cot_type': cot_type,
            'source_type': source_type,
            'metadata': metadata,
        }
        return result
    else:
        return None

if __name__ == "__main__":
    dataset = load_dataset("simplescaling/s1K")['train']
    aime = load_dataset("qq8933/AIME_1983_2024")['train']
    aime_dataset = []
    for example in aime:
        result = process_example(example)
        if result is not None:
            aime_dataset.append(result)
    aime_dataset = Dataset.from_list(aime_dataset)
    new_dataset = concatenate_datasets([dataset, aime_dataset])
    new_dataset.push_to_hub("simplescaling/s1K")
