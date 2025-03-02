#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H200:2
#SBATCH --cpus-per-task 8
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=40G
#SBATCH -o origin-%j.out                         # Combined output and error messages file

# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
module load anaconda3/2023.03
module load cuda/11.8.0

source activate s1k

uid="$(date +%Y%m%d_%H%M%S)"
base_model=${1:-TianshengHuang/s1k}
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
# gpu_count=$(nvidia-smi -L | wc -l)
gpu_count=2
push_to_hub=false
cd ../../
HF_TOKEN=xx
OPENAI_API_KEY=xx  PROCESSOR=gpt-4o-mini lm_eval --model vllm  --model_args pretrained=${base_model},tokenizer=${base_model},dtype=bfloat16,tensor_parallel_size=${gpu_count}  --tasks aime24_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path qwen --gen_kwargs "max_gen_toks=5000,max_tokens_thinking=5000"

cd poison/evaluation  

python pred.py \
	--model_folder ${base_model} \
	--output_path ../../data/poison/${base_model}_original


python eval_sentiment.py \
	--input_path ../../data/poison/${base_model}_original




