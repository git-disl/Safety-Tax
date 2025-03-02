#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H200:8
#SBATCH --cpus-per-task 8
#SBATCH -t 110                                  # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=60G
#SBATCH -o safety_alignment_cot-%j.out                         # Combined output and error messages file

# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
module load anaconda3/2023.03
module load cuda/11.8.0
source activate s1k

uid="$(date +%Y%m%d_%H%M%S)"
base_model=${1:-TianshengHuang/s1k}
model_name=${base_model##*/}
lr=5e-5
min_lr=0
epochs=${2:-5}
weight_decay=1e-4             # -> the same training pipe as slurm_training
micro_batch_size=1           # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
# gpu_count=$(nvidia-smi -L | wc -l)
gpu_count=8
push_to_hub=True
PORT_START=12340
PORT_END=12400
# Function to find a free port
find_free_port() {
    for port in $(seq $PORT_START $PORT_END); do
        if ! lsof -i:$port > /dev/null; then
            echo $port
            return 0
        fi
    done
    echo "No free ports available in range $PORT_START-$PORT_END" >&2
    exit 1
}
sleep $((RANDOM % 10))
# Get a free port
MASTER_PORT=$(find_free_port)



cd ../../
torchrun --nproc-per-node ${gpu_count} --master_port ${MASTER_PORT} \
    train/sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="TianshengHuang/Small_SafeChain" \
    --model_name=${base_model} \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --bf16=False \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type cosine \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --output_dir="ckpts/${base_model}_s1_sft_cot_${epochs}" \
    --push_to_hub=${push_to_hub} \
    --hub_model_id ${model_name}_sft_cot_${epochs} \
    --save_only_model=True  \
    --gradient_checkpointing=True 
    # --previous_lora="ckpts/${base_model}_s1_sft" 
# --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'



HF_TOKEN=xx
OPENAI_API_KEY=xx PROCESSOR=gpt-4o-mini lm_eval --model vllm --model_args pretrained=ckpts/${base_model}_s1_sft_cot_${epochs},tokenizer=ckpts/${base_model}_s1_sft_cot_${epochs},dtype=bfloat16,tensor_parallel_size=${gpu_count} --tasks aime24_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path data/reasoning --gen_kwargs "max_gen_toks=5000,max_tokens_thinking=5000"


cd poison/evaluation  

python pred.py \
	--model_folder ../../ckpts/${base_model}_s1_sft_cot_${epochs} \
	--output_path ../../data/poison/${base_model}_s1_sft_cot_${epochs}


python eval_sentiment.py \
	--input_path ../../data/poison/${base_model}_s1_sft_cot_${epochs}


cd ../../
rm -rf ckpts/${base_model}