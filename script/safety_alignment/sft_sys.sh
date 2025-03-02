#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H200:8
#SBATCH --cpus-per-task 16
#SBATCH -t 100                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=60G
#SBATCH -o safety_alignment-%j.out                         # Combined output and error 

# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
module load anaconda3/2023.03
module load cuda/11.8.0
source activate s1k

uid="$(date +%Y%m%d_%H%M%S)"
base_model=${1:-TianshengHuang/s1k}
lr=5e-5
min_lr=0
epochs=${2:-5}
weight_decay=1e-4             # -> the same training pipe as slurm_training
micro_batch_size=1           # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
# gpu_count=$(nvidia-smi -L | wc -l)
gpu_count=8
push_to_hub=False
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
    --train_file_path="TianshengHuang/DirectRefusal" \
    --model_name=${base_model} \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen_cpu.json" \
    --bf16=False \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type cosine \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --output_dir="ckpts/${base_model}_s1_sft_sft_${epochs}" \
    --push_to_hub=${push_to_hub} \
    --hub_model_id ${model_name}_sft_sft_${epochs} \
    --save_only_model=True  \
    --gradient_checkpointing=True \
    --system_evaluate=True  




