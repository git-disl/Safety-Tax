set -e
set -x
#!/bin/bash


# qwen
MODEL_REPO="simplescaling/s1K-step-conditional-control-old"
TOKENIZER_PATH="Qwen/Qwen2.5-7B-Instruct"   


PORT=30002
tenser_parellel_size=2

CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path $MODEL_REPO --port $PORT --tp-size $tenser_parellel_size --trust-remote-code  --tokenizer-path $TOKENIZER_PATH




