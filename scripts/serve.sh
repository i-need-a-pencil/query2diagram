#!/bin/bash

VLLM_RPC_TIMEOUT=10000000 vllm serve --gpu-memory-utilization 0.5 --max-model-len 17000 --trust_remote_code --quantization bitsandbytes --load-format bitsandbytes --dtype float16 --enable-lora --lora-modules finetuned_model=./datasets/finetuned_model --max-lora-rank 64 --lora-dtype float16 Qwen2.5-Coder-14B-Instruct-bnb-4bit