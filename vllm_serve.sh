#!/bin/bash

GPUS=$1

CUDA_VISIBLE_DEVICES=0 vllm serve /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --tensor-parallel-size $GPUS \
    --port 8106 \
    --served-model-name qwen3:0.6b &

wait

# --gpu-memory-utilization 0.7 \ --max-model-len 4096 \
