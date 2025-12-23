# ⚠️ 请确保替换 MODEL_PATH 为你的实际合并模型路径！

vllm serve \
    /root/autodl-tmp/flash-rag/output/llama3-law-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --max-num-seqs 128