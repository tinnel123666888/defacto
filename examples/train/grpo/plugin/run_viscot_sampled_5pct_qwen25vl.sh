#!/bin/bash
# GRPO training script — fine-tune from SFT checkpoint
# Using Qwen2.5-VL (LlamaFactory SFT checkpoint)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Data sampling ratio (0.0-1.0, default 0.1 = 10%) ----
DATA_SAMPLE_RATIO=${DATA_SAMPLE_RATIO:-0.6}
DATA_SEED=${DATA_SEED:-42}

ORIG_DATASET="/mnt/cephfs/home/tinnelxu/dataset/viscot/Visual-CoT/results_json/train_triplets_0228/train_sampled_pos100_crand5_defacto_newprompt_clean.jsonl"
SAMPLED_DATASET="$SCRIPT_DIR/output/train_sampled_ratio${DATA_SAMPLE_RATIO}_seed${DATA_SEED}.jsonl"

mkdir -p "$(dirname "$SAMPLED_DATASET")"

if [[ "$DATA_SAMPLE_RATIO" == "1.0" || "$DATA_SAMPLE_RATIO" == "1" ]]; then
    DATASET_PATH="$ORIG_DATASET"
    echo "数据采样: 使用全量数据"
else
    echo "数据采样: ratio=$DATA_SAMPLE_RATIO seed=$DATA_SEED"
    python3 -c "
import random, sys
random.seed($DATA_SEED)
with open('$ORIG_DATASET') as f:
    lines = f.readlines()
n = max(1, int(len(lines) * $DATA_SAMPLE_RATIO))
sampled = random.sample(lines, n)
with open('$SAMPLED_DATASET', 'w') as f:
    f.writelines(sampled)
print(f'采样完成: {len(lines)} -> {n} 条 ({$DATA_SAMPLE_RATIO*100:.1f}%)')
"
    DATASET_PATH="$SAMPLED_DATASET"
fi

# Detect GPU count
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    GPU_COUNT=4
fi
echo "检测到 $GPU_COUNT 个 GPU"

# Multi-node distributed config (can be injected by scheduler).
# NOTE: Do NOT export WORLD_SIZE or RANK here — torchrun will set per-process values.
# Only provide NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT.
export NNODES=${WORLD_SIZE:-${NNODES:-1}}
export NODE_RANK=${RANK:-${NODE_RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.
}
export MASTER_PORT=${MASTER_PORT:-29500}
unset WORLD_SIZE RANK

echo "分布式配置: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NODE_RANK=$NODE_RANK"

# Multi-node stability/diagnostic defaults. These are no-op for single node.
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1200}

# Optional: override NCCL socket interface for multi-node
# export NCCL_SOCKET_IFNAME=eth0  # Uncomment and set as needed
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
    echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
fi
echo "NCCL配置: ASYNC_ERR=$NCCL_ASYNC_ERROR_HANDLING BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT DEBUG=$NCCL_DEBUG"

# Also set Gloo socket interface (used by vLLM for CPU group)
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
    echo "Gloo网卡: $GLOO_SOCKET_IFNAME"
fi

# Disable vLLM V1 engine (experimental, unstable with multi-node Gloo backend)
export VLLM_USE_V1=${VLLM_USE_V1:-0}
echo "vLLM引擎版本: V1=$VLLM_USE_V1 (0=stable V0, 1=experimental V1)"

export OPENAI_LOG=${OPENAI_LOG:-error}
export VISCOT_API_MAX_WORKERS=${VISCOT_API_MAX_WORKERS:-64}
export VISCOT_API_TIMEOUT=${VISCOT_API_TIMEOUT:-8}
export VISCOT_API_MAX_RETRIES=${VISCOT_API_MAX_RETRIES:-0}
export VISCOT_PROFILE_REWARD=${VISCOT_PROFILE_REWARD:-1}
export VISCOT_PROFILE_LOG_EVERY=${VISCOT_PROFILE_LOG_EVERY:-10}
echo "奖励API配置: workers=$VISCOT_API_MAX_WORKERS timeout=${VISCOT_API_TIMEOUT}s retries=$VISCOT_API_MAX_RETRIES profile=$VISCOT_PROFILE_REWARD/$VISCOT_PROFILE_LOG_EVERY"

# Ensure conda runtime libs are resolved before system libs, otherwise
# vLLM/trl import chain can fail with CXXABI symbol errors from /usr/lib.
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    echo "Conda库优先: LD_LIBRARY_PATH前缀=$CONDA_PREFIX/lib"
fi
export CUDA_LAUNCH_BLOCKING=1
export VISCOT_SAVE_REQUEST_EVERY=${VISCOT_SAVE_REQUEST_EVERY:-10}
export VISCOT_SAVE_REQUEST_FILE=${VISCOT_SAVE_REQUEST_FILE:-$SCRIPT_DIR/output/viscot_api_requests_every10.jsonl}
export VISCOT_SAVE_REWARD_TRACE_EVERY=${VISCOT_SAVE_REWARD_TRACE_EVERY:-20}
export VISCOT_SAVE_REWARD_TRACE_FILE=${VISCOT_SAVE_REWARD_TRACE_FILE:-$SCRIPT_DIR/output/viscot_reward_trace_every${VISCOT_SAVE_REWARD_TRACE_EVERY}.jsonl}
echo "请求采样保存: every=$VISCOT_SAVE_REQUEST_EVERY file=$VISCOT_SAVE_REQUEST_FILE"
echo "奖励明细保存: every=$VISCOT_SAVE_REWARD_TRACE_EVERY file=$VISCOT_SAVE_REWARD_TRACE_FILE"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
USE_HF=1 \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1))) \
NPROC_PER_NODE=$GPU_COUNT \
MAX_PIXELS=${MAX_PIXELS:-$((16384*28*28))} \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/cephfs/home/tinnelxu/models/LlamaFactory/monkey_train0408_all_sft_qwen2_5_vl/checkpoint-9222_export \
    --template qwen2_5_vl \
    --external_plugins "$SCRIPT_DIR/viscot_reward_plugin_parallel.py" \
    --reward_funcs viscot_answer viscot_format viscot_selection \
    --reward_weights 1.0 0.2 0.2 \
    --remove_unused_columns false \
    --use_hf true \
    --dataset "$DATASET_PATH" \
    --max_length 8192 \
    --max_completion_length 2048 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 8192 \
    --vllm_max_num_seqs 2 \
    --vllm_enable_prefix_caching false \
    --vllm_enforce_eager true \
    --vllm_mm_processor_cache_gb 0 \
    --tuner_type full \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --steps_per_generation 4 \
    --eval_steps 10000 \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --dataloader_num_workers 16 \
    --num_generations 4 \
    --temperature 0.9 \
    --deepspeed zero2 \
    --offload_optimizer true \
    --offload_model true \
    --move_model_batches 40 \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --async_generate false \
    --beta 0 \
    --loss_type grpo \
    --vllm_enable_lora false \
    --advantage_estimator grpo \
    --sleep_level 1
