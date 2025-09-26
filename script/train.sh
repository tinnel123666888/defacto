export CUDA_HMOE=/usr/local/cuda-12.5
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

# 相对路径
export DATA_PATH=../../0917
export CKPT_PATH=../../../models/Qwen2.5-VL-7B-Instruct
export SAVE_PATH=../../save_models/Qwen2.5-VL-7B-Instruct_GRPO_defacto

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH=../../debug_log_7b_GRPO_defacto.txt

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12323" \
    ../virft/src/open_r1/grpo_defacto.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed ../virft/local_scripts/zero3_offload.json \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 705600 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-7B_GRPO_test \
    --save_steps 500 \
    --save_only_model false \
    --num_generations 4
