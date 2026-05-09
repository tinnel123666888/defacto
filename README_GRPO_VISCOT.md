# Visual-CoT GRPO Training Framework

This repository contains the code for training vision-language models using **GRPO (Group Relative Policy Optimization)** on **Visual Chain-of-Thought (Visual-CoT)** tasks, powered by the **MS-Swift** framework and optimized with **vLLM** for efficient inference.

## Overview

**Visual-CoT** combines visual reasoning with chain-of-thought explanations, where the model:
1. Analyzes visual regions using bounding boxes
2. Provides step-by-step reasoning in `<think>` tags
3. Gives a final structured answer in `<answer>` tags

**GRPO Training** optimizes the model through reinforcement learning with multiple reward signals:
- **viscot_answer**: Correctness of the final answer
- **viscot_format**: Proper formatting of output (correct tags)
- **viscot_selection**: Accuracy of spatial region selection (bounding boxes)

## Key Components

### Framework & Model
- **Model**: Qwen2.5-VL (7B) - A vision-language model with visual understanding capabilities
- **Training Framework**: MS-Swift (RLHF trainer with GRPO support)
- **Inference Engine**: vLLM (with tensor parallelism for multi-GPU inference)
- **Optimization**: DeepSpeed Zero-2 (memory-efficient distributed training)

### Multi-GPU & Multi-Node Support
- Automatic GPU detection and scaling
- Distributed training via `torchrun`
- Multi-node training support with NCCL/Gloo coordination
- Optional configurable network interfaces (for cloud environments)

---

## Data Format

Training data must be in **JSONL format** (one JSON object per line). Each line represents one training example:

```json
{
  "query": "<image>\n[Question text]\nThink step by step. If helpful, identify the key image region using <box>[[x1,y1,x2,y2],...]</box> inside your reasoning. Format:\n<think>\n<box>[[x1,y1,x2,y2],...]</box>\nYour reasoning.\n</think>\n<answer>Your answer.</answer>",
  "response": "<think>\n<box>[[x1,y1,x2,y2], [x3,y3,x4,y4]]</box>\nStep-by-step reasoning explaining the answer...\n</think>\n<answer>Final answer text.</answer>",
  "images": [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg"
  ],
  "solution": {
    "type": "pos",
    "reward_case": "pos",
    "y_star": "Ground truth answer text",
    "all_boxes": [
      [x1, y1, x2, y2],
      [x3, y3, x4, y4],
      ...
    ]
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Multi-modal query with `<image>` placeholder and instructions for reasoning format |
| `response` | string | Model's response with `<think>` (reasoning) and `<answer>` (final answer) tags |
| `images` | list[string] | Absolute paths to images referenced in the query |
| `solution` | dict | Ground truth information for reward computation |
| `solution.type` | string | Example type: "pos" (positive) or "neg" (negative) |
| `solution.reward_case` | string | Reward category for analysis |
| `solution.y_star` | string | Ground truth answer for comparison |
| `solution.all_boxes` | list[list[int]] | All valid bounding boxes as `[x1, y1, x2, y2]` coordinates |

### Example Query Format

```
<image>
Can you describe the attire of the individuals operating the pulley system?
Think step by step. If helpful, identify the key image region using <box>[[x1,y1,x2,y2],...]</box> inside your reasoning. Format:
<think>
<box>[[x1,y1,x2,y2],...]</box>
Your reasoning.
</think>
<answer>Your answer.</answer>
```

### Expected Response Format

```
<think>
<box>[[368, 375, 508, 748], [671, 416, 836, 875]]</box>
These boxes highlight the individuals operating the pulley system. Their attire includes hard hats and dark work clothing, which is visible and relevant to answering the question.
</think>
<answer>They are wearing safety gear which includes hard hats.</answer>
```

---

## Installation & Setup

### 1. Environment Setup

```bash
# Clone or navigate to the defacto directory
cd /path/to/defacto

# Install dependencies
pip install -r requirements.txt

# Or install from setup.py for development
pip install -e .
```

### 2. Download/Prepare Data

Prepare your Visual-CoT dataset in JSONL format (see Data Format section above).

Example path structure:
```
dataset/
  viscot/
    Visual-CoT/
      results_json/
        train_triplets_0228/
          train_sampled_pos100_crand5_defacto_newprompt_clean.jsonl
```

### 3. Prepare Base Model Checkpoint

The training uses a **pre-trained SFT (Supervised Fine-Tuning) checkpoint** as the starting point:

```bash
# SFT checkpoint should be at:
/path/to/LlamaFactory/monkey_train0408_all_sft_qwen2_5_vl/checkpoint-9222_export
```

You can either:
- Use an existing LlamaFactory SFT checkpoint
- Train your own using: [LlamaFactory](https://github.com/hiyouga/LlamaFactory)

---

## Training

### Quick Start (Single Node, 4 GPUs)

```bash
cd /path/to/defacto/examples/train/grpo/plugin

# Run with default settings
bash run_viscot_sampled_5pct_qwen25vl.sh
```

### Configuration & Environment Variables

Before running, you can customize training behavior via environment variables:

```bash
# Data sampling (0.0-1.0, default 60% of data)
export DATA_SAMPLE_RATIO=0.6
export DATA_SEED=42

# vLLM inference engine
export VLLM_USE_V1=0                           # 0=stable V0, 1=experimental V1
export VLLM_GPU_MEMORY_UTILIZATION=0.5        # GPU memory for vLLM

# Reward API (external reward server if used)
export VISCOT_API_MAX_WORKERS=64               # Parallel worker threads
export VISCOT_API_TIMEOUT=8                   # Timeout in seconds
export VISCOT_API_MAX_RETRIES=0                # Retry attempts
export VISCOT_PROFILE_REWARD=1                # Enable reward profiling
export VISCOT_PROFILE_LOG_EVERY=10             # Log interval

# Save intermediate states for debugging
export VISCOT_SAVE_REQUEST_EVERY=10            # Save API requests
export VISCOT_SAVE_REWARD_TRACE_EVERY=20      # Save reward traces

# Multi-node training
export NNODES=1                                # Number of nodes
export NODE_RANK=0                             # Current node rank (0-indexed)
export MASTER_ADDR=127.0.0.1                   # Master node IP
export MASTER_PORT=29500                       # Master node port
export NCCL_SOCKET_IFNAME=eth0                 # Network interface (optional)

# Data paths
export ORIG_DATASET="/path/to/your/data.jsonl"
```

### Example: Custom Training

```bash
#!/bin/bash
cd /path/to/defacto/examples/train/grpo/plugin

# Use 50% of data, custom seed
export DATA_SAMPLE_RATIO=0.5
export DATA_SEED=123

# Multi-node: 2 nodes, this is node 0
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=192.168.1.100
export NCCL_SOCKET_IFNAME=eth1

# Run training
bash run_viscot_sampled_5pct_qwen25vl.sh
```

### Multi-Node Training

For training across multiple nodes:

```bash
# On node 0 (master):
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
bash run_viscot_sampled_5pct_qwen25vl.sh

# On node 1 (worker):
export NNODES=2
export NODE_RANK=1
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
bash run_viscot_sampled_5pct_qwen25vl.sh
```

---

## Training Parameters

Key hyperparameters in `run_viscot_sampled_5pct_qwen25vl.sh`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rlhf_type` | `grpo` | Reinforcement learning algorithm |
| `model` | Qwen2.5-VL SFT checkpoint | Base model |
| `template` | `qwen2_5_vl` | Model's chat template |
| `reward_funcs` | viscot_answer, viscot_format, viscot_selection | Reward functions to use |
| `reward_weights` | 1.0, 0.2, 0.2 | Relative weights for each reward |
| `max_length` | 8192 | Max input+output tokens |
| `max_completion_length` | 2048 | Max output tokens |
| `per_device_train_batch_size` | 2 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | Effective batch = 2×8=16 per GPU |
| `learning_rate` | 1e-6 | RLHF learning rate |
| `num_generations` | 4 | Samples per prompt during rollout |
| `vllm_tensor_parallel_size` | 4 | Tensor parallelism (for large models) |
| `deepspeed` | `zero2` | Memory optimization strategy |
| `num_train_epochs` | 1 | Training epochs |

### Recommended Adjustments

**For Memory Issues (OOM):**
```bash
--per_device_train_batch_size 1
--vllm_gpu_memory_utilization 0.4
--gradient_accumulation_steps 16
```

**For Faster Training (if resources available):**
```bash
--per_device_train_batch_size 4
--vllm_gpu_memory_utilization 0.7
--num_generations 8
--gradient_accumulation_steps 4
```

**For Multi-GPU Inference Scaling:**
```bash
--vllm_tensor_parallel_size 2  # Spread model across 2 GPUs per process
```

---

## Output & Monitoring

### Training Outputs

Training saves checkpoints and logs to:
```
examples/train/grpo/plugin/output/
├── Qwen2.5-VL-7B-Instruct/
│   ├── v0-<timestamp>/
│   │   ├── checkpoint-*.safetensors       # Model weights
│   │   ├── training_args.json              # Hyperparameters
│   │   ├── args.json                       # Full config
│   │   ├── completions.jsonl               # Model generations
│   │   ├── logging.jsonl                   # Training metrics
│   │   ├── training_timing.jsonl           # Performance logs
│   │   └── runs/                           # TensorBoard events
│   └── v1-<timestamp>/
│       └── ...
├── train_sampled_ratio0.6_seed42.jsonl     # Sampled training data
├── viscot_api_requests_every10.jsonl       # Reward computation trace
└── viscot_reward_trace_every20.jsonl       # Reward statistics
```

### Monitoring Training

**Via TensorBoard:**
```bash
cd output/Qwen2.5-VL-7B-Instruct/v0-<timestamp>/runs/
tensorboard --logdir . --port 6006
# Visit: http://localhost:6006
```

**Via Logs:**
```bash
# Check training progress
tail -f output/Qwen2.5-VL-7B-Instruct/v0-<timestamp>/logging.jsonl | python3 -m json.tool

# Monitor reward signals
tail -f output/viscot_reward_trace_every20.jsonl | python3 -m json.tool
```

---

## Understanding Reward Functions

### 1. `viscot_answer`
- **Purpose**: Measures if the final answer (`<answer>...</answer>`) matches ground truth
- **Weight**: 1.0 (primary signal)
- **Implementation**: String similarity comparison with `y_star`

### 2. `viscot_format`
- **Purpose**: Ensures output follows required format (has `<think>` and `<answer>` tags)
- **Weight**: 0.2 (supplementary)
- **Implementation**: Validates tag presence and structure

### 3. `viscot_selection`
- **Purpose**: Evaluates if selected bounding boxes match ground truth regions
- **Weight**: 0.2 (supplementary)
- **Implementation**: Compares predicted boxes against `all_boxes`

---

## Troubleshooting

### Issue: `CUDA out of memory`
**Solution:**
```bash
export DATA_SAMPLE_RATIO=0.2  # Use less data
export VLLM_GPU_MEMORY_UTILIZATION=0.3
# Or reduce batch size in the script
```

### Issue: `NCCL timeout` on multi-node
**Solution:**
```bash
export NCCL_TIMEOUT=1800  # Increase timeout
export NCCL_DEBUG=TRACE   # Enable debugging
export NCCL_SOCKET_IFNAME=<correct_interface>  # Specify network interface
```

### Issue: `ModuleNotFoundError: swift`
**Solution:**
```bash
pip install -e /path/to/defacto/
```

### Issue: Model inference too slow
**Solution:**
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.7  # Use more GPU memory
# Reduce generation samples:
--num_generations 2  # Instead of 4
```

---

## Project Structure

```
defacto/
├── swift/                          # MS-Swift framework
│   ├── cli/                        # Command-line interfaces
│   │   ├── rlhf.py                 # RLHF training entry point
│   │   └── main.py
│   ├── rlhf_trainers/              # RLHF algorithm implementations
│   │   ├── grpo_trainer.py         # GRPO trainer
│   │   ├── rollout_mixin.py        # Generation/rollout logic
│   │   └── ...
│   ├── rewards/                    # Reward function framework
│   │   ├── orm.py                  # Reward model base class
│   │   └── rm_plugin.py            # Plugin system for custom rewards
│   └── ...
│
├── examples/train/grpo/plugin/
│   ├── run_viscot_sampled_5pct_qwen25vl.sh  # Main training script
│   ├── viscot_reward_plugin_parallel.py      # Custom reward functions
│   ├── viscot_reward_plugin.py               # Alternative reward implementation
│   └── output/                               # Training outputs (generated)
│
├── setup.py                        # Package configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Citation & References

This code implements training for Visual-CoT tasks using GRPO, combining:
- **Visual reasoning** with spatial region selection (bounding boxes)
- **Chain-of-thought explanations** for interpretability
- **GRPO algorithm** for efficient reward-based optimization
- **Distributed training** for large-scale model adaptation

For research using this code, please cite:
- MS-Swift framework
- vLLM for efficient inference
- Your Visual-CoT dataset paper

---

## License & Acknowledgments

This framework builds on:
- **MS-Swift** (Alibaba): Swift RLHF training framework
- **vLLM** (UC Berkeley): High-throughput LLM inference
- **DeepSpeed** (Microsoft): Distributed training optimization

See individual repositories for their licenses.

---

## Support & Contact

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `output/` directory
3. Verify data format matches the specification
4. Ensure all dependencies are installed: `pip install -r requirements.txt`
