
# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning (ICML 2026)


This repository provides the official code for our paper:

**DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning**

Tianrun Xu, Haoda Jing, Ye Li, Yuquan Wei, Jun Feng, Guanyu Chen, Haichuan Gao, Tianren Zhang, Feng Chen

ICML 2026

The code supports vision-language model training with reinforcement learning and counterfactual reasoning on the DeFacto dataset.


## Overview

DeFacto enables evidence-grounded and faithful reasoning on images. The framework supports counterfactual and visual reasoning with flexible reward functions. For details, please refer to our paper.
## Judging (Reward/Judge Customization)

You can use your own judge for answer evaluation. The default judge logic is implemented in:

`examples/train/grpo/plugin/viscot_reward_plugin_parallel.py`

You can modify or replace the reward functions in this file to use your own judging logic. For example, to change the answer correctness judge, edit the `ViscotAnswerReward` class. The script supports both exact match and LLM-based judging. See comments in the file for details.

After修改后，重新运行训练脚本即可生效。

## Key Components

### Framework & Model
- **Model**: Qwen2.5-VL (7B) - A vision-language model with visual understanding
- **Training Framework**: MS-Swift (RLHF trainer with GRPO support)
- **Inference Engine**: vLLM (with tensor parallelism for multi-GPU inference)
- **Optimization**: DeepSpeed Zero-2 (memory-efficient distributed training)

### Multi-GPU & Multi-Node Support
- Automatic GPU detection and scaling
- Distributed training via `torchrun`
- Multi-node training support with NCCL/Gloo coordination
- Configurable network interfaces for cloud environments

---

## Data Format

Training data should be in **JSONL format** (one JSON object per line):

```json
{
  "query": "<image>\n[Question text]\nThink step by step. If helpful, identify the key image region using <box>[[x1,y1,x2,y2],...]</box> inside your reasoning. Format:\n<think>\n<box>[[x1,y1,x2,y2],...]</box>\nYour reasoning.\n</think>\n<answer>Your answer.</answer>",
  "response": "<think>\n<box>[[x1,y1,x2,y2], [x3,y3,x4,y4]]</box>\nStep-by-step reasoning explaining the answer...\n</think>\n<answer>Final answer text.</answer>",
  "images": [
    "image1.jpg",
    "image2.jpg"
  ],
  "solution": {
    "type": "pos",
    "reward_case": "pos",
    "y_star": "Ground truth answer text",
    "all_boxes": [
      [x1, y1, x2, y2],
      [x3, y3, x4, y4]
    ]
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Multi-modal query with `<image>` placeholder and instructions for reasoning format |
| `response` | string | Model's response with reasoning and answer tags |
| `images` | list[string] | Image file names or relative paths |
| `solution` | dict | Ground truth information for reward computation |
| `solution.type` | string | Example type: "pos" (positive) or "neg" (negative) |
| `solution.reward_case` | string | Reward category for analysis |
| `solution.y_star` | string | Ground truth answer for comparison |
| `solution.all_boxes` | list[list[int]] | Valid bounding boxes as `[x1, y1, x2, y2]` |

### Expected Response Format

```
<think>
<box>[[x1, y1, x2, y2], [x3, y3, x4, y4]]</box>
Step-by-step reasoning about the visual regions and the question.
</think>
<answer>Final answer text here.</answer>
```

---

## Getting Started

### 1. Installation

```bash
# Clone repository
git clone https://github.com/tinnel123666888/defacto.git
cd defacto

# Install dependencies
pip install -r requirements.txt

# Or install for development
pip install -e .
```

### 2. Download Dataset

The training data is available on Hugging Face:

```bash
# Download Defacto dataset
huggingface-cli download tinnel123/defacto_dataset --repo-type dataset --local-dir ./data/

# Or use the dataset directly with the script
```

### 3. Prepare Model Checkpoint

The training uses a pre-trained SFT (Supervised Fine-Tuning) checkpoint. You can either:
- Use an existing checkpoint from LlamaFactory
- Train your own using [LlamaFactory](https://github.com/hiyouga/LlamaFactory)

---

## Training

### Quick Start

```bash
cd examples/train/grpo/plugin

# Run with default settings (single node, 4 GPUs)
bash run_viscot_sampled_5pct_qwen25vl.sh
```

### Configuration

Customize training via environment variables:

```bash
# Data sampling (0.0-1.0, default 60%)
export DATA_SAMPLE_RATIO=0.6
export DATA_SEED=42

# vLLM inference engine
export VLLM_USE_V1=0                           # 0=stable, 1=experimental
export VLLM_GPU_MEMORY_UTILIZATION=0.5        # GPU memory allocation

# Reward API configuration
export VISCOT_API_MAX_WORKERS=64               # Parallel workers
export VISCOT_API_TIMEOUT=8                   # Timeout (seconds)
export VISCOT_API_MAX_RETRIES=0                # Retry attempts

# Save debugging information
export VISCOT_SAVE_REQUEST_EVERY=10            # Save requests
export VISCOT_SAVE_REWARD_TRACE_EVERY=20      # Save reward traces

# Multi-node training
export NNODES=1                                # Number of nodes
export NODE_RANK=0                             # Node rank (0-indexed)
export MASTER_ADDR=127.0.0.1                   # Master IP
export MASTER_PORT=29500                       # Master port
```

### Custom Training Example

```bash
#!/bin/bash
cd examples/train/grpo/plugin

# Use 50% of data with custom seed
export DATA_SAMPLE_RATIO=0.5
export DATA_SEED=123

# Run training
bash run_viscot_sampled_5pct_qwen25vl.sh
```

### Multi-Node Training

```bash
# On master node (rank 0):
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
bash run_viscot_sampled_5pct_qwen25vl.sh

# On worker node (rank 1):
export NNODES=2
export NODE_RANK=1
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
bash run_viscot_sampled_5pct_qwen25vl.sh
```

---

## Training Parameters

Key hyperparameters in the training script:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rlhf_type` | `grpo` | Reinforcement learning algorithm |
| `template` | `qwen2_5_vl` | Model's chat template |
| `reward_weights` | 1.0, 0.2, 0.2 | Weights for answer, format, selection rewards |
| `max_length` | 8192 | Max input+output tokens |
| `max_completion_length` | 2048 | Max output tokens |
| `per_device_train_batch_size` | 2 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | Effective batch per GPU = 16 |
| `learning_rate` | 1e-6 | RLHF learning rate |
| `num_generations` | 4 | Samples per prompt |
| `deepspeed` | `zero2` | Memory optimization |
| `num_train_epochs` | 1 | Training epochs |

### Performance Tuning

**For resource constraints:**
```bash
--per_device_train_batch_size 1
--vllm_gpu_memory_utilization 0.4
--gradient_accumulation_steps 16
```

**For faster training:**
```bash
--per_device_train_batch_size 4
--vllm_gpu_memory_utilization 0.7
--num_generations 8
--gradient_accumulation_steps 4
```

---

## Monitoring Training

### Output Structure

Training saves outputs to `examples/train/grpo/plugin/output/`:
```
output/
├── Qwen2.5-VL-7B-Instruct/
│   ├── v0-<timestamp>/
│   │   ├── checkpoint-*.safetensors       # Model weights
│   │   ├── training_args.json              # Configuration
│   │   ├── completions.jsonl               # Generated samples
│   │   ├── logging.jsonl                   # Training metrics
│   │   └── runs/                           # TensorBoard logs
│   └── v1-<timestamp>/
│       └── ...
├── training_data_sampled.jsonl             # Sampled training set
└── reward_trace.jsonl                      # Reward statistics
```

### TensorBoard

```bash
cd output/Qwen2.5-VL-7B-Instruct/v0-<timestamp>/runs/
tensorboard --logdir . --port 6006
# Visit: http://localhost:6006
```

### View Logs

```bash
# Training progress
tail -f output/Qwen2.5-VL-7B-Instruct/v0-<timestamp>/logging.jsonl

# Reward signals
tail -f output/reward_trace.jsonl
```

---

## Reward Functions

### 1. Answer Correctness
Measures if the final answer matches ground truth through string similarity.
- Weight: 1.0 (primary signal)
- Primary objective for model optimization

### 2. Format Compliance
Validates proper output structure with required tags.
- Weight: 0.2 (auxiliary signal)
- Ensures consistent formatting

### 3. Region Selection
Evaluates bounding box predictions against ground truth regions.
- Weight: 0.2 (auxiliary signal)
- Assesses spatial understanding

---

## Project Structure

```
defacto/
├── swift/                          # MS-Swift training framework
│   ├── cli/                        # Command-line interfaces
│   ├── rlhf_trainers/              # RLHF implementations
│   │   └── grpo_trainer.py         # GRPO trainer
│   ├── rewards/                    # Reward functions
│   └── ...
│
├── examples/train/grpo/plugin/
│   ├── run_viscot_sampled_5pct_qwen25vl.sh  # Training script
│   ├── viscot_reward_plugin_parallel.py      # Reward functions
│   └── output/                               # Training outputs
│
├── setup.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 4+ GPUs recommended (can run on 1 GPU with reduced batch size)

See `requirements.txt` for complete dependency list.

---

## Dataset

The Defacto dataset is available on Hugging Face:
- **Repository**: [tinnel123/defacto_dataset](https://huggingface.co/datasets/tinnel123/defacto_dataset)
- **Format**: JSONL with image paths and annotations
- **Size**: Large-scale vision-language dataset

Download and use:
```bash
huggingface-cli download tinnel123/defacto_dataset --repo-type dataset
```

---


## Citation

If you use this code or the DeFacto dataset, please cite:

```bibtex
@article{xu2025defacto,
  title={DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning},
  author={Xu, Tianrun and Jing, Haoda and Li, Ye and Wei, Yuquan and Feng, Jun and Chen, Guanyu and Gao, Haichuan and Zhang, Tianren and Chen, Feng},
  journal={arXiv preprint arXiv:2509.20912},
  year={2025}
}
```

---

## License & Acknowledgments

This framework builds on:
- **MS-Swift** (Alibaba): RLHF training framework
- **vLLM** (UC Berkeley): High-throughput LLM inference  
- **DeepSpeed** (Microsoft): Distributed training optimization

See individual repositories for licensing details.

---

## Contact

For questions or issues, please open an issue on GitHub at:
https://github.com/tinnel123666888/defacto
