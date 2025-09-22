cd src/virft
pip install -e ".[dev]"

# Addtional modules

pip install tensorboardx
pip install qwen_vl_utils torchvision
# pip install flash-attn --no-build-isolation

# vLLM support 
pip install vllm==0.7.2

# fix transformers version
# pip install git+https://bgithub.xyz/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
# pip install git+https://bgithub.xyz/huggingface/transformers.git@8ee50537fe7613b87881cd043a85971c85e99519

# pip install trl==0.16.0
# pip install torch==2.6.0
pip install flash_attn-2.8.2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install wandb==0.16.6
pip install rapidfuzz