#!/bin/bash
#SBATCH --job-name=Deepseek8NER

#SBATCH -A dmx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3      #qos_gpu_h100-t3        # Quality of service qos_gpu_a100-dev (up to 2 h) or qos_gpu_a100-t3 (up to 20h)
#SBATCH --time=12:00:00

#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --gres=gpu:4                 # Request 4 GPUs
#SBATCH --cpus-per-task=4            # 4 CPUs per task
#SBATCH --hint=nomultithread         # Disable hyperthreading

#SBATCH --output=grpo_out/grpo_%j.out  
#SBATCH --error=grpo_out/grpo_%j.err

# clean and load env
module purge
module load arch/h100
module load cuda/12.4.1
module load nccl/2.21.5-1-cuda
module load cudnn/9.2.0.82-cuda
#module load intel-compilers/2021.9.0   # for a100 only?

# load conda env with pytorch 2.5.1 cuda12.4.1, vllm==0.7.2, flash_attn, deepspeed, accelerate ...
source ~/.bashrc
conda activate venv_py26

export HF_HOME=$WORK/.cache/.huggingface
export XDG_CACHE_HOME=$WORK/.cache
export PYTHONPATH=$WORK/SLIMER

# ensure offline execution since nodes do not have internet access
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Add this right after the module loading section
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONMP_START_METHOD=spawn

export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread

# Enable debugging & fail on errors
set -x -e

# Get allocated node from Slurm
NODE_NAME=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export NODE_NAME

# Ensure correct GPU allocation
export CUDA_VISIBLE_DEVICES=3

# Run vLLM on GPU 3 (single GPU for inference)
nohup trl vllm-serve \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tensor_parallel_size 1 \
    --host $NODE_NAME \
    --port 2020 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 5000 \
    --enable_prefix_caching false > vllm_server_Deepseek_p0.log 2>&1 & disown

# Give vLLM time to initialize
sleep 60

# Set GPUs for training (0,1,2)
export CUDA_VISIBLE_DEVICES=0,1,2

# Run training on GPUs 0, 1, 2
accelerate launch \
    --config_file src/RLtraining/recipes/accelerate_configs/ddp.yaml \
    --main_process_port 29700 \
    src/RLtraining/src/training/default_grpo.py \
    --config src/RLtraining/recipes/Deepseek/DeepseekLlama_8B.yaml &
wait