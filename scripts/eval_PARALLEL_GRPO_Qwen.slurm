#!/bin/bash
#SBATCH --job-name=eval_IT_Qwen

#SBATCH -A dmx@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3  # Quality of service up 2h otherwise qos_gpu_a100-t3
#SBATCH --time=03:00:00

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --gres=gpu:2            # number of GPUs per node

#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread    # disable hyperthreading

#SBATCH --output=eval_LLM/eval_LLM_%j.out 
#SBATCH --error=eval_LLM/eval_LLM_%j.err   

module purge
module load arch/h100
module load cuda/12.4.1
module load nccl/2.21.5-1-cuda
module load cudnn/9.2.0.82-cuda
#module load intel-compilers/2021.9.0   # for a100 only?

# load conda env with pytorch 2.5.1 cuda12.4.1, vllm==0.7.2, flash_attn, deepspeed, accelerate ...
source ~/.bashrc
conda activate venv_py26  # vllm==0.7.2

export PYTHONPATH=$WORK/SLIMER_IT

export HF_HOME=$WORK/.cache/.huggingface
export XDG_CACHE_HOME=$WORK/.cache

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

set -x -e

# Set your base model and output directories
BASE_MODEL="Qwen/Qwen3-8B"

TRAINED_MODEL="./trained_models/Qwen3-8B-GRPO-SLIMER-PARALLEL-p0mask/checkpoint-160"
MERGED_MODEL="./merged_models/Qwen3-8B-GRPO-SLIMER-PARALLEL-p0mask-cp-160"

# Merge LoRA weights
#python src/SFT_finetuning/commons/merge_lora_weights.py "$BASE_MODEL" "$TRAINED_MODEL" "$MERGED_MODEL"

python src/SFT_finetuning/evaluating/evaluate_SLIMER_PARALLEL_GRPO.py \
--merged_model_name "$BASE_MODEL" \
--with_guidelines \
--max_tagNames_per_prompt 20 \
--temperature 0.6 \
--top_p 0.9 \
--top_k 20 \
--min_p 0 \
--num_GPUs 2 \
--system_message "Sei un assistente utile per il riconoscimento di entità nominate (NER). Segui le istruzioni qui sotto per annotare le entità richieste da un testo fornito. Impegnati in un processo di ragionamento approfondito: esamina attentamente le definizioni e le linee guida per guidare l'annotazione, discuti i casi ambigui e risolvi l'assegnazione delle classi confrontando i candidati strettamente correlati. Racchiudi il tuo ragionamento tra i tag <think> e </think>. Infine, restituisci le entità annotate nel formato JSON richiesto, racchiuse tra i tag ```json e ``` . Nulla deve essere restituito al di fuori di questi tag. Ragiona e rispondi in Italiano."

#--with_guidelines \
#--mask_labels \


# non thinking
# --temperature 0.7 \
# --top_p 0.8 \
# --system_message "You are a helpful NER assistant. Follow the instruction below to annotate the asked entities from a provided text. Return the annotated entities in the requested JSON format, enclosed between ```json and ``` tags. Nothing else should be returned outside these tags."

# thinking
# --temperature 0.6 \
# --top_p 0.9 \
#--system_message "You are a helpful NER assistant. Follow the instruction below to annotate the asked entities from a provided text. Engage in a thorough reasoning process: carefully scrutinize the definitions and guidelines to guide your annotation, discuss ambiguous cases, and resolve class assignments by comparing closely related candidates. Enclose your reasoning between <think> and </think> tags. Finally, return the annotated entities in the requested JSON format, enclosed between ```json and ``` tags. Nothing else should be returned outside these tags."
