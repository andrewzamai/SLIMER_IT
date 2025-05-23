"""GRPO trainer, modified from https://github.com/huggingface/open-r1/tree/main/src/open_r1"""

import os
import sys
import wandb
import torch
import logging
import datasets
import transformers

from pathlib import Path
from transformers import set_seed
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# custom imports
from src.RLtraining.src.training.configs import GRPOConfig
from src.RLtraining.src.utils import get_tokenizer
from src.RLtraining.src.utils.callbacks import get_callbacks
from src.RLtraining.src.training.v17_my_grpo_trainers import MyCustomv17GRPOTrainer

from src.RLtraining.src.training.rewards import (
    macrof1_reward,
    microf1_reward,
    format_reward
)
from src.SFT_finetuning.commons.prompter import Prompter


logger = logging.getLogger(__name__)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions."
        },
    )

@dataclass
class CustomTrainingConfig:
    pad_token_id: int
    template_path: str
    prediction_prompter: str
    

def main(script_args, training_args, model_args, custom_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    ##############################
    # W&B Initialization (Only Rank 0)
    ##############################
    if training_args.local_rank == 0 and os.environ["WANDB_MODE"] != "disabled":
        logger.info("Setting up W&B offline mode...")

        # Resolve $WORK directory properly
        wandb_dir = os.path.join(os.environ.get("WORK", "/tmp"), "wandb/logs")  # Default to /tmp if $WORK not set
        os.makedirs(wandb_dir, exist_ok=True)  # Ensure the directory exists
        logger.info(f"WandB logs will be saved to: {wandb_dir}")
        # Set environment variables for WandB
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = wandb_dir  # Set W&B log directory

        try:
            wandb.init(
                project="zeroshotNER",  # Your W&B project name
                #entity="VolBrain",  # Your W&B team/organization
                mode="offline",
                dir=wandb_dir,
                name=Path(training_args.output_dir).name,
                settings=wandb.Settings(init_timeout=300),
            )
            logger.info("WandB initialization successful.")
        except Exception as e:
            # In case of failure, switch to TensorBoard logging
            logger.error(f"Error during WandB initialization: {e}")
            
            # Modify `training_args` to log to TensorBoard instead of WandB
            logger.info("Switching to TensorBoard logging only.")

            # Update the training arguments to use TensorBoard and disable WandB
            training_args.logging_dir = os.path.join(training_args.output_dir, "logs")  # Set logging directory for TensorBoard
            training_args.report_to = ["tensorboard"]  # Disable WandB and enable TensorBoard
            training_args.disable_tqdm = False  # Optional: Enable TQDM progress bar for TensorBoard (if desired)
            
            # Optionally, add further settings related to TensorBoard here.
            # You can define a TensorBoard callback if you wish to log custom metrics.
    else:
        # Modify `training_args` to log to TensorBoard instead of WandB
        logger.info("Switching to TensorBoard logging only.")
        # Update the training arguments to use TensorBoard and disable WandB
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")  # Set logging directory for TensorBoard
        training_args.report_to = ["tensorboard"]  # Disable WandB and enable TensorBoard
        training_args.disable_tqdm = False  # Optional: Enable TQDM progress bar for TensorBoard (if desired)
        
        
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    training_args.vllm_server_host = os.getenv('NODE_NAME', 'default_node_name') 

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer
    ################
    # get_tokenizer is a custom function that returns a tokenizer with the chat template set
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token_id = custom_args.pad_token_id # e.g. "<|finetune_right_pad_id|>"
    # GRPO trainer requires left padding contrary to SFT (it pads to the left to prompt n generations on which then trains on)
    tokenizer.padding_side = "left" 
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Tokenizer pad token set to: {tokenizer.decode(tokenizer.pad_token_id)}")
        print(f"Tokenizer padding side (for training): {tokenizer.padding_side}")
    
    assert tokenizer.padding_side == "left", "Tokenizer padding side must be set to 'left' for GRPO training."

    ################
    # Register reward functions
    ################
    REWARD_FUNCS_REGISTRY = {
        "macrof1_reward": macrof1_reward,
        "microf1_reward": microf1_reward,
        "format": format_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    ################
    # Load dataset
    ################
    # subject, txt_report, gold_diagnosis (eg. CN, AD etc.) columns
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    prompter = Prompter(custom_args.prediction_prompter, template_path=custom_args.template_path)

    # Compute token lengths and update dataset
    def compute_txt_report_tokens_len(example):
        example["txt_report_len"] = len(tokenizer(example['input'])['input_ids'])
        return example
    dataset = dataset.map(compute_txt_report_tokens_len)

    # Format into sytem/user prompt, grpo trainer expects prompt column name
    # formats txt report into instruction
    def format_sys_user_prompt(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": prompter.generate_prompt(instruction=example["instruction"], input=example["input"])})
        return {"prompt": prompt}
    dataset = dataset.map(format_sys_user_prompt)

    # ensure there is no messages column, only prompt
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\nSample from dataset:")
        train_sample = dataset['train'][0]
        print(train_sample.keys())
        # print the first prompt by key value pair
        for kv in train_sample['prompt']:
            print(f"\n{kv['role']} --> {kv['content']}")
   
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map="auto"
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = MyCustomv17GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


# python3 src/RLtraining/src/training/grpo.py --config src/RLtraining/src/training_configs/Llama-3B-GRPO.yaml
if __name__ == "__main__":

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, CustomTrainingConfig))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, custom_args)