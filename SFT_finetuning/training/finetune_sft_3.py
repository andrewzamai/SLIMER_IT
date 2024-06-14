""" Supervised Fine-tuning Trainer (SFT) for generative LLMs """

__package__ = "SFT_finetuning.training"

import os
import sys
import shutil

import safetensors
import yaml
import argparse
from typing import List

import torch
import transformers
from datasets import load_dataset
from peft import set_peft_model_state_dict
from transformers import EarlyStoppingCallback

from ..commons.initialization import init_model, wrap_model_for_peft, get_HF_access_token
from ..commons.preprocessing import generate_and_tokenize_prompt
from ..commons.prompter import Prompter


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "",
        val_data_path: str = "",
        output_dir: str = "./lora-alpaca",
        task_instruction: str = '',
        text_col: str = '',
        output_col: str = '',
        dataset_name: str = '',
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 4096,
        select_train_portion: int = -1,
        val_set_size: int = 2000,
        # lora hyperparams
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ("q_proj", "v_proj"),
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        eos_text: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        save_total_limit: int = 5,
        warmup_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10,
        max_grad_norm: float = 0.3,
        use_flash_attention: bool = False,
        shuffle: bool = True,
        gradient_checkpointing: bool = False,
        early_stopping_patience=5,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"LoRA SFT training with params:\n"
            f"base_model: {base_model}\n"
            f"task instruction: {task_instruction}\n"
            f"text col: {text_col}\n"
            f"output col: {output_col}\n"
            f"load_8bit: {load_8bit}\n"
            f"load_4bit: {load_4bit}\n"
            f"dataset_name: {dataset_name}\n"
            f"data_path: {data_path}\n"
            f"val_data_path: {val_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"early_stopping_patience: {early_stopping_patience}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"select_train_portion: {select_train_portion}\n"
            f"use_lora: {use_lora}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"eos_text: {eos_text}\n"
            f"use flash attn: {use_flash_attention}\n"
            f"shuffle: {shuffle}\n"
            f"gradient checkpointing: {gradient_checkpointing}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='mosaicml/mpt-7b-instruct'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name, template_path='./src/SFT_finetuning/templates', eos_text=eos_text)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print(f"World size: {world_size}")
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    tokenizer, model = init_model(
        base_model,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        cutoff_len=cutoff_len,
        device_map=device_map,
        use_flash_attention=use_flash_attention
    )

    if use_lora:
        model = wrap_model_for_peft(
            model,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout
        )
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    """ Loading train/validation datasets """

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path, dataset_name) if dataset_name else load_dataset(data_path)

    if task_instruction:
        def add_instruction(example):
            example["instruction"] = task_instruction
            return example
        data = data.map(lambda x: add_instruction(x), desc="Adding instruction to examples")

    if text_col:
        data = data.rename_column(text_col, "input")

    if output_col:
        data = data.rename_column(output_col, "output")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.safetensors"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            # adapters_weights = torch.load(checkpoint_name)
            # TODO: fix this
            adapters_weights = {}
            with safetensors.safe_open(checkpoint_name, framework="pt") as f:
                for k in f.keys():
                    adapters_weights[k] = f.get_tensor(k)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    train_data = data["train"]
    print("Train statistics: ")
    dataset_statistics = data_handler_pileNER.get_statistics_for_QA_dataset(train_data,
                                                                            input_column_name='input',
                                                                            instruction_column_name='instruction',
                                                                            output_column_name='output')
    print(dataset_statistics)

    print(train_data['instruction'][0])

    # TODO: masking for enhanced training
    from MSEQA_4_NER.data_handlers.data_handler_pileNER import mask_named_entities
    # train_data = mask_named_entities(train_data, corruption_prob=0.5, masking_prob=0.8, default_mask='<unk>')

    #if os.path.exists(data_path[:-len(".json") + '_' + base_model.split("/")[-1] + '_tokenized'):
    #train_data = load_dataset()

    if shuffle:
        # shuffle train_data
        train_data = train_data.shuffle(seed=42)

    if val_set_size > 0 and not val_data_path:
        train_val = train_data.train_test_split(test_size=val_set_size, shuffle=False)
        train_data = (
            train_val["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        )

        train_data = train_data.filter(lambda x: len(x["input_ids"]) > 5)

        val_data = (
            train_val["test"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        )

        val_data = val_data.filter(lambda x: len(x["input_ids"]) > 5)

    elif val_data_path:
        train_data = data["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        val_data = load_dataset("json", data_files=val_data_path)
        # if -1 use all validation data
        if val_set_size == -1:
            val_set_size = len(val_data["train"])
        val_data = val_data["train"].select(list(range(min(val_set_size, len(val_data["train"])))))  # no more than val_set_size 5k examples
        # TODO: masking for enhanced training
        # val_data = mask_named_entities(val_data, corruption_prob=0.5, masking_prob=0.8, default_mask='<unk>')
        print("Validation statistics: ")
        dataset_statistics = data_handler_pileNER.get_statistics_for_QA_dataset(val_data,
                                                                                input_column_name='input',
                                                                                instruction_column_name='instruction',
                                                                                output_column_name='output')
        print(dataset_statistics)

        val_data = val_data.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        val_set_size = len(val_data)
    else:
        train_data = data["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer, prompter, cutoff_len, train_on_inputs), num_proc=30)
        if "validation" in data:
            print("Validation is in the dataset")
            val_data = data["validation"]
            val_set_size = len(val_data)
        else:
            val_data = None

    if select_train_portion > 0:
        train_data = train_data.select(list(range(min(select_train_portion, len(train_data)))))

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if not train_on_inputs:
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type='cosine',
        bf16=True,
        logging_steps=logging_steps,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 or val_data_path else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 or val_data_path else None,
        save_steps=eval_steps,
        max_grad_norm=max_grad_norm,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 or val_data_path else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=world_size
    )

    print(train_data)
    print("====================")
    print(val_data)
    print("\n\n")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > -1 else []
    )
    model.config.use_cache = False

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f'Saving model at: {output_dir}')
    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":

    """ Fine-tune LLAMA-3-8B Instruct for NER """

    # load HuggingFace access token with permissions to LLAMA3 repo
    from huggingface_hub import login
    HF_ACCESS_TOKEN = get_HF_access_token('./.env')
    login(token=HF_ACCESS_TOKEN)

    # with_guidelines, number_NEs, number_pos_samples_per_NE, number_neg_samples_per_NE
    # use number_NEs=391 or -1 for using top423NEs/MIT/crossNER labels
    parser = argparse.ArgumentParser(description='''pileNER dataset constructor for NER Instuction-Tuning - same instructions''')
    # adding arguments
    parser.add_argument('--with_guidelines', action='store_true', help='Whether to use guidelines')
    parser.add_argument('number_NEs', type=int, help='Number of NEs')
    parser.add_argument('number_pos_samples_per_NE', type=int, help='Number of positive samples per NE')
    parser.add_argument('number_neg_samples_per_NE', type=int, help='Number of negative samples per NE')
    # parsing arguments
    args = parser.parse_args()

    print("pileNER dataset will be constructed with the following specifications:")
    print("with_guidelines:", args.with_guidelines)
    print("number_NEs:", args.number_NEs)
    print("number_pos_samples_per_NE:", args.number_pos_samples_per_NE)
    print("number_neg_samples_per_NE:", args.number_neg_samples_per_NE)

    # TODO: added -SI as SameInstruction
    dataset_name = f"{args.number_NEs}x{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_GenQA_{args.with_guidelines}Def-SI"
    from MSEQA_4_NER.data_handlers import data_handler_pileNER

    """
    pileNER_MSEQA_FalseDef = data_handler_pileNER.build_dataset_MSEQA_format_with_n_samples_per_NE_pos_neg(args.number_pos_samples_per_NE,
                                                                                                           args.number_neg_samples_per_NE,
                                                                                                           removeTestDatasetsNEs=True if args.number_NEs < 423 else False,
                                                                                                           keep_only_top_tagNames=args.number_NEs)
    # convert from FalseDef to TrueDef if args.with_guidelines==True
    if args.with_guidelines:
        # adding guidelines to MSEQA format datasetDict before converting to GenQA
        pileNER_MSEQA_TrueDef = data_handler_pileNER.build_dataset_MSEQA_format_with_guidelines("./src/MSEQA_4_NER/data_handlers/questions/pileNER/all_423_NE_definitions.json", pileNER_MSEQA_FalseDef)
        # TODO: same instruction for both FalseDef and TrueDef
        data_handler_pileNER.convert_MSEQA_dataset_to_GenQA_format_SI(pileNER_MSEQA_TrueDef, with_definition=args.with_guidelines, path_to_save_to=f"./datasets/pileNER/{dataset_name}")
    else:
        data_handler_pileNER.convert_MSEQA_dataset_to_GenQA_format_SI(pileNER_MSEQA_FalseDef, with_definition=args.with_guidelines, path_to_save_to=f"./datasets/pileNER/{dataset_name}")
    """

    # now loading training config from yml and overriding some variables like dataset name and output_dir
    path_to_training_config = './src/SFT_finetuning/training_config/llama3_4_NER_XDef_NsamplesPerNE.yml'
    with open(path_to_training_config, 'rb') as f:
        configs = yaml.safe_load(f.read())
    configs['data_path'] = f"./datasets/pileNER/{dataset_name}/train.jsonl"
    configs['val_data_path'] = f"./datasets/pileNER/{dataset_name}/validation.jsonl"
    configs['output_dir'] = f"./trained_models/llama3_8B_{args.number_pos_samples_per_NE}pos_{args.number_neg_samples_per_NE}neg_perNE_top{args.number_NEs}NEs_{args.with_guidelines}Def-SI-plain"

    train(**configs)

    # copy training config file
    shutil.copy(src=path_to_training_config, dst=os.path.join(configs['output_dir'], 'training_configs.yml'))

    torch.cuda.empty_cache()

    print("Training DONE :)\n\n")
