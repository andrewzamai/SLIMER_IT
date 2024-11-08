from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    T5ForConditionalGeneration
)
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

import logging
import torch
import sys


def get_HF_access_token(path_to_env_file):
    """ return my personal HuggingFace access token stored in .env (git-ignored)"""
    with open(path_to_env_file, 'r') as file:
        api_keys = file.readlines()
    api_keys_dict = {}
    for api_key in api_keys:
        api_name, api_value = api_key.split('=')
        api_keys_dict[api_name] = api_value
    return api_keys_dict['AZ_HUGGINGFACE_TOKEN']


def init_model(base_model, **kwargs):
    load_8bit = kwargs.get("load_8bit", False)
    load_4bit = kwargs.get("load_4bit", False)
    eval_mode = kwargs.get("eval", False)
    cutoff_len = kwargs.get("cutoff_len", 2048)
    device_map = kwargs.get("device_map", "auto")
    use_flash_attention = kwargs.get("use_flash_attention", False)
    lora_weights = kwargs.get("lora_weights", '')
    padding_side = kwargs.get("padding_side", "left")

    # added padding_side
    # usually left for generative LLMs
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side=padding_side, trust_remote_code=True)

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.update({"max_seq_len": cutoff_len})

    """
    if "llama" in base_model and use_flash_attention:
        from src.SFT_finetuning.commons.patch_models.modeling_flash_llama import LlamaForCausalLM as LlamaForCausalLMFlash
        logging.warning("Using Flash Attention for LLaMA model.")
        load_fn = LlamaForCausalLMFlash
    else:
        logging.warning("Unable to use Flash Attention.")
        load_fn = AutoModelForCausalLM
    
    model = load_fn.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        load_in_4bit=load_4bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    """

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        load_in_4bit=load_4bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa"
    )

    if "Llama-2" in base_model:
        print("\nSetting LLaMA pad_token_id to <unk>")
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    elif "Llama-3" in base_model:
        model.config.pad_token_id = tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if lora_weights:
        print("\nLoading Lora weights...\n")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            repo_type='model'
        )

    # if not load_8bit and not load_4bit and not lora_weights:
    #      model.half()  # seems to fix bugs for some users.

    if eval_mode:
        print("Setting model in EVAL MODE")
        model.eval()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    return tokenizer, model


def wrap_model_for_peft(model, **kwargs):
    load_8bit = kwargs.get("load_8bit", False)
    load_4bit = kwargs.get("load_4bit", False)
    lora_r = kwargs.get("lora_r", 8)
    lora_alpha = kwargs.get("lora_alpha", 16)
    lora_target_modules = kwargs.get("lora_target_modules", "[q_proj,k_proj,v_proj,o_proj]")
    lora_dropout = kwargs.get("lora_dropout", 0.05)
    use_flash_attention = kwargs.get("use_flash_attention", False)

    if load_8bit or load_4bit:
        model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    return model


def init_base_s2s_model(base_model, **kwargs):
    load_8bit = kwargs.get("load_8bit", False)
    load_4bit = kwargs.get("load_4bit", False)
    cutoff_len = kwargs.get("cutoff_len", 2048)
    device_map = kwargs.get("device_map", "auto")
    lora_weights = kwargs.get("lora_weights", '')

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    config = AutoConfig.from_pretrained(base_model)
    config.update({"max_seq_len": cutoff_len})

    model = T5ForConditionalGeneration.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        load_in_4bit=load_4bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )

    # todo add here fixes for other kind of models, if any

    if lora_weights:
        print("\nLoading Lora weights...\n")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )

    if not load_8bit and not load_4bit:
        model.half()  # seems to fix bugs for some users.

    return tokenizer, model


