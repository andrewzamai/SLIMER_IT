__package__ = "SFT_finetuning.commons"

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, T5ForConditionalGeneration
from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


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
    padding_side = kwargs.get("padding_side", "right")

    # assert not use_flash_attention or "llama" in base_model, "Cannot use flash attention in a non llama architecture"

    # if use_flash_attention and torch.cuda.get_device_capability()[0] >= 8:
    #     from elmi.commons.llama_patch import replace_attn_with_flash_attn
    #     print("Using flash attention")
    #     replace_attn_with_flash_attn()

    # added cache_dir
    # added padding_side
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side=padding_side, trust_remote_code=True, cache_dir='./hf_cache_dir')

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True, cache_dir='./hf_cache_dir')
    config.update({"max_seq_len": cutoff_len})

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        load_in_4bit=load_4bit,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        use_flash_attention_2=use_flash_attention,
        cache_dir='./hf_cache_dir'
    )

    # if use_flash_attention:
    #     from elmi.commons.llama_patch import forward
    #     assert model.model.layers[
    #                0].self_attn.forward.__doc__ == forward.__doc__, "Model is not using flash attention"

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    # todo add here fixes for other kind of models, if any

    if "llama" in base_model:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

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

    # if use_flash_attention:
    #     from elmi.commons.llama_patch import upcast_layer_for_flash_attention
    #     model = upcast_layer_for_flash_attention(model, torch_dtype=torch.bfloat16)

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


