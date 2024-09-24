import torch
from transformers import GenerationConfig

from .preprocessing import truncate_input


def evaluate(
        tokenizer,
        model,
        prompter,
        instruction,
        input=None,
        cutoff_len=2048,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        num_return_sequences: int = 1,
        max_new_tokens=128,
        prompt_template: str = '',
        renormalize_logits: bool = True,
        early_stopping: bool = True,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,  # -0.1
        no_repeat_ngram_size: int = 0,
        do_sample: bool = False,
        device: str = "cuda",
        **kwargs,
):
    prompt = prompter.generate_prompt(
        instruction,
        truncate_input({"input": input, "instruction": instruction}, tokenizer, prompter, cutoff_len)
    )

    # prompt = prompter.generate_prompt(instruction, input)

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    #print(f"\nNumber of input tokens: {input_ids.shape}\n")
    # force_words_ids = [tokenizer("</you>", add_special_tokens=False)["input_ids"]]
    force_words_ids = None

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        # eos_token_id=[29958, model.config.eos_token_id], # [model.config.eos_token_id, 29958], # [model.config.eos_token_id, tokenizer.vocab["###"]],
        do_sample=do_sample,
        early_stopping=early_stopping,
        remove_invalid_values=True,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        renormalize_logits=renormalize_logits,
        force_words_ids=force_words_ids,
        num_return_sequences=num_return_sequences,
        **kwargs
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]

    prompt_length = input_ids.shape[1]
    return tokenizer.decode(s[prompt_length:]), s.shape[-1]


def batch_evaluate(
        tokenizer,
        model,
        prompter,
        instructions,
        inputs,
        cutoff_len=2048,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        num_return_sequences: int = 1,
        max_new_tokens=128,
        renormalize_logits: bool = True,
        early_stopping: bool = True,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,  # -0.1
        no_repeat_ngram_size: int = 0,
        do_sample: bool = False,
        device: str = "cuda",
        verbose: bool = True,
        **kwargs
):

    batch_instruction_input_pairs = [
        (instruction, truncate_input({"input": context, "instruction": instruction}, tokenizer, prompter, cutoff_len))
        for context, instruction in zip(inputs, instructions)
    ]
    # prompts = [prompter.generate_prompt(instruction, for context, instruction in zip(inputs, instructions)]
    prompts = [prompter.generate_prompt(instruction, input) for instruction, input in batch_instruction_input_pairs]

    if verbose:
        print(prompts)

    encoder_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=cutoff_len,
        padding=True,
        return_tensors="pt"
    )
    # print(encoder_inputs["input_ids"][0])

    encoder_inputs = encoder_inputs.to(device)
    if verbose:
        print(f"\nNumber of input tokens: {encoder_inputs['input_ids'].shape}\n")

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        # eos_token_id=[29958, model.config.eos_token_id], # [model.config.eos_token_id, 29958], # [model.config.eos_token_id, tokenizer.vocab["###"]],
        do_sample=do_sample,
        early_stopping=early_stopping,
        remove_invalid_values=True,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
        renormalize_logits=renormalize_logits,
        num_return_sequences=num_return_sequences,
        **kwargs
    )

    with torch.no_grad():
        generation_output = model.generate(
            **encoder_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )

    s = generation_output.sequences
    prompt_length = encoder_inputs['input_ids'].shape[1]
    return tokenizer.batch_decode(s[:, prompt_length:], skip_special_tokens=True), s.shape[-1]
