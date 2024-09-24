

def tokenize(tokenizer, prompt, max_seq_len, add_eos_token=True, add_special_tokens=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_special_tokens
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_seq_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point, tokenizer, prompter, cutoff_len, train_on_inputs=False):
    if train_on_inputs:
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len)

    else:
        truncated_input = ''
        if "input" in data_point and data_point["input"]:
            truncated_input = truncate_input(data_point, tokenizer, prompter, cutoff_len)

        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            truncated_input,
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len)

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], truncated_input
        )
        tokenized_user_prompt = tokenize(tokenizer, user_prompt, cutoff_len, add_eos_token=False, add_special_tokens=True)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

    return tokenized_full_prompt


def truncate_input(data_point, tokenizer, prompter, cutoff_len):
    """
    Applies truncation in the input context, to avoid truncation of other parts
    of the prompt template (e.g. instruction, output, etc..).

    :return a string with the truncated input text.
    """
    if "input" not in data_point or not data_point["input"]:
        return ''

    prompt = prompter.generate_prompt(
        data_point["instruction"], ' '
    )  # prompt without the input

    prompt_tokens = tokenize(tokenizer, prompt, cutoff_len, add_eos_token=False, add_special_tokens=True)
    prompt_length = len(prompt_tokens["input_ids"])  # length of prompt without the input

    labels_length = 0
    if "output" in data_point:
        tokenized_labels = tokenize(
            tokenizer, data_point["output"], min(cutoff_len - prompt_length, 1024),
            add_eos_token=True,
            add_special_tokens=False
        )
        labels_length = len(tokenized_labels["input_ids"])  # output length
    elif "max_gen_length" in data_point:
        labels_length = data_point["max_gen_length"]

    len_margin = 16  # tokenization may be different after combining text in template. So we take a small tokens margin
    truncated_input = ''  # new truncated input
    input_residual_len = max(cutoff_len - (labels_length + prompt_length + len_margin), 0)
    if input_residual_len > 0:
        tokenized_input = tokenize(
            tokenizer, data_point["input"],
            max_seq_len=input_residual_len,
            add_eos_token=False,
            add_special_tokens=False
        )

        # decodes truncated input
        truncated_input = tokenizer.decode(
            tokenized_input["input_ids"],
            skip_special_tokens=True
        )

    return truncated_input
