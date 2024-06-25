"""
Evaluating ExtremITLLaMA for zero-shot NER on Italian datasets

- Using provided uniNER official evaluation script

- Using vLLM library for faster inference
"""

__package__ = "SFT_finetuning.evaluating"

import shutil

# use vllm_pip_container.sif
# noinspection PyUnresolvedReferences
from vllm import LLM, SamplingParams

from datasets import Dataset, DatasetDict, load_dataset
from collections import defaultdict
import numpy as np
import argparse
import json
import sys
import os
import re

# copy of uniNER official eval script from their github
import uniNER_official_eval_script

# my libraries
from src.data_handlers.KIND import KIND
from src.data_handlers.Multinerd_it import Multinerd_it

from ..commons.initialization import get_HF_access_token
from ..commons.preprocessing import truncate_input
from ..commons.prompter import Prompter

def load_or_build_dataset_GenQA_format(datasets_cluster_name, subdataset_name, data_handler, with_definition):
    path_to_BIO = f'./datasets/{datasets_cluster_name}'
    if datasets_cluster_name == 'KIND':
        path_to_BIO += '/evalita-2023'
    path_to_guidelines = None
    dataset_manager = data_handler(
        path_to_BIO,
        path_to_templates='./src/templates',
        SLIMER_prompter_name='SLIMER_instruction_it',  # will use ExtremITLLaMA instruction in convert_dataset_for_ExtremITLLaMA function
        path_to_DeG='',
        test_only=True)

    test_dataset = dataset_manager.convert_dataset_for_ExtremITLLaMA()['test']
    if subdataset_name != []:
        def filter_by_prefix(dataset, subdata_name):
            def filter_function(example):
                id_prefix = example["id"].split(":")[0]  # id and not doc_tag_id
                return id_prefix == subdata_name
            return dataset.filter(filter_function)

        # Apply the filter
        test_dataset = filter_by_prefix(test_dataset, subdataset_name)

    return test_dataset


if __name__ == '__main__':

    to_eval_on = [
        {'datasets_cluster_name': 'KIND', 'data_handler': KIND, 'subdataset_names': ['WN', 'FIC', 'ADG']},
        {'datasets_cluster_name': 'Multinerd_it', 'data_handler': Multinerd_it, 'subdataset_names': ['it']}
    ]

    HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    print("ExtremITLLaMA ZERO-SHOT NER EVALUATIONS with UniNER official eval script:\n")

    WITH_DEFINITION = False
    print(f"\nWith definition: {WITH_DEFINITION}")

    partial_evaluate = False
    print(f"\npartial_evaluate: {partial_evaluate}")

    prompter = Prompter('camoscio_italian', template_path='./src/SFT_finetuning/templates', eos_text='')

    model_path_or_name = "./hf_cache_dir/ExtremITLLaMA"
    print(f"LLM model: {model_path_or_name}")

    max_new_tokens = 256
    print(f"max_new_tokens {max_new_tokens}")

    vllm_model = LLM(model=model_path_or_name, download_dir='./hf_cache_dir')

    tokenizer = vllm_model.get_tokenizer()

    #sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop=['</s>'])
    #sampling_params = SamplingParams(temperature=0.2, top_p=0.75)

    # beam search generation
    sampling_params = SamplingParams(
        n=1,  # number of output sequences to return for the given prompt,
        best_of=4,  # from these `best_of` sequences, the top `n` are returned, treated as the beam width when `use_beam_search` is True
        use_beam_search=True,
        early_stopping='never',  # stopping condition for beam search
        temperature=0,
        top_p=1,
        top_k=-1
    )

    print(sampling_params)

    for data in to_eval_on:

        for subdataset_name in data['subdataset_names']:

            print(f"\n\nEvaluating model named '{model_path_or_name.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            cutoff_len = 768
            print(f"cutoff_len: {cutoff_len}")

            dataset_GenQA_format = load_or_build_dataset_GenQA_format(data['datasets_cluster_name'], subdataset_name, data['data_handler'], WITH_DEFINITION)
            print(dataset_GenQA_format)
            print(dataset_GenQA_format[0])
            sys.stdout.flush()

            all_gold_answers = dataset_GenQA_format['output']

            instructions = dataset_GenQA_format['instruction']
            print(instructions[0])
            sys.stdout.flush()

            inputs = dataset_GenQA_format['input']

            batch_instruction_input_pairs = [
                (instruction,
                 truncate_input({"input": context, "instruction": instruction}, tokenizer, prompter, cutoff_len))
                for context, instruction in zip(inputs, instructions)
            ]

            prompts = [prompter.generate_prompt(instruction, input) for instruction, input in batch_instruction_input_pairs]

            responses = vllm_model.generate(prompts, sampling_params)

            # should be already ordered by the vLLM engine
            responses_corret_order = []
            response_set = {response.prompt: response for response in responses}
            for prompt in prompts:
                assert prompt in response_set
                responses_corret_order.append(response_set[prompt])
            responses = responses_corret_order
            all_pred_answers = [output.outputs[0].text.strip() for output in responses]

            preds_to_save = []
            for i, sample in enumerate(dataset_GenQA_format):
                preds_to_save.append({
                    'id': sample['id'],
                    'gold_answers': all_gold_answers[i],
                    'pred_answers': all_pred_answers[i]
                })

            path_to_save_predictions = os.path.join("./predictions", model_path_or_name.split('/')[-1])
            if not os.path.exists(path_to_save_predictions):
                os.makedirs(path_to_save_predictions)
            with open(os.path.join(path_to_save_predictions, subdataset_name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(preds_to_save, f, ensure_ascii=False, indent=2)
            print("\n")

    print("\nDONE :)")
    sys.stdout.flush()
