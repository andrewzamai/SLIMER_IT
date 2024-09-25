"""
Evaluating SLIMER-IT for zero-shot NER on Italian datasets

- with input prefix-caching

- Using vLLM library for faster inference
"""
import time

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

from src.SFT_finetuning.commons.initialization import get_HF_access_token
from src.SFT_finetuning.commons.preprocessing import truncate_input
from src.SFT_finetuning.commons.prompter import Prompter
from src.SFT_finetuning.evaluating.eval_utils import (
    chunk_document_with_sliding_window,
    aggregate_preds_from_chunks,
    filter_by_prefix
)


def load_or_build_dataset_GenQA_format(
    datasets_cluster_name, subdataset_name, data_handler, with_definition
):
    path_to_BIO = f"./datasets/{datasets_cluster_name}"
    if datasets_cluster_name == "KIND":
        path_to_BIO += "/evalita-2023"
    path_to_guidelines = None
    if with_definition:
        path_to_guidelines = f"./src/def_and_guidelines/{datasets_cluster_name}.json"
    dataset_manager = data_handler(
        path_to_BIO,
        path_to_templates="./src/templates",
        SLIMER_prompter_name="SLIMER_instruction_it",
        path_to_DeG=path_to_guidelines,
        test_only=True
    )

    test_dataset = dataset_manager.convert_dataset_for_SLIMER_prefix_caching()["test"]
    if subdataset_name:
        def filter_by_prefix(dataset, subdata_name):
            def filter_function(example):
                id_prefix = example["id"].split(":")[0]
                return id_prefix == subdata_name
            return dataset.filter(filter_function)

        test_dataset = filter_by_prefix(test_dataset, subdataset_name)

    return test_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Evaluate SLIMER-IT's Zero-Shot NER performance"""
    )
    parser.add_argument("merged_model_name", type=str, help="path_to_merged_model")
    parser.add_argument("model_template_name", type=str, help="e.g. llama3_italian")
    parser.add_argument("--with_guidelines", action="store_true", help="Whether to use Def & Guidelines")
    args = parser.parse_args()

    to_eval_on = [
        {
            "datasets_cluster_name": "KIND",
            "data_handler": KIND,
            "subdataset_names": ["WN", "FIC", "ADG"],
        },
        {
            "datasets_cluster_name": "Multinerd_it",
            "data_handler": Multinerd_it,
            "subdataset_names": ["it"],
        },
    ]

    # HF_ACCESS_TOKEN = get_HF_access_token('./.env')

    print("SLIMER-IT ZERO-SHOT NER EVALUATIONS with UniNER official eval script:\n")

    WITH_DEFINITION = args.with_guidelines
    print(f"\nWith definition: {WITH_DEFINITION}")

    partial_evaluate = False
    print(f"\npartial_evaluate: {partial_evaluate}")

    # base_model = "LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    model_path_or_name = args.merged_model_name
    print(f"LLM model: {model_path_or_name}")

    max_new_tokens = 128
    print(f"max_new_tokens {max_new_tokens}")

    cutoff_len = 768
    print(f"cutoff_len: {cutoff_len}")

    vllm_model = LLM(
        model=model_path_or_name, max_model_len=cutoff_len, enable_prefix_caching=True
    )  # , download_dir='./hf_cache_dir')

    tokenizer = vllm_model.get_tokenizer()

    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    print(sampling_params)

    prompter = Prompter(
        args.model_template_name,
        template_path="./src/SFT_finetuning/templates",
        eos_text="",
    )

    for data in to_eval_on:

        for subdataset_name in data["subdataset_names"]:

            print(f"\n\nEvaluating model named '{model_path_or_name.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n")

            dataset_SLIMER_format = load_or_build_dataset_GenQA_format(
                data["datasets_cluster_name"],
                subdataset_name,
                data["data_handler"],
                WITH_DEFINITION,
            )
            print(dataset_SLIMER_format)
            print(dataset_SLIMER_format[0])
            sys.stdout.flush()

            for sample in dataset_SLIMER_format:
                input = sample['input']

                batch_instruction_input_pairs = [
                    (
                        x['instruction'],
                        truncate_input(
                            {"input": input, "instruction": x['instruction']},
                            tokenizer,
                            prompter,
                            cutoff_len,
                        ),
                    )
                    for x in sample['entities'].values()
                ]

                prompts = [
                    prompter.generate_prompt(instruction, input)
                    for instruction, input in batch_instruction_input_pairs
                ]

                prefix_end = prompts[0].find("Istruzione: ")
                prefix = prompts[0][0:prefix_end]
                print(prefix)

                start_time = time.time()
                responses = vllm_model.generate(prompts, sampling_params)
                end_time = time.time()
                print(f"Generation time: {end_time - start_time} seconds.")

                # 7) retrieve pred answers, aggregate them from chunks back to document level
                all_pred_answers = [output.outputs[0].text.strip() for output in responses]


