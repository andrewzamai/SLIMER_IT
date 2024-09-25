"""
Evaluating SLIMER-IT (Llama-3-8B) for zero-shot NER on Italian datasets

- Using provided uniNER official evaluation script

- Using vLLM library for faster inference
"""

import shutil

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
        test_only=True,
    )

    test_dataset = dataset_manager.dataset_dict_SLIMER["test"]
    if subdataset_name:
        test_dataset = filter_by_prefix(test_dataset, subdataset_name)

    return test_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Evaluate SLIMER-IT's Zero-Shot NER performance"""
    )
    parser.add_argument("merged_model_name", type=str, help="path_to_merged_model")
    parser.add_argument("model_template_name", type=str, help="e.g. llama3_italian")
    parser.add_argument(
        "--with_guidelines", action="store_true", help="Whether to use Def & Guidelines"
    )
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
        model=model_path_or_name, max_model_len=cutoff_len
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

            print(
                f"\n\nEvaluating model named '{model_path_or_name.split('/')[-1]}' on '{subdataset_name}' test fold in ZERO-SHOT setting\n"
            )

            dataset_SLIMER_format = load_or_build_dataset_GenQA_format(
                data["datasets_cluster_name"],
                subdataset_name,
                data["data_handler"],
                WITH_DEFINITION,
            )
            print(dataset_SLIMER_format)
            print(dataset_SLIMER_format[0])
            sys.stdout.flush()

            # 2) for each tagName save the indices of the associated samples
            indices_per_tagName = defaultdict(list)
            for i, sample in enumerate(dataset_SLIMER_format):
                indices_per_tagName[sample["tagName"]].append(i)

            # 3) input, instructions and gold answers
            all_gold_answers = dataset_SLIMER_format["output"]

            instructions = dataset_SLIMER_format["instruction"]
            print("\n" + instructions[0] + "\n")
            sys.stdout.flush()

            inputs = dataset_SLIMER_format["input"]

            # 4) chunk long inputs into multiple chunks
            inputs_chunked_for_inference = False
            if data["datasets_cluster_name"] != "BUSTER":
                batch_instruction_input_pairs = [
                    (
                        instruction,
                        truncate_input(
                            {"input": context, "instruction": instruction},
                            tokenizer,
                            prompter,
                            cutoff_len,
                        ),
                    )
                    for context, instruction in zip(inputs, instructions)
                ]
            else:
                inputs_chunked_for_inference = True
                # for each doc_tag_pairID save a list of indices of its generated chunks
                batch_instruction_input_pairs, chunks_per_sample = [], defaultdict(list)
                chunk_id = 0
                for sample in dataset_SLIMER_format:
                    chunks = chunk_document_with_sliding_window(
                        sample["input"], window_size=900, overlap=15
                    )
                    chunks_per_sample[sample["doc_tag_pairID"]].extend(
                        range(chunk_id, chunk_id + len(chunks))
                    )
                    batch_instruction_input_pairs.extend(
                        (sample["instruction"], chunk) for chunk in chunks
                    )
                    chunk_id += len(chunks)
                print(
                    f"Number of samples num_NE x n_chunks: {len(batch_instruction_input_pairs)}"
                )
                sys.stdout.flush()

            # 5) run inference on SLIMER via vLLM
            prompts = [
                prompter.generate_prompt(instruction, input)
                for instruction, input in batch_instruction_input_pairs
            ]
            responses = vllm_model.generate(prompts, sampling_params)

            # 6) should be already ordered by the vLLM engine, we ensure that
            responses_corret_order = []
            response_set = {response.prompt: response for response in responses}
            for prompt in prompts:
                assert prompt in response_set
                responses_corret_order.append(response_set[prompt])
            responses = responses_corret_order

            # 7) retrieve pred answers, aggregate them from chunks back to document level
            all_pred_answers = [output.outputs[0].text.strip() for output in responses]

            if inputs_chunked_for_inference:
                all_pred_answers = aggregate_preds_from_chunks(
                    dataset_SLIMER_format, chunks_per_sample, all_pred_answers
                )

            # 8) Compute micro, perTagName, macro and weighted metrics

            print("\ngold_answers")
            print(all_gold_answers[0:10])
            print("pred_answers")
            print(all_pred_answers[0:10])

            if partial_evaluate:
                eval_result = (
                    uniNER_official_eval_script.NEREvaluator().partial_evaluate(
                        all_pred_answers, all_gold_answers
                    )
                )
            else:
                eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(
                    all_pred_answers, all_gold_answers
                )

            precision = round(eval_result["precision"] * 100, 2)
            recall = round(eval_result["recall"] * 100, 2)
            f1 = round(eval_result["f1"] * 100, 2)
            print(
                "\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(
                    subdataset_name, precision, recall, f1
                )
            )

            print("\nMetrics per NE category (100%):\n")
            this_dataset_metrics = {}
            for tagName, indices_for_this_tagName in indices_per_tagName.items():
                this_tagName_golds = [
                    gold_ans
                    for idx, gold_ans in enumerate(all_gold_answers)
                    if idx in indices_for_this_tagName
                ]
                this_tagName_preds = [
                    pred_ans
                    for idx, pred_ans in enumerate(all_pred_answers)
                    if idx in indices_for_this_tagName
                ]
                if partial_evaluate:
                    eval_result = (
                        uniNER_official_eval_script.NEREvaluator().partial_evaluate(
                            this_tagName_preds, this_tagName_golds
                        )
                    )
                else:
                    eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(
                        this_tagName_preds, this_tagName_golds
                    )
                # eval json dumps to list before counting support

                print("{} --> support: {}".format(tagName, eval_result["support"]))
                print(
                    "{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(
                        tagName,
                        eval_result["TP"],
                        eval_result["FN"],
                        eval_result["FP"],
                        -1,
                    )
                )
                precision = round(eval_result["precision"] * 100, 2)
                recall = round(eval_result["recall"] * 100, 2)
                f1 = round(eval_result["f1"] * 100, 2)
                print(
                    "{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(
                        tagName, precision, recall, f1
                    )
                )
                print("---------------------------------------------------------- ")
                this_dataset_metrics[tagName] = {
                    "support": eval_result["support"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

            # computing MACRO scores
            this_dataset_precisions = [
                this_dataset_metrics[tagName]["precision"]
                for tagName in this_dataset_metrics
            ]
            this_dataset_recalls = [
                this_dataset_metrics[tagName]["recall"]
                for tagName in this_dataset_metrics
            ]
            this_dataset_f1s = [
                this_dataset_metrics[tagName]["f1"] for tagName in this_dataset_metrics
            ]
            this_dataset_supports = [
                this_dataset_metrics[tagName]["support"]
                for tagName in this_dataset_metrics
            ]
            print(
                "\n{} ==> MACRO-Precision: {:.2f} +- {:.2f}, MACRO-Recall: {:.2f} +- {:.2f}, MACRO-F1: {:.2f} +- {:.2f}".format(
                    subdataset_name,
                    np.average(this_dataset_precisions),
                    np.std(this_dataset_precisions),
                    np.average(this_dataset_recalls),
                    np.std(this_dataset_recalls),
                    np.average(this_dataset_f1s),
                    np.std(this_dataset_f1s),
                )
            )

            # computing WEIGHTED scores
            this_dataset_supports_sum = sum(this_dataset_supports)
            this_dataset_precisions_weighted = [
                this_dataset_metrics[tagName]["precision"]
                * (this_dataset_metrics[tagName]["support"] / this_dataset_supports_sum)
                for tagName in this_dataset_metrics
            ]
            this_dataset_recalls_weighted = [
                this_dataset_metrics[tagName]["recall"]
                * (this_dataset_metrics[tagName]["support"] / this_dataset_supports_sum)
                for tagName in this_dataset_metrics
            ]
            this_dataset_f1s_weighted = [
                this_dataset_metrics[tagName]["f1"]
                * (this_dataset_metrics[tagName]["support"] / this_dataset_supports_sum)
                for tagName in this_dataset_metrics
            ]
            print(
                "\n{} ==> Weighted-Precision: {:.2f}, Weighted-Recall: {:.2f}, Weighted-F1: {:.2f}".format(
                    subdataset_name,
                    np.sum(this_dataset_precisions_weighted),
                    np.sum(this_dataset_recalls_weighted),
                    np.sum(this_dataset_f1s_weighted),
                )
            )

            # Finally, save predictions
            preds_to_save = []
            for i, sample in enumerate(dataset_SLIMER_format):
                preds_to_save.append(
                    {
                        "doc_tag_pairID": sample["doc_tag_pairID"],
                        "tagName": sample["tagName"],
                        "gold_answers": all_gold_answers[i],
                        "pred_answers": all_pred_answers[i],
                    }
                )

            path_to_save_predictions = os.path.join(
                "./predictions", args.merged_model_name.split("/")[-1]
            )
            if not os.path.exists(path_to_save_predictions):
                os.makedirs(path_to_save_predictions)
            with open(
                os.path.join(path_to_save_predictions, subdataset_name + ".json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(preds_to_save, f, ensure_ascii=False, indent=2)
            print("\n")

        print("\nDONE :)")
        sys.stdout.flush()
