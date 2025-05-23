""" Custom NER reward functions for GRPO training."""

from collections import defaultdict
import numpy as np
import math
import json
import re


from src.SFT_finetuning.evaluating import uniNER_official_eval_script


def parse_json_pred(response, input, output):
    """
    Evaluate json prediction to dictionary.
    """
    all_good_parsing = True
    try:
        parsed_response = json.loads(response)
    except json.JSONDecodeError:
        all_good_parsing = False
        parsed_response = {}
    try:
        parsed_gold_output = json.loads(output)
    except json.JSONDecodeError:
        all_good_parsing = False
        parsed_gold_output = {}

    # check for hallucinated types (unexpected keys)
    expected_keys = set(parsed_gold_output.keys())
    keys_in_response = set(parsed_response.keys())

    # identify (and remove) unexpected (hallucinated) keys
    unexpected_keys = keys_in_response - expected_keys
    if unexpected_keys:
        all_good_parsing = False
    #for key in unexpected_keys:
        #parsed_response.pop(key)

    # check for missing keys or not parsable
    for key in expected_keys:
        # if missing key set all_good_parsing to False
        value = parsed_response.get(key, None)
        #value = parsed_response.get(key, [])
        if not value or not isinstance(value, list):
            all_good_parsing = False
            #parsed_response[key] = []
        else:
            # if any is not string set all_good_parsing to False
            if not all(isinstance(x, str) for x in value):
                all_good_parsing = False
            # if not str pop it
            value = [x for x in value if isinstance(x, str)]
            #parsed_response[key] = value

    # remove hallucinated text spans
    #for key, values in parsed_response.items():
        #parsed_response[key] = [text_span for text_span in values if text_span in input]

    return parsed_gold_output, parsed_response, all_good_parsing


def format_reward(completions, input, output, **kwargs):
    """
    Format reward function scoring:
    - 0.33 for <think> tags
    - 0.33 for presence of valid ```json block
    - 0.33 for valid JSON parsing of the response without hallucinated types
    """
    pattern_json = r"```json\n([\s\S]*?)\n```$"
    pattern_think = r"<think>([\s\S]*?)</think>"

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, inp, outp in zip(completion_contents, input, output):
        all_good_parsing = False
        reward = 0.0
        # Check for <think>...</think>
        # if <think> suffixed to assistant add generation prompt opening tag will be not in the completion
        closing_think_count = content.lower().count("</think>\n\n```json")
        if closing_think_count == 1:
            reward += 0.5

        matches_json = re.findall(pattern_json, content, re.DOTALL | re.MULTILINE)
        if len(matches_json) == 1:
            reward += 0.5

        match_json = re.search(pattern_json, content, re.DOTALL | re.MULTILINE)
        if match_json:
            json_block = match_json.group(1)
            try:
                parsed_gold_output, parsed_response, all_good_parsing = parse_json_pred(json_block, inp, outp)
            except Exception:
                all_good_parsing = False
                parsed_response = {}
                parsed_gold_output = {}
        
        #if all_good_parsing:
            #reward += 0.5
        
        rewards.append(reward)

    return rewards


def macrof1_reward(completions, input, output, **kwargs):
    """
    Compute macro-F1 reward by comparing predicted entities with the ground-truth output.
    
    Args:
        completions (list of dict): List of model-generated responses.
        output (list of str): List of ground-truth entities in JSON output.
        
    Returns:
        list of float: Reward scores for each response.
    """
    pattern_json = r"</think>\n\n```json\n([\s\S]*?)\n```$"
    evaluator = uniNER_official_eval_script.NEREvaluator()

    rewards = []
    for pred_completion, inp, outp in zip(completions, input, output):
        # Extract assistant's content if in conversational format
        pred_completion = pred_completion[0]['content']
        try:
            parsed_gold_output = json.loads(outp)
        except:
            parsed_gold_output = {}
        parsed_response = {}
        all_good_parsing = False

        # Extract JSON content
        match_json = re.search(pattern_json, pred_completion, re.DOTALL | re.MULTILINE)
        json_extracted = False
        if match_json:
            json_block = match_json.group(1)
            json_extracted = True
            try:
                parsed_gold_output, parsed_response, all_good_parsing = parse_json_pred(json_block, inp, outp)
            except Exception:
                pass  # Parsing failed, keep default empty dicts
        
        f1_scores = []
        for tagName, gold_vals in parsed_gold_output.items():
            if tagName not in parsed_response:
                f1_scores.append(0.0)
            else:
                eval_result = evaluator.evaluate_single(parsed_response[tagName], gold_vals)
                f1_scores.append(eval_result['f1'])

        """
        all_pred_answers_per_type = defaultdict(list)
        all_gold_answers_per_type = defaultdict(list)

        for tagName, pred_vals in parsed_response.items():
            all_pred_answers_per_type[tagName].extend(pred_vals)
        for tagName, gold_vals in parsed_gold_output.items():
            all_gold_answers_per_type[tagName].extend(gold_vals)

        f1_scores = []
        for tagName in all_gold_answers_per_type.keys():
            tag_gold = all_gold_answers_per_type[tagName]
            tag_pred = all_pred_answers_per_type[tagName]

            eval_result = evaluator.evaluate_single(tag_pred, tag_gold)

            f1_scores.append(eval_result['f1'])  # raw float [0, 1]
        """

        # If no tag evaluated, fallback to 0
        if f1_scores:
            macro_f1 = round(sum(f1_scores) / len(f1_scores), 4)
        else:
            macro_f1 = 0.0

        rewards.append(macro_f1)
        
    return rewards


def microf1_reward(completions, input, output, **kwargs):
    """
    Compute micro-F1 reward by comparing predicted entities with the ground-truth output.
    
    Args:
        completions (list of dict): List of model-generated responses.
        output (list of str): List of ground-truth entities in JSON output.
        
    Returns:
        list of float: Reward scores for each response.
    """
    pattern_json = r"</think>\n\n```json\n([\s\S]*?)\n```$"
    evaluator = uniNER_official_eval_script.NEREvaluator()

    rewards = []
    for pred_completion, inp, outp in zip(completions, input, output):
        # Extract assistant's content if in conversational format
        pred_completion = pred_completion[0]['content']
        try:
            parsed_gold_output = json.loads(outp)
        except:
            parsed_gold_output = {}
        parsed_response = {}
        all_good_parsing = False

        # Extract JSON content
        match_json = re.search(pattern_json, pred_completion, re.DOTALL | re.MULTILINE)
        json_extracted = False
        if match_json:
            json_block = match_json.group(1)
            json_extracted = True
            try:
                parsed_gold_output, parsed_response, all_good_parsing = parse_json_pred(json_block, inp, outp)
            except Exception:
                pass  # Parsing failed, keep default empty dicts
        
        # Aggregate TP, FP, FN for all tags
        total_tp = total_fp = total_fn = 0

        for tagName, gold_vals in parsed_gold_output.items():
            pred_vals = parsed_response.get(tagName, [])
            eval_result = evaluator.evaluate_single(pred_vals, gold_vals)
            total_tp += eval_result.get('TP', 0)
            total_fp += eval_result.get('FP', 0)
            total_fn += eval_result.get('FN', 0)

       # If both gold and predicted outputs are effectively empty, it's a perfect match
        if total_tp == 0 and total_fp == 0 and total_fn == 0:
            f1 = 1.0
        else:
            # Compute micro F1 for non-empty cases
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1 = 1 if f1 == 1 else 0 # Apply the binary reward (keeps 1.0 as 1, others as 0)

        if not json_extracted:
            f1 = 0.0
    
        rewards.append(f1)
        
    return rewards


if __name__ == "__main__":

    from datasets import load_dataset
    import os
    dataset = load_dataset("andrewzamai/SLIMER_PARALLEL_pileNER_top391NEs_TrueDef_GRPO", keep_in_memory=True)
    # Format into fake conversation just to have prompt column
    def make_conversation(example):
        prompt = []
        prompt.append({"role": "user", "content": example["instruction"]})
        return {"prompt": prompt}
    dataset = dataset.map(make_conversation)

    # load an example of possible reasoning and final pred output
    file_path = os.path.join('./src/RLtraining/src/training', 'debug_sample.md')
    with open(file_path, "r", encoding="utf-8") as file:
        example_pred_completion_content = file.read()
    print(example_pred_completion_content)

    # duplicate pred_completion as many samples in the dataset
    import copy
    # enclose it conversational format
    pred_completion = [{"role": "assistant", "content": example_pred_completion_content}]
    pred_completion = [copy.deepcopy(pred_completion) for _ in range(len(dataset['train']['output']))]
   
    #rewards = accuracy_reward(pred_completion, dataset['train']['gold_diagnosis'])
    #print(rewards)

    format_rewards = format_reward(pred_completion, dataset['train']['input'], dataset['train']['output'])
    print(format_rewards)

    rewards = microf1_reward(pred_completion, dataset['train']['input'], dataset['train']['output'])
    print(rewards)
    