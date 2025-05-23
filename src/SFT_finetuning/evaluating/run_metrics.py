import os
import json
import argparse
import numpy as np
from collections import defaultdict
import re
import uniNER_official_eval_script


def parse_json_pred(sample, response):
    """
    Extract JSON from model response and match it against gold annotations.
    Returns (gold_dict, pred_dict, all_good_parsing)
    """
    all_good_parsing = True

    try:
        # Step 1: Extract code block if enclosed with ```json ... ```
        match_json = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match_json:
            response = match_json.group(1).strip()
        else:
            # Step 2: Trim anything before/including </think> if present
            think_end = re.search(r"</think>", response, re.IGNORECASE)
            if think_end:
                response = response[think_end.end():].strip()

                # Step 2.1: Fallback - extract JSON from first { to last }
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1 and end > start:
                    response = response[start:end+1].strip()
                
        # Step 4: Try to parse cleaned response as JSON
        parsed_response = json.loads(response)

    except (json.JSONDecodeError, ValueError):
        all_good_parsing = False
        parsed_response = {}

    # Step 5: Parse gold standard JSON
    try:
        parsed_gold_output = json.loads(sample['gold_answers'])
    except json.JSONDecodeError:
        all_good_parsing = False
        parsed_gold_output = {}

    # Step 6: Normalize and validate keys and value types
    expected_keys = set(parsed_gold_output.keys())
    for key in expected_keys:
        value = parsed_response.get(key, [])
        if not isinstance(value, list):
            parsed_response[key] = []
        else:
            parsed_response[key] = [x for x in value if isinstance(x, str)]

    return parsed_gold_output, parsed_response, all_good_parsing


def evaluate_predictions(data, filename): # Added filename argument
    """
    Given a list of input dicts with keys input, gold_answers, pred_answers
    compute per-tag and aggregate metrics.
    """
    all_gold_answers_per_type = defaultdict(list)
    all_pred_answers_per_type = defaultdict(list)
    safely_parsed = 0

    # Define entities to discard based on the inferred dataset name
    to_discard_NEs = []
    
    # Infer dataset cluster name from the filename
    # Example filename: "it.json"
    match = re.match(r"(\w+)\.json", filename) # Updated regex
    if match:
        dataset_name = match.group(1)
        if dataset_name == "it": # Check for "it" instead of "Multinerd_it"
            to_discard_NEs = ["entità biologica", "persona", "organizzazione", "luogo"]
            print(f"Discarding entities for dataset '{dataset_name}': {to_discard_NEs}")
    else:
        print(f"Could not infer dataset name from filename: {filename}")


    for sample in data:
        gold, pred, good = parse_json_pred(sample, sample['pred_answers'])
        if good:
            safely_parsed += 1

        # Filter gold answers
        filtered_gold = {
            tag: entities for tag, entities in gold.items() if tag.lower() not in to_discard_NEs
        }
        for tag, gold_list in filtered_gold.items():
            all_gold_answers_per_type[tag].append(gold_list)

        # Filter predicted answers
        filtered_pred = {
            tag: entities for tag, entities in pred.items() if tag.lower() not in to_discard_NEs
        }
        for tag, pred_list in filtered_pred.items():
            all_pred_answers_per_type[tag].append(pred_list)

    print(f"\nParsed safely: {safely_parsed}/{len(data)} ({safely_parsed/len(data)*100:.2f}%)\n")

    metrics_per_tag = {}
    micro_tp = micro_fp = micro_fn = 0

    # Ensure we only iterate over tags present in gold answers after filtering
    for tag in all_gold_answers_per_type:
        golds = all_gold_answers_per_type[tag]
        preds = all_pred_answers_per_type.get(tag, [[]] * len(golds))
        evaluator = uniNER_official_eval_script.NEREvaluator()
        result = evaluator.evaluate(preds, golds)

        precision = round(result["precision"] * 100, 2)
        recall = round(result["recall"] * 100, 2)
        f1 = round(result["f1"] * 100, 2)

        print(f"{tag} --> P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}, support: {result['support']}")
        metrics_per_tag[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": result["support"],
            "TP": result["TP"],
            "FP": result["FP"],
            "FN": result["FN"],
        }

        micro_tp += result["TP"]
        micro_fp += result["FP"]
        micro_fn += result["FN"]

    # MICRO
    micro_precision = 100 * micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0
    micro_recall = 100 * micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0

    print(f"\nMicro Precision: {micro_precision:.2f}, Recall: {micro_recall:.2f}, F1: {micro_f1:.2f}")

    # MACRO
    if metrics_per_tag:
        precisions = [v["precision"] for v in metrics_per_tag.values()]
        recalls = [v["recall"] for v in metrics_per_tag.values()]
        f1s = [v["f1"] for v in metrics_per_tag.values()]
        print(f"Macro Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}, "
              f"Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}, "
              f"F1: {np.mean(f1s):.2f} ± {np.std(f1s):.2f}")
    else:
        print("No entities to calculate Macro metrics after filtering.")


    # WEIGHTED
    total_support = sum([v["support"] for v in metrics_per_tag.values()])
    if total_support == 0:
        weighted_p = weighted_r = weighted_f1 = 0.0
    else:
        weighted_p = sum([v["precision"] * v["support"] for v in metrics_per_tag.values()]) / total_support
        weighted_r = sum([v["recall"] * v["support"] for v in metrics_per_tag.values()]) / total_support
        weighted_f1 = sum([v["f1"] * v["support"] for v in metrics_per_tag.values()]) / total_support

    print(f"Weighted Precision: {weighted_p:.2f}, Recall: {weighted_r:.2f}, F1: {weighted_f1:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=False, default="./output/Llama-3.1-8B-Instruct/", help='Directory containing prediction json files')
    args = parser.parse_args()

    files = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    assert files, f"No JSON files found in directory: {args.input_dir}"

    for file in files:
        print(f"\nEvaluating {file}...\n")
        path = os.path.join(args.input_dir, file)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        evaluate_predictions(data, file)

if __name__ == '__main__':
    main()