import numpy as np
import json
import os
import re

from src.SFT_finetuning.evaluating import uniNER_official_eval_script

if __name__ == '__main__':

    dataset_name = "FIC"
    w_def = False
    model_name = "LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    #model_name = ""
    with open(os.path.join(f'../../../exp_outputs/predictions/{model_name}_-1pos_-1neg_perNE_top-1NEs_{w_def}Def-IT2', f'{dataset_name}.json')) as fp:
    #with open(os.path.join(f'../../../exp_outputs/predictions/SLIMER', f'{dataset_name}.json')) as fp:
        data_w_preds = json.load(fp)

    #exclude_this_nes = ['PER', 'ORG', 'LOC', 'BIO']
    exclude_this_nes = []

    all_pred_answers = []
    all_gold_answers = []
    indices_per_tagName = {}
    count = 0
    for i, sample in enumerate(data_w_preds):
        tagName = sample['tagName']
        if tagName not in indices_per_tagName and tagName not in exclude_this_nes:
            indices_per_tagName[tagName] = []
        if tagName not in exclude_this_nes:
            indices_per_tagName[tagName].append(count)
            count += 1
            all_pred_answers.append(sample['pred_answers'])
            all_gold_answers.append(sample['gold_answers'])

    eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(all_pred_answers, all_gold_answers)
    precision = round(eval_result["precision"] * 100, 2)
    recall = round(eval_result["recall"] * 100, 2)
    f1 = round(eval_result["f1"] * 100, 2)
    print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(dataset_name, precision, recall, f1))

    print("\nMetrics per NE category (100%):\n")
    this_dataset_metrics = {}
    for tagName, indices_for_this_tagName in indices_per_tagName.items():
        #print(indices_for_this_tagName)
        #print(len(all_gold_answers))
        this_tagName_golds = [all_gold_answers[idx] for idx in indices_for_this_tagName]
        this_tagName_preds = [all_pred_answers[idx] for idx in indices_for_this_tagName]

        eval_result = uniNER_official_eval_script.NEREvaluator().evaluate(this_tagName_preds, this_tagName_golds)

        print("{} --> support: {}".format(tagName, eval_result['support']))
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(tagName, eval_result['TP'], eval_result['FN'], eval_result['FP'], -1))
        precision = round(eval_result["precision"] * 100, 2)
        recall = round(eval_result["recall"] * 100, 2)
        f1 = round(eval_result["f1"] * 100, 2)
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(tagName, precision, recall, f1))
        print("------------------------------------------------------- ")
        this_dataset_metrics[tagName] = {
            'support': eval_result['support'],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # computing MACRO scores
    this_dataset_precisions = [this_dataset_metrics[tagName]['precision'] for tagName in this_dataset_metrics]
    this_dataset_recalls = [this_dataset_metrics[tagName]['recall'] for tagName in this_dataset_metrics]
    this_dataset_f1s = [this_dataset_metrics[tagName]['f1'] for tagName in this_dataset_metrics]
    this_dataset_supports = [this_dataset_metrics[tagName]['support'] for tagName in this_dataset_metrics]
    print(
        "\n{} ==> MACRO-Precision: {:.2f} +- {:.2f}, MACRO-Recall: {:.2f} +- {:.2f}, MACRO-F1: {:.2f} +- {:.2f}".format(
            dataset_name,
            np.average(this_dataset_precisions),
            np.std(this_dataset_precisions),
            np.average(this_dataset_recalls),
            np.std(this_dataset_recalls),
            np.average(this_dataset_f1s),
            np.std(this_dataset_f1s)))

    this_dataset_supports_sum = sum(this_dataset_supports)
    this_dataset_precisions_weighted = [this_dataset_metrics[tagName]['precision'] * (
                this_dataset_metrics[tagName]['support'] / this_dataset_supports_sum) for tagName in
                                        this_dataset_metrics]
    this_dataset_recalls_weighted = [
        this_dataset_metrics[tagName]['recall'] * (this_dataset_metrics[tagName]['support'] / this_dataset_supports_sum)
        for tagName in this_dataset_metrics]
    this_dataset_f1s_weighted = [
        this_dataset_metrics[tagName]['f1'] * (this_dataset_metrics[tagName]['support'] / this_dataset_supports_sum) for
        tagName in this_dataset_metrics]
    print(
        "\n{} ==> Weighted-Precision: {:.2f}, Weighted-Recall: {:.2f}, Weighted-F1: {:.2f}".format(
            dataset_name,
            np.sum(this_dataset_precisions_weighted),
            np.sum(this_dataset_recalls_weighted),
            np.sum(this_dataset_f1s_weighted)))
