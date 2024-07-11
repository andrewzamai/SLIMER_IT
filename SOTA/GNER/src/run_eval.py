"""
Run uniNER official eval script over GNER predictions
"""
import os.path

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

from src.SOTA.GNER.src import gner_evaluator


if __name__ == '__main__':

    # load tokenizer to tokenize and parse GNER predictions
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    #tokenizer = AutoTokenizer.from_pretrained("swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA")
    tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")

    test_name = 'ADG'
    # test datasets with predictions
    test_set = load_dataset("json", data_files=f'../data/GNER-EN-vllm/{test_name}.jsonl')['train']
    print(test_set)
    print(test_set[0])

    """ Load labels set """
    path_to_test_data = f'../data/{test_name}'
    with open(os.path.join(path_to_test_data, 'label.txt'), 'r') as f:
        ne_types_list = f.readlines()
    ne_types_list = [n.strip() for n in ne_types_list]
    print(ne_types_list)

    """ 3) compute scores """
    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}
    #exclude_this_nes = ['persona', 'organizzazione', 'luogo', 'entità biologica']
    exclude_this_nes = []
    exc = [scores_per_NE.pop(ne) for ne in exclude_this_nes]

    for sample in test_set:
        words = sample['instance']['words']
        labels = sample['instance']['labels']
        predictions = gner_evaluator.extract_predictions(sample, tokenizer)
        gold_tuples = gner_evaluator.parser(words, labels)
        pred_tuples = gner_evaluator.parser(words, predictions)
        for t in pred_tuples:
            if t in gold_tuples and t[-1] not in exclude_this_nes:
                n_correct += 1
                scores_per_NE[t[-1]]['n_correct'] += 1
            n_pos_pred += 1
            if t[-1] not in exclude_this_nes:
                scores_per_NE[t[-1]]['n_pos_pred'] += 1
        n_pos_gold += len(gold_tuples)
        for g_t in gold_tuples:
            if g_t[-1] not in exclude_this_nes:
                scores_per_NE[g_t[-1]]['n_pos_gold'] += 1

    prec = n_correct / (n_pos_pred + 1e-10)
    recall = n_correct / (n_pos_gold + 1e-10)
    f1 = 2 * prec * recall / (prec + recall + 1e-10)

    precision = round(prec * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)
    print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(test_name, precision, recall, f1))

    macro_precision = []
    macro_recall = []
    macro_f1 = []
    for ne, ne_scores in scores_per_NE.items():
        prec = ne_scores['n_correct'] / (ne_scores['n_pos_pred'] + 1e-10)
        recall = ne_scores['n_correct'] / (ne_scores['n_pos_gold'] + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)

        precision = round(prec * 100, 2)
        macro_precision.append(precision)
        recall = round(recall * 100, 2)
        macro_recall.append(recall)
        f1 = round(f1 * 100, 2)
        macro_f1.append(f1)
        print("{} --> support: {}".format(ne, ne_scores['n_pos_gold']))
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(ne, ne_scores['n_correct'], ne_scores['n_pos_gold'] - ne_scores['n_correct'], ne_scores['n_pos_pred'] - ne_scores['n_correct'], -1))
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(ne, precision, recall, f1))
        print("------------------------------------------------------- ")
    print("\n{} ==> Macro-Precision: {:.2f}, Macro-Recall: {:.2f}, Macro-F1: {:.2f}".format(test_name, np.average(macro_precision), np.average(macro_recall), np.average(macro_f1)))
