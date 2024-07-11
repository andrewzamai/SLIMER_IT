import json
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from src.SOTA.GNER.src import gner_evaluator
from src.data_handlers.KIND import KIND


if __name__ == '__main__':

    path_to_KIND_BIO = '../../../../datasets/KIND/evalita-2023'
    test_KIND_BIO = KIND(path_to_KIND_BIO).datasetdict_BIO['test']
    print(test_KIND_BIO[0])

    #path_to_predictions = "../data/predictions_bert-base-italian-cased.json"
    path_to_predictions = "../data/predictions_BERT_base_trained_on_WN_ITA_ziorufus.json"
    with open(path_to_predictions, "r") as file:
        BIOpreds_per_sample = json.load(file)
    #print(BIOpreds_per_sample)
    # from list to dict
    BIOpreds_per_sample = {x['id']: x for x in BIOpreds_per_sample}

    dataset_name = 'ADG'
    test_KIND_BIO = test_KIND_BIO.filter(lambda sample: sample['id'].split(':')[0] == dataset_name)
    print(len(test_KIND_BIO))

    all_gold_tuples_per_doc = defaultdict(set)
    all_pred_tuples_per_doc = defaultdict(set)
    for sample in test_KIND_BIO:

        gold_tuples = gner_evaluator.parser(sample['tokens'], sample['labels'])
        this_sample_preds = BIOpreds_per_sample[sample['id']]['prediction']
        pred_tuples = gner_evaluator.parser(sample['tokens'], this_sample_preds)

        print(gold_tuples)
        print(pred_tuples)
        print("----------------------")

        id = sample['id']
        all_gold_tuples_per_doc[id].update(gold_tuples)
        all_pred_tuples_per_doc[id].update(pred_tuples)

    print(len(all_gold_tuples_per_doc.keys()))
    print(len(all_pred_tuples_per_doc.keys()))
    print(list(all_gold_tuples_per_doc.values())[:10])
    print(list(all_pred_tuples_per_doc.values())[:10])

    """ 3) compute scores """
    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    ne_types_list = ['per', 'org', 'loc']
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}

    for id in set(test_KIND_BIO['id']):
        pred_tuples = all_pred_tuples_per_doc[id]
        gold_tuples = all_gold_tuples_per_doc[id]
        for t in pred_tuples:
            if t in gold_tuples:
                n_correct += 1
                scores_per_NE[t[-1]]['n_correct'] += 1
            n_pos_pred += 1
            scores_per_NE[t[-1]]['n_pos_pred'] += 1
        n_pos_gold += len(gold_tuples)
        for g_t in gold_tuples:
            scores_per_NE[g_t[-1]]['n_pos_gold'] += 1

    prec = n_correct / (n_pos_pred + 1e-10)
    recall = n_correct / (n_pos_gold + 1e-10)
    f1 = 2 * prec * recall / (prec + recall + 1e-10)

    precision = round(prec * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)
    print("\n{} ==> micro-Precision: {:.2f}, micro-Recall: {:.2f}, micro-F1: {:.2f}".format(dataset_name, precision, recall, f1))

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
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(ne, ne_scores['n_correct'],
                                                             ne_scores['n_pos_gold'] - ne_scores['n_correct'],
                                                             ne_scores['n_pos_pred'] - ne_scores['n_correct'], -1))
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(ne, precision, recall, f1))
        print("------------------------------------------------------- ")

    print("\n{} ==> Macro-Precision: {:.2f}, Macro-Recall: {:.2f}, Macro-F1: {:.2f}".format(dataset_name, np.average(macro_precision), np.average(macro_recall), np.average(macro_f1)))