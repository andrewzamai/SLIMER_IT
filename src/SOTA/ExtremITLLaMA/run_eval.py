import numpy as np
import json
import os
import re

from src.SFT_finetuning.evaluating import uniNER_official_eval_script

def extract_list_from_gold_answers_dict(ga_dict):
    result = []
    for entity_type, entities in ga_dict.items():
        for entity in entities:
            text_span = entity[0]
            normalized_text_span = uniNER_official_eval_script.normalize_answer(text_span)
            if (entity_type, normalized_text_span) not in result:
                result.append((entity_type, normalized_text_span))
    return result

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def extract_entities(label):
    # input example: [ORG] Partito popolare italiano [LOC] Italia
    # output example: [ (ORG, Partito popolare italiano), (LOC, Italia) ]
    entities = []
    while label != "" and label != " ":
        entity_label = ""

        try:
            # extract the label
            start_i = label.index("[")
            end_i = label.index("]", start_i) + 1
            entity_label = label[start_i:end_i]
            # consume the label
            label = label.replace(entity_label, "", 1).strip()
            entity_label = entity_label.replace("[", "").replace("]", "")
        except:
            print("EXCEPTION not well formatted")
            print(label)

        # extract the span
        if "[" in label:
            span = label.split(" [")[0]
        else:
            span = label
        # consume the span
        label = label.replace(span, "", 1).strip()

        # add touple to list
        if entity_label != "":
            normalized_text_span = uniNER_official_eval_script.normalize_answer(span)
            if (entity_label, span) not in entities:
                entities.append((entity_label, normalized_text_span))

    return entities


if __name__ == '__main__':

    dataset_name = "ADG"
    with open(os.path.join('../../../exp_outputs/predictions/ExtremITLLaMA_T0', f'{dataset_name}.json')) as fp:
        data_w_preds = json.load(fp)

    print(data_w_preds[0])

    gold = json.loads(data_w_preds[10]['gold_answers'])
    print(extract_list_from_gold_answers_dict(gold))

    pred = json.loads(data_w_preds[10]['pred_answers'])
    print(pred)

    cleaned_pred = clean_input_text(pred)
    print(cleaned_pred)

    extracted = extract_entities(cleaned_pred)
    print(extracted)

    """ 3) compute scores """

    n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
    # ne_types_list = ['PER', 'ORG', 'LOC']
    # extract tag set from gold answers dict keys
    ne_types_list = list(json.loads(data_w_preds[0]['gold_answers']).keys())
    # add tag for hallucinated types
    ne_types_list.append('HALLUCINATED')
    scores_per_NE = {ne: {"n_correct": 0, "n_pos_gold": 0, "n_pos_pred": 0} for ne in ne_types_list}

    exclude_this_nes = [] #['PER', 'ORG', 'LOC']
    exc = [scores_per_NE.pop(ne) for ne in exclude_this_nes]

    for sample in data_w_preds:
        gold_tuples = extract_list_from_gold_answers_dict(json.loads(sample['gold_answers']))
        pred_tuples = extract_entities(json.loads(sample['pred_answers']))
        for t in pred_tuples:
            if t in gold_tuples and t[0] not in exclude_this_nes:
                n_correct += 1
                scores_per_NE[t[0]]['n_correct'] += 1
            n_pos_pred += 1
            if t[0] in scores_per_NE and t[0] not in exclude_this_nes:
                scores_per_NE[t[0]]['n_pos_pred'] += 1
            elif t[0] in exclude_this_nes:
                pass
            else:
                scores_per_NE['HALLUCINATED']['n_pos_pred'] += 1
        n_pos_gold += len(gold_tuples)
        for g_t in gold_tuples:
            if g_t[0] not in exclude_this_nes:
                scores_per_NE[g_t[0]]['n_pos_gold'] += 1

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
        if ne != 'HALLUCINATED':
            macro_precision.append(precision)
        recall = round(recall * 100, 2)
        if ne != 'HALLUCINATED':
            macro_recall.append(recall)
        f1 = round(f1 * 100, 2)
        if ne != 'HALLUCINATED':
            macro_f1.append(f1)
        print("{} --> support: {}".format(ne, ne_scores['n_pos_gold']))
        print("{} --> TP: {}, FN: {}, FP: {}, TN: {}".format(ne, ne_scores['n_correct'],
                                                             ne_scores['n_pos_gold'] - ne_scores['n_correct'],
                                                             ne_scores['n_pos_pred'] - ne_scores['n_correct'], -1))
        print("{} --> Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(ne, precision, recall, f1))
        print("------------------------------------------------------- ")

    print("\n{} ==> Macro-Precision: {:.2f}, Macro-Recall: {:.2f}, Macro-F1: {:.2f}".format(dataset_name,
                                                                                            np.average(macro_precision),
                                                                                            np.average(macro_recall),
                                                                                            np.average(macro_f1)))

