"""
Universal-NER official evaluation script from https://github.com/universal-ner/universal-ner/tree/main/src/eval

For each document-query samplem, the list of pred answers is compared with the list of gold answers

1. both inputs (preds and golds) are assumed to be JSON dumped lists json.dumps(list)
2. for both the enclosing [] brakets are searched and then json.load is performed
3. each text item in the lists is normalized removing punctuation, articles and extra whitespaces
4. comparison is performed using == between strings BUT in parser function a SET of preds and golds is constructed,
    thus we compute scores on SET of unique normalized strings

Because SETs of pred and gold lists are compared, the support of each NE is different from original BIO-labeling

In Zero-shot evaluation this metric relaxation may help for better evaluation of what is able to hit,
without penalizing not returning all occurrences for a same NE span

"""

import re
import json
import string

def normalize_answer(s):
    """ FROM SQUAD2: lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parser(text):
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = []
        for item in items:
            if isinstance(item, list) or isinstance(item, tuple):
                item = tuple([normalize_answer(element) for element in item])
            else:
                item = normalize_answer(item)
            if item not in formatted_items:
                formatted_items.append(item)
        return formatted_items
    except Exception:
        return []


def corrected_parser(text):
    try:
        match = re.match(r'\[(.*?)\]', text)
        if match:
            text = match.group()
        else:
            text = '[]'
        items = json.loads(text)
        formatted_items = set()  # Use set to ensure uniqueness
        for item in items:
            if isinstance(item, (list, tuple)):
                formatted_items.add(tuple(normalize_answer(element) for element in item))
            else:
                formatted_items.add(normalize_answer(item))
        return list(formatted_items)  # Convert set back to list before returning
    except Exception as e:
        print(e)
        return []


def compute_overlap_percentage(normalized_str1, normalized_str2):
    # split the strings into sets of words
    words1 = set(normalized_str1.split())
    words2 = set(normalized_str2.split())

    # Calculate the number of overlapping words
    overlap_count = len(words1.intersection(words2))

    # Calculate the percentage of overlapping words relative to the total number of words in the longer string
    max_word_count = max(len(words1), len(words2))
    if max_word_count == 0:  # Handle division by zero
        return 0.0
    return overlap_count / max_word_count


def compute_words_overlap_score(normalized_pred_text, normalized_gold_text):
    pred_words = set(normalized_pred_text.split())
    gold_words = set(normalized_gold_text.split())

    # precision
    true_positives = len(pred_words.intersection(gold_words))
    if true_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / len(pred_words)

    # recall
    if len(gold_words) == 0:
        recall = 0.0
    else:
        recall = true_positives / len(gold_words)

    # F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


class NEREvaluator:
    @staticmethod
    def evaluate(preds: list, golds: list):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for pred, gold in zip(preds, golds):
            gold_tuples = parser(gold)
            pred_tuples = parser(pred)
            for t in pred_tuples:
                if t in gold_tuples:
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'TP': n_correct,
            'FP': n_pos_pred - n_correct,
            'FN': n_pos_gold - n_correct,
            'support': n_pos_gold
        }

    @staticmethod
    def partial_evaluate(preds: list, golds: list):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for pred, gold in zip(preds, golds):
            gold_tuples = parser(gold)
            pred_tuples = parser(pred)
            for t in pred_tuples:
                # if t in gold_tuples:
                # if any(compute_overlap_percentage(t, gold) >= 0.5 for gold in gold_tuples):
                if any(compute_words_overlap_score(t, gold) >= 0.5 for gold in gold_tuples):
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'TP': n_correct,
            'FP': n_pos_pred - n_correct,
            'FN': n_pos_gold - n_correct,
            'support': n_pos_gold
        }


"""
def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    data_path: str = './src/eval/test_data/CrossNER_AI.json',
    tensor_parallel_size: int = 1,
):
    with open(data_path, 'r') as fh:
        examples = json.load(fh)
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    golds = [example['conversations'][-1]['value'] for example in examples]
    outputs = inference(llm, examples)

    eval_result = NEREvaluator().evaluate(outputs, golds)
    print(f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}')

if __name__ == "__main__":
    # universal-ner function
    # from src.serve.inference import inference
    # from vllm import LLM  # for faster LLMs inference
    # import fire
    fire.Fire(main)
"""


if __name__ == "__main__":
    #preds = ['expert.ai', 'expert AI']
    #golds = ['expert.ai', 'expertAI', 'expert AI']

    preds = ['pippo', 'caio', 'caio']
    golds = ['pippo', 'pluto', 'paperino', 'pippo']

    preds = json.dumps(preds)
    golds = json.dumps(golds)
    print(preds)
    print(golds)

    scores = NEREvaluator().evaluate([preds], [golds])
    print(scores)
