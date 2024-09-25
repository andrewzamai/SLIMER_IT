from collections import defaultdict
from typing import List, Tuple
import json


def chunk_document_with_sliding_window(
        document_input: str,
        window_size: int = 900,
        overlap: int = 15
) -> List[str]:
    """Splits a long document into chunks of specified window size with overlapping words.

    Args:
        document_input (str): The input document text.
        window_size (int): The length of each chunk. Default is 900 words.
        overlap (int): The number of overlapping words between chunks. Default is 15.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    end = window_size
    while start < len(document_input):
        chunk_inputs = document_input[start:end]
        chunks.append(chunk_inputs)
        start += window_size - overlap
        end += window_size - overlap
    # discard last chunk if shorter than 20 words
    if len(chunks[-1].split()) < 20:
        chunks = chunks[:-1]

    return chunks

def aggregate_preds_from_chunks(dataset_SLIMER_format, chunks_per_sample, all_pred_answers):
    all_pred_answers_aggregated = []
    # for sample_ID, chunks_indices in chunks_per_sample.items():
    for sample in dataset_SLIMER_format:
        sampleID = sample['doc_tag_pairID']
        chunks_indices = chunks_per_sample[sampleID]
        document_level_preds = set()
        for idx in chunks_indices:
            this_chunk_preds = all_pred_answers[idx]
            try:
                this_chunk_preds = json.loads(this_chunk_preds)
            except:
                this_chunk_preds = []
            for pred in this_chunk_preds:
                # add only if text prediction and not evaluates to other types e.g. dict
                if isinstance(pred, str):
                    document_level_preds.add(pred)
        document_level_preds = json.dumps(list(document_level_preds))
        all_pred_answers_aggregated.append(document_level_preds)

    return all_pred_answers_aggregated


def filter_by_prefix(dataset, subdata_name):
    def filter_function(example):
        id_prefix = example["doc_tag_pairID"].split(":")[0]
        return id_prefix == subdata_name

    return dataset.filter(filter_function)
