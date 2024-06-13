"""
MultiNER-ITA partition
https://huggingface.co/datasets/tner/multinerd
"""

from datasets import Dataset, DatasetDict, load_dataset
import json
import os

from Data_Interface import Data_Interface

class Multinerd_it(Data_Interface):

    def load_datasetdict_BIO(self, path_to_BIO):
        data_files = {
            "train": os.path.join(path_to_BIO, 'train_it' + '.jsonl'),
            "validation": os.path.join(path_to_BIO, 'val_it' + '.jsonl'),
            "test": os.path.join(path_to_BIO, 'test_it' + '.jsonl')
        }
        dataset_dict_BIO = load_dataset("json", data_files=data_files)

        with open(os.path.join(path_to_BIO, 'tag_to_id_map.jsonl'), 'r') as file:
            tag_to_id_map = json.load(file)
        id_to_tag_map = {v: k for k, v in tag_to_id_map.items()}

        new_dataset_dict_BIO_list = {split: [] for split in dataset_dict_BIO.keys()}
        for split_name, dataset in dataset_dict_BIO.items():
            progressive_ID = 0
            for sample in dataset:
                ner_tags_ids = sample['ner_tags']
                ner_tags_BIO = [id_to_tag_map[x] for x in ner_tags_ids]

                new_dataset_dict_BIO_list[split_name].append({
                    "tokens": sample['tokens'],
                    "labels": ner_tags_BIO,
                    "document_id": split_name + ':' + str(progressive_ID)
                })
                progressive_ID += 1

        return DatasetDict({split: Dataset.from_list(values) for split, values in new_dataset_dict_BIO_list.items()})


if __name__ == '__main__':

    path_to_BIO = '../../datasets/multinerd'

    Multinerd_it_dataset = Multinerd_it(path_to_BIO)
    dataset_statistics = Multinerd_it_dataset.get_dataset_statistics()
    print(dataset_statistics)

    dataset_dict_BIO = Multinerd_it_dataset.datasetdict_BIO

    print(dataset_dict_BIO.keys())
    print(dataset_dict_BIO['train'][0:10])

    ne_categories = Multinerd_it_dataset.get_ne_categories()
    print(ne_categories)

    sample_BIO = dataset_dict_BIO['train'][0]
    sample_w_gold_spans = Multinerd_it_dataset.extract_gold_spans_per_ne_category(sample_BIO)
    print(sample_w_gold_spans)

