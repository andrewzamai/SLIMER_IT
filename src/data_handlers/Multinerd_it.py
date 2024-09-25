"""
Multinerd ITALIAN-partition
https://huggingface.co/datasets/Babelscape/multinerd
"""

from datasets import Dataset, DatasetDict, load_dataset
import json
import os

# ABSTRACT class which inherits from
from src.data_handlers.Data_Interface import Data_Interface

class Multinerd_it(Data_Interface):

    def load_datasetdict_BIO(self, path_to_BIO, test_only=False):
        if test_only:
            data_files = {
                "test": os.path.join(path_to_BIO, 'test_it' + '.jsonl')
            }
        else:
            data_files = {
                "train": os.path.join(path_to_BIO, 'train_it' + '.jsonl'),
                "validation": os.path.join(path_to_BIO, 'val_it' + '.jsonl'),
                "test": os.path.join(path_to_BIO, 'test_it' + '.jsonl')
            }
        dataset_dict_BIO = load_dataset("json", data_files=data_files)

        # mapping ID to BIO label
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
                    "id": 'it:' + split_name + ':' + str(progressive_ID)
                })
                progressive_ID += 1

                #if progressive_ID == 1000:
                #    break

        return DatasetDict({split: Dataset.from_list(values) for split, values in new_dataset_dict_BIO_list.items()})

    def get_map_to_extended_NE_name(self):
        with open(os.path.join(self.path_to_BIO, 'extended_labels_map.jsonl')) as fp:
            extended_labels_map = json.load(fp)
        return extended_labels_map


if __name__ == '__main__':

    path_to_BIO = '../../datasets/Multinerd_it'

    Multinerd_it_manager = Multinerd_it(path_to_BIO,
                                        path_to_templates='../templates/',
                                        SLIMER_prompter_name='SLIMER_instruction_it',
                                        test_only=True,
                                        path_to_DeG='../def_and_guidelines/Multinerd_it.json')

    dataset_statistics = Multinerd_it_manager.get_dataset_statistics()
    print(dataset_statistics)

    dataset_dict_BIO = Multinerd_it_manager.datasetdict_BIO

    print(dataset_dict_BIO.keys())
    print(dataset_dict_BIO['test'][0:10])

    ne_categories = Multinerd_it_manager.get_ne_categories()
    print(ne_categories)

    sample_BIO = dataset_dict_BIO['test'][0]
    sample_w_gold_spans = Multinerd_it_manager.extract_gold_spans_per_ne_category(sample_BIO)
    print(sample_w_gold_spans)

    dataset_SLIMER_prefix_caching = Multinerd_it_manager.convert_dataset_for_SLIMER_prefix_caching()
    print(dataset_SLIMER_prefix_caching)
    print(dataset_SLIMER_prefix_caching['test'][0])

    #extremITLLaMA_test = Multinerd_it_manager.convert_dataset_for_ExtremITLLaMA()['test']
    #print(extremITLLaMA_test)
    #print(extremITLLaMA_test[0])

    #dataset_dict_SLIMER = Multinerd_it_manager.dataset_dict_SLIMER
    #for split_name, dataset in dataset_dict_SLIMER.items():
        #dataset.to_json(f'../../datasets/Multinerd_it/SLIMER/{split_name}.jsonl')

    #sentences_per_ne_type = Multinerd_it_manager.get_n_sentences_per_ne_type(n_sentences_per_ne=3)
    #with open("../../datasets/multinerd/SLIMER/sentences_per_ne_type.json", 'w') as f:
    #   json.dump(sentences_per_ne_type, f, indent=2)

    """

    def process_file(input_file, output_file, entity_mapping):
       
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            if line.strip():  # Check if the line is not empty
                parts = line.split('\t')
                if len(parts) == 2:  # Assuming each line is "entity<TAB>label\n"
                    entity = parts[1].strip()
                    if entity != 'O':
                        prefix, entity = entity.split('-')  # Extract the entity (e.g., "PER", "LOC", "ORG")
                        entity = entity_mapping[entity]  # Replace with the mapped value
                        parts[1] = prefix + '-' + entity
                    else:
                        parts[1] = entity
                    modified_lines.append('\t'.join(parts).strip() + '\n')
            else:
                modified_lines.append('\n')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)


    path_to_BIO = '../../datasets/Multinerd_it'
    Multinerd_it_manager = Multinerd_it(path_to_BIO,
                                        path_to_templates='../templates/',
                                        SLIMER_prompter_name='SLIMER_instruction_it',
                                        test_only=True)
    test_split_BIO = Multinerd_it_manager.datasetdict_BIO['test']
    map_tag_to_extended_name = Multinerd_it_manager.get_map_to_extended_NE_name()

    to_write = []
    for sample in test_split_BIO:
        tokens = sample['tokens']
        labels = sample['labels']
        for token, label in zip(tokens, labels):
            if label != 'O':
                prefix, entity = label.split('-')  # Extract the entity (e.g., "PER", "LOC", "ORG")
                entity = map_tag_to_extended_name[entity]  # Replace with the mapped value
                label = prefix + '-' + entity
            to_write.append('\t'.join([token, label]).strip() + '\n')
        to_write.append('\n')

    with open('../../datasets/KIND/GNER/MultinerdIT/test.txt', 'w', encoding='utf-8') as f:
        f.writelines(to_write)

    file_path = '../../datasets/KIND/GNER/MultinerdIT/label.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        for value in map_tag_to_extended_name.values():
            f.write(value + '\n')

    """