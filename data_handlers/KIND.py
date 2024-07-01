"""
KIND (Kessler Italian Named-entities Dataset), evalita-2023 dataset handler
https://github.com/dhfbk/KIND/tree/main/evalita-2023

Subdatasets: WN (wikinews), FIC (fiction), ADG (Alcide De Gasperi)
(Letters from AldoMoro not used because silver annotations)
"""

from datasets import Dataset, DatasetDict, load_dataset
import json
import os
import re

# ABSTRACT class which inherits from
from src.data_handlers.Data_Interface import Data_Interface

class KIND(Data_Interface):

    def load_datasetdict_BIO(self, path_to_BIO, test_only=False):

        # match only files subdataset_split.tsv
        pattern = r'^[^_]*_[^_]*\.tsv$'
        all_files = os.listdir(path_to_BIO)
        matching_files = [f for f in all_files if re.match(pattern, f)]

        # merge subdatasets per split giving ID e.g. 'WN:train:0'
        dataset_dict = {'test': []} if test_only else {split: [] for split in ['train', 'validation', 'test']}
        for file_name in matching_files:
            ds_name, split_name = file_name[:-len('.tsv')].split('_')
            split_name = 'validation' if split_name == 'dev' else split_name
            # train only on WN subdataset to have ADG/FIC Out-Of-Domain
            if split_name in ['train', 'validation'] and test_only:
                pass
            elif split_name == 'train' and ds_name in ['FIC', 'ADG']:
                pass
            else:
                ds_content = self.__read_bio_file(os.path.join(path_to_BIO, file_name), ds_name, split_name)
                dataset_dict[split_name].extend(ds_content)

        return DatasetDict({split: Dataset.from_list(values) for split, values in dataset_dict.items()})

    @staticmethod
    def __read_bio_file(path_to_bio_txt, ds_name, split_name):
        """ read BIO content from TSV file """

        with open(path_to_bio_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        progressive_ID = 0
        sentences = []
        tokens = []
        labels = []

        for line in lines:
            line = line.strip()
            if not line:
                if tokens and labels:
                    sentences.append({
                        'id': ds_name + ':' + split_name + ':' + str(progressive_ID),
                        'tokens': tokens,
                        'labels': labels
                    })
                    tokens = []
                    labels = []
                    progressive_ID += 1
            else:
                token, label = line.split()
                tokens.append(token)
                labels.append(label)

        return sentences

    def get_map_to_extended_NE_name(self):
        return {
            'PER': 'persona',
            'LOC': 'luogo',
            'ORG': 'organizzazione'
        }


if __name__ == '__main__':

    """
    path_to_BIO = '../../datasets/KIND/evalita-2023'

    dataset_KIND_manager = KIND(path_to_BIO,
                                path_to_templates='../templates',
                                SLIMER_prompter_name='SLIMER_instruction_it',
                                path_to_DeG='',
                                test_only=False)
    #path_to_DeG='../def_and_guidelines/KIND.json',

    # statistics from BIO dataset
    dataset_statistics = dataset_KIND_manager.get_dataset_statistics()
    print(dataset_statistics)

    dataset_dict_BIO = dataset_KIND_manager.datasetdict_BIO

    print(dataset_dict_BIO.keys())
    print(dataset_dict_BIO['train'][0:10])

    ne_categories = dataset_KIND_manager.get_ne_categories()
    print(ne_categories)

    sample_BIO_list = Dataset.from_dict(dataset_dict_BIO['train'][0:10])
    for sample_BIO in sample_BIO_list:
        print(sample_BIO['tokens'])
        print(sample_BIO['labels'])
        sample_w_gold_spans = dataset_KIND_manager.extract_gold_spans_per_ne_category(sample_BIO)
        print(sample_w_gold_spans)

    dataset_dict_SLIMER = dataset_KIND_manager.get_Npos_Mneg_per_topXtags(N_pos=-1, M_neg=-1)
    for split_name, dataset in dataset_dict_SLIMER.items():
        dataset.to_json(f'../../datasets/KIND/SLIMER/{split_name}.json')

    
    split = 'train'
    kind_SLIMER = load_dataset("json", data_files=f'../../datasets/KIND/SLIMER/{split}.json')['train']
    print(kind_SLIMER)

    unique_ids = set([int(x['doc_tag_pairID'].split(':')[2]) for x in kind_SLIMER])
    print(sorted(unique_ids))
    print(len(unique_ids))

    #with open(os.path.join('../../datasets/KIND/evalita-2023', f'WN_{split}.tsv'), 'r', encoding='utf-8') as file:
        #lines = file.read()

    #print(lines.split('\n\n'))
    """

    def process_file(input_file, output_file, entity_mapping):
        """
        Function to read a file with labeled entities (PER, LOC, ORG),
        replace them with values from a provided dictionary mapping,
        and save the modified content back into a TSV file.

        Args:
        - input_file (str): Path to the input file.
        - output_file (str): Path to the output TSV file.
        - entity_mapping (dict): Dictionary mapping entities to new values.

        Returns:
        - None
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            if line.strip():  # Check if the line is not empty
                parts = line.split('\t')
                if len(parts) == 2:  # Assuming each line is "entity<TAB>label\n"
                    entity = parts[1].strip()
                    if entity != 'O':
                        prefix, entity = entity.split('-')# Extract the entity (e.g., "PER", "LOC", "ORG")
                        entity = entity_mapping[entity]  # Replace with the mapped value
                        parts[1] = prefix + '-' + entity
                    else:
                        parts[1] = entity
                    modified_lines.append('\t'.join(parts).strip() + '\n')
            else:
                modified_lines.append('\n')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)


    map_tag_to_extended_name = {
        'PER': 'persona',
        'LOC': 'luogo',
        'ORG': 'organizzazione'
    }

    process_file('../../datasets/KIND/evalita-2023/FIC_test.tsv', '../../datasets/KIND/GNER/FIC/test.txt', map_tag_to_extended_name)




