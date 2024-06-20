"""
An abstract class for SLIMER to interface with any NER dataset.

Initialize with path to json w/ D&G for each NE; default w/o D&G.

Inherit from this class and define the abstract methods:
- load_datasetdict_BIO: load the BIO dataset and return a DatasetDict of Datasets with (tokens, labels, id) features
"""

__package__ = "SLIMER_IT.src.data_handlers"

from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from collections import OrderedDict
from typing import Union
import numpy as np
import random
import math
import json
import os
import re

# SLIMER prompter to format a ne_tag, Def and Guidelines into a prompt for NER
from src.SLIMER_Prompter import SLIMER_Prompter

class Data_Interface(ABC):

    def __init__(self, path_to_BIO, path_to_templates, SLIMER_prompter_name, path_to_DeG: Union[None, str] = None, test_only=False):
        """
        Instantiate a NER dataset for SLIMER w/ D&G if provided path to json.

        :param path_to_BIO: the path to the folder with BIO data
        :param path_to_templates: path to folder with SLIMER prompt templates
        :param SLIMER_prompter_name: the name of a SLIMER prompt template
        :param path_to_DeG: optional path to json with Def & Guidelines for each NE, if not provided SLIMER w/o D&G instantiated
        """
        self.path_to_BIO = path_to_BIO
        self.datasetdict_BIO = self.load_datasetdict_BIO(path_to_BIO, test_only)
        self.ne_categories = self.get_ne_categories()  # list of NE tags from BIO labels
        self.slimer_prompter = SLIMER_Prompter(SLIMER_prompter_name, path_to_templates)
        self.path_to_DeG = path_to_DeG
        self.dataset_dict_SLIMER = self.convert_dataset_for_SLIMER()

    @abstractmethod
    def load_datasetdict_BIO(self, path_to_BIO, test_only=False):
        """
        Different NER datasets format their data in different ways: csv, tsv, ... and different column names
        Define your implementation to load the data and return a DatasetDict of Datasets (train, validation, test)
        with features: tokens (list), labels (list), id (int)
        """
        pass

    @abstractmethod
    def get_map_to_extended_NE_name(self):
        """ provide a dictionary that maps each NE tag to its extended name (e.g. PER to PERSON) """
        pass

    def get_ne_categories(self):
        """
        Parse the BIO labels to extract the NE tags: e.g. PER, ORG, LOC ...
        """
        ne_categories = {}
        for split in self.datasetdict_BIO.keys():
            if split != 'dataset_name':
                for document in self.datasetdict_BIO[split]:
                    doc_labels = document["labels"]
                    for lbl in doc_labels:
                        if lbl not in ne_categories:
                            ne_categories[lbl] = 0
        ne_categories_sorted = dict(sorted(ne_categories.items())).keys()
        return [lbl[2:] for lbl in ne_categories_sorted if lbl[0] == 'B']

    def get_dataset_statistics(self):
        """
        Computes statistics for each BIO split.
        """
        per_split_statistics = {split: {} for split in self.datasetdict_BIO.keys()}
        for split in per_split_statistics:
            context_lengths = []
            per_split_statistics[split]['occurrences_per_ne'] = {ne: 0 for ne in self.ne_categories}
            for sample in self.datasetdict_BIO[split]:
                context_length = len(sample['tokens'])
                context_lengths.append(context_length)
                for label in sample['labels']:
                    if label[0] == 'B':
                        per_split_statistics[split]['occurrences_per_ne'][label[2:]] += 1

            per_split_statistics[split]['number_input_texts'] = len(self.datasetdict_BIO[split])
            per_split_statistics[split]['input_avg_number_words'] = int(np.average(context_lengths))
            per_split_statistics[split]['input_min_number_words'] = int(np.min(context_lengths))
            per_split_statistics[split]['input_max_number_words'] = int(np.max(context_lengths))

        return per_split_statistics

    def extract_gold_spans_per_ne_category(self, sample_BIO):
        """ given a BIO sample parse its labels-list to extract the gold spans (with start/end positions in chars) for each NE tag """
        sample_gold_spans_per_ne = {ne: [] for ne in self.ne_categories}
        i = 0
        index = 0
        startIndex = index
        entity = ''  # entity being reconstructed
        while i < len(sample_BIO['labels']):
            # if the token is labelled as part of an entity
            if sample_BIO['labels'][i] != 'O':
                if entity == '':
                    startIndex = index
                entity = entity + ' ' + sample_BIO['tokens'][i]  # this will add an initial space (to be removed)
                # if next label is Other or the beginning of another entity
                # or end of document, the current entity is complete
                if (i < len(sample_BIO['labels']) - 1 and sample_BIO['labels'][i + 1][0] in ["O", "B"]) or (
                        i == len(sample_BIO['labels']) - 1):
                    # add to metadata
                    tagName = sample_BIO['labels'][i][2:]
                    # adding also if same name but will have != start-end indices
                    sample_gold_spans_per_ne[tagName].append((entity[1:], startIndex, startIndex + len(entity[1:])))
                    # cleaning for next entity
                    entity = ''

            index = index + len(sample_BIO['tokens'][i]) + 1
            i += 1

        return sample_gold_spans_per_ne

    def load_DeG_per_NEs(self):
        """ load json and eval to dictionary the D&G for each NE """
        if not self.path_to_DeG:
            raise Exception("Path to Def & Guidelines not provided")
        if not os.path.exists(self.path_to_DeG):
            raise ValueError(f"Can't find or read D&G at {self.path_to_DeG}")
        with open(self.path_to_DeG) as fp:
            DeG_per_NEs_raw = json.load(fp)
        # converting list to dict for fast access
        if DeG_per_NEs_raw and isinstance(DeG_per_NEs_raw, list):
            DeG_per_NEs_raw = {x['ne_tag']: x for x in DeG_per_NEs_raw}
        for ne_tag, values in DeG_per_NEs_raw.items():
            gpt_definition = values['gpt_DeG']
            if not gpt_definition.endswith("}"):
                if not gpt_definition.endswith("\""):
                    gpt_definition += "\""
                gpt_definition += "}"

            this_ne_guidelines = eval(gpt_definition)
            # replacing ne types occurrences between single quotes to their UPPERCASE
            ne_type_in_natural_language = values['real_name']
            pattern = re.compile(rf'\'{re.escape(ne_type_in_natural_language)}\'', re.IGNORECASE)
            this_ne_guidelines = {k: pattern.sub(f'{ne_type_in_natural_language.upper()}', v) for k, v in this_ne_guidelines.items()}
            values['gpt_DeG'] = this_ne_guidelines

        return DeG_per_NEs_raw

    def convert_dataset_for_SLIMER(self):
        """ convert Dataset from BIO to SLIMER format
        with features: doc_tag_pairID, tagName, input, instruction (with D&G if path_to_DeG provided) and output the gold answers spans """
        dataset_dict_SLIMER = {split: [] for split in self.datasetdict_BIO.keys()}
        if self.path_to_DeG:
            DeG_per_NEs = self.load_DeG_per_NEs()
        for split_name, dataset_BIO in self.datasetdict_BIO.items():
            for sample_BIO in dataset_BIO:
                sample_gold_spans_per_ne = self.extract_gold_spans_per_ne_category(sample_BIO)
                tag_ID = 0  # assign id to each tag: input x |NE|
                for ne_tag, gold_spans in sample_gold_spans_per_ne.items():
                    definition = ''
                    guidelines = ''
                    # get tag in its extended form and convert it to uppercase e.g. PERSON
                    ne_tag_extended = self.get_map_to_extended_NE_name()[ne_tag].upper()
                    if self.path_to_DeG:
                        ne_tag_extended = DeG_per_NEs[ne_tag]['real_name'].upper()
                        definition = DeG_per_NEs[ne_tag]['gpt_DeG']['Definition']
                        guidelines = DeG_per_NEs[ne_tag]['gpt_DeG']['Guidelines']

                    instruction = self.slimer_prompter.generate_prompt(ne_tag=ne_tag_extended,
                                                                       definition=definition,
                                                                       guidelines=guidelines)

                    # sort text answers by increasing start positions
                    ga_sorted_by_start_pos = sorted(gold_spans, key=lambda x: x[1])
                    # retrieve only text answers
                    ga_sorted_text_only = [item[0] for item in ga_sorted_by_start_pos]
                    # deleting any duplicate while preserving order (order within document context)
                    ga_sorted_text_only_wo_duplicates = list(OrderedDict.fromkeys(ga_sorted_text_only).keys())
                    ga_dumped = json.dumps(ga_sorted_text_only_wo_duplicates)  # stringifying list

                    dataset_dict_SLIMER[split_name].append(
                        {"doc_tag_pairID": sample_BIO['id'] + ":" + str(tag_ID),
                         "input": ' '.join(sample_BIO['tokens']),
                         "tagName": ne_tag,
                         "instruction": instruction,
                         "output": ga_dumped
                         })
                    tag_ID += 1

        return DatasetDict({split: Dataset.from_list(values) for split, values in dataset_dict_SLIMER.items()})

    def get_Npos_Mneg_per_topXtags(self, N_pos, M_neg, topXtags=-1):
        """
        build dataset with N positive samples per NE and M negative samples per NE
        train fold with N + M samples per NE
        validation fold with ceil(N/4) + ceil(M/4) samples per NE
        test fold is copied unchanged
        """
        # if keep_only_top_tagNames > -1 and keep_only_top_tagNames != 391:
            # dataset_MSEQA_format = keep_only_top_N_tagNames(dataset_MSEQA_format, keep_only_top_tagNames)

        n_samples_per_NE_dataset = {split: [] for split in self.dataset_dict_SLIMER.keys()}
        n_samples_per_NE_dataset['test'] = self.dataset_dict_SLIMER['test']  # copy test fold unchanged
        for split in self.dataset_dict_SLIMER.keys():
            # shuffle dataset so input texts are not grouped
            self.dataset_dict_SLIMER[split] = self.dataset_dict_SLIMER[split].shuffle(seed=42)
            # draw reduced samples only for train and validation
            if split != 'test':
                # count how many pos/neg samples we have per NE
                ne_list = {}
                for sample in self.dataset_dict_SLIMER[split]:
                    ne_type = sample['tagName']
                    if ne_type not in ne_list:
                        ne_list[ne_type] = {'yes_answer': 0, 'no_answer': 0}
                    if sample['output'] == '[]':
                        ne_list[ne_type]['no_answer'] += 1
                    else:
                        ne_list[ne_type]['yes_answer'] += 1

                # if N_pos == -1 use all positive examples for a NE and corresponding number of negative samples
                if N_pos == -1 and M_neg == -1:
                    ne_list = {ne: {'yes_answer': values['yes_answer'],
                                    'no_answer': values['yes_answer'] if values['no_answer'] > values['yes_answer'] else values['no_answer']} for
                               ne, values in ne_list.items()}
                    if split == 'validation':
                        ne_list = {ne: {'yes_answer': math.ceil(values['yes_answer']/4.0),
                                        'no_answer': math.ceil(values['no_answer']/4.0)} for ne, values in ne_list.items()}
                else:
                    # if validation use 1/4 samples per NE
                    if split == 'validation':
                        N_pos = math.ceil(N_pos/4.0)
                        M_neg = math.ceil(M_neg/4.0)
                    ne_list = {ne: {'yes_answer': N_pos if values['yes_answer'] > N_pos else values['yes_answer'], 'no_answer': M_neg if values['no_answer'] > M_neg else values['no_answer']} for ne, values in ne_list.items()}

                print(f"{split} statistics: ")
                for ne, values in ne_list.items():
                    print(f"NE: {ne}")
                    print(values)

                for sample in self.dataset_dict_SLIMER[split]:
                    has_answer = 'yes_answer'
                    if sample['output'] == '[]':
                        has_answer = 'no_answer'
                    if ne_list[sample['tagName']][has_answer] > 0:
                        n_samples_per_NE_dataset[split].append(sample)
                        ne_list[sample['tagName']][has_answer] -= 1

                # random.shuffle(n_samples_per_NE_dataset[split])

        return DatasetDict({split: Dataset.from_list(values) for split, values in n_samples_per_NE_dataset.items()})

    """ Functions to extract N sentences per NE to show examples of NE in context for prompting ChatGPT for D&G"""

    @staticmethod
    def __split_into_sentences(passage):
        # split the passage into sentences based on punctuation .?! while not splitting "Dr." or "Fig.1"
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?! \d)(?!\d)', passage)
        return [sentence for sentence in sentences if sentence.strip()]

    @staticmethod
    def __count_target_words(sentence, target_words):
        # count occurrences of the target words in the sentence
        matches = re.finditer(r'\b(?:' + '|'.join(map(re.escape, target_words)) + r')\b', sentence)
        # get the list of target words found in this sentence
        target_words_found = [match.group() for match in matches]
        # count the number of target words found in this sentence
        occurrences_count = len(target_words_found)
        return occurrences_count, target_words_found

    def __get_one_sentence_from_sample(self, sample):
        """ given a sample and the gold answer spans for a NE return a sentence with (max number of) occurrences in it"""
        document_context = sample['input']
        target_words = json.loads(sample['output'])
        ne_type = sample['tagName']
        # split in sentences according to punctuation .?!
        sentences = self.__split_into_sentences(document_context)
        # count the occurrences of target words in each sentence
        # to return the one with at least 1/highest number of occ.
        target_word_counts = []
        for sentence in sentences:
            occurrences_count, target_words_found = self.__count_target_words(sentence, target_words)
            target_word_counts.append({"sentence": sentence,
                                       "target_words_in_it": target_words_found,
                                       "occurrences_count": occurrences_count
                                       })

        # sort sentences by decreasing occurrences_count
        target_word_counts = sorted(target_word_counts, key=lambda x: x['occurrences_count'], reverse=True)
        # returning the sentence with highest number of occurrences, but with some contraints
        sentence_to_ret = None
        i = 0
        while i < len(target_word_counts):
            if target_word_counts[i]['occurrences_count'] != 0:
                if 50 < len(target_word_counts[i]['sentence']) < 100:
                    sentence_to_ret = target_word_counts[i]
                    break
            i += 1
        return sentence_to_ret

    def get_n_sentences_per_ne_type(self, n_sentences_per_ne=3):
        # getting from training set n_sentences_per_ne as positive examples from which to let gpt infer NE definition
        sentences_per_ne_type = {ne: [] for ne in self.ne_categories}
        trainDataset = self.dataset_dict_SLIMER['train'].to_list()
        random.seed(4)
        random.shuffle(trainDataset)
        for ne_type in self.ne_categories:
            i = 0
            while len(sentences_per_ne_type[ne_type]) < n_sentences_per_ne and i < len(trainDataset):
                sample = trainDataset[i]
                if sample['tagName'] == ne_type and json.loads(sample['output']) != []:
                    sentence_target_words = self.__get_one_sentence_from_sample(sample)
                    if sentence_target_words is not None:
                        # removing duplicates in list of target words
                        sentence_target_words['target_words_in_it'] = list(set(sentence_target_words['target_words_in_it']))
                        sentences_per_ne_type[ne_type].append(sentence_target_words)
                i += 1

        not_enough_sentences = []
        for ne_type, sentences in sentences_per_ne_type.items():
            if len(sentences) < n_sentences_per_ne:
                # raise ValueError(f"not enough sentences for {ne_type}")
                not_enough_sentences.append((ne_type, len(sentences)))
        print(f"NE types with less than n_sentences_per_ne: {len(not_enough_sentences)}")
        print(not_enough_sentences)

        return sentences_per_ne_type
