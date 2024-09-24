from datasets import load_dataset
import os

if __name__ == '__main__':

    """
    raw_datasets = load_dataset(
                os.path.join(os.path.dirname(__file__), "gner_dataset.py"),
                data_dir='../../../../datasets/KIND/GNER/',
                instruction_file='../configs/instruction_configs/instruction.json',
                data_config_dir='../configs/dataset_configs/task_adaptation_configs',
                add_dataset_name=False,
    )
    print(raw_datasets)
    #raw_datasets['test'].to_json('../../../../datasets/KIND/test_gner_format.json')
    raw_datasets['test'].to_json('../data/MultinerdIT/test_GNER_format.json')
    """

    data = load_dataset("json", data_files=f'../data/MultinerdIT/test_GNER_format.json')['train']
    print(data)
    print(data[0])
