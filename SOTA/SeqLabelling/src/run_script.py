import os

from src.data_handlers.KIND import KIND
from src.data_handlers.Multinerd_it import Multinerd_it

from datasets import concatenate_datasets

if __name__ == "__main__":
    train_batch_size = 32
    eval_batch_size = 32
    max_seq_length = 512
    n_epochs = 25

    model_name = "bert-base-cased"
    #model_name = "dbmdz/bert-base-italian-cased"

    path_to_BIO = './data/SLIMER_IT_datasets/KIND/evalita-2023'
    #path_to_BIO = '../../../../datasets/KIND/evalita-2023'
    dataset_dict_KIND_BIO = KIND(path_to_BIO).datasetdict_BIO

    print(dataset_dict_KIND_BIO)

    #dataset_dict_MultinerdIT_BIO = Multinerd_it(path_to_BIO).datasetdict_BIO

    #print(dataset_dict_MultinerdIT_BIO)

    #dataset_dict_KIND_BIO['validation'].to_json(f'./data/KIND_validation.json')

    for split_name, dataset in dataset_dict_KIND_BIO.items():
        dataset.to_json(f'./data/SLIMER_IT_datasets/KIND/BIO_4_SeqLabelling/{split_name}.json')

    train_file = os.path.join('./data/SLIMER_IT_datasets/KIND/BIO_4_SeqLabelling', 'train.json')
    val_file = os.path.join('./data/SLIMER_IT_datasets/KIND/BIO_4_SeqLabelling', 'validation.json')
    test_file = os.path.join('./data/SLIMER_IT_datasets/KIND/BIO_4_SeqLabelling', 'test.json')

    #output_dir = os.path.join('./outputs', model_name, 'trained_on_WN_FIC_ADG_BERT_ITA_2')
    output_dir = os.path.join('./outputs', model_name, 'trained_on_WN_ITA_ziorufus')

    print("\n\nNow finetuning model {} on KIND\n".format(model_name))

    res = os.system(
        f"python3 ./src/run_ner.py "
        f"--model_name_or_path {model_name} \
            --train_file {train_file} \
            --validation_file {val_file} \
            --test_file {test_file} \
            --text_column_name tokens\
            --label_column_name labels\
            --output_dir {output_dir} \
            --num_train_epochs {n_epochs} \
            --per_device_train_batch_size {train_batch_size} \
            --per_device_eval_batch_size {eval_batch_size} \
            --max_seq_length {max_seq_length} \
            --do_train \
            --do_predict \
            --do_eval"
    )


