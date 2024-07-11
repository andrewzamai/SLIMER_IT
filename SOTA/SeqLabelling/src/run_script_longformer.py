import os

if __name__ == "__main__":
    train_batch_size = 2
    eval_batch_size = 2
    max_seq_length = 4096 # Longformer model, do not use chunked documents
    n_epochs = 15

    # models_name = ["nlpaueb/sec-bert-base", "bert-base-cased"] # ["roberta-base"]

    # models_name = ["roberta-base", "bert-base-cased"]  #, "distilbert-base-cased"]
    # models_name = ["nlpaueb/sec-bert-base"]  #, "distilbert-base-cased"]
    # models_name = ["bert-base-cased"]  #, "distilbert-base-cased"]
    models_name = ["allenai/longformer-base-4096"]  #, "distilbert-base-cased"]

    data_path = "./data/BUSTER_23_c/NER_LONGFORMER_KFOLDS_PERMUTATIONS"
    permutations = ['123_4_5', '234_5_1', '345_1_2', '451_2_3', '512_3_4']

    for model_name in models_name:
        for permutation_name in permutations:
            train_file = os.path.join(data_path, permutation_name, 'train.json')
            val_file = os.path.join(data_path, permutation_name, 'validation.json')
            test_file = os.path.join(data_path, permutation_name, 'test.json')

            output_dir = os.path.join('./outputs', model_name, permutation_name)

            print("\n\nNow finetuning model {} on permutation {}\n".format(model_name, permutation_name))

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
