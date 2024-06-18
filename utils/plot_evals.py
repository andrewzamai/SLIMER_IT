import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np
import os.path
import math
import re


def read_eval_outputs(path_to_file):
    """
    read txt containing evaluation outputs of different model configurations on each test Dataset
    e.g. llama2_7B_1pos_1neg_perNE_top391NEs_FalseDef-D on ai, literature, ...
    returns a Dataframe where each row represents micro-scores relative to a model config on a Dataset
    """
    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # micro-score on this Dataset
    micro_scores_pattern = re.compile(r'^([\w\s.]+) ==> micro-Precision: (\d+\.\d+), micro-Recall: (\d+\.\d+), micro-F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            print(f"{model_name} on {dataset_name} scores...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        micro_scores_match = micro_scores_pattern.match(line)
        if micro_scores_match:
            dataset_name_2 = micro_scores_match.group(1)
            # double check on dataset name
            if dataset_name_2 != dataset_name:
                raise ValueError("This dataset name differs from previously read dataset name!")
            micro_precision = float(micro_scores_match.group(2))
            micro_recall = float(micro_scores_match.group(3))
            micro_f1 = float(micro_scores_match.group(4))

            micro_scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'micro-Precision': micro_precision,
                'micro-Recall': micro_recall,
                'micro-F1': micro_f1
            }
            df = pd.DataFrame(micro_scores, index=['m_on_ds'])
            df.drop(columns='model', inplace=True)
            df_list.append(df)

    overall_df = pd.concat(df_list)
    return overall_df


def collect_tp_fp_from_eval_outputs(path_to_file, per_dataset_metrics=False):
    """
    collects TP/FN/FP count on each Named Entity in test, for each different model configuration
    we can then compute micro/macro scores per Dataset (i.e. micro/macro scores already computed in the txt)
    or across all datasets considering them as a single merged dataset (MIT+CrossNER+BUSTER) where all NEs have equal contribution
    """
    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # metrics on a single NE pattern
    support_pattern = re.compile(r'^([\w\s.]+) --> support: (\d+)$')
    tp_fp_fn_pattern = re.compile(r'^([\w\s.]+) --> TP: (\d+), FN: (\d+), FP: (\d+), TN: -1$')
    metrics_pattern = re.compile(r'^([\w\s.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    support = -1
    current_ne = ""
    tp = fn = fp = this_ne_precision = this_ne_recall = this_ne_f1 = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            # print(f"{model_name} on {dataset_name} scores ...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        support_match = support_pattern.match(line)
        if support_match:
            support = int(support_match.group(2))
        tp_fp_fn_pattern_match = tp_fp_fn_pattern.match(line.strip())
        if tp_fp_fn_pattern_match:
            tp = int(tp_fp_fn_pattern_match.group(2))
            fn = int(tp_fp_fn_pattern_match.group(3))
            fp = int(tp_fp_fn_pattern_match.group(4))
            if (tp + fn) != support:
                raise ValueError("TP+FN != support")
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            current_ne = metrics_match.group(1)
            if '.' in current_ne:
                current_ne = current_ne.split('.')[-1]
            this_ne_precision = float(metrics_match.group(2))
            this_ne_recall = float(metrics_match.group(3))
            this_ne_f1 = float(metrics_match.group(4))

            scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'test_NE': current_ne,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'this_NE_F1': this_ne_f1
            }
            df = pd.DataFrame(scores, index=['m_on_ne'])  # model on NE scores
            df = df.drop(columns='model')
            df_list.append(df)

    overall_df = pd.concat(df_list)
    #print("\nCollected the following TP/FP/FN per model configuration on each test NE:\n")
    #print(overall_df)

    # TODO: remove BUSTER if needed
    # overall_df = overall_df[overall_df['dataset'] != 'BUSTER']

    if per_dataset_metrics:
        overall_df = overall_df.groupby(['dataset', 'w_def', 'num_NEs', 'samples_per_NE']).agg(
            {'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'this_NE_F1': 'mean'}).reset_index()
    else:
        overall_df = overall_df.groupby(['w_def', 'num_NEs', 'samples_per_NE']).agg(
            {'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'this_NE_F1': 'mean'}).reset_index()

    overall_df.rename(columns={'this_NE_F1': 'macro-f1'}, inplace=True)

    # compute precision and recall
    overall_df['precision'] = 100 * overall_df['TP'] / (overall_df['TP'] + overall_df['FP'])
    overall_df['recall'] = 100 * overall_df['TP'] / (overall_df['TP'] + overall_df['FN'])
    overall_df['micro-f1'] = 2 * overall_df['precision'] * overall_df['recall'] / (overall_df['precision'] + overall_df['recall'])
    overall_df['micro-f1'].fillna(0, inplace=True)
    overall_df = overall_df.drop(columns=['precision', 'recall'])

    return overall_df


def collect_tp_fp_from_eval_outputs_perNE(path_to_file):
    """
    collects TP/FN/FP count on each Named Entity in test, for each different model configuration
    no further computing F1 metrics
    """
    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # metrics on a single NE pattern
    support_pattern = re.compile(r'^([\w\s.]+) --> support: (\d+)$')
    tp_fp_fn_pattern = re.compile(r'^([\w\s.]+) --> TP: (\d+), FN: (\d+), FP: (\d+), TN: -1$')
    metrics_pattern = re.compile(r'^([\w\s.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    support = -1
    current_ne = ""
    tp = fn = fp = this_ne_precision = this_ne_recall = this_ne_f1 = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            # print(f"{model_name} on {dataset_name} scores ...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        support_match = support_pattern.match(line)
        if support_match:
            support = int(support_match.group(2))
        tp_fp_fn_pattern_match = tp_fp_fn_pattern.match(line.strip())
        if tp_fp_fn_pattern_match:
            tp = int(tp_fp_fn_pattern_match.group(2))
            fn = int(tp_fp_fn_pattern_match.group(3))
            fp = int(tp_fp_fn_pattern_match.group(4))
            if (tp + fn) != support:
                raise ValueError("TP+FN != support")
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            current_ne = metrics_match.group(1)
            if '.' in current_ne:
                current_ne = current_ne.split('.')[-1]
            this_ne_precision = float(metrics_match.group(2))
            this_ne_recall = float(metrics_match.group(3))
            this_ne_f1 = float(metrics_match.group(4))

            scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'test_NE': current_ne,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'this_NE_precision': this_ne_precision,
                'this_NE_recall': this_ne_recall,
                'this_NE_F1': this_ne_f1
            }
            df = pd.DataFrame(scores, index=['m_on_ne'])  # model on NE scores
            df = df.drop(columns='model')
            df_list.append(df)

    overall_df = pd.concat(df_list)
    return overall_df


def collect_BUSTER_eval_outputs(path_to_file, num_NEs, samples_per_NE):

    with open(path_to_file, 'r') as file:
        logs = file.readlines()

    # to extract model name and dataset the scores that will follow belong to
    evaluating_pattern = re.compile(r'^Evaluating model named \'(.+?)\' on \'(.+?)\' test fold in ZERO-SHOT setting$')
    # metrics on a single NE pattern
    support_pattern = re.compile(r'^([\w\s.]+) --> support: (\d+)$')
    tp_fp_fn_pattern = re.compile(r'^([\w\s.]+) --> TP: (\d+), FN: (\d+), FP: (\d+), TN: -1$')
    metrics_pattern = re.compile(r'^([\w\s.]+) --> Precision: (\d+\.\d+), Recall: (\d+\.\d+), F1: (\d+\.\d+)$')

    model_name = ""
    dataset_name = ""
    number_NEs = -1
    with_definition = None
    number_samples_per_NE = -1
    support = -1
    current_ne = ""
    tp = fn = fp = this_ne_precision = this_ne_recall = this_ne_f1 = -1
    df_list = []
    for i, line in enumerate(logs):
        evaluate_match = evaluating_pattern.match(line)
        if evaluate_match:
            model_name = evaluate_match.group(1)
            dataset_name = evaluate_match.group(2)
            # print(f"{model_name} on {dataset_name} scores ...")

            # extracting number of distinct NEs it has been trained on
            number_NEs_pattern = re.compile(r'top(\d+)NEs')
            number_NEs_match = number_NEs_pattern.search(model_name)
            number_NEs = int(number_NEs_match.group(1)) if number_NEs_match else -1

            # with or w/o guidelines
            with_definition = True if 'True' in model_name else False

            # number of pos training samples per NE
            number_samples_pattern = re.compile(r'llama2_7B_(\d+)pos_')
            number_samples_match = number_samples_pattern.search(model_name)
            number_samples_per_NE = int(number_samples_match.group(1)) if number_samples_match else -1

        support_match = support_pattern.match(line)
        if support_match:
            support = int(support_match.group(2))
        tp_fp_fn_pattern_match = tp_fp_fn_pattern.match(line.strip())
        if tp_fp_fn_pattern_match:
            tp = int(tp_fp_fn_pattern_match.group(2))
            fn = int(tp_fp_fn_pattern_match.group(3))
            fp = int(tp_fp_fn_pattern_match.group(4))
            if (tp + fn) != support:
                raise ValueError("TP+FN != support")
        metrics_match = metrics_pattern.match(line)
        if metrics_match:
            current_ne = metrics_match.group(1)
            if '.' in current_ne:
                current_ne = current_ne.split('.')[-1]
            this_ne_precision = float(metrics_match.group(2))
            this_ne_recall = float(metrics_match.group(3))
            this_ne_f1 = float(metrics_match.group(4))

            scores = {
                'model': model_name,
                'dataset': dataset_name,
                'w_def': with_definition,
                'num_NEs': number_NEs,
                'samples_per_NE': number_samples_per_NE,
                'test_NE': current_ne,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'this_NE_precision': this_ne_precision,
                'this_NE_recall': this_ne_recall,
                'this_NE_F1': this_ne_f1
            }
            df = pd.DataFrame(scores, index=['m_on_ne'])  # model on NE scores
            df = df.drop(columns='model')
            df_list.append(df)

    overall_df = pd.concat(df_list)

    overall_df = overall_df[overall_df['dataset'] == 'BUSTER']
    overall_df = overall_df[overall_df['num_NEs'] == num_NEs]
    overall_df = overall_df[overall_df['samples_per_NE'] == samples_per_NE]

    return overall_df


if __name__ == '__main__':

    save_images_to = '../../exp_outputs/plots'
    plots_fontsize = 12  # 16

    metric_to_plot_to_label = {
        'micro-f1': u"\u03bc-F1",
        'macro-f1': "M-F1"
    }

    """ 
    3) Plotting FalseDef vs TrueDef as Number of samples per NE in training increase

    Micro scores computed considering test datasets as a single merged dataset (to not average runs on datasets averages)
    Macro scores computed averaging F1 on each NE considering test datasets as a single merged dataset
    """
    per_dataset_metrics = False
    path_to_eval_folder = '../../exp_outputs/increasing_samples_KIND'
    run_names = ['IT-1']
    number_samples_per_NE_list = [100, 250, 500, 1000]

    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'FalseDef_{run}.txt'), per_dataset_metrics)
        print(f"FalseDef_{run}")
        print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'TrueDef_{run}.txt'), per_dataset_metrics)
        print(f"TrueDef_{run}")
        print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)
    results_avg_across_runs = all_runs_eval_results.groupby(['w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': ['mean', np.std], 'micro-f1': ['mean', np.std]}).reset_index()
    print(results_avg_across_runs)

    results_avg_across_runs.columns = ['w_def', 'num_NEs', 'samples_per_NE', 'macro-f1', 'macro-f1-std', 'micro-f1', 'micro-f1-std']
    # PLOTTING
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['samples_per_NE'].isin(number_samples_per_NE_list)]
    n_samples_per_NE = sorted(list(set(results_avg_across_runs['samples_per_NE'])))
    print(n_samples_per_NE)

    metrics_to_plot = ['micro-f1', 'macro-f1']
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6))
    fig.suptitle("Training on NEs from KIND, evaluating on KIND + MultiNERd_IT")

    for i, metric_to_plot in enumerate(metrics_to_plot):

        TrueDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == True]
        FalseDef_avg_score = results_avg_across_runs[results_avg_across_runs['w_def'] == False]

        axs[i].errorbar(n_samples_per_NE, FalseDef_avg_score[metric_to_plot],
                        yerr=FalseDef_avg_score[metric_to_plot + '-std'],
                        label='baseline w/o D&G', fmt='--o', capsize=5)
        axs[i].errorbar(n_samples_per_NE, TrueDef_avg_score[metric_to_plot],
                        yerr=TrueDef_avg_score[metric_to_plot + '-std'],
                        label='SLIMER', fmt='-o', capsize=5)

        #axs[i].plot(n_samples_per_NE, FalseDef_avg_score[metric_to_plot], marker='o', label='w/o guidelines')
        #axs[i].plot(n_samples_per_NE, TrueDef_avg_score[metric_to_plot], marker='D', label='w guidelines')
        axs[i].set_xticks(n_samples_per_NE, labels=n_samples_per_NE, fontsize=plots_fontsize)
        axs[i].grid(axis='y', linestyle='--', color='lightgray')
        axs[i].set_ylabel(metric_to_plot_to_label[metric_to_plot]+'\n', fontsize=plots_fontsize)
        axs[i].set_yticks([i for i in range(0, int(max(TrueDef_avg_score[metric_to_plot])) + 10, 10)], labels=[i for i in range(0, int(max(TrueDef_avg_score[metric_to_plot])) + 10, 10)], fontsize=plots_fontsize)

        axs[i].legend(loc='lower right', fontsize=plots_fontsize)  # Reversing both handles and labels

    axs[-1].set_xlabel('\nPositive samples per NE type', fontsize=plots_fontsize+2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_images_to, 'IncreaseSamplesPerNE.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


    """ 
    4) PER-DATASET plot FalseDef vs TrueDef as Number samples per NE in training increase
    Micro scores within each dataset
    Macro scores computed averaging F1 on each NE within a dataset
    """
    per_dataset_metrics = True
    path_to_eval_folder = '../../exp_outputs/increasing_samples_KIND'
    run_names = ['IT-1']

    all_runs_eval_results = []
    for run in run_names:
        eval_results_FalseDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'FalseDef_{run}.txt'), per_dataset_metrics)
        print(f"FalseDef_{run}")
        print(eval_results_FalseDef)
        eval_results_TrueDef = collect_tp_fp_from_eval_outputs(os.path.join(path_to_eval_folder, f'TrueDef_{run}.txt'), per_dataset_metrics)
        print(f"TrueDef_{run}")
        print(eval_results_TrueDef)

        eval_results_FalseDef['run'] = run
        eval_results_TrueDef['run'] = run

        all_runs_eval_results.append(eval_results_FalseDef)
        all_runs_eval_results.append(eval_results_TrueDef)

    all_runs_eval_results = pd.concat(all_runs_eval_results).reset_index(drop=True)
    print(all_runs_eval_results)

    # grouping on dataset also!
    results_avg_across_runs = all_runs_eval_results.groupby(['dataset', 'w_def', 'num_NEs', 'samples_per_NE']).agg(
        {'macro-f1': 'mean', 'micro-f1': 'mean'}).reset_index()
    print(results_avg_across_runs)

    # PLOTTING
    metric_to_plot = 'macro-f1'
    results_avg_across_runs = results_avg_across_runs[results_avg_across_runs['samples_per_NE'].isin(number_samples_per_NE_list)]
    n_samples_per_NE = sorted(list(set(results_avg_across_runs['samples_per_NE'])))
    print(n_samples_per_NE)

    datasets = sorted(set(results_avg_across_runs['dataset']))
    colors = plt.cm.tab10.colors[:len(datasets)]  # generate unique colors for each dataset
    plt.figure(figsize=(15, 5))
    plt.grid(axis='y', linestyle='--', color='lightgray')
    plt.xticks(n_samples_per_NE)

    for dataset, color in zip(datasets, colors):
        # select data corresponding to the current dataset
        dataset_data = results_avg_across_runs[results_avg_across_runs['dataset'] == dataset]

        TrueDef_avg_score = dataset_data[dataset_data['w_def'] == True][metric_to_plot]
        FalseDef_avg_score = dataset_data[dataset_data['w_def'] == False][metric_to_plot]

        plt.plot(n_samples_per_NE, FalseDef_avg_score, marker='o', linestyle='--', color=color, label=f'{dataset} - baseline w/o D&G')

        plt.plot(n_samples_per_NE, TrueDef_avg_score, marker='D', linestyle='-', color=color, label=f'{dataset} - SLIMER')

    plt.xlabel('\nPositive samples per NE', fontsize=12)
    plt.ylabel(metric_to_plot_to_label[metric_to_plot])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_images_to, 'IncreaseSamplesPerNE_perDataset.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
