import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    # Example data: a dictionary of datasets where each dataset contains a dictionary of models and their F1 scores
    f1_scores = {
        'WN (supervised)': {'Camoscio': 81.50, 'LLaMA2': 83.24, 'Mistral': 85.55, 'LLaMA3': 85.38, 'LLaMAntino3': 85.78}, #, 'GNER': 90.3, 'extremITLLaMA': 89.1, 'GLiNER-MultiLingual': 70.6, 'Seq.Lab. RoBERTa': 84.1},
        'FIC (OOD)': {'Camoscio': 85.08, 'LLaMA2': 88.81, 'Mistral': 92.78, 'LLaMA3': 84.38, 'LLaMAntino3': 82.52}, # 'GNER': 88.9, 'extremITLLaMA': 90.3, 'GLiNER-MultiLingual': 46.5, 'Seq.Lab. RoBERTa': 70.1},
        'ADG (OOD)': {'Camoscio': 76.00, 'LLaMA2': 79.26, 'Mistral': 80.56, 'LLaMA3': 78.29, 'LLaMAntino3': 81.65}, # 'GNER': 82.5, 'extremITLLaMA': 83.4, 'GLiNER-MultiLingual': 49.4, 'Seq.Lab. RoBERTa': 74.0},
        'MN (unseen NEs)': {'Camoscio': 38.68, 'LLaMA2': 35.16, 'Mistral': 40.64, 'LLaMA3': 50.74, 'LLaMAntino3': 54.65} # 'GNER': 1.2, 'extremITLLaMA': 0.2, 'GLiNER-MultiLingual': 17.4, 'Seq.Lab. RoBERTa': 0}
    }

    # Extract datasets and models
    datasets = list(f1_scores.keys())
    models = list(f1_scores[datasets[0]].keys())

    # Number of datasets and models
    n_datasets = len(datasets)
    n_models = len(models)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 5))
    #ax.grid(True, which='both', axis='y', color='lightgrey', linestyle='--', linewidth=0.5)

    # Define the width of each bar
    bar_width = 0.05
    # Define the separation between groups
    group_offset = 0.1
    # Define the separation between bars within a group
    bar_offset = 0.01

    # Set positions of the bars on the x-axis
    indices = np.arange(n_datasets) * (n_models * (bar_width + bar_offset) + group_offset)

    # Colors for each model (optional)
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #hatch_patterns = ['/', '*', 'x', '-', '+', 'x', 'o']
    #hatch_patterns = ['/', '/', '/', '/', '/', '', '', '', '']

    # Plotting each model's F1 scores for each dataset
    for i, model in enumerate(models):
        f1_model_scores = [f1_scores[dataset][model] for dataset in datasets]
        ax.bar(indices + i * (bar_width + bar_offset), f1_model_scores, bar_width, label=model) #, hatch=hatch_patterns[i % len(hatch_patterns)])#, color=colors[i % len(colors)])

    # Add labels, title, and legend
    #ax.set_xlabel('Datasets')
    ax.set_ylabel(u"\u03bc-F1", fontsize=16)
    #ax.set_title('F1 Scores of Models on Different Datasets')
    ax.set_xticks(indices + (n_models - 1) * (bar_width + bar_offset) / 2)
    ax.set_xticklabels(datasets, fontsize=14)
    ax.set_yticks(range(10, 101, 10), labels=range(10, 101, 10), fontsize=16)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join('../../exp_outputs/plots', 'comparing_all_IT_models_hist.pdf'), dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

    """
    # micro-F1 per model on supervised WN
    supervised_micro_scores = {
        'LLaMA2': {'SLIMER-IT': 83.24, 'w/o D&G': 80.69},
        'Camoscio': {'SLIMER-IT': 81.5, 'w/o D&G': 81.8},
        'Mistral': {'SLIMER-IT': 85.55, 'w/o D&G': 82.71},
        'LLaMA3': {'SLIMER-IT': 82.91, 'w/o D&G': 85.93},
        'LLaMAntino3': {'SLIMER-IT': 83.93, 'w/o D&G': 84.11}
    }

    ood_FIC_micro_scores = {
        'LLaMA2': {'SLIMER-IT': 88.81, 'w/o D&G': 80.45},
        'Camoscio': {'SLIMER-IT': 85.08, 'w/o D&G': 82.44},
        'Mistral': {'SLIMER-IT': 92.78, 'w/o D&G': 85.61},
        'LLaMA3': {'SLIMER-IT': 84.48, 'w/o D&G': 82.85},
        'LLaMAntino3': {'SLIMER-IT': 83.65, 'w/o D&G': 85.96}
    }

    ood_ADG_micro_scores = {
        'LLaMA2': {'SLIMER-IT': 79.26, 'w/o D&G': 73.81},
        'Camoscio': {'SLIMER-IT': 76, 'w/o D&G': 79.01},
        'Mistral': {'SLIMER-IT': 80.56, 'w/o D&G': 75.8},
        'LLaMA3': {'SLIMER-IT': 74.31, 'w/o D&G': 80},
        'LLaMAntino3': {'SLIMER-IT': 76.33, 'w/o D&G': 73.72}
    }

    unseen_Multinerd_micro_scores = {
        'LLaMA2': {'SLIMER-IT': 35.16, 'w/o D&G': 32.38},
        'Camoscio': {'SLIMER-IT': 38.38, 'w/o D&G': 32.28},
        'Mistral': {'SLIMER-IT': 40.64, 'w/o D&G': 35.63},
        'LLaMA3': {'SLIMER-IT': 47.83, 'w/o D&G': 27.62},
        'LLaMAntino3': {'SLIMER-IT': 53.56, 'w/o D&G': 42.08}
    }

    datasets = [supervised_micro_scores, ood_FIC_micro_scores, ood_ADG_micro_scores, unseen_Multinerd_micro_scores]
    titles = ['WikiNews (supervised)', 'FICtion (OOD)', 'ADeGasperi (OOD)', 'MultinerdIT (OOD & Unseen NEs)']

    fig, axes = plt.subplots(nrows=1, ncols=len(titles), figsize=(15, 5))

    for idx, (data, ax, title) in enumerate(zip(datasets, axes, titles)):
        # Extracting data for plotting
        models = list(data.keys())
        sub_names = list(data[models[0]].keys())

        models.append('GNER')
        gner_scores = {
            'WikiNews (supervised)': 90.31,
            'FICtion (OOD)': 88.91,
            'ADeGasperi (OOD)': 82.47,
            'MultinerdIT (OOD & Unseen NEs)': 1.22
        }

        gner_score = gner_scores[title]

        n_models = len(models)
        n_subs = len(sub_names)

        # Creating sub-model data
        sub1_scores = [data[model][sub_names[1]] for model in models if model not in ['GNER']]
        sub2_scores = [data[model][sub_names[0]] for model in models if model not in ['GNER']]

        # Creating arrays for the x positions and heights
        x = np.arange(n_models-1)  # the label locations
        width = 0.3  # the width of the bars

        # Plotting
        rects1 = ax.bar(x - width / 2, sub1_scores, width, label=sub_names[1])
        rects2 = ax.bar(x + width / 2, sub2_scores, width, label=sub_names[0])

        rects3 = ax.bar(x[-1] + 3.3*width, gner_score, width, label='GNER', color='green')  # Plotting GNER in red

        # Adding labels, title, and custom x-axis tick labels, etc.
        if idx == 0:
            ax.set_ylabel(u"\u03bc-F1\n", fontsize=12)
        ax.set_title(title)
        # model_labels = ['\n' + model if i % 2 else model for i, model in enumerate(models)]
        ax.set_xticks(np.arange(len(models)), labels=models, rotation=90, ha='center')
        #ax.set_xticklabels(models)
        ax.legend()
        #ax.grid()

        # Clipping the y-axis to a maximum of 100
        min_clip_x = 50
        if min(sub1_scores) > min_clip_x or min(sub2_scores) > min_clip_x:
            ax.set_ylim(min_clip_x, 100)
        else:
            ax.set_ylim(0, 100)

    fig.tight_layout()

    # Save the figure as a PDF
    plt.savefig('../../paper/plots/models_comparison_hist.pdf')

    plt.show()
    
    """