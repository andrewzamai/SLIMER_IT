import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # micro-F1 per model on supervised WN
    supervised_micro_scores = {
        'LLaMA-2-7B': {'SLIMER-IT': 83.24, 'w/o D&G': 80.69},
        'Camoscio-7b': {'SLIMER-IT': 81.5, 'w/o D&G': 81.8},
        'Mistral-7B': {'SLIMER-IT': 85.55, 'w/o D&G': 82.71},
        'LLaMA-3-8B': {'SLIMER-IT': 82.91, 'w/o D&G': 85.93},
        'LLaMAntino-3-8B': {'SLIMER-IT': 83.93, 'w/o D&G': 84.11}
    }

    ood_FIC_micro_scores = {
        'LLaMA-2-7B': {'SLIMER-IT': 88.81, 'w/o D&G': 80.45},
        'Camoscio-7b': {'SLIMER-IT': 85.08, 'w/o D&G': 82.44},
        'Mistral-7B': {'SLIMER-IT': 92.78, 'w/o D&G': 85.61},
        'LLaMA-3-8B': {'SLIMER-IT': 84.48, 'w/o D&G': 82.85},
        'LLaMAntino-3-8B': {'SLIMER-IT': 83.65, 'w/o D&G': 85.96}
    }

    ood_ADG_micro_scores = {
        'LLaMA-2-7B': {'SLIMER-IT': 79.26, 'w/o D&G': 73.81},
        'Camoscio-7b': {'SLIMER-IT': 76, 'w/o D&G': 79.01},
        'Mistral-7B': {'SLIMER-IT': 80.56, 'w/o D&G': 75.8},
        'LLaMA-3-8B': {'SLIMER-IT': 74.31, 'w/o D&G': 80},
        'LLaMAntino-3-8B': {'SLIMER-IT': 76.33, 'w/o D&G': 73.72}
    }

    unseen_Multinerd_micro_scores = {
        'LLaMA-2-7B': {'SLIMER-IT': 35.16, 'w/o D&G': 32.38},
        'Camoscio-7b': {'SLIMER-IT': 38.38, 'w/o D&G': 32.28},
        'Mistral-7B': {'SLIMER-IT': 40.64, 'w/o D&G': 35.63},
        'LLaMA-3-8B': {'SLIMER-IT': 47.83, 'w/o D&G': 27.62},
        'LLaMAntino-3-8B': {'SLIMER-IT': 53.56, 'w/o D&G': 42.08}
    }

    datasets = [supervised_micro_scores, ood_FIC_micro_scores, ood_ADG_micro_scores, unseen_Multinerd_micro_scores]
    titles = ['Supervised - WN', 'OOD - FICtion', 'OOD - ADeGasperi', 'OOD Unseen NEs - Multinerd-IT']

    fig, axes = plt.subplots(nrows=1, ncols=len(titles), figsize=(15, 5))

    for idx, (data, ax, title) in enumerate(zip(datasets, axes, titles)):
        # Extracting data for plotting
        models = list(data.keys())
        sub_names = list(data[models[0]].keys())

        models.append('GNER')
        gner_scores = [80.00]

        n_models = len(models)
        n_subs = len(sub_names)

        # Creating sub-model data
        sub1_scores = [data[model][sub_names[1]] for model in models if model not in ['GNER']]
        sub2_scores = [data[model][sub_names[0]] for model in models if model not in ['GNER']]

        # Creating arrays for the x positions and heights
        x = np.arange(n_models-1)  # the label locations
        width = 0.25  # the width of the bars

        # Plotting
        rects1 = ax.bar(x - width / 2, sub1_scores, width, label=sub_names[1])
        rects2 = ax.bar(x + width / 2, sub2_scores, width, label=sub_names[0])

        rects3 = ax.bar(x[-1] + 4*width, gner_scores, width, label='GNER', color='green')  # Plotting GNER in red

        # Adding labels, title, and custom x-axis tick labels, etc.
        if idx == 0:
            ax.set_ylabel(u"\u03bc-F1\n", fontsize=12)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(models)), labels=models, rotation=90)
        #ax.set_xticklabels(models)
        ax.legend()


        # Clipping the y-axis to a maximum of 100
        min_clip_x = 50
        if min(sub1_scores) > min_clip_x or min(sub2_scores) > min_clip_x:
            ax.set_ylim(min_clip_x, 100)
        else:
            ax.set_ylim(0, 100)

    fig.tight_layout()

    # Save the figure as a PDF
    #plt.savefig('micro_f1_scores.pdf')

    plt.show()