import matplotlib.pyplot as plt
import numpy as np

def plot_bar(values, groups, title='Model Comparison by Metric', save_path=None,
                      n_groups=5, figsize=(12, 6), legend=False, fontsize=22):
    """
    Plots a group grouped bar chart with predefined models, colors, and hatching.
    
    Parameters:
        values (ndarray): A 2 x N array of values (2 groups x number of models).
        groups (list): Names of the two groups (e.g., ['ROC-AUC', 'PR-AUC']).
        title (str): Plot title.
        save_path (str or None): Path to save the figure. If None, the figure is not saved.
    """
    fontsize = fontsize

    models = ['State-Sets', 'Pert Mean', 'Cell Type Mean', 'Linear', 'scVI', 'CPA', 'scGPT']
    models = models[:n_groups]
    
    n_groups = len(groups)
    n_models = len(models)

    # Color and hatch maps
    color_map = {
        'State-Sets': 'royalblue',
        'Pert Mean': '#B8860B',         # dark goldenrod
        'Cell Type Mean': '#DAA520',    # goldenrod
        'Linear': '#505050',            # custom dark gray
        'scVI': '#808080',              # gray
        'CPA': '#A9A9A9',               # darkgray (lighter than gray)
        'scGPT': '#D3D3D3'              # lightgray
    }



    bar_colors = [color_map.get(model, 'gray') for model in models]
    ##bar_hatches = [hatch_map.get(model, '') for model in models]

    bar_width = 0.12
    group_spacing = 0.3
    index = np.arange(n_groups) * (n_models * bar_width + group_spacing)

    plt.figure(figsize=figsize)
    
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width,
                values[:, i],
                bar_width,
                label=model,
                color=bar_colors[i],
                edgecolor='black',
                linewidth=1.2)

    plt.xlabel('', fontsize=fontsize)
    #plt.ylabel('Score', fontsize=fontsize)
    plt.title('', fontsize=fontsize)
    plt.xticks(index + bar_width * (n_models - 1) / 2, groups, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if legend:
        plt.legend(fontsize=fontsize - 2, bbox_to_anchor=[1.0, 0.8])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.ylim([0,1.1])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()