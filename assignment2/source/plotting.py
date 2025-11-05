import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def plot_runges(x_line, y_line, x_scatter, y_scatter, alpha_val=0.5, title='remember title',
                filename='dummy', save_image=True, show_plot=True):
    fig, ax = plt.subplots(figsize=(6, 4)) 

    ax.scatter(x_scatter, y_scatter, alpha=alpha_val, label='Predictions')
    ax.plot(x_line, y_line, color='red', label='Runge function')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()

    if save_image:
        fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_heatmap(dataset, heat_metric, title='remember title',
                 heat_index='Lambda', heat_column='Learning Rate',
                 y_axis_scientific=True, scientific_precision=3,
                 filename='dummy', save_image=True, show_plot=True):
    """
    Generates and displays a heatmap from a given dataset using specified metrics and formatting options.
    """

    # Pivot data for heatmap
    heatmap_data = dataset.pivot_table(index=heat_index, columns=heat_column, values=heat_metric)



    rows, cols = heatmap_data.shape
    fig_width = max(4, cols * 1.2)  
    fig_height = max(3, rows * 0.5)  

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  
    if y_axis_scientific:
        def sci_label(v):
            return np.format_float_scientific(v, precision=scientific_precision, unique=False)
        yticklabels = [sci_label(v) for v in heatmap_data.index.values]

        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='viridis',
            fmt=".2f",
            cbar_kws={'label': heat_metric},
            yticklabels=yticklabels
        )
    else:
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='viridis',
            fmt=".2f",
            cbar_kws={'label': heat_metric}
        )

    ax.set_ylabel("$\\eta$")
    ax.set_xlabel("$\\lambda$")
    plt.title(title)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_image:
        plt.savefig(f"{filename}.png")

    if show_plot: plt.show()
    else: plt.close(fig)

