import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt


def plot_heatmap(dataset, heat_metric, heat_index='Lambda', heat_column='Learning Rate', y_axis_scientific=True, scientific_precision=3):
    """
    Docstring generated with Copilot
    if y_axis_scientific: ... part is generated with Copilot

    Generates and displays a heatmap from a given dataset using specified metrics and formatting options.

    Parameters:
    ----------
    dataset : pandas.DataFrame
        The input dataset containing the data to be visualized.
    
    heat_metric : str
        The name of the column in `dataset` whose values will be used to populate the heatmap cells.
    
    heat_index : str, optional (default='Lambda')
        The column to use for the heatmap's y-axis (rows).
    
    heat_column : str, optional (default='Learning Rate')
        The column to use for the heatmap's x-axis (columns).
    
    y_axis_scientific : bool, optional (default=True)
        If True, formats the y-axis tick labels using scientific notation.
    
    scientific_precision : int, optional (default=3)
        Number of significant digits to use when formatting y-axis labels in scientific notation.

    Returns:
    -------
    None
        Displays the heatmap using matplotlib and seaborn.
    
    """

    # Pivot and choose relevant data for dataframe - heatmap
    heatmap_data = dataset.pivot_table(index=heat_index, columns=heat_column, values=heat_metric)
    
    
    if y_axis_scientific:
        def sci_label(v):
            # Adjust precision as needed; unique=False keeps trailing zeros where appropriate
            return np.format_float_scientific(v, precision=scientific_precision, unique=False)
        yticklabels = [sci_label(v) for v in heatmap_data.index.values]

        ax = heatmap(
        heatmap_data,
        annot=True,
        cmap='viridis',
        fmt=".2f",
        cbar_kws={'label': heat_metric},
        yticklabels=yticklabels
    )
    else:
        ax = heatmap(
        heatmap_data,
        annot=True,
        cmap='viridis',
        fmt=".2f",
        cbar_kws={'label': heat_metric})


    plt.figure(figsize=(12, 6)) 
    plt.show()