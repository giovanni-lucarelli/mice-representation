import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_cka_comparison(median_scores_random, median_scores_inet, metric_name):
    # Add a 'model' column to each dataframe to distinguish them
    median_scores_random['model'] = 'Random'
    median_scores_inet['model'] = 'ImageNet'

    # Concatenate the two dataframes
    combined_scores = pd.concat([median_scores_random, median_scores_inet], ignore_index=True)

    # Define the order of layers for a more intuitive plot
    layer_order = sorted(combined_scores['layer'].unique(), key=lambda x: int(x.replace('conv', '')))

    # Create a FacetGrid to generate a plot for each area, with different colors for each model
    g = sns.FacetGrid(combined_scores, col="area", hue="model", col_wrap=3, height=4, aspect=1.2, sharey=False, palette={'Random': 'blue', 'ImageNet': 'orange'})

    # Define a function to plot the line and the ribbon
    def plot_with_ribbon(data, **kwargs):
        # Sort the data by the specified layer order to ensure correct line plotting
        data = data.set_index('layer').loc[layer_order].reset_index()
        ax = plt.gca()
        # Plot the median score as a line
        sns.lineplot(data=data, x='layer', y='score', ax=ax, **kwargs)
        # Add the SEM as a shaded ribbon
        ax.fill_between(data['layer'], data['score'] - data['sem'], data['score'] + data['sem'], alpha=0.2)

    # Map the plotting function to the FacetGrid
    g.map_dataframe(plot_with_ribbon)

    # Add a legend
    g.add_legend(title='Model')

    # Set titles and labels
    g.fig.suptitle(f'Median {metric_name} Score by Layer for Each Brain Area (Random vs. ImageNet)', y=1.03, fontsize=16)
    g.set_titles("Area: {col_name}")
    g.set_axis_labels("Layer", f"Median {metric_name} Score")

    # Improve readability of x-axis labels
    g.set_xticklabels(rotation=45)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()