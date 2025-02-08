import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    """
    Applies the sigmoid function

    Args:
        x: vector to be transformed

    Returns:
        np.array: sigmoid(x)
    """
    return 1/(1 + np.exp(-x))

def get_p_from_bivariate_gaussian(s: float, size: int):
        """
        Generate probability of edge creation from a bivariate gaussian distribution.

        Args:
            s: covariance between gaussian random variables.
            size: size of the output matrix.

        Returns:
            probability of edge creation.
        """

        p = np.random.multivariate_normal(mean=[0, 0], cov=[[1, s], [s, 1]], size=size)

        return p


def calculate_rates(p_values, threshold):
    """
    Calculate the proportion of rejected null hypothesis (p=0) on Simulation 2

    Args:
        p_values: p_value list from Spearman's correlation test
        threshold: scalar p_value threshold for rejecting the null hypothesis

    Returns:
        float: proportion of rejected null hypothesis [0,1] 
    """
    fp, tn = 0, 0
    for p_value in p_values:
        if p_value > threshold:
            fp += 1
        else:
            tn += 1
    return tn / (fp + tn) if (fp + tn) > 0 else 0


def riemann_integral(x, y, method='left'):
    """
    Computes the Riemann integral of a curve defined by (x, y) points.

    Parameters:
        x (array-like): x-coordinates (must be sorted in increasing order).
        y (array-like): y-coordinates (function values at x).
        method (str): 'left', 'right', or 'mid' (default is 'left' Riemann sum).

    Returns:
        float: Approximate integral value.
    """
    x = np.array(x)
    y = np.array(y)

    dx = np.diff(x)  # Compute rectangle widths

    if method == 'left':
        integral = np.sum(y[:-1] * dx)
    elif method == 'right':
        integral = np.sum(y[1:] * dx)
    elif method == 'mid':
        mid_y = (y[:-1] + y[1:]) / 2
        integral = np.sum(mid_y * dx)
    else:
        raise ValueError("Method must be 'left', 'right', or 'mid'.")

    return integral

def bootstrap_ci(x, y, n_bootstrap=1000, ci=95, method='left'):
    """
    Computes a confidence interval for the Riemann integral using bootstrapping.

    Parameters:
        x (array-like): x-coordinates
        y (array-like): y-coordinates
        n_bootstrap (int): Number of bootstrap samples
        ci (float): Confidence interval percentage (default 95%)
        method (str): Method for Riemann sum ('left', 'right', or 'mid')

    Returns:
        (float, float, float): Mean integral estimate, lower bound, upper bound
    """
    bootstrapped_integrals = []
    n = len(y)

    for _ in range(n_bootstrap):
        # Resample y with replacement
        y_sample = np.random.choice(y, size=n, replace=True)
        bootstrapped_integrals.append(riemann_integral(x, y_sample, method))

    mean_integral = np.mean(bootstrapped_integrals)
    lower_bound, upper_bound = np.percentile(bootstrapped_integrals, [(100-ci)/2, 100 - (100-ci)/2])

    return mean_integral, lower_bound, upper_bound

    
class GenerateSimulationPlot():

    def __init__(self):
        """
        Results plots generation class.
        """
        pass

    def plot_sim1_regression(self,df_results,n_graphs,n_nodes):
         """
         Plot the regressions lines of each method (GNN-Siamese and Spectral Radius) predicted correlation against the real parameteres correlation. This is done for each class of graph.

         Args:
            df_results: pandas DataFrame object containg the results from the specific simulation. 

         Returns:
            None
         """
         # Plot regression 
         graph_names = df_results['graph_name'].unique()
         num_graphs = len(graph_names)
         fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
         # Flatten axes array for easy iteration
         axes = axes.flatten()
         # Loop through unique graph names and plot them in separate subplots
         for i, graph in enumerate(graph_names):
            subset = df_results[df_results['graph_name'] == graph]
            # Overlay regression line (without hue)
            for method in subset["method"].unique():
                method_data = subset[subset["method"] == method]
                sns.regplot(x="real_corr", y="corr", data=method_data, scatter=True, ax=axes[i], label=method)
            axes[i].set_title(f"Graph: {graph}", fontsize=12)
            axes[i].set_xlabel("Real Correlation", fontsize=10)
            axes[i].set_ylabel("Predicted Correlation", fontsize=10)
         # Hide any unused subplots if num_graphs < 6
         for j in range(i + 1, len(axes)):
             fig.delaxes(axes[j])
        
         # Add main title
         plt.suptitle(f"Real parameter Correlation vs Methods Correlation\n {n_graphs} graphs and {n_nodes} nodes", fontsize=16)
         # Adjust layout and place legend outside
         plt.tight_layout(rect=[0, 0, 1, 0.95])
         plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
         plt.savefig(f"/graph_correlation/results/simulation1_regression_{n_graphs}_graphs_{n_nodes}_nodes.png", dpi=300, bbox_inches="tight")

    def plot_sim2_roc_curves(self,df_results,simulation_name):
        """
         Plot the ROC curves for Simulation 2a (type I error) and 2b (power) for each class of graph. This is done for various numbers of graphs. 

         Args:
            df_results: pandas DataFrame object containg the results from the specific simulation
            simulation_name: Name of the simulation. Should be "simulation_2a" or "simulation_2b"

         Returns:
            None
        """
        method = df_results.method.unique()
        graph_types = ['erdos','geometric','k_regular','watts_strogatz','barabasi']
        graph_labels = {'erdos':'Erdös-Rényi','geometric':'Random Geometric','k_regular':'Random Regular','watts_strogatz':'Watts-Strogatz','barabasi':'Barabási-Albert'}
        linestyle = {20:(0, (5, 2, 1, 2)),40:'dashed',60:'dotted',80:'dashdot',100:'solid'}
        fig, axs = plt.subplots(len(graph_types), len(graph_types), figsize=(20, 20))
        for n_graph in df_results.number_of_graphs.unique():
            for i, gt1 in enumerate(graph_types):
                for j, gt2 in enumerate(graph_types):
                    if j >= i:  # This condition ensures we only fill the upper triangle
                        df = df_results[df_results['number_of_graphs']==n_graph]
                        df = df[df['graph_pair'].astype(str) == f"['{gt1}', '{gt2}']"]
                        pval = df['p_value'].to_list()
                        thresholds = np.linspace(0, 1, len(pval))
                        fprs = [calculate_rates(pval, th) for th in thresholds]
                        axs[i, j].plot(thresholds, fprs,  linestyle=linestyle[n_graph],label=f'{n_graph} graphs')
                        axs[i, j].legend(loc='lower right')
                        axs[0, j].set_title(f"{graph_labels[gt1]}", fontsize=10)
                        if i==j:
                          axs[i, j].set_ylabel(f"{graph_labels[gt1]}", fontsize=10)
        
                    else:
                        axs[i, j].axis('off')
                        
        # Add main title
        plt.suptitle(f"ROC curves {method} for {simulation_name} ", fontsize=16)
        plt.savefig(f"/graph_correlation/results/{simulation_name}_{method[0]}_ROC_curve.png", dpi=300, bbox_inches="tight")
        

    def plot_compare_power(self,spectral_radius_results,gnn_siamese_results):
        """
        Visualize the bar plot comparing the power of the Spearman's correlation test on Simulation 2b (power) for both the Spectral Radius and GNN-Siamese methods. 

         Args:
            spectral_radius_results: pandas DataFrame object containg the results from the spectral radius simulation 2b
            gnn_siamese_results: pandas DataFrame object containg the results from the GNN-Siamese simulation 2b

         Returns:
            None
        """
        graph_types = ['erdos', 'geometric', 'k_regular', 'watts_strogatz', 'barabasi']
        graph_labels = {'erdos': 'Erdös-Rényi', 'geometric': 'Random Geometric', 'k_regular': 'Random Regular',
                        'watts_strogatz': 'Watts-Strogatz', 'barabasi': 'Barabási-Albert'}
        linestyle = {20: (0, (5, 2, 1, 2)), 40: 'dashed', 60: 'dotted', 80: 'dashdot', 100: 'solid'}
        
        fig, axs = plt.subplots(len(graph_types), len(graph_types), figsize=(18, 18))
        
        for i, gt1 in enumerate(graph_types):
            for j, gt2 in enumerate(graph_types):
                if j >= i:  # This condition ensures we only fill the upper triangle
                    integrals = []
                    conf_intervals = []
                    curve_labels = []
                    methods = []
                    datasets = [("Spectral Radius", spectral_radius_results), ("GNN-Siamese", gnn_siamese_results)]
                    # Loop through unique number_of_graphs values
                    for n_graphs in [20, 40, 60, 80]:
                        curve_labels.append(f'{n_graphs} graphs')
                        for method_name, df in datasets:
                            df_aux = df[df['number_of_graphs'] == n_graphs]
                            df_aux = df_aux[df_aux['graph_pair'].astype(str) == f"['{gt1}', '{gt2}']"]
                            pval = df_aux['p_value'].to_list()
                            thresholds = np.linspace(0, 1, len(pval))
                            fprs = [calculate_rates(pval, th) for th in thresholds]
        
                            # Compute integral and confidence interval
                            mean_integral, lower_ci, upper_ci = bootstrap_ci(thresholds, fprs, method='left')
                            integrals.append(mean_integral)
                            conf_intervals.append((upper_ci - lower_ci) / 2)  # Error bar length
                            methods.append(method_name)
        
                    # Convert methods to categorical indices
                    unique_methods = list(set(methods))
                    bar_width = 0.4
                    x_positions = np.arange(len(set(curve_labels)))  # X positions for groups
        
                    for idx, method_name in enumerate(unique_methods):
                        indices = [j for j, m in enumerate(methods) if m == method_name]
                        axs[i, j].bar(x_positions + idx * bar_width,
                                      [integrals[j] for j in indices],
                                      yerr=[conf_intervals[j] for j in indices],
                                      capsize=5, width=bar_width, label=method_name, alpha=0.7)
        
                    axs[i, j].set_xticks(x_positions + bar_width / 2)
                    axs[i, j].set_xticklabels(curve_labels, rotation=20, fontsize=8)
                    axs[i, j].legend(title='Method', fontsize=8, loc='lower right')
                    axs[0, j].set_title(f"{graph_labels[gt1]}", fontsize=10)
                    if i == j:
                        axs[i, j].set_ylabel(f"{graph_labels[gt1]}", fontsize=10)
                else:
                    axs[i, j].axis('off')


        plt.savefig(f"/graph_correlation/results/Power_compare.png", dpi=300, bbox_inches="tight")


