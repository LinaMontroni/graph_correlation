import pandas as pd 
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

def calculate_rates(p_values, threshold):
        fp, tn = 0, 0
        for p_value in p_values:
            if p_value > threshold:
                fp += 1
            else:
                tn += 1
        return tn / (fp + tn) if (fp + tn) > 0 else 0

class GenerateSimulationPlot():

    def __init__(self):
        """
        Results plots generation class.
        """
        pass

    def plot_sim1_regression(self,df_results):
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
            subset = df_res[df_res['graph_name'] == graph]
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
         plt.tight_layout(rect=[0, 0, 1, 0.95])ß
         plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
         plt.savefig(f"/results/simulation1_regression_{args.n_graphs}_graphs_{args.n_nodes}_nodes.png", dpi=300, bbox_inches="tight")

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
        for n_graph in df_res.number_of_graphs.unique():
            for i, gt1 in enumerate(graph_types):
                for j, gt2 in enumerate(graph_types):
                    if j >= i:  # This condition ensures we only fill the upper triangle
                        df = df_res[df_res['number_of_graphs']==n_graph]
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
         plt.savefig(f"/results/{simulation_name}_{method[0]}_ROC_curve.png", dpi=300, bbox_inches="tight")

