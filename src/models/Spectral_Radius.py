import numpy as np 
from scipy.sparse.linalg import eigs

class SpectralRadius:
    def _init_():
        """
        Spectral Radius method implementation
    
        Args:
            None
            
        Returns:
            None
        """
        pass

    def scipy_max_eigenvalue(self,adj):
        """
        Get the largest eigenvalue from one single adjacent matrix.
    
        Args:
            adj (torch tensor): adjacent matrix 
            
        Returns:
            int: largest eigenvalue from adj
         """
        adj = adj.numpy().astype(np.float64) 
        largest_eigenvalue, _ = eigs(adj, k=1, which='LM')
        max_eigenvalue = largest_eigenvalue[0].real        
        return max_eigenvalue

    def get_max_eigenvalue(self,dataset,graph_name,n_graphs):
        """
        Get the largest eigenvalue from all adjacent matrix in a dataset.
    
        Args:
            dataset (torch dataset): dataset. Adjacent matrix must be stored in data.adj
            graph_name (str): Graph class name 
            n_graphs (int): Number of graphs in each set on the dataset
            
        Returns:
            DataFrame: pandas DataFrame with stored eigenvalues
            list: max eigenvalues list
        """
        eigenvalues_calculated_list = []
        for data in dataaset:
            max_engenvalue = self.scipy_max_eigenvalue(data.adj)
            eigenvalues_calculated_list.append(max_eigenvalue)
            
        df = pd.DataFrame(data={'max_eig':eigenvalues_calculated_list,'graph_name':[graph_name]*len(eigenvalues_calculated_list),'n_graphs':[n_graphs]*len(eigenvalues_calculated_list)})
        return df,eigenvalues_calculated_list

  
        
            




