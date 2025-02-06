import numpy as np
from tqdm import tqdm
import torch
import networkx as nx
from scipy.sparse import csgraph
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, DataLoader
import igraph
from torchmetrics.regression import SpearmanCorrCoef
import joblib
from utils import sigmoid


class GraphSim:
    def __init__(self, graph_name: str):
        """
        Random Graph simulation class.

        Args:
            graph_name: name of the graph to be generated.
            seed: random seed.

        Returns:
            None
        """
        self.graph_name = graph_name


    def get_p_from_bivariate_gaussian(self, s: float, size: int):
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

    def update_seed(self, seed: int=None):
        """
        Update random seed.

        Args:
            seed: random seed.

        Returns:
            None
        """
        self.seed = np.random.randint(0, 100000) if seed is not None else seed

    def simulate_erdos(self, n: int, prob: float):
        """
        Simulate a random Erdős-Rényi graph.

        Args:
            n: number of nodes.
            prob: probability of edge creation.

        Returns:
            networkx graph object.
        """
        return nx.erdos_renyi_graph(n=n, p=prob, seed=self.seed),prob

    def simulate_k_regular(self, n: int, k: int):
        """
        Simulate a random k-regular graph.

        Args:
            n: number of nodes.
            k: egree of each node.

        Returns:
            networkx graph object.
        """
        return nx.random_regular_graph(d=k, n=n, seed=self.seed),k

    def simulate_geometric(self, n: int, radius: float):
        """
        Simulate a random geometric graph.

        Args:
            n: number of nodes.
            radius: radius for edge creation.

        Returns:
            networkx graph object.
        """
        return nx.random_geometric_graph(n=n, radius=radius, seed=self.seed),radius

    def simulate_barabasi_albert(self, n: int, m: int, ps: float):
        """
        Simulate a random Barabási-Albert preferential attachment graph.

        Args:
            n: number of nodes.
            m: number of edges to attach from a new node to existing nodes.
            ps: scaling exponent

        Returns:
            igraph object.
        """
        #return nx.barabasi_albert_graph(n=n, m=m, seed=self.seed),m
        return igraph.Graph.Barabasi(n=n, m=m, power=ps,start_from=igraph.Graph.Full(3)).to_networkx(), ps

    def simulate_watts_strogatz(self, n: int, k: int, p: float):
        """
        Simulate a random Watts-Strogatz small-world graph.

        Args:
            n: number of nodes.
            k: each node is joined with its k nearest neighbors in a ring topology.
            p: probability of rewiring each edge.

        Returns:
            networkx graph object.
        """
        return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=self.seed), p

    def simulate(self,graph_name,n_nodes,p):
      if self.graph_name == "erdos":
        return self.simulate_erdos(n=n_nodes, prob=p)
      if self.graph_name == "geometric":
        return self.simulate_geometric(n=n_nodes, radius=p)
      if self.graph_name == "watts_strogatz":ß
        return self.simulate_watts_strogatz(n=n_nodes,k=3, p=p)
      if self.graph_name == "k_regular":
        k = [int(10*p)+1 if n_nodes * int(10*p) % 2 != 0 else (int(10*p)) for j in range(1)]
        if k == [0]: k=[1]
        return self.simulate_k_regular(n=n_nodes, k=k[0])
      if self.graph_name == "barabasi":
        ps = [int(10*p)+1 if n_nodes * int(10*p) % 2 != 0 else (int(10*p)) for j in range(1)]
        return self.simulate_barabasi_albert(n=n_nodes, m=3, ps=ps[0])


class NodeFeaturesGeneration():
    def _init__(self,graph, adj_matrix: torch.Tensor):
        """
        Node features generation class.

        Args:
            graph: graph data (nx object)
            adj_matrix: graph adjacent matrix 

        Returns:
            None
        """
        self.graph = graph
        self.adj_matrix = adj_matrix

    def DegreeEncoding(self):
        """
        Compute degree encoding for nodes in a graph given its adjacency matrix.
    
        Parameters:
        adj_matrix (torch.Tensor): Adjacency matrix of the graph (n x n)
    
        Returns:
        torch.Tensor: Degree encoding of the nodes (n x 1)
        """
        # Ensure the adjacency matrix is square
        assert self.adj_matrix.size(0) == self.adj_matrix.size(1), "Adjacency matrix must be square"
    
        # Compute the degree of each node (sum of each row in the adjacency matrix)
        degree = self.adj_matrix.sum(dim=1, keepdim=True)
    
        return degree
    
    
    def LaplacianEncoding(self, l: int=3):
        """
        Compute Laplacian (positional) encoding for nodes in a graph given its adjacency matrix.
    
        Parameters:
           l (int): Number of principal eigenvectors to use in the encoding
    
        Returns:
           torch.Tensor: Laplacian encoding of the nodes (n x l)
        """
        # Ensure the adjacency matrix is square
        assert self.adj_matrix.size(0) == self.adj_matrix.size(1), "Adjacency matrix must be square"
    
        # Convert adjacency matrix to numpy array for compatibility with scipy
        adj_matrix_np = self.adj_matrix.numpy()
    
        # Compute the degree matrix
        degree_matrix = np.diag(adj_matrix_np.sum(axis=1))
    
        # Compute the normalized Laplacian matrix
        d_root_inv = 1.0 / np.sqrt(degree_matrix)
        d_root_inv[np.isinf(d_root_inv)] = 0
    
        laplacian = np.eye(adj_matrix_np.shape[0]) - d_root_inv @ adj_matrix_np @ d_root_inv
    
        # Compute the eigenvalues and eigenvectors of the Laplacian matrix
        eigvals, eigvecs = eigh(laplacian)
    
        # Select the first l eigenvectors (corresponding to the smallest eigenvalues)
        encoding = eigvecs[:, :l]
    
        return torch.tensor(encoding, dtype=torch.float)
   
    
    def ClusteringCoefficient(self):
        """
        Compute the Clustering Coefficient for all nodes in a graph.

        Returns:
         torch.Tensor: clustering coefficient 
        """
        # Ensure the adjacency matrix is square
        assert self.adj_matrix.size(0) == self.adj_matrix.size(1), "Adjacency matrix must be square"
        n = adj_matrix.size(0)
    
        # Compute the degree of each node (sum of each row in the adjacency matrix)
        clustering_coefficient = list(nx.clustering(self.graph).values())
    
        return torch.tensor(clustering_coefficient).view(n,1)
    
    def BetweennessCentrality(self):
        """
        Compute the Betweenness Centrality for all nodes in a graph.

        Returns:
         torch.Tensor: betweenness centrality 
        """
    
        n = self.adj_matrix.size(0)
    
        betweenness_centrality = list(nx.betweenness_centrality(self.graph).values())
    
        return torch.tensor(betweenness_centrality).view(n,1)
    
    def EigenvectorCentrality(self):
         """
        Compute the Eigenvector Centrality for all nodes in a graph.

        Returns:
         torch.Tensor: eigenvector centrality
        """
    
        n = self.adj_matrix.size(0)
    
        adj_matrix_np = self.adj_matrix.numpy()
        G = igraph.Graph.Adjacency((adj_matrix_np > 0).tolist())
        eigenvector_centrality = G.eigenvector_centrality()
       
        return torch.tensor(eigenvector_centrality).view(n,1)
    
    
    def ClosenessCentrality(self):
         """
        Compute the Closeness Centrality for all nodes in a graph.

        Returns:
         torch.Tensor: closeness centrality
        """
    
        n = self.adj_matrix.size(0)
    
        closeness_centrality = list(nx.closeness_centrality(self.graph).values())
    
        return torch.tensor(closeness_centrality).view(n,1)
    
    
    
    def Pagerank(self):
         """
        Compute Pagerank for all nodes in a graph.

        Returns:
         torch.Tensor: pagerank
        """
    
        n = self.adj_matrix.size(0)
    
        pagerank = list(nx.pagerank(self.graph).values())
    
        return torch.tensor(pagerank).view(n,1)
    
    def LocalGraphDensity(self):
         """
        Compute the Local Graph Density for all nodes in a graph.

        Returns:
         torch.Tensor: local graph density
        """
        n = self.adj_matrix.size(0)
    
        source_node = 0
        graph_distance = nx.single_source_shortest_path_length(self.graph, source_node)
    
        def local_graph_density(self.graph, node):
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                return 0
            subgraph = self.graph.subgraph(neighbors + [node])  # include the node itself
            num_edges = len(subgraph.edges())
            return num_edges / (len(neighbors) * (len(neighbors) - 1) / 2) if len(neighbors) > 1 else 0
    
        local_densities = list({node: local_graph_density(self.graph, node) for node in self.graph.nodes()}.values())
        return torch.tensor(local_densities).view(n,1)

    def TriangleCount(self):
        """
        Count how many traingles (comple graphs of size 3) are in the graph. 
        
        Returns:
        torch.Tensor: triangle count
        """
        # Triangle Count
        n = self.adj_matrix.size(0)
        triangle_count = nx.triangles(self.graph)
        triangle_count = list(triangle_count.values())
     
        return torch.tensor(triangle_count).view(n,1)


    def MotifBasedLocalDensity(self):
         """
        Compute the Motif Based Local Density for all nodes in a graph.

        Returns:
         torch.Tensor: motif based local density
        """
        # Motif-Based Local Density: Density of the subgraph formed by neighbors
        n = self.adj_matrix.size(0)
        motif_local_density = {
          node: nx.density(nx.ego_graph(self.graph, node)) for node in graph.nodes
        }
        motif_local_density = list(motif_local_density.values())
    
        return torch.tensor(motif_local_density).view(n,1)



    def LocalClusteringChange(self):
         """
        Compute the Local Clustering Change for all nodes in a graph.

        Returns:
         torch.Tensor: local clustering change
        """
        # Local Clustering Change: Not directly computed in NetworkX, simulate a change
        # Assuming you compare with a base graph G_base
        n = self.adj_matrix.size(0)
        G_base = nx.watts_strogatz_graph(n, 3, 0)  # Base graph without rewiring
        local_clustering_change = {
          node: nx.clustering(self.graph)[node] - nx.clustering(G_base)[node]
          for node in self.graph.nodes
        }
        local_clustering_change = list(local_clustering_change.values())
    
        return torch.tensor(local_clustering_change).view(n,1)


class DataGeneration():
    
    def __init__(self,simulation_name,n_simulations):
        """
        Graph data generation class.

        Args:
            simulation_name: name of the simulation to run. Must be one of the followings: "simulation_1", "simulation_2a", "simulation_2b"
            n_simulations: number of simulations to run

        Returns:
            None
        """
        self.simulation_name = simulation_name
        self.n_simulations = n_simulations
        self.train_x = torch.tensor([])
        self.p_norm1 = []
        self.p_norm2 = []
        self.simulation_list = []
        self.original_p_list1 = []
        self.original_p_list2 = []
        
        self.covs_xy = [np.arange(-1, 1.1, 0.1) if self.simulation_name=='simulation_1' 
                 else np.array([0]) if self.simulation_name=='simulation_2a' 
                 else np.array([0.5]) if self.simulation_name=='simulation_2b']



    def simulate_graph_and_nodes(self,p_list,graph_name,n_nodes):
        """
        Generates a torch geometric dataset with all graphs from a simulation. Generates node features. 

        Args:
            p_list: normalized parameter. Used to generate the random graphs
            graph_name: Graph class name, example: Erdos, Geometric, etc. 
            n_nodes: Size of each graph. 

        Returns:
            torch geometric dataset object.
        """
        dataset = []
        for j in range(p_list.shape[0]):
          # simulate graph
          gs = GraphSim(graph_name)
          gs.update_seed()
          graph,y = gs.simulate(n_nodes,p_list[j])
          adj_np = nx.adjacency_matrix(graph).toarray()
          adj = torch.tensor(adj_np)
          edge_index = adj.nonzero().t().contiguous()
            
          #Node features
          node_features = NodeFeaturesGeneration(graph, adj)
          degree = node_features.DegreeEncoding() 
          laplacian = node_features.LaplacianEncoding() 
          clustering = node_features.ClusteringCoefficient()
          bet_central = node_features.BetweennessCentrality()
          eig_cen = node_features.EigenvectorCentrality()
          clos_cen = node_features.ClosenessCentrality()
          pagerank = node_features.Pagerank()
          local_density = node_features.LocalGraphDensity()
          tri_count = node_features.TriangleCount()
          modf_local_dens = node_features.MotifBasedLocalDensity()
          local_clust = node_features.LocalClusteringChange()
          node_features = torch.cat((degree,laplacian,clustering,bet_central,eig_cen,clos_cen,pagerank,local_density,tri_count,modf_local_dens,local_clust),dim=1)
          node_features = node_features.to(torch.float32)
        
          data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float32),adj=adj, graph_name=graph_name,n_simulation=self.simulation_list[j])
          dataset.append(data)
        return dataset

    
    def generate_siamese_data(self,n_graphs):
        """
        Samples the parameteres of the graphs from a bivariate normal distribuition with covariance s. 

        Args:
            n_graphs: number of graphs (in each set). 
        """
        p_list1 = np.array([])
        p_list2 = np.array([])
        original_p_list1 = []
        original_p_list2 = []
        simulation_list = []
        
        for s in self.covs_xy:
          for i in range(self.n_simulations):
            p = get_p_from_bivariate_gaussian(s=s,size=n_graphs)
            org_p1 = p[:,0]
            org_p2 = p[:,1]
            p = sigmoid(p)#utils.sigmoid
            p1 = p[:,0]
            p2 = p[:,1]
            p_list1 = np.concatenate([p_list1,p1])
            p_list2 = np.concatenate([p_list2,p2])
            original_p_list1.append(org_p1)
            original_p_list2.append(org_p2)
            simulation_list += [i+1 for j in range(n_graphs)]
        
          p_list1 = torch.from_numpy(p_list1)
          p_list1 = p_list1.to(torch.float32)
          p_list2 = torch.from_numpy(p_list2)
          p_list2 = p_list2.to(torch.float32)
        self.p_norm1 = p_list1
        self.p_norm2 = p_list2
        self.simulation_list = simulation_list
        self.original_p_list1 = original_p_list1
        self.original_p_list2 = original_p_list2

    def generate_scaler(self,dataset):
        """
        Fit the training data into a MinMaxScaler sklearn object. 

        Args:
            dataset: training dataset - torch geometric data object.

        Returns:
            fited sclaer 
        """
        loader = DataLoader(dataset, batch_size=len(dataset),shuflle=False,drop_last=False)
        if set_type == 'train':
        train_x = torch.tensor([])
        for data in loader:
          train_x = torch.cat([train_x,torch.tensor(data.x)],dim=0)
    
        scaler_x = MinMaxScaler()
        scaler_x.fit(train_x)
        self.scaler = scaler_x

        return self.scaler




 