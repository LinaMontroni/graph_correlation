import numpy as np
import torch
import networkx as nx
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from torch_geometric.data import Data, DataLoader
from src.models.GNN_Siamese import *
from src.models.Spectral_Radius import *
from scipy.stats import spearmanr


class Simulation1Evaluation:

    def __init__(self,n_nodes,n_graphs,graph_name):
        self.graph_name = graph_name
        self.n_nodes = n_nodes
        self.n_graphs = n_graphs
        self.test_set1 = []
        self.test_set2 = []
        self.scaler_x = joblib.load(f'/graph_correlation/data/scaler.pkl')

    def read_test_data(self):
        test_dataset1 = torch.load(f'/graph_correlation/data/simulation_1/test_set1_simulation_1_{self.graph_name}_{self.n_nodes}_nodes_{self.n_graphs}_graphs.pth')     
        test_dataset2 = torch.load(f'/graph_correlation/data/simulation_1/test_set2_simulation_1_{self.graph_name}_{self.n_nodes}_nodes_{self.n_graphs}_graphs.pth')

        self.test_set1 = test_dataset1
        self.test_set2 = test_dataset2
    
    def fit_test_data(self):
        test_loader1 = DataLoader(self.test_set1,batch_size = len(self.test_set1),shuffle=False,drop_last=True)
        test_loader2 = DataLoader(self.test_set2,batch_size = len(self.test_set2),shuffle=False,drop_last=True)
        for data1,data2 in zip(test_loader1,test_loader2):
          data1.x = torch.tensor(self.scaler_x.transform(data1.x)).to(torch.float32)
          data2.x = torch.tensor(self.scaler_x.transform(data2.x)).to(torch.float32)
    
        self.test_set1 = test_loader1.dataset
        self.test_set2 = test_loader2.dataset
    
    def evaluate_GNN_Siamese(self):
      
        checkpoint_path = '/graph_correlation/data/GNN_siamese_best_model_state_dict.pth'
        state_dict = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        
        original_params = state_dict['params']
        
        updated_params = {"batch_size": 1 , "n_graphs": self.n_graphs, 'conv_hidden_channels':original_params['conv_hidden_channels'], 'activation':original_params['activation'],  'readout_layer':original_params['readout_layer'], 'num_node_features':original_params['num_node_features'],'gat_heads':original_params['gat_heads']}
        
        # Step 3: Create a new model instance with the updated parameters
        model = GNNSiamese(**updated_params)
        
        # Step 4: Load the state dictionary into the new model
        model.load_state_dict(state_dict['model_state_dict'])
        
        real_corr_list = []
        network_corr_list =[]
        
        test_loader1 = DataLoader(self.test_set1,batch_size = self.n_graphs,shuffle=False,drop_last=True)
        test_loader2 = DataLoader(self.test_set2,batch_size = self.n_graphs,shuffle=False,drop_last=True)
        for data1,data2 in zip(test_loader1,test_loader2):
            real_corr,p_value = spearmanr(data1.y.numpy(), data2.y.numpy())
            real_corr_list.append(real_corr)
    
            #Newtork p
            model.eval()
            pred_corr = model(data1.x, data1.edge_index, data1.batch,
                            data2.x, data2.edge_index, data2.batch)  # Perform a single forward pass.
            network_corr_list.append(pred_corr.item())
        
        return real_corr_list,network_corr_list
    
    def evaluate_spectral(self):
          
        sr = SpectralRadius()
        test_loader1 = DataLoader(self.test_set1,batch_size = 1,shuffle=False,drop_last=False)
        test_loader2 = DataLoader(self.test_set2,batch_size = 1,shuffle=False,drop_last=False)
        
        eig_corr_list = []
        p_corr_list = []
        eigenvalues_calculated_list1 = []
        eigenvalues_calculated_list2 = []
        p_list1 = []
        p_list2 = []
        count = 0
        for data1,data2 in zip(test_loader1,test_loader2):
            count += 1
            max_eigenvalue1 = sr.scipy_max_eigenvalue(data1.adj)
            eigenvalues_calculated_list1.append(max_eigenvalue1)
            p_list1.append(data1.y.item())
        
            max_eigenvalue2 = sr.scipy_max_eigenvalue(data2.adj)
            eigenvalues_calculated_list2.append(max_eigenvalue2)
            p_list2.append(data2.y.item())
        
            if count % self.n_graphs == 0:
                eig_corr,p_value  = spearmanr(eigenvalues_calculated_list1, eigenvalues_calculated_list2)
                eig_corr_list.append(eig_corr)
                p_corr,p_value = spearmanr(p_list1,p_list2)
                p_corr_list.append(p_corr)
                
                eigenvalues_calculated_list1 = []
                eigenvalues_calculated_list2 = []
                p_list1 = []
                p_list2 = []
        
        return eig_corr_list,p_corr_list



                                            
                                            



