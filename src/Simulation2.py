import numpy as np
import torch
import networkx as nx
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import torch_geometric.nn as nn
from torch_geometric.data import Data, DataLoader
from scipy.stats import spearmanr
from src.models.Spectral_Radius import *

class Simulation2Evaluation:
    
    
    def __init__(self,n_nodes,n_graphs,graph_names,simulation_name):
        self.graph_name1 = graph_names[0]
        self.graph_name2 = graph_names[1]
        self.n_nodes = n_nodes
        self.n_graphs = n_graphs
        self.test_set1 = []
        self.test_set2 = []
        self.simulation_name = simulation_name
        self.scaler_x = joblib.load( f'/graph_correlation/data/scaler.pkl')

    def read_test_data(self):
        test_dataset1 = torch.load(f'/graph_correlation/data/{self.simulation_name}/test_set1_{self.simulation_name}_{self.graph_name1}_{self.n_nodes}_nodes_{self.n_graphs}_graphs.pth')     
        test_dataset2 = torch.load(f'/graph_correlation/data/{self.simulation_name}/test_set2_{self.simulation_name}_{self.graph_name2}_{self.n_nodes}_nodes_{self.n_graphs}_graphs.pth')

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
        
        network_p_value_list =[]
        
        test_loader1 = DataLoader(self.test_set1,batch_size = self.n_graphs,shuffle=False,drop_last=True)
        test_loader2 = DataLoader(self.test_set2,batch_size = self.n_graphs,shuffle=False,drop_last=True)
        for data1,data2 in zip(test_loader1,test_loader2):
          model.eval()
          p_value = model(data1.x, data1.edge_index, data1.batch,
                            data2.x, data2.edge_index, data2.batch)  # Perform a single forward pass.
          network_p_value_list.append(p_value)
    
        return network_p_value_list
    
    def evaluate_spectral(self):
        sr = SpectralRadius()
        test_loader1 = DataLoader(self.test_set1,batch_size = 1,shuffle=False,drop_last=False)
        test_loader2 = DataLoader(self.test_set2,batch_size = 1,shuffle=False,drop_last=False)
          
        eig_p_value_list = []
        eigenvalues_calculated_list1 = []
        eigenvalues_calculated_list2 = []
        count = 0
        for data1,data2 in zip(test_loader1,test_loader2):
            max_eigenvalue1 = sr.scipy_max_eigenvalue(data1.adj)
            eigenvalues_calculated_list1.append(max_eigenvalue1)
        
            max_eigenvalue2 = sr.scipy_max_eigenvalue(data2.adj)
            eigenvalues_calculated_list2.append(max_eigenvalue2)
            
        df_eig_graph = pd.DataFrame(data={'graph_name':[self.graph_name1]*len(eigenvalues_calculated_list1),
                                        'eigenvalues_calculated_list1':eigenvalues_calculated_list1,
                                        'eigenvalues_calculated_list2':eigenvalues_calculated_list2})
        return df_eig_graph

    def get_p_value_from_spectral(self,df_eigevalues,classes_comb):
        df_p_value_results = pd.DataFrame()
        for class_pair in classes_comb:
            eigenvalues_calculated_list1 = df_eigevalues[df_eigevalues['graph_name']==class_pair[0]]['eigenvalues_calculated_list1'].tolist()
            eigenvalues_calculated_list2 = df_eigevalues[df_eigevalues['graph_name']==class_pair[1]]['eigenvalues_calculated_list2'].tolist()
            total_data = len(eigenvalues_calculated_list1)
            eig_p_value_list = []
            for i in range(total_data):
                if i % self.n_graphs == 0:
                    corr, p_value = spearmanr(eigenvalues_calculated_list1[i-self.n_graphs:i], eigenvalues_calculated_list2[i-self.n_graphs:i])
                    eig_p_value_list.append(p_value)    
            results_len = len(eig_p_value_list)
            df_partial_results = pd.DataFrame(data={'p_value':eig_p_value_list,'graph_pair':[class_pair]*results_len,'method':['spectral_radius']*results_len,'number_of_graphs':[self.n_graphs]*results_len})
            df_p_value_results = pd.concat([df_p_value_results,df_partial_results])
        return df_p_value_results

class GNNSiamese(torch.nn.Module):
    def __init__(self, conv_hidden_channels,activation,batch_size,readout_layer,num_node_features,n_graphs,gat_heads):
        super(GNNSiamese, self).__init__()
        self.readout_layer = readout_layer
        self.activation = activation
        self.n_graphs = n_graphs
        self.last_conv_hidden_channel = conv_hidden_channels[-1]
        self.batch_size = batch_size

        self.convs = torch.nn.ModuleList()
        for i in range(len(conv_hidden_channels)):
          in_channels = num_node_features if i == 0 else conv_hidden_channels[i - 1] * gat_heads
          out_channels = conv_hidden_channels[i]
          self.convs.append(nn.GATConv(in_channels, out_channels,heads=gat_heads))
          if i == len(conv_hidden_channels)-1:
              self.convs.append(nn.GATConv(out_channels * gat_heads, 1,heads=gat_heads))


    def forward_once(self, x, edge_index, batch):

        for conv in self.convs:
            x = self.activation(conv(x, edge_index))

        # Compute the mean of heads
        x = x.mean(dim=-1)  # Averaging over the head dimension

        # 2. Readout layer
        x = self.readout_layer(x, batch)

        #reshape
        x = x.view(-1)
        x = x.reshape(self.batch_size,self.n_graphs).T #reshape for n_graphs per set 

        return x

    def forward(self, x1, edge_index1, batch1,x2, edge_index2, batch2):
        # 1. Call foward_once for both pair graphs
        out1 = self.forward_once(x1, edge_index1, batch1)
        out2 = self.forward_once(x2, edge_index2, batch2)

        # 2. Calculate Spearman correlation using the Lambda-like layer
        pred_corr, p_value = spearmanr(out1.detach().numpy(), out2.detach().numpy())

        return p_value



                                            
                                            



