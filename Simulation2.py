import numpy as np
import torch
import networkx as nx
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from torch_geometric.data import Data, DataLoader
from scipy.stats import spearmanr



class Simulation2Evaluation():

  def __init__(self,n_nodes,n_graphs,graph_name):
    self.graph_name = graph_name
    self.n_nodes = n_nodes
    self.n_graphs = n_graphs
    self.test_set1 = []
    self.test_set2 = []
    self.scaler_x = joblib.load( f'/data/scaler.pkl')
   
  def fit_test_data(self):
    test_dataset1 = torch.load(,f'/data/test/test_set1_simulation2_{self.graph_name}_{self.n_graphs}_graphs_{self.n_nodes}_nodes.pkl')     
    test_dataset2 = torch.load(,f'/data/test/test_set1_simulation2_{self.graph_name}_{self.n_graphs}_graphs_{self.n_nodes}_nodes.pkl')
    test_loader1 = DataLoader(test_dataset1,batch_size = len(test_dataset1),shuffle=False,drop_last=True)
    test_loader2 = DataLoader(test_dataset2,batch_size = len(test_dataset1),shuffle=False,drop_last=True)
    for data1,data2 in zip(test_loader1,test_loader2):
      data1.x = torch.tensor(self.scaler_x.transform(data1.x)).to(torch.float32)
      data2.x = torch.tensor(self.scaler_x.transform(data2.x)).to(torch.float32)

    self.test_set1 = test_loader1.datasets
    self.test_set2 = test_loader2.datasets
    return test_dataset,test_dataset2

  def evaluate_GNN_Siamese(self):
      
    checkpoint_path = '/models/GNN_siamese_best_model_state_dict.pth'
    if device == 'cpu':
      state_dict = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    else:
      state_dict = torch.load(checkpoint_path)

    original_params = state_dict['params']

    updated_params = {"batch_size": 1 , "n_graphs": self.n_graphs, 'conv_hidden_channels':original_params['conv_hidden_channels'], 'activation':original_params['activation'],  'readout_layer':original_params['readout_layer'], 'num_node_features':original_params['num_node_features'],'gat_heads':original_params['gat_heads']}

      # Step 3: Create a new model instance with the updated parameters
      model = GNNSiamese(**updated_params)

      # Step 4: Load the state dictionary into the new model
      model.load_state_dict(state_dict['model_state_dict'])

    network_p_value_list =[]

    test_loader = DataLoader(self.test_set1,batch_size = self.n_graphs,shuffle=False,drop_last=True)
    test_loader2 = DataLoader(self.test_set2,batch_size = self.n_graphs,shuffle=False,drop_last=True)
    for data1,data2 in zip(test_loader,test_loader2):
      model.eval()
      p_value = model(data1.x.to(device), data1.edge_index.to(device), data1.batch.to(device),
                        data2.x.to(device), data2.edge_index.to(device), data2.batch.to(device))  # Perform a single forward pass.
      network_p_value_list.append(pred_corr.item())

    return network_p_value_list

  def evaluate_spectral(self):
    test_loader = DataLoader(self.test_set1,batch_size = 1,shuffle=False,drop_last=False)
    test_loader2 = DataLoader(self.test_set2,batch_size = 1,shuffle=False,drop_last=False)
      
    eig_p_value_list = []
    eigenvalues_calculated_list1 = []
    eigenvalues_calculated_list2 = []
    count = 0
    for data1,data2 in zip(test_loader,test_loader2):
        max_eigenvalue1 = sr.scipy_max_eigenvalue(data1.adj)
        eigenvalues_calculated_list1.append(max_eigenvalue1)

        max_eigenvalue2 = sr.scipy_max_eigenvalue(data2.adj)
        eigenvalues_calculated_list2.append(max_eigenvalue2)

      if count % self.n_graphs == 0:
        eig_corr,eig_p_value = spearmanr(eigenvalues_calculated_list1.numpy(), eigenvalues_calculated_list2.numpy())
        eig_p_value_list.append(eig_p_value)

        eigenvalues_calculated_list1 = []
        eigenvalues_calculated_list2 = []
    return eig_p_value_list

class GNNSiamese(torch.nn.Module):
    def __init__(self, conv_hidden_channels,activation,batch_size,readout_layer,num_node_features,n_graphs,gat_heads):
        super(GCNSimpleSiamese, self).__init__()
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
        pred_corr, p_value = spearmanr(out1.numpy(), out2.numpy())

        return p_value



                                            
                                            



