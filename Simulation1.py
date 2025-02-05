import numpy as np
import torch
import networkx as nx
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from torch_geometric.data import Data, DataLoader
from GNN_Siamese import DifferentialSpearmanCorrCoef, GNNSiamese
from .models.Specral_Radius import SpectralRadius


class Simulation1Evaluation():

  def __init__(self,n_nodes,n_graphs,graph_name):
    self.graph_name = graph_name
    self.n_nodes = n_nodes
    self.n_graphs = n_graphs
    self.test_set1 = []
    self.test_set2 = []
    self.scaler_x = joblib.load( f'/data/scaler.pkl')
   
  def fit_test_data(self):
    test_dataset1 = torch.load(,f'/data/test/test_set1_simulation1_{self.graph_name}_{self.n_graphs}_graphs_{self.n_nodes}_nodes.pkl')     
    test_dataset2 = torch.load(,f'/data/test/test_set1_simulation1_{self.graph_name}_{self.n_graphs}_graphs_{self.n_nodes}_nodes.pkl')
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

    spearman = DifferentialSpearmanCorrCoef(num_outputs=1)
    real_corr_list = []
    network_corr_list =[]

    test_loader = DataLoader(self.test_set1,batch_size = self.n_graphs,shuffle=False,drop_last=True)
    test_loader2 = DataLoader(self.test_set2,batch_size = self.n_graphs,shuffle=False,drop_last=True)
    for data1,data2 in zip(test_loader,test_loader2):
      real_corr = spearman(data1.y, data2.y)
      real_corr_list.append(real_corr.item())
      #Newtork p
      model.eval()
      pred_corr = model(data1.x.to(device), data1.edge_index.to(device), data1.batch.to(device),
                        data2.x.to(device), data2.edge_index.to(device), data2.batch.to(device))  # Perform a single forward pass.
      network_corr_list.append(pred_corr.item())

    return real_corr_list,network_corr_list

  def evaluate_spectral(self):
    sr = SpectralRadius()
    test_loader1 = DataLoader(self.test_set1,batch_size = 1,shuffle=False,drop_last=False)
    test_loader2 = DataLoader(self.test_set2,batch_size = 1,shuffle=False,drop_last=False)
    spearman = SpearmanCorrCoef(num_outputs=1)

    eig_corr_list = []
    p_corr_list = []
    eigenvalues_calculated_list1 = []
    eigenvalues_calculated_list2 = []
    p_list1 = []
    p_list2 = []
    count = 0
    for data1,data2 in zip(test_loader,test_loader2):
        max_eigenvalue1 = sr.scipy_max_eigenvalue(data1.adj)
        eigenvalues_calculated_list1.append(max_eigenvalue1)
        p_list1.append(data1.y.item())

        max_eigenvalue2 = sr.scipy_max_eigenvalue(data2.adj)
        eigenvalues_calculated_list2.append(max_eigenvalue2)
        p_list2.append(data1.y.item())

      if count % self.n_graphs == 0:
        eig_corr = spearman(torch.tensor(eigenvalues_calculated_list1), torch.tensor(eigenvalues_calculated_list2))
        eig_corr_list.append(eig_corr.item())
        p_corr = spearman(torch.tensor(p_list1), torch.tensor(p_list2))
        p_corr_list.append(p_corr.item())

        eigenvalues_calculated_list1 = []
        eigenvalues_calculated_list2 = []
        p_list1 = []
        p_list2 = []

    return eig_corr_list,p_corr_list



                                            
                                            



