"""
Reference: https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py
"""

import logging
import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
import sys, os
import requests
from torch_geometric.data import Data
from zipfile import ZipFile
import tarfile
from sklearn.linear_model import TheilSenRegressor
from dcor import distance_correlation
from multiprocessing import Pool
from .io import dataverse_download, tar_data_download_wrapper

logger = logging.getLogger(__name__)


class GeneSimNetwork:
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """

    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(
            self.edge_list,
            source="source",
            target="target",
            edge_attr=["importance"],
            create_using=nx.DiGraph(),
        )
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        # self.edge_weight = torch.Tensor(self.edge_list['importance'].values)

        edge_attr = nx.get_edge_attributes(self.G, "importance")
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)


def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): whether condition is 'ctrl' or not
        pert_names (list): list of perturbations
    """

    return (condition == "ctrl") or (condition in pert_names)

def create_cell_graph_for_prediction(X, pert_idx, pert_gene):
    """
    Create a perturbation specific cell graph for inference

    Args:
        X (np.array): gene expression matrix
        pert_idx (list): list of perturbation indices
        pert_gene (list): list of perturbations

    """

    if pert_idx is None:
        pert_idx = [-1]
    return Data(x=torch.Tensor(X).T, pert_idx=pert_idx, pert=pert_gene)


def create_cell_graph_dataset_for_prediction(
    pert_gene, ctrl_adata, gene_names, device, num_samples=300
):
    """
    Create a perturbation specific cell graph dataset for inference

    Args:
        pert_gene (list): list of perturbations
        ctrl_adata (anndata): control anndata
        gene_names (list): list of gene names
        device (torch.device): device to use
        num_samples (int): number of samples to use for inference (default: 300)

    """

    # Get the indices (and signs) of applied perturbation
    pert_idx = [np.where(p == np.array(gene_names))[0][0] for p in pert_gene]

    Xs = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :].X.toarray()
    # Create cell graphs
    cell_graphs = [
        create_cell_graph_for_prediction(X, pert_idx, pert_gene).to(device) for X in Xs
    ]
    return cell_graphs
