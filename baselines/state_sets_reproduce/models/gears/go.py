"""
Reference: https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py
"""

import os
import pickle
import pandas as pd
import tempfile

from .io import tar_data_download_wrapper, dataverse_download


def load_go_graph(url: str, k: int):
    """
    Load the GO graph from harvard dataverse

    Args:
        url: url of the GO graph
        k: number of edges to keep for each gene

    Returns:
        df_out: pandas dataframe of the GO graph
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        tar_data_download_wrapper(
            url, os.path.join(temp_dir, "go_essential_all"), temp_dir
        )
        df_jaccard = pd.read_csv(
            os.path.join(temp_dir, "go_essential_all/go_essential_all.csv")
        )

        df_out = (
            df_jaccard.groupby("target")
            .apply(lambda x: x.nlargest(k + 1, ["importance"]))
            .reset_index(drop=True)
        )

    return df_out


def load_gene2go(
    gene2go_url: str = 'https://dataverse.harvard.edu/api/access/datafile/6153417', 
    essential_genes_url: str = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
):
    """
    Load the gene2go dictionary from harvard dataverse
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        download_path = os.path.join(temp_dir, 'gene2go_all.pkl')
        dataverse_download(gene2go_url,
                           download_path)
        with open(download_path, 'rb') as f:
            gene2go = pickle.load(f)
            
        download_path = os.path.join(temp_dir, 'essential_all_data_pert_genes.pkl')
        dataverse_download(essential_genes_url,
                           download_path)
        with open(download_path, 'rb') as f:
            essential_genes = pickle.load(f)
            
    gene2go = {i: gene2go[i] for i in essential_genes if i in gene2go}

    return gene2go
