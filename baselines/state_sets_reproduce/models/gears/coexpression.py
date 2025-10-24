"""
Reference: https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py
"""

import os
import pandas as pd
import numpy as np

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def get_coexpression_network_from_train(
    adata,
    threshold,
    k,
    data_path,
    data_name,
    split,
    seed,
    train_gene_set_size,
    set2conditions,
):
    """
    Infer co-expression network from training data

    Args:
        adata (anndata.AnnData): anndata object
        threshold (float): threshold for co-expression
        k (int): number of edges to keep
        data_path (str): path to data
        data_name (str): name of dataset
        split (str): split of dataset
        seed (int): seed for random number generator
        train_gene_set_size (int): size of training gene set
        set2conditions (dict): dictionary of perturbations to conditions
    """

    fname = os.path.join(
        os.path.join(data_path, data_name),
        split
        + "_"
        + str(seed)
        + "_"
        + str(train_gene_set_size)
        + "_"
        + str(threshold)
        + "_"
        + str(k)
        + "_co_expression_network.csv",
    )

    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.gene_name.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list))
        X = adata.X
        train_perts = set2conditions["train"]
        X_tr = X[np.isin(adata.obs.condition, [i for i in train_perts if "ctrl" in i])]
        gene_list = adata.var["gene_name"].values

        X_tr = X_tr.toarray()
        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1) :]
        out_sort_val = np.sort(out)[:, -(k + 1) :]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(
            columns={0: "source", 1: "target", 2: "importance"}
        )
        df_co_expression.to_csv(fname, index=False)
        return df_co_expression
