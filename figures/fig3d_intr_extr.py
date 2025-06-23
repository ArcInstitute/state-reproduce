"""
Script to generate figure 3d.

Requires a .h5 with embedding keys and corresponding toml files.
1. uses toml file to get held-out perturbations from .h5 
2. for each cell type, splits .h5ad into train/val/test and trains a MLP to classify perturbations on these embeddings
2. computes AUROC/ACCURACY on test set (INTRINSIC)
3. if --extrinsic-root provided, evaluates the trained model on predicted data from ST and outputs AUROC/ACCURACY (EXTRINSIC)
"""

import argparse
import os
import sys
import numpy as np
import scanpy as sc
import torch
import toml
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark all embeddings using perturbations listed in TOML test sets")
    p.add_argument("--data-dir", type=str, required=True, help="Directory containing .h5ad and TOML files")
    p.add_argument("--val-split", type=float, default=0.20)
    p.add_argument("--perturb-key", type=str, default="perturbation")
    p.add_argument("--cell-type-key", type=str, default="cell_type")
    p.add_argument("--extrinsic-root", type=str, required=False, help="Base directory for ST predicted output")
    p.add_argument("--extrinsic-step", type=str, default="eval_step=80000.ckpt", help="Step directory name under each cell type")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-layers", type=int, default=1)
    return p.parse_args()

def load_test_perturbations(data_dir):
    test_perturbs = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".toml"):
            cell_type = fname.replace(".toml", "")
            conf = toml.load(os.path.join(data_dir, fname))
            try:
                test_list = conf['fewshot'][f'{cell_type}']['test']
                test_perturbs[cell_type] = test_list
            except KeyError:
                print(f"Warning: No test set in {fname}, skipping")
    return test_perturbs

def filter_by_test_perturbations(adata, embed_key, perturb_key, cell_type_key, test_perturbations):
    valid_data = {}
    for cell_type, perturbs in test_perturbations.items():
        subset = adata[(adata.obs[cell_type_key] == cell_type) & 
                       (adata.obs[perturb_key].isin(perturbs))].copy()
        if subset.n_obs == 0:
            print(f"Skipping {cell_type}: No valid cells after filtering for test perturbations")
            continue
        labels = subset.obs[perturb_key].astype("category")
        features = subset.obsm[embed_key]
        valid_data[cell_type] = (features, labels.cat.codes.values, list(labels.cat.categories))
    return valid_data

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout=0.1):
        super().__init__()
        layers, norms = [], []
        layers.append(nn.Linear(in_dim, hidden_dim))
        norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            norms.append(nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        for lin, norm in zip(self.layers, self.norms):
            x = self.drop(self.act(norm(lin(x))))
        return self.out(x)

def split_indices_fraction(labels, val_frac, n_groups, seed):
    idx = np.arange(len(labels))
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for g in range(n_groups):
        inds = idx[labels == g]
        if len(inds) == 0:
            continue
        rng.shuffle(inds)
        n = len(inds)
        holdout = int(np.floor(val_frac * n))
        n_train = n - holdout
        n_val = holdout // 2
        n_test = holdout - n_val
        train_idx.extend(inds[:n_train])
        val_idx.extend(inds[n_train:n_train + n_val])
        test_idx.extend(inds[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def make_loaders(features, labels, train_idx, val_idx, test_idx, bs):
    def mk(sub):
        if len(sub) == 0: return None
        return TensorDataset(torch.FloatTensor(features[sub]), torch.LongTensor(labels[sub]))
    return tuple(DataLoader(mk(idx), batch_size=bs, shuffle=(i==0)) if mk(idx) else None
                 for i, idx in enumerate([train_idx, val_idx, test_idx]))

def train_and_select(model, loaders, epochs, lr, device):
    train_loader, val_loader, _ = loaders
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        if val_loader:
            model.eval()
            total_val = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    total_val += loss_fn(model(X), y).item() * X.size(0)
            avg_val = total_val / len(val_loader.dataset)
            if avg_val < best_val:
                best_val = avg_val
                best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate(model, loader, device):
    if loader is None: return float('nan'), float('nan'), float('nan')
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_preds, all_probs, all_labels = 0, [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * X.size(0)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    n_classes = all_probs.shape[1]
    
    if n_classes > 2:
        auroc_scores = []
        for i in range(n_classes):
            binary_labels = (all_labels == i).astype(int)
            class_probs = all_probs[:, i]
            try:
                auroc = roc_auc_score(binary_labels, class_probs)
                auroc_scores.append(auroc)
            except ValueError:
                pass
        auroc = np.nanmean(auroc_scores) if auroc_scores else float("nan")
    else:
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auroc = float("nan")
    
    return avg_loss, accuracy, auroc

def evaluate_extrinsic(model, embed_key, cell_type, label_names, args, device):
    ext_path = os.path.join(args.extrinsic_root, emb_name, cell_type, args.extrinsic_step, "adata_pred.h5ad")
    if not os.path.exists(ext_path):
        print(f"[EXTRINSIC] Missing: {ext_path}")
        return float("nan"), float("nan"), float("nan")
    adata_ext = sc.read_h5ad(ext_path)
    if args.perturb_key not in adata_ext.obs:
        print(f"[EXTRINSIC] Missing perturbation key in {ext_path}")
        return float("nan"), float("nan"), float("nan")

    adata_ext = adata_ext[adata_ext.obs[args.cell_type_key] == cell_type]
    adata_ext = adata_ext[adata_ext.obs[args.perturb_key].isin(label_names)].copy()
    if adata_ext.n_obs == 0:
        print(f"[EXTRINSIC] No matching cells for {cell_type} in extrinsic")
        return float("nan"), float("nan"), float("nan")

    labels = adata_ext.obs[args.perturb_key].astype("category")
    label_map = {name: i for i, name in enumerate(label_names)}
    label_indices = np.array([label_map.get(lbl, -1) for lbl in labels])
    valid_mask = label_indices >= 0
    if emb_name == 'hvg':
        features = adata_ext.X[valid_mask]
    else:
        features = adata_ext.obsm[embed_key][valid_mask]
    labels = label_indices[valid_mask]

    test_loader = DataLoader(TensorDataset(torch.FloatTensor(features), torch.LongTensor(labels)), batch_size=args.batch_size)
    return evaluate(model, test_loader, device)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    adata = sc.read_h5ad(os.path.join(args.data_dir, "processed.h5"))
    test_perturbations = load_test_perturbations(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for embed_key in adata.obsm_keys():
        print(f"\n{'='*60}\nBenchmarking embedding: {embed_key}\n{'='*60}")
        celltype_data = filter_by_test_perturbations(adata, embed_key, args.perturb_key, args.cell_type_key, test_perturbations)
        all_results = []

        for cell_type, (features, labels, label_names) in celltype_data.items():
            tr_idx, va_idx, te_idx = split_indices_fraction(labels, args.val_split, len(label_names), args.seed)
            loaders = make_loaders(features, labels, tr_idx, va_idx, te_idx, args.batch_size)
            model = MLPClassifier(features.shape[1], 1024, len(label_names), args.n_layers).to(device)
            model = train_and_select(model, loaders, args.epochs, args.lr, device)
            test_loss, test_acc, test_auroc = evaluate(model, loaders[2], device)
            print(f"{cell_type} [INTR]: loss={test_loss:.4f}, acc={test_acc:.4f}, AUROC={test_auroc:.4f}")

            extr_loss, extr_acc, extr_auroc = float('nan'), float('nan'), float('nan')
            if args.extrinsic_root:
                extr_loss, extr_acc, extr_auroc = evaluate_extrinsic(model, embed_key, cell_type, label_names, args, device)
                print(f"{cell_type} [EXTR]: loss={extr_loss:.4f}, acc={extr_acc:.4f}, AUROC={extr_auroc:.4f}")

            all_results.append((test_loss, test_acc, test_auroc, extr_loss, extr_acc, extr_auroc))

        if all_results:
            arr = np.array(all_results)
            print(f"\nSummary for {embed_key}:")
            print(f"INTR: Loss={np.nanmean(arr[:,0]):.4f}, Accuracy={np.nanmean(arr[:,1]):.4f}, AUROC={np.nanmean(arr[:,2]):.4f}")
            if args.extrinsic_root:
                print(f"EXTR: Loss={np.nanmean(arr[:,3]):.4f}, Accuracy={np.nanmean(arr[:,4]):.4f}, AUROC={np.nanmean(arr[:,5]):.4f}")


if __name__ == "__main__":
    main()
