#!/usr/bin/env python

import argparse
import os
import sys
import pickle
import re
import gc
import yaml
import logging
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from scipy.sparse import csr_matrix
from tqdm import tqdm

from cell_load.mapping_strategies import (
    BatchMappingStrategy,
    RandomMappingStrategy,
)
from cell_load.data_modules import PerturbationDataModule
from cell_load.utils.modules import get_datamodule

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    """
    CLI for evaluation. The arguments mirror some of the old script_lightning/eval_lightning.py.
    """
    parser = argparse.ArgumentParser(
        description="Get predictions from a trained PerturbationModel."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    return parser.parse_args()


def post_process_preds(batch_preds, model_class_name):
    if "cell_mask" in batch_preds:
        new_batch_preds = {}
        cell_mask = batch_preds["cell_mask"]  # (batch_size, )
        for k, v in batch_preds.items():
            if k == "cell_mask":
                continue
            elif isinstance(v, torch.Tensor):
                new_batch_preds[k] = v[cell_mask]
            elif isinstance(v, list) and v[0] is not None:
                new_batch_preds[k] = [v[i] for i in range(len(v)) if cell_mask[i]]
            else:
                new_batch_preds[k] = v
    else:
        new_batch_preds = batch_preds

    if "gene_mask" in batch_preds:
        gene_mask = (
            batch_preds["gene_mask"].detach().cpu().numpy()
        )  # (batch_size, num_genes)
        gene_mask = gene_mask[0, :]
        gene_mask = gene_mask.astype(bool)
    else:
        gene_mask = None

    return new_batch_preds, gene_mask


def load_config(cfg_path: str) -> dict:
    """
    Load config from the YAML file that was dumped during training.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config file: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_latest_step_checkpoint(directory):
    # Get all checkpoint files
    files = os.listdir(directory)

    # Extract step numbers using regex, excluding files with 'val_loss'
    step_numbers = []
    for f in files:
        if f.startswith("step=") and "val_loss" not in f:
            # Extract the number between 'step=' and '.ckpt'
            match = re.search(r"step=(\d+)(?:-v\d+)?\.ckpt", f)
            if match:
                step_numbers.append(int(match.group(1)))

    if not step_numbers:
        raise ValueError("No checkpoint files found")

    # Get the maximum step number
    max_step = max(step_numbers)

    # Construct the checkpoint path
    checkpoint_path = os.path.join(directory, f"step={max_step}.ckpt")

    return checkpoint_path


def main():
    args = parse_args()

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])

    try:
        cell_sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
    except:
        cell_sentence_len = 1

    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # 3. Load the data module
    data_module = get_datamodule(
        name=cfg["data"]["name"],
        kwargs=cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=cell_sentence_len,
    )
    data_module.setup()

    # seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 4. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    if cfg["model"]["name"].lower() == "lowranklinear":
        if (
            cfg["model"]["kwargs"]["pert_emb"] == "identity"
        ):  # Use the identity matrix as the perturbation embeddings (one-hot encoding)
            cfg["model"]["kwargs"]["pert_emb_path"] = "identity"
        elif (
            cfg["model"]["kwargs"]["pert_emb"] == "scgpt"
        ):  # scGPT: Genetic perturbation data
            cfg["model"]["kwargs"][
                "pert_emb_path"
            ] = f"/large_storage/goodarzilab/userspace/mohsen/VCI-models/scGPT/scGPT_human/gene_embeddings.h5"
        elif (
            cfg["model"]["kwargs"]["pert_emb"] == "tahoe_rdkit"
        ):  # Tahoe: Chemical perturbation data
            cfg["model"]["kwargs"][
                "pert_emb_path"
            ] = "/large_storage/goodarzilab/userspace/mohsen/VCI/tahoe/tahoe_rdkit_embs.h5"
        elif (
            cfg["model"]["kwargs"]["pert_emb"] == "gears_norman"
        ):  # Extract GEARS perturbation embeddings from the trained GEARS on Norman2019 dataset
            cfg["model"]["kwargs"][
                "pert_emb_path"
            ] = "/large_storage/goodarzilab/userspace/mohsen/VCI-models/GEARS/gears_norman.h5"
        else:
            raise ValueError(
                f"Unknown perturbation embedding: {cfg['model']['kwargs']['pert_emb']}"
            )

        if (
            cfg["model"]["kwargs"]["gene_emb"] == "training_data"
        ):  # Use the training data as the gene embeddings
            # 1. Perform PCA on the training data
            raise NotImplementedError("PCA on training data is not implemented yet")
        elif (
            cfg["model"]["kwargs"]["gene_emb"] == "gears_norman"
        ):  # Extract GEARS gene embeddings from the trained GEARS on Norman2019 dataset
            cfg["model"]["kwargs"][
                "gene_emb_path"
            ] = "/large_storage/goodarzilab/userspace/mohsen/VCI-models/GEARS/gears_norman.h5"
        elif (
            cfg["model"]["kwargs"]["gene_emb"] == "scgpt"
        ):  # Extract scGPT's vocabulary embeddings
            cfg["model"]["kwargs"][
                "gene_emb_path"
            ] = f"/large_storage/goodarzilab/userspace/mohsen/VCI-models/scGPT/scGPT_human/gene_embeddings.h5"
        else:
            raise ValueError(f"Unknown gene embedding: {cfg['model']['gene_emb']}")

    # The model architecture is determined by the config
    model_class_name = cfg["model"]["name"]  # e.g. "EmbedSum" or "NeuralOT"
    model_kwargs = cfg["model"]["kwargs"]  # dictionary of hyperparams

    # Build the correct class
    if model_class_name.lower() == "cpa":
        from state_sets_reproduce.models.cpa import CPAPerturbationModel

        ModelClass = CPAPerturbationModel
    elif model_class_name.lower() in ["scgpt-genetic", "scgpt-chemical", "scgpt"]:
        from state_sets_reproduce.models.scgpt import scGPTForPerturbationModel

        ModelClass = scGPTForPerturbationModel
    elif model_class_name.lower() == "scvi":
        from state_sets_reproduce.models.scvi import SCVIPerturbationModel

        ModelClass = SCVIPerturbationModel
    elif model_class_name.lower() == "lowranklinear":
        from state_sets_reproduce.models.low_rank_linear import LowRankLinearModel

        ModelClass = LowRankLinearModel
    elif model_class_name.lower() == "gears":
        from state_sets_reproduce.models.gears import GEARSPerturbationModel

        ModelClass = GEARSPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()  # e.g. input_dim, output_dim, pert_dim
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        # "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        # other model_kwargs keys to pass along:
        **model_kwargs,
    }

    # load checkpoint
    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    if model_class_name.lower() in ["scgpt", "scgpt-genetic", "scgpt-chemical"]:
        model.to(torch.bfloat16)

    logger.info("Model loaded successfully.")

    baseline_models = [
        "scvi",
        "cpa",
        "lowranklinear",
        "scgpt-genetic",
        "scgpt-chemical",
        "scgpt",
    ]

    # 5. Run inference on test set
    if model_class_name.lower() in [
        "scvi",
        "cpa",
        "lowranklinear",
        "gears",
    ] or model_class_name.lower().startswith("scgpt"):
        if (
            "cell_sentence_len" in cfg["model"]["kwargs"]
            and cfg["model"]["kwargs"]["cell_sentence_len"] > 1
        ):
            data_module.cell_sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
        else:
            data_module.cell_sentence_len = 1
    else:
        data_module.cell_sentence_len = cfg["model"]["kwargs"][
            "transformer_backbone_kwargs"
        ]["n_positions"]

    test_loader = data_module.test_dataloader()

    print(f"DEBUG: data_module.batch_size: {data_module.batch_size}")

    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    # num_cells = test_loader.batch_sampler.tot_num
    # output_dim = var_dims["output_dim"]
    # gene_dim = var_dims["gene_dim"]
    # hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    final_preds = []
    final_reals = []

    store_raw_expression = (
        data_module.embed_key is not None
        and data_module.embed_key != "X_hvg"
        and cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (
        data_module.embed_key is not None
        and cfg["data"]["kwargs"]["output_space"] == "all"
    )

    if store_raw_expression:
        # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
        if cfg["data"]["kwargs"]["output_space"] == "gene":
            final_X_hvg = []
            final_gene_preds = []
        if cfg["data"]["kwargs"]["output_space"] == "all":
            final_X_hvg = []
            final_gene_preds = []
    else:
        # Otherwise, use lists for later concatenation.
        final_X_hvg = None
        final_gene_preds = None

    logger.info("Generating predictions on test set ...")

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_ctrl_cell_barcodes = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Predicting", unit="batch")
        ):
            # Move each tensor in the batch to the model's device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            # Get predictions
            with torch.autocast(device_type="cuda", enabled=True):
                if model_class_name.lower() in [
                    "scgpt",
                    "scgpt-genetic",
                    "scgpt-chemical",
                ]:
                    batch = {
                        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                batch_preds = model.predict_step(batch, batch_idx, padded=False)

                if batch_preds["preds"] is None:
                    continue

                batch_preds, gene_mask = post_process_preds(
                    batch_preds, model_class_name
                )

            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
            else:
                all_pert_names.append(batch_preds["pert_name"])

            # Handle ctrl_cell_barcode
            if batch_preds["ctrl_cell_barcode"] is not None and isinstance(
                batch_preds["ctrl_cell_barcode"], list
            ):
                all_ctrl_cell_barcodes.extend(batch_preds["ctrl_cell_barcode"])
            else:
                all_ctrl_cell_barcodes.append(batch_preds["ctrl_cell_barcode"])

            # Handle celltype_name
            if isinstance(batch_preds["cell_type"], list):
                all_celltypes.extend(batch_preds["cell_type"])
            elif isinstance(batch_preds["cell_type"], torch.Tensor):
                if batch_preds["cell_type"].ndim == 2:
                    all_celltypes.extend(
                        batch_preds["cell_type"].argmax(dim=1).cpu().numpy().tolist()
                    )  # backward compatibility
                else:
                    all_celltypes.extend(batch_preds["cell_type"].cpu().numpy())
            else:
                all_celltypes.append(batch_preds["cell_type"])

            # Handle gem_group
            if isinstance(batch_preds["batch_name"], list):
                all_gem_groups.extend(batch_preds["batch_name"])
            elif isinstance(batch_preds["batch_name"], torch.Tensor):
                if batch_preds["batch_name"].ndim == 2:
                    all_gem_groups.extend(
                        batch_preds["batch_name"].argmax(dim=1).cpu().numpy().tolist()
                    )  # backward compatibility
                else:
                    all_gem_groups.extend(batch_preds["batch_name"].cpu().numpy())
            else:
                all_gem_groups.append(batch_preds["batch_name"])

            batch_pred_np = batch_preds["preds"].float().cpu().numpy()
            batch_real_np = batch_preds["X"].float().cpu().numpy()
            batch_size = batch_pred_np.shape[0]

            # print(f"DEBUG: batch_pred_np.shape: {batch_pred_np.shape}, batch_real_np.shape: {batch_real_np.shape}")

            final_preds.append(batch_pred_np)
            final_reals.append(batch_real_np)

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["X_hvg"].cpu().numpy()
                final_X_hvg.append(batch_real_gene_np)

            # Handle decoded gene predictions if available
            if final_gene_preds is not None:
                batch_gene_pred_np = batch_preds["gene_preds"].cpu().numpy()
                final_gene_preds.append(batch_gene_pred_np)

    logger.info("Creating anndatas from predictions from manual loop...")

    # Build pandas DataFrame for obs
    obs_data = {
        "pert_name": all_pert_names,
        "celltype_name": all_celltypes,
        "gem_group": all_gem_groups,
    }

    if len(all_ctrl_cell_barcodes) > 0:
        obs_data["ctrl_cell_barcode"] = all_ctrl_cell_barcodes

    obs = pd.DataFrame(obs_data)

    final_preds = np.concatenate(final_preds, axis=0)
    final_reals = np.concatenate(final_reals, axis=0)

    print(
        f"DEBUG: obs.shape: {obs.shape}, final_preds.shape: {final_preds.shape}, final_reals.shape: {final_reals.shape}"
    )

    # Create adata for predictions
    adata_pred = anndata.AnnData(X=final_preds, obs=obs)
    # Create adata for real
    adata_real = anndata.AnnData(X=final_reals, obs=obs)

    # Create adata for real data in gene space (if available)
    adata_real_gene = None
    if (
        final_X_hvg and len(final_X_hvg) > 0
    ):  # either this is available, or we are already working in gene space
        final_X_hvg = np.concatenate(final_X_hvg, axis=0)
        if "int_counts" in data_module.__dict__ and data_module.int_counts:
            final_X_hvg = np.log1p(final_X_hvg)
        adata_real_gene = anndata.AnnData(X=final_X_hvg, obs=obs)

    # Create adata for gene predictions (if available)
    adata_pred_gene = None
    if final_gene_preds and len(final_gene_preds) > 0:
        final_gene_preds = np.concatenate(final_gene_preds, axis=0)
        if "int_counts" in data_module.__dict__ and data_module.int_counts:
            final_gene_preds = np.log1p(final_gene_preds)
        adata_pred_gene = anndata.AnnData(X=final_gene_preds, obs=obs)

    # save out adata_real to the output directory
    adata_real_out = os.path.join(args.output_dir, "adata_real.h5ad")
    adata_real.write_h5ad(adata_real_out)
    logger.info(f"Saved adata_real to {adata_real_out}")

    adata_pred_out = os.path.join(args.output_dir, "adata_pred.h5ad")
    adata_pred.write_h5ad(adata_pred_out)
    logger.info(f"Saved adata_pred to {adata_pred_out}")

    if adata_real_gene is not None:
        adata_real_gene_out = os.path.join(args.output_dir, "adata_real_gene.h5ad")
        adata_real_gene.write_h5ad(adata_real_gene_out)
        logger.info(f"Saved adata_real_gene to {adata_real_gene_out}")

    if adata_pred_gene is not None:
        adata_pred_gene_out = os.path.join(args.output_dir, "adata_pred_gene.h5ad")
        adata_pred_gene.write_h5ad(adata_pred_gene_out)
        logger.info(f"Saved adata_pred_gene to {adata_pred_gene_out}")


if __name__ == "__main__":
    main()
