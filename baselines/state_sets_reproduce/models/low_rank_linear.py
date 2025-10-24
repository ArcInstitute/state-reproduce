import ast
import os
import numpy as np
import torch
import torch.nn as nn
import logging
from collections import defaultdict
from typing import Dict
import h5py
from tqdm import tqdm

from .base import PerturbationModel  # Adjust the import path as needed

logger = logging.getLogger(__name__)


def parse_tahoe_style_pert_col(pert_str):
    drug, dosage, unit = ast.literal_eval(pert_str)[0]
    return drug, dosage


class LowRankLinearModel(PerturbationModel):
    """
    Low Rank Linear Baseline Model
    This model learns a low-rank

    Implementation details:
      - During training (in on_fit_start), we iterate over the training dataloader,
        and for each cell type, we accumulate sums and counts for cells with
        pert_name != self.control_pert.
      - For each cell type, we compute:
            celltype_mean = sum(expression) / count
      - At inference, for each sample in the batch, we look up its cell type and return
        the corresponding average perturbed expression.
      - If the cell type of a sample at inference was not observed during training,
        an assertion error is raised.
    """

    @classmethod
    def from_pretrained_embeddings(
        cls,
        gene_emb_path: str,
        pert_emb_path: str,
        gene_names: list,
        pert_names: list,
        control_pert: str,
        **kwargs,
    ):
        if os.path.exists(gene_emb_path):
            with h5py.File(gene_emb_path, "r") as f:
                emb_gene_names = [
                    n.decode("utf-8") for n in f["gene_names"][:]
                ]  # (n_genes, )
                gene_emb = np.array(f["gene_emb_X"][:])  # (n_genes, gene_emb_dim)

            print(f"Found {len(emb_gene_names)} genes in the pretrained embeddings")

        else:
            raise ValueError(f"Unknown gene embedding: {gene_emb_path}")

        if os.path.exists(pert_emb_path):
            with h5py.File(pert_emb_path, "r") as f:
                try:
                    emb_pert_names = [n.decode("utf-8") for n in f["pert_names"][:]]
                    pert_emb = np.array(f["pert_emb_X"][:])  # (n_perts, pert_emb_dim)
                except:
                    emb_pert_names = [n.decode("utf-8") for n in f["gene_names"][:]]
                    pert_emb = np.array(f["gene_emb_X"][:])  # (n_perts, pert_emb_dim)

            print(
                f"Found {len(emb_pert_names)} perturbations in the pretrained embeddings"
            )

        elif (
            pert_emb_path == "identity"
        ):  # Use the identity matrix as the perturbation embeddings (one-hot encoding)
            emb_pert_names = list(pert_names)
            pert_emb = np.eye(len(pert_names))
        else:
            raise ValueError(f"Unknown perturbation embedding: {pert_emb_path}")

        process_pert_names = kwargs.pop("process_pert_names", None)

        if process_pert_names == "tahoe_style":
            logger.info(f"Processing perturbation names in tahoe style")
            pert_names = [parse_tahoe_style_pert_col(pert)[0] for pert in pert_names]
            # control_pert = parse_tahoe_style_pert_col(control_pert)[0]

        num_genes, gene_emb_dim = gene_emb.shape
        num_perts, pert_emb_dim = pert_emb.shape

        invalid_genes = set(gene_names) - set(emb_gene_names)
        invalid_perts = set(pert_names) - set(emb_pert_names) - {control_pert}

        logger.info(f"pert_names: {sorted(pert_names)[:10]}")
        logger.info(f"emb_pert_names: {sorted(emb_pert_names)[:10]}")
        logger.info(f"control_pert: {control_pert}")

        if len(invalid_genes) > 0:
            logger.warning(
                f"LowRankLinearModel: {len(invalid_genes)}/{len(gene_names)} genes not found in the pretrained embeddings"
            )

        if len(invalid_perts) > 0:
            logger.warning(
                f"LowRankLinearModel: {len(invalid_perts)}/{len(pert_names)} perturbations not found in the pretrained embeddings"
            )

        if process_pert_names == "tahoe_style":
            control_pert_idx = pert_names.index(
                parse_tahoe_style_pert_col(control_pert)[0]
            )
        else:
            control_pert_idx = pert_names.index(control_pert)

        gene_embeddings = torch.zeros(len(gene_names), gene_emb_dim)
        pert_embeddings = torch.zeros(len(pert_names), pert_emb_dim)

        emb_gene_names_indices = [
            emb_gene_names.index(gene)
            for i, gene in enumerate(gene_names)
            if gene not in invalid_genes
        ]
        emb_pert_names_indices = [
            emb_pert_names.index(pert)
            for i, pert in enumerate(pert_names)
            if pert not in invalid_perts and pert != control_pert
        ]

        gene_emb = gene_emb[
            emb_gene_names_indices, :
        ]  # (num_valid_genes, gene_emb_dim)
        pert_emb = pert_emb[
            emb_pert_names_indices, :
        ]  # (num_valid_perts, pert_emb_dim)

        valid_gene_indices = torch.LongTensor(
            [i for i, gene in enumerate(gene_names) if gene not in invalid_genes]
        )
        valid_pert_indices = torch.LongTensor(
            [
                i
                for i, pert in enumerate(pert_names)
                if pert not in invalid_perts and pert != control_pert
            ]
        )

        gene_embeddings[valid_gene_indices, :] = torch.as_tensor(gene_emb)
        pert_embeddings[valid_pert_indices, :] = torch.as_tensor(pert_emb).float()

        invalid_genes_indices = [gene_names.index(gene) for gene in invalid_genes]
        invalid_perts_indices = [pert_names.index(pert) for pert in invalid_perts]

        model = cls(
            gene_emb_dim=gene_emb_dim,
            pert_emb_dim=pert_emb_dim,
            num_perts=len(pert_names),
            control_pert_idx=control_pert_idx,
            hidden_dim=None,
            invalid_genes=invalid_genes_indices,
            invalid_perts=invalid_perts_indices,
            **kwargs,
        )

        model.G.weight.data = gene_embeddings
        model.P.weight.data = pert_embeddings

        model.G.weight.requires_grad = False
        model.P.weight.requires_grad = False

        model.pert_names = pert_names
        model.gene_names = gene_names

        return model

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_perts: int,
        gene_emb_dim: int,
        pert_emb_dim: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        loss_fn=nn.MSELoss(),
        embed_key: str = None,
        output_space: str = "gene",
        decoder=None,
        gene_names=None,
        ridge_lambda: float = 0.1,
        center_Y: bool = True,
        control_pert_idx: int = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: Size of input features (genes or embedding).
            hidden_dim: Not used here, but required by base-class signature.
            output_dim: Dimension of the output (typically number of genes).
            pert_dim: Dimension of perturbation embeddings (not used here).
            n_decoder_layers: (Unused) provided for config compatibility.
            dropout: (Unused) provided for config compatibility.
            lr: Learning rate for the optimizer (dummy param only).
            loss_fn: Loss function for training (default MSELoss).
            embed_key: Optional embedding key (unused).
            output_space: 'gene' or 'latent'. Determines which key from the batch to use.
            decoder: Optional separate decoder (unused).
            gene_names: Optional gene names (for logging or reference).
            kwargs: Catch-all for any extra arguments.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            lr=lr,
            loss_fn=loss_fn,
            embed_key=embed_key,
            output_space=output_space,
            decoder=decoder,
            gene_names=gene_names,
            **kwargs,
        )

        self.G = nn.Embedding(input_dim, gene_emb_dim)  # (input_dim, gene_emb_dim)
        self.P = nn.Embedding(num_perts, pert_emb_dim)

        self.Y = nn.Parameter(torch.zeros(input_dim, num_perts), requires_grad=False)
        self.Y_counts = nn.Parameter(torch.zeros(num_perts), requires_grad=False)

        self.K = nn.Parameter(
            torch.randn(self.G.embedding_dim, self.P.embedding_dim), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(input_dim, device=self.device), requires_grad=False
        )

        self.ridge_lambda = ridge_lambda
        self.center_Y = center_Y
        self.control_pert_idx = control_pert_idx

        self.invalid_gene_indices = torch.tensor(
            list(kwargs.get("invalid_genes", [])), dtype=torch.long
        )
        self.invalid_pert_indices = torch.tensor(
            list(kwargs.get("invalid_perts", [])), dtype=torch.long
        )

    def load_state_dict(self, state_dict, strict=True):
        # Fix bias shape mismatch
        for key, param in state_dict.items():
            if "bias" in key and param.dim() == 2 and param.shape[1] == 1:
                state_dict[key] = param.squeeze(-1)  # [2000, 1] -> [2000]

        return super().load_state_dict(state_dict, strict=strict)

    def on_fit_start(self):
        """Called by Lightning before training starts.

        Computes, for each cell type, the mean perturbed expression from the training data.
        Only cells with a perturbation (pert_name) different from self.control_pert are used.
        """
        super().on_fit_start()

        train_loader = self.trainer.datamodule.train_dataloader()
        if train_loader is None:
            logger.warning("No train dataloader found. Cannot compute cell type means.")
            return

        # Initialize dictionary to accumulate sum and count for each cell type.
        # Need to aggregate the expression change matrices from all training samples
        # and then solve the linear basline with the aggregated expression change matrix.
        # Eq: (Y) = G * K * P^T + b where G and P are the gene and perturbation embedding matrices
        #   with shapes (n_genes, gene_emb_dim) and (n_perts, pert_emb_dim), respectively.
        # K is the low-rank linear transformation matrix with shape (gene_emb_dim, pert_emb_dim).
        # Y is the expression change matrix with shape (n_genes, n_perts).
        # We also have pre-computed Y based on the training data.
        # We want to learn K and b.

        with torch.no_grad():
            for batch in tqdm(
                train_loader,
                desc="Computing expression change matrix Y",
                total=len(train_loader),
            ):
                # Select the proper expression space

                X_pert = batch["pert_cell_emb"]  # (batch_size, n_genes)

                # Ensure the expression values are in float and on CPU
                X_ctrl = batch["ctrl_cell_emb"]  # (batch_size, n_genes)

                batch_size, n_genes = X_pert.shape[0], X_pert.shape[1]

                Y_batch = X_pert - X_ctrl  # (batch_size, n_genes)
                pert_ids = batch["pert_emb"].argmax(dim=1)  # (batch_size)

                cell_line_ids = batch["cell_type_onehot"].argmax(dim=1)  # (batch_size)

                self.Y.scatter_add_(
                    dim=1,
                    index=pert_ids.unsqueeze(0).expand(n_genes, -1).to(self.device),
                    src=Y_batch.T.to(self.device),
                )

                self.Y_counts.scatter_add_(
                    dim=0,
                    index=pert_ids.to(self.device),
                    src=torch.ones_like(pert_ids, dtype=self.Y_counts.dtype).to(
                        self.device
                    ),
                )

        logger.info(
            f"LowRankLinearModel: Computed Expression Change matrix Y with shape {self.Y.shape}."
        )

        self.solve_closed_form_ridge()

        # stop trainer
        self.trainer.should_stop = True

    def solve_closed_form_ridge(self):
        """
        Solves the closed form ridge regression problem:
            min_K ||Y - G * K * P^T - b||^2 + lambda ||K||^2
        where Y is the expression change matrix, G and P are the gene and perturbation embedding matrices,
        b is the bias term, and lambda is the regularization parameter.
        The closed-form solution for the above objective is:
            K = (G^T G + lambda I)^{-1} G^T Y P (P^T P + lambda I)^{-1}
        """
        Y = self.Y  # (n_genes, n_perts)
        Y_counts = self.Y_counts  # (n_perts)
        G = self.G.weight  # (n_genes, gene_emb_dim)
        P = self.P.weight  # (n_perts, pert_emb_dim)

        # Only use perturbations that have been seen during training
        pert_mask = Y_counts > 0
        Y = Y[:, pert_mask]
        P = P[pert_mask]
        Y_counts = Y_counts[pert_mask]

        Y = Y / (Y_counts.unsqueeze(0) + 1e-6)  # (n_genes, n_perts)

        if self.center_Y:
            Y = Y - Y.mean(dim=1, keepdim=True)
            self.bias.data = Y.mean(dim=1)

        if Y.shape[1] == 0:
            raise ValueError(
                "No perturbations seen during training. Cannot solve closed form ridge regression."
            )

        # try:
        # Solution: K = (G^T G + lambda_gene I)^{-1} G^T Y P (P^T P + lambda_pert I)^{-1}
        # gene_term = (G^T G + lambda_gene I)^{-1} G^T
        # pert_term = (P^T P + lambda_pert I)^{-1}
        # K = gene_term @ Y @ P @ pert_term
        gene_part = torch.mm(G.T, G) + self.ridge_lambda * torch.eye(
            G.shape[1], device=self.device
        )
        gene_term = torch.linalg.solve(
            gene_part, torch.mm(G.T, Y)
        )  # (gene_emb_dim, n_genes)

        pert_part = torch.mm(P.T, P) + self.ridge_lambda * torch.eye(
            P.shape[1], device=self.device
        )  # (pert_emb_dim, pert_emb_dim)
        pert_term = torch.linalg.inv(pert_part)  # (pert_emb_dim, pert_emb_dim)

        self.K.data = gene_term @ P @ pert_term  # (gene_emb_dim, pert_emb_dim)

        logger.info(
            f"LowRankLinearModel: Solved closed form ridge regression for K with shape {self.K.shape}."
        )

        # except Exception as e:
        #     raise ValueError(f"Error solving closed form ridge regression: {e}")

    def configure_optimizers(self):
        """
        Returns an optimizer for our dummy parameter if available.
        """
        return None

    def forward(self, batch: dict) -> torch.Tensor:
        """
        For each sample in the batch:
          - If the cell is a control (pert_name == self.control_pert), return the control cell's expression.
          - Otherwise, look up and return the stored average perturbed expression for its cell type.

        Args:
            batch (dict): Dictionary containing at least the keys "cell_type", "pert_name", and the expression key
                          ("X_hvg" if output_space == "gene", else "X").

        Returns:
            torch.Tensor: Predicted expression tensor of shape (B, output_dim).
        """
        batch_size = batch["cell_type_onehot"].size(0)
        # Determine which key to use for the expression values.
        output_key = "pert_cell_emb"

        device = batch[output_key].device

        preds = torch.zeros_like(batch[output_key], device=device)

        pert_ids = batch["pert_emb"].argmax(dim=1)  # (batch_size, )
        pert_embeds = self.P(pert_ids)  # (batch_size, pert_emb_dim)

        preds = (
            torch.mm(self.G.weight, self.K) @ pert_embeds.T
        )  # (num_genes, batch_size)

        ctrl_mask = torch.eq(pert_ids, self.control_pert_idx)

        preds = preds.T  # (batch_size, num_genes)

        # For control cells, simply return the control cell's expression
        preds[ctrl_mask] = batch[output_key][ctrl_mask].to(preds.dtype)

        return preds

    def training_step(self, batch, batch_idx):
        """
        Computes the training loss (MSE) for the entire batch.
        For control cells (where pert_name == self.control_pert), the prediction is simply the control cell's expression.
        For perturbed cells, the prediction is the cell type's average perturbed expression computed during on_fit_start.

        Args:
            batch (dict): Batch dictionary containing keys such as "cell_type", "pert_name", and the expression key.
            batch_idx (int): Batch index (unused here).

        Returns:
            torch.Tensor: The computed loss.
        """
        pred = self(batch)
        output_key = "pert_cell_emb"
        target = batch[output_key]
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return None

    def validation_step(self, batch, batch_idx):
        """
        Computes the validation loss (MSE) for the entire batch.
        """
        pred = self(batch)
        output_key = "pert_cell_emb"
        target = batch[output_key]
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss, prog_bar=True)
        return None

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Computes the prediction for the entire batch.
        """
        preds = self.forward(batch)  # (batch_size, num_genes)
        preds = preds + self.bias.unsqueeze(0)
        basal = batch["ctrl_cell_emb"]  # (batch_size, num_genes)

        final_preds = basal + preds

        batch_size, num_genes = preds.shape
        device = preds.device
        valid_gene_mask = ~torch.isin(
            torch.arange(num_genes, device=device), self.invalid_gene_indices.to(device)
        )

        valid_pert_mask = ~torch.isin(
            batch["pert_emb"].argmax(dim=1), self.invalid_pert_indices.to(device)
        )

        final_preds = torch.nan_to_num(
            final_preds[valid_pert_mask][:, valid_gene_mask],
            nan=0.0,
            posinf=1e3,
            neginf=0,
        )
        X = batch["pert_cell_emb"][valid_pert_mask][:, valid_gene_mask]

        pert_names = [p for i, p in enumerate(batch["pert_name"]) if valid_pert_mask[i]]
        cell_types = [c for i, c in enumerate(batch["cell_type"]) if valid_pert_mask[i]]
        batch_names = [
            b for i, b in enumerate(batch["batch_name"]) if valid_pert_mask[i]
        ]

        return {
            "preds": final_preds,
            "X": X,
            "pert_name": pert_names,
            "cell_type": cell_types,
            "batch_name": batch_names,
        }

    def on_save_checkpoint(self, checkpoint):
        """
        Save the computed cell type means to the checkpoint.
        """
        super().on_save_checkpoint(checkpoint)
        # Convert each tensor to a CPU numpy array for serialization.
        checkpoint["Y"] = self.Y.cpu().numpy()
        checkpoint["Y_counts"] = self.Y_counts.cpu().numpy()
        logger.info("LowRankLinearModel: Saved Y and Y_counts to checkpoint.")

    def on_load_checkpoint(self, checkpoint):
        """
        Load the cell type means from the checkpoint.
        """
        super().on_load_checkpoint(checkpoint)
        if "Y" in checkpoint:
            self.Y.data = torch.tensor(checkpoint["Y"], dtype=torch.float32)
            self.Y_counts.data = torch.tensor(
                checkpoint["Y_counts"], dtype=torch.float32
            )
            logger.info(f"LowRankLinearModel: Loaded Y and Y_counts from checkpoint.")
        else:
            logger.warning("LowRankLinearModel: No Y or Y_counts found in checkpoint.")

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """
        Identity encoding; no perturbation embedding is learned.
        """
        return self.P(pert.argmax(dim=1))

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """
        Identity encoding; no basal expression transformation.
        """
        return expr

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Not used in the normal forward pass. Returns basal unchanged.
        """
        return basal + self.forward(pert)

    def _build_networks(self):
        """
        No networks to build for this baseline model.
        """
        pass
