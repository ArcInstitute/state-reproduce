# File: benchmark/models/simple_sum.py

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict
from collections import defaultdict

from models.base import PerturbationModel
from models.utils import build_mlp

logger = logging.getLogger(__name__)


class SimpleSumPerturbationModel(PerturbationModel):
    """
    A simple baseline model that:
      1) Computes the mean offset for each perturbation (X - X_control) over the entire training set.
      2) At inference time, for a cell with basal state 'basal' and perturbation 'pert_name',
         we predict 'basal + mean_offset[pert_name]'.

    If output_space == "gene", we additionally learn a one-layer MLP (or linear layer) on top
    of the sum in order to map from the 'summed space' to gene expression dimension.

    Otherwise, if output_space == "latent", we return the 'basal + offset' directly (no learned layer).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn=nn.MSELoss(),  # the base class will parse from config
        embed_key: str = None,
        output_space: str = "gene",
        decoder=None,
        gene_names=None,
        **kwargs,
    ):
        """
        Args:
            input_dim: Dimensionality of input features (genes or latent).
            hidden_dim: Not strictly used (since we do no big model),
                        but we accept it to conform with base-class signature.
            output_dim: Dim of the output (genes if output_space='gene', else same as input_dim).
            pert_dim:  Dimension of the perturbation one-hot embedding or other encoding.
            dropout:   Not used in this baseline (we do no big net).
            lr:        Learning rate for optimizer (used if we have a final MLP for gene space).
            loss_fn:   The chosen PyTorch loss function.
            embed_key: Possibly the name of the embedding key.
            output_space: "gene" or "latent".
            decoder:   Possibly a decoder interface if output_space='latent'.
            gene_names: Names of genes (useful for logging).
            **kwargs:   Catch-all for any other config arguments.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            dropout=dropout,
            lr=lr,
            loss_fn=loss_fn,
            embed_key=embed_key,
            output_space=output_space,
            decoder=decoder,
            gene_names=gene_names,
            **kwargs,
        )

        # This will store {pert_name -> mean_offset_vector}, shape of mean_offset_vector == [input_dim].
        self.n_decoder_layers = n_decoder_layers
        self.pert_mean_offsets: Dict[str, torch.Tensor] = None
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

        # We'll build the minimal "MLP" only if output_space='gene'.
        # If in latent space, we just do (basal + offset).
        self._build_networks()

    def _build_networks(self):
        """
        Build the optional linear layer if output_space == 'gene'.
        If output_space == 'latent', we do nothing.
        """
        if self.output_space == "gene":
            # build mlp for the projection layer from UCE to gene space
            self.project_out = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_decoder_layers,
                dropout=self.dropout,
            )
        else:
            self.project_out = None

    def on_fit_start(self):
        """
        Called by PyTorch Lightning right before training begins.
        We'll compute the mean offsets from the entire training set in a single pass.
        """
        super().on_fit_start()

        # We'll gather sums, counts for each perturbation in the training loader.
        offset_sums = defaultdict(lambda: torch.zeros(self.input_dim))
        offset_counts = defaultdict(int)

        logger.info("SimpleSum: collecting offsets from the entire training set ...")
        train_loader = self.trainer.datamodule.train_dataloader()

        with torch.no_grad():
            for batch in train_loader:
                # The base class's collate_fn returns dictionary with "X", "basal", "pert_name", ...
                X = batch["X"]  # shape: (B, input_dim)
                basal = batch["basal"]  # shape: (B, input_dim)
                pert_names = batch["pert_name"]  # list of strings length B

                X_cpu = X.float()
                basal_cpu = basal.float()
                offset_cpu = X_cpu - basal_cpu  # shape (B, input_dim)

                for i, p_name in enumerate(pert_names):
                    # convert to string in case it's a single-element list or tensor
                    p_name_str = str(p_name)
                    offset_sums[p_name_str] += offset_cpu[i]
                    offset_counts[p_name_str] += 1

        self.pert_mean_offsets = {}
        for p_name_str, sum_vec in offset_sums.items():
            c = offset_counts[p_name_str]
            if c < 1:
                continue
            mean_offset = sum_vec / float(c)
            # store it as a CPU float tensor, shape [input_dim]
            self.pert_mean_offsets[p_name_str] = mean_offset

        logger.info("SimpleSum: offsets are computed for %d unique perturbations." % len(self.pert_mean_offsets))

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Not really used here, but required by abstract base. We do no param-based encoding."""
        return pert

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """No param-based encoding of basal. Just identity."""
        return expr

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Not used in the normal forward pass, because we look up offset by 'pert_name' strings.
        But we must define it to meet the abstract contract.
        """
        return basal

    def forward(self, batch: dict) -> torch.Tensor:
        """
        The main forward pass. We do:
            1) For each sample i in the batch, get the precomputed offset for that perturbation.
            2) sum with basal -> if output_space='gene', pass into self.project_out -> return
        """
        if not self.pert_mean_offsets:
            raise RuntimeError("SimpleSum: offsets not yet computed; please run fit first.")

        basal = batch["basal"]  # shape [B, input_dim]
        B = basal.shape[0]

        # We'll gather the offsets for each sample
        offsets_list = []
        for i, p_name in enumerate(batch["pert_name"]):
            p_name_str = str(p_name)
            if p_name_str not in self.pert_mean_offsets:
                # If we never saw this perturbation in training, fallback to zero offset
                offset_vec = torch.zeros(self.input_dim)
            else:
                offset_vec = self.pert_mean_offsets[p_name_str]
            offsets_list.append(offset_vec)

        # stack => shape [B, input_dim]
        offsets_tensor = torch.stack(offsets_list, dim=0).to(basal.device)
        perturbed = basal + offsets_tensor

        # If output_space == 'gene', we apply the project_out linear layer
        if self.project_out is not None:
            perturbed = self.project_out(perturbed)

        return perturbed

    def configure_optimizers(self):
        """Return optimizer only if we have trainable parameters."""
        if self.project_out is not None:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        return None  # No parameters to optimize

    def training_step(self, batch, batch_idx):
        """Override to handle parameter-free case."""
        pred = self(batch)
        target = batch["X_hvg"] if self.embed_key and self.output_space == "gene" else batch["X"]
        loss = self.loss_fn(pred, target)

        # Only log the loss if we're actually training parameters
        if self.project_out is not None:
            self.log("train_loss", loss, prog_bar=True)
            return loss
        return None  # Skip optimizer step when no parameters

    def on_save_checkpoint(self, checkpoint):
        """
        Save self.pert_mean_offsets into the checkpoint dictionary
        so it can be restored at test-time or for inference.
        """
        # convert the offsets to CPU numpy for safe serialization
        offset_dict = {k: v.cpu().numpy() for k, v in self.pert_mean_offsets.items()}
        checkpoint["simple_sum_offsets"] = offset_dict

    def on_load_checkpoint(self, checkpoint):
        """
        Load self.pert_mean_offsets from the checkpoint dictionary.
        """
        if "simple_sum_offsets" in checkpoint:
            offset_np_dict = checkpoint["simple_sum_offsets"]
            loaded_offsets = {}
            for k, arr in offset_np_dict.items():
                loaded_offsets[k] = torch.tensor(arr, dtype=torch.float32)
            self.pert_mean_offsets = loaded_offsets
            self.offsets_computed = True
            logger.info(f"SimpleSum: loaded {len(self.pert_mean_offsets)} offsets from checkpoint.")
        else:
            self.offsets_computed = False
            logger.warning("SimpleSum: no precomputed offsets found in checkpoint. Predictions may fail.")
