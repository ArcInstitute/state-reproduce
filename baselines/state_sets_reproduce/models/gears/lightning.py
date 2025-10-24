import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


from ..base import PerturbationModel
from .model import GEARS_Model
from .coexpression import get_coexpression_network_from_train
from .go import load_go_graph, load_gene2go
from .utils import GeneSimNetwork
from .loss import loss_fct, uncertainty_loss_fct

logger = logging.getLogger(__name__)


class GEARSPerturbationModel(PerturbationModel):
    """
    GEARS Perturbation Model
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pert_dim: int,
        gene_dim: int,
        coexpression_graph_path: str,  # Path to co-expression graph
        batch_dim: int = None,
        hidden_dim: int = 64,
        num_similar_genes_go_graph: int = 20,
        num_go_gnn_layers: int = 1,
        num_gene_gnn_layers: int = 1,
        decoder_hidden_size: int = 16,
        coexpress_threshold: float = 0.4,
        uncertainty: bool = False,
        uncertainty_reg: float = 1.0,
        direction_lambda: float = 1e-1,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        no_perturb: bool = False,
        pert_type: str = "genetic",
        gene_names: Optional[List[str]] = None,
        pert_names: Optional[List[str]] = None,
        control_pert: str = "non-targeting",
        go_graph_url: str = "https://dataverse.harvard.edu/api/access/datafile/6934319",
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            lr=lr,
            gene_names=gene_names,
            control_pert=control_pert,
            **kwargs,
        )

        self.gene_dim = gene_dim
        self.coexpression_graph_path = coexpression_graph_path
        self.go_graph_url = go_graph_url
        self.data_pert_names = pert_names or []
        self.hidden_size = hidden_dim
        self.num_similar_genes_go_graph = num_similar_genes_go_graph
        self.num_go_gnn_layers = num_go_gnn_layers
        self.num_gene_gnn_layers = num_gene_gnn_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.coexpress_threshold = coexpress_threshold
        self.uncertainty = uncertainty
        self.uncertainty_reg = uncertainty_reg
        self.direction_lambda = direction_lambda
        self.weight_decay = weight_decay
        self.no_perturb = no_perturb
        self.pert_type = pert_type

        assert self.pert_type in [
            "genetic",
            "chemical",
        ], "pert_type must be either 'genetic' or 'chemical'"

        if self.pert_type == "genetic":
            self.gene2go = load_gene2go()

            self.pert_names = np.unique(list(self.gene2go.keys()))
            self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}

            invalid_pert_names = [
                p
                for p in self.data_pert_names
                if p not in self.pert_names and p != self.control_pert
            ]

            if len(invalid_pert_names) > 0:
                print(
                    f"GEARSPerturbationModel: Found {len(invalid_pert_names)} invalid perturbations in data_pert_names. Will ignore them in loss calculation."
                )

            self.data_pert_idx_map = {
                idx: self.pert_names.tolist().index(pert)
                for idx, pert in enumerate(self.data_pert_names)
                if pert not in invalid_pert_names + [self.control_pert]
            }
            self.data_pert_idx_map[self.data_pert_names.index(self.control_pert)] = -1
        else:  # chemical
            self.pert_names = self.data_pert_names
            self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            self.data_pert_idx_map = {
                idx: self.pert_names.index(pert)
                for idx, pert in enumerate(self.data_pert_names)
            }

        self._build_networks()

    def _build_networks(self):
        self.gears_config = {
            "hidden_size": self.hidden_size,
            "num_similar_genes_go_graph": self.num_similar_genes_go_graph,
            "num_go_gnn_layers": self.num_go_gnn_layers,
            "num_gene_gnn_layers": self.num_gene_gnn_layers,
            "decoder_hidden_size": self.decoder_hidden_size,
            "uncertainty": self.uncertainty,
            "uncertainty_reg": self.uncertainty_reg,
            "direction_lambda": self.direction_lambda,
            "num_genes": len(self.gene_names),
            "num_perts": len(self.pert_names),
            "no_perturb": False,
            "G_go": None,  # Will be set during setup
            "G_go_weight": None,
            "G_coexpress": None,  # Will be set during setup
            "G_coexpress_weight": None,
            "pert_type": self.pert_type,
        }

        self.setup_graphs()

        self.gears_model = GEARS_Model(self.gears_config)

    def setup_graphs(
        self,
    ):
        """
        Set up the co-expression and GO graphs needed by GEARS.
        """
        # Set up co-expression graph
        if self.gears_config["G_coexpress"] is None:
            logger.info("Building co-expression similarity graph...")
            if not os.path.exists(self.coexpression_graph_path):
                raise FileNotFoundError(
                    f"Co-expression graph not found at {self.coexpression_graph_path}. If you haven't computed it, please run `python -m models.gears.coexpression`"
                )  # TODO: complete the command

            edge_list = pd.read_csv(self.coexpression_graph_path)

            gene_list = self.gene_names
            node_map = {gene: i for i, gene in enumerate(gene_list)}

            sim_network = GeneSimNetwork(edge_list, gene_list, node_map=node_map)

            self.gears_config["G_coexpress"] = sim_network.edge_index.to(self.device)
            self.gears_config["G_coexpress_weight"] = sim_network.edge_weight.to(
                self.device
            )

        # Set up GO graph
        if self.gears_config["G_go"] is None and self.pert_type == "genetic":
            logger.info("Building GO similarity graph...")
            edge_list = load_go_graph(
                url=self.go_graph_url,
                k=self.num_similar_genes_go_graph,
            )

            node_map_pert = {pert: i for i, pert in enumerate(self.pert_names)}

            sim_network = GeneSimNetwork(
                edge_list, self.data_pert_names, node_map=node_map_pert
            )
            self.gears_config["G_go"] = sim_network.edge_index.to(self.device)
            self.gears_config["G_go_weight"] = sim_network.edge_weight.to(self.device)
        elif self.pert_type == "chemical":
            self.gears_config["G_go"] = None
            self.gears_config["G_go_weight"] = None

    def extract_batch_tensors(self, batch: Dict[str, torch.Tensor]):
        batch = self.transfer_batch_to_device(batch, self.device, 0)

        x_pert = batch["pert_cell_emb"]
        x_basal = batch["ctrl_cell_emb"]
        pert = batch["pert_emb"]
        cell_type = batch["cell_type_onehot"]
        batch_ids = batch["batch"]

        # if pert is one-hot, convert to index
        if pert.dim() == 2:
            pert = pert.argmax(1)

        pert = torch.LongTensor(
            [self.data_pert_idx_map.get(idx.item(), -2) for idx in pert]
        ).to(self.device)

        # mask the samples with invalid perturbations
        mask = pert != -2
        x_pert = x_pert[mask]
        x_basal = x_basal[mask]
        pert = pert[mask]
        cell_type = cell_type[mask]
        batch_ids = batch_ids[mask]

        unique_pert = torch.unique(pert)
        unique_cell_lines = torch.unique(cell_type)

        batch["mask"] = mask
        batch["unique_pert"] = unique_pert
        batch["unique_cell_lines"] = unique_cell_lines

        if cell_type.dim() == 2:
            cell_type = cell_type.argmax(1)

        if batch_ids.dim() == 2:
            batch_ids = batch_ids.argmax(1)

        return (
            x_pert,
            x_basal,
            pert,
            cell_type,
            batch_ids,
            unique_pert,
            unique_cell_lines,
        )

    def forward(self, x, pert_idx, batch) -> torch.Tensor:
        """
        Forward pass through GEARS model.

        Args:
            batch: Dictionary containing:
                - X: Gene expression data [batch_size, gene_dim]
                - pert_idx: Perturbation indices [batch_size, max_perts]
                - (other keys as needed)

        Returns:
            Predicted gene expression [batch_size, gene_dim]
        """
        if self.uncertainty:
            pred, logvar = self.gears_model(x=x, pert_idx=pert_idx, batch=batch)
            return pred, logvar
        else:
            pred = self.gears_model(x=x, pert_idx=pert_idx, batch=batch)
            return pred, None

    def _convert_batch_to_gears_format(self, batch):
        from torch_geometric.data import Data

        x = batch["X"]
        batch_size, num_genes = x.shape

        # Create perturbation indices
        # GEARS expects a list of lists, where each inner list contains
        # the perturbation indices for that sample
        if "pert_idx" in batch:
            pert_idx = batch["pert_idx"]
        elif "pert_name" in batch:
            # Convert perturbation names to indices
            pert_idx = []
            for i, pert_name in enumerate(batch["pert_name"]):
                if pert_name == self.control_pert or pert_name == "ctrl":
                    pert_idx.append([-1])  # Control condition
                else:
                    # Find perturbation index
                    if pert_name in self.pert_names:
                        idx = self.pert_names.index(pert_name)
                        pert_idx.append([idx])
                    else:
                        pert_idx.append([-1])  # Unknown perturbation, treat as control
        else:
            # Default to control condition for all samples
            pert_idx = [[-1] for _ in range(batch_size)]

        # Create batch indices - each sample gets its own batch index
        batch_indices = torch.arange(batch_size, device=x.device)

        # Create Data object in the format expected by GEARS
        # x should be the gene expression values
        # pert_idx should be the perturbation indices
        # batch should indicate which sample each gene belongs to
        data = Data(
            x=x,  # Keep original shape [batch_size, num_genes]
            pert_idx=pert_idx,  # List of perturbation indices
            batch=batch_indices,  # Batch indices
        )

        return data

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for GEARS model."""
        x_pert, x_basal, pert, cell_type, batch_ids, unique_pert, unique_cell_lines = (
            self.extract_batch_tensors(batch)
        )
        pred, logvar = self.forward(x_basal, pert, batch_ids)
        if self.uncertainty:
            raise NotImplementedError("Uncertainty is not yet implemented for GEARS")
        else:
            # loss = self.loss_fn(pred, x_pert)
            loss = loss_fct(
                pred=pred,
                y=x_pert,
                x_basal=x_basal,
                cell_lines=cell_type,
                pert_indices=pert,
                unique_pert=unique_pert,
                unique_cell_lines=unique_cell_lines,
                direction_lambda=self.direction_lambda,
            )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step for GEARS model."""
        x_pert, x_basal, pert, cell_type, batch_ids, unique_pert, unique_cell_lines = (
            self.extract_batch_tensors(batch)
        )
        pred, logvar = self.forward(x_basal, pert, batch_ids)

        if self.uncertainty:
            raise NotImplementedError("Uncertainty is not yet implemented for GEARS")
        else:
            # loss = self.loss_fn(pred, x_pert)
            loss = loss_fct(
                pred=pred,
                y=x_pert,
                x_basal=x_basal,
                cell_lines=cell_type,
                pert_indices=pert,
                unique_pert=unique_pert,
                unique_cell_lines=unique_cell_lines,
                direction_lambda=self.direction_lambda,
            )

        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss, "predictions": pred}

    def predict_step(self, batch, batch_idx, **kwargs):
        x_pert, x_basal, pert, cell_type, batch_ids, unique_pert, unique_cell_lines = (
            self.extract_batch_tensors(batch)
        )
        if x_pert.shape[0] == 0:
            return {
                "preds": None,
                "X": None,
                "pert_name": None,
                "cell_type": None,
                "batch_name": None,
            }

        pred, logvar = self.forward(x_basal, pert, batch_ids)
        outputs = {
            "preds": pred,
            "X": batch.get("pert_cell_emb", None),
            "pert_name": batch.get("pert_name", None),
            "cell_type": batch.get("cell_type", None),
            "batch_name": batch.get("batch_name", None),
        }

        outputs = {k: v for k, v in outputs.items() if v is not None}

        return outputs

    def configure_optimizers(self):
        """Configure optimizer for GEARS model."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Use step scheduler as in original GEARS
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
