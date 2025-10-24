import torch
import torch.nn as nn

import lightning as L
from torchmetrics.functional import r2_score, pearson_corrcoef
from .generation_model import ChemicalTransformerGenerator, TransformerGenerator
from .utils import map_raw_id_to_vocab_id
from .loss import masked_mse_loss

from typing import Any, Union, Optional, Literal, List
from ..base import PerturbationModel


class scGPTForPerturbationModel(PerturbationModel):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: dict,
        n_drug_tokens: int = 0,
        dropout: float = 0.5,
        pad_token_id: int = 0,
        pad_value: int = 0,
        pert_pad_id: int = 2,
        do_mvc: bool = False,
        domain_spec_batchnorm: Union[bool, str] = False,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        lr: float = 1e-4,
        step_size_lr: int = 1,
        include_zero_gene: Optional[Literal["all", "batch-wise"]] = "all",
        max_seq_len: int = 1536,
        do_CLS: bool = True,
        do_CCE: bool = False,
        do_MVC: bool = False,
        do_ECS: bool = False,
        gene_names: Optional[List[str]] = None,
        embed_key: Optional[str] = None,
        perturbation_type: Literal["chemical", "genetic"] = "chemical",
        **kwargs,
    ):
        super().__init__(
            input_dim=None,
            hidden_dim=None,
            output_dim=None,
            pert_dim=None,
            dropout=dropout,
            lr=lr,
            loss_fn="mse",
            embed_key=embed_key,
            output_space="gene",
            decoder=None,
            gene_names=gene_names,
            batch_size=64,
        )
        self.ntoken = ntoken
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nlayers_cls = nlayers_cls
        self.n_cls = n_cls
        self.n_drug_tokens = n_drug_tokens
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.do_mvc = do_mvc
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.cell_emb_style = cell_emb_style
        self.mvc_decoder_style = mvc_decoder_style
        self.ecs_threshold = ecs_threshold
        self.explicit_zero_prob = explicit_zero_prob
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend
        self.pre_norm = pre_norm
        self.perturbation_type = perturbation_type.lower()

        for k, v in kwargs.items():
            print(f"WARNING: scGPTForPerturbation Model unused kwarg: {k}")

        self.lr = lr
        self.step_size_lr = step_size_lr
        self.include_zero_gene = include_zero_gene
        self.max_seq_len = max_seq_len
        self.do_CLS = do_CLS
        self.do_CCE = do_CCE
        self.do_MVC = do_MVC
        self.do_ECS = do_ECS

        assert self.perturbation_type in [
            "chemical",
            "genetic",
        ], "perturbation_type must be either 'chemical' or 'genetic'"

        if self.perturbation_type == "chemical":
            assert (
                self.n_drug_tokens > 0
            ), "n_drug_tokens must be greater than 0 for chemical perturbation"

        self.vocab = vocab
        self.gene_ids = torch.tensor(
            [
                vocab[gene] if gene in vocab else vocab["<pad>"]
                for gene in self.gene_names
            ],
            dtype=int,
        )[torch.randperm(len(self.gene_names))[:2000]]

        num_invalid_genes = torch.sum(self.gene_ids == vocab["<pad>"])
        if num_invalid_genes > 0:
            print(
                f"scGPTForPerturbationModel: Found {num_invalid_genes} invalid genes in vocab (len = {len(self.gene_names)})"
            )

        if self.perturbation_type == "genetic":
            num_genes_X = len(self.gene_names)
            self.pert_flags = {}

            pert_names = kwargs.get("pert_names", [])
            control_pert = kwargs.get(
                "control_pert",
            )
            num_invalid_perts = 0
            for pert in pert_names:
                self.pert_flags[pert] = torch.zeros(num_genes_X)
                if pert in self.gene_names:
                    self.pert_flags[pert][self.gene_names.index(pert)] = 1
                else:
                    if pert != control_pert:
                        num_invalid_perts += 1

            if num_invalid_perts > 0:
                print(
                    f"scGPTForPerturbationModel: Found {num_invalid_perts} invalid perturbations in pert_names"
                )

            pert_flag_embs = torch.stack([self.pert_flags[pert] for pert in pert_names])

            self.pert_flag_emb = nn.Embedding.from_pretrained(
                pert_flag_embs, freeze=True
            )
            self.control_pert_idx = pert_names.index(control_pert)

        self.save_hyperparameters()

        self.validation_outputs = []

        self._build_networks()

    def _build_networks(self):
        generator_params = dict(
            ntoken=self.ntoken,
            d_model=self.d_model,
            nhead=self.nhead,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            nlayers_cls=self.nlayers_cls,
            n_cls=self.n_cls,
            dropout=self.dropout,
            pad_token_id=self.pad_token_id,
            pad_value=self.pad_value,
            pert_pad_id=self.pert_pad_id,
            do_mvc=self.do_mvc,
            domain_spec_batchnorm=self.domain_spec_batchnorm,
            cell_emb_style=self.cell_emb_style,
            mvc_decoder_style=self.mvc_decoder_style,
            ecs_threshold=self.ecs_threshold,
            explicit_zero_prob=self.explicit_zero_prob,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend=self.fast_transformer_backend,
            pre_norm=self.pre_norm,
        )
        if self.perturbation_type == "chemical":
            self.model = ChemicalTransformerGenerator(
                n_drug_tokens=self.n_drug_tokens,
                **generator_params,
            )
        elif self.perturbation_type == "genetic":
            self.model = TransformerGenerator(**generator_params)

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Map perturbation to an effect vector in embedding space."""
        raise NotImplementedError("Perturbation encoding not supported for scGPT model")

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Expression is already in embedding space, pass through."""
        raise NotImplementedError(
            "Basal expression encoding not supported for scGPT model"
        )

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Given a perturbation and basal embeddings, compute the perturbed embedding.
        """
        # Project perturbation and basal cell state to latent space
        raise NotImplementedError("Perturb function not supported for scGPT model")

    def _log_normalize_expression(
        self, X: torch.Tensor, target_sum: int = 10000
    ) -> torch.Tensor:
        """
        Normalize expression to have a desired sum.
        """
        counts_per_cell = X.sum(dim=1, keepdim=True)
        safe_counts = torch.where(
            counts_per_cell > 0, counts_per_cell, torch.ones_like(counts_per_cell)
        )
        safe_counts = safe_counts / target_sum

        X = torch.true_divide(X, safe_counts)
        X = torch.log1p(X)

        return X

    def preprocess_batch(self, batch):
        X_pert = batch["pert_cell_emb"]
        X_ctrl = batch["ctrl_cell_emb"]

        if X_pert.max() > 25:  # Raw counts
            batch["pert_cell_emb"] = self._log_normalize_expression(
                X_pert, target_sum=1e4
            )
            batch["ctrl_cell_emb"] = self._log_normalize_expression(
                X_ctrl, target_sum=1e4
            )
        else:
            batch["pert_cell_emb"] = self._log_normalize_expression(
                torch.expm1(X_pert), target_sum=1e4
            )
            batch["ctrl_cell_emb"] = self._log_normalize_expression(
                torch.expm1(X_ctrl), target_sum=1e4
            )

        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.step_size_lr, gamma=0.9
        )
        return [optimizer], [scheduler]

    def shared_step(
        self,
        batch,
        truncate=True,
    ):
        batch = self.preprocess_batch(batch)

        x_ctrl = batch["ctrl_cell_emb"]  # (batch_size, n_genes)
        x_pert = batch["pert_cell_emb"]  # (batch_size, n_genes)
        pert_ids = batch["pert_emb"].argmax(dim=1)

        if self.perturbation_type == "chemical":
            pert_flags = torch.zeros_like(
                x_pert, dtype=torch.long
            )  # no genes are perturbed
            cell_mask = torch.ones_like(pert_ids, dtype=torch.bool)
        else:
            pert_flags = self.pert_flag_emb(pert_ids)
            cell_mask = (pert_flags.sum(dim=1) > 0) | (
                pert_ids == self.control_pert_idx
            )

        gene_ids = self.gene_ids.to(x_ctrl.device)

        nonpad_genes_mask = gene_ids != self.pad_token_id

        x_ctrl = x_ctrl[:, nonpad_genes_mask]
        x_pert = x_pert[:, nonpad_genes_mask]
        gene_ids = gene_ids[nonpad_genes_mask]
        pert_flags = pert_flags[:, nonpad_genes_mask]

        batch_size, n_genes = x_ctrl.size()

        if self.include_zero_gene == "all":
            input_gene_ids = torch.arange(
                n_genes, device=x_ctrl.device, dtype=torch.long
            )
        else:
            input_gene_ids = (
                x_ctrl.nonzero()[:, 1].flatten().unique().sort()[0]
            )  # TODO: need double-check

        # sample input_gene_id
        if truncate:
            if len(input_gene_ids) > self.max_seq_len:
                input_gene_ids = torch.randperm(
                    len(input_gene_ids), device=x_ctrl.device
                )[: self.max_seq_len]

            x_ctrl = x_ctrl[:, input_gene_ids]
            x_pert = x_pert[:, input_gene_ids]
            pert_flags = pert_flags[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(
            batch_size, 1
        )  # (batch_size, max_seq_len)

        src_key_padding_mask = mapped_input_gene_ids.eq(self.pad_token_id)
        # src_key_padding_mask = torch.zeros_like(
        #     x_basal, dtype=torch.bool, device=x_basal.device
        # )

        if self.perturbation_type == "genetic":
            pert_ids = None

        output_dict = self.model(
            mapped_input_gene_ids.long(),
            pert_ids,
            x_ctrl,
            pert_flags.long(),
            src_key_padding_mask=src_key_padding_mask,
            CLS=self.do_CLS and self.training,
            CCE=self.do_CCE and self.training,
            MVC=self.do_MVC and self.training,
            ECS=self.do_ECS and self.training,
        )
        output_values = output_dict["mlm_output"]  # batch_size, max_seq_len

        masked_positions = torch.ones_like(x_ctrl, dtype=torch.bool)  # Use all genes
        loss = loss_mse = masked_mse_loss(
            output_values[cell_mask], x_pert[cell_mask], masked_positions[cell_mask]
        )

        output_dict["x_pred"] = output_dict["mlm_output"].float()
        output_dict["x_true"] = x_pert
        output_dict["x_basal"] = x_ctrl

        return loss, output_dict

    def compute_metrics(self, x_basal, x_pred, x_true):
        """
        Computes a sets of evaluation metrics for assessing perturbation response prediction's quality.

        Parameters
        ----------
        x_pred : torch.Tensor of shape (batch_size, n_genes)
            The predicted perturbation response.
        x_true : torch.Tensor of shape (batch_size, n_genes)
            The true perturbation response.
        """
        r2_mean = r2_score(
            x_pred.mean(0), x_true.mean(0)
        )  # R2 score for mean gene expression

        change_pred = x_pred - x_basal
        change_true = x_true - x_basal

        pearson_mean_lfc = pearson_corrcoef(change_pred.mean(0), change_true.mean(0))

        return {
            "r2_mean": r2_mean,
            "pearson_mean_lfc": pearson_mean_lfc,
        }

    def training_step(self, batch, batch_idx):
        loss, batch_outputs = self.shared_step(batch)

        metrics = self.compute_metrics(
            x_basal=batch_outputs["x_basal"],
            x_pred=batch_outputs["x_pred"],
            x_true=batch_outputs["x_true"],
        )

        self.log("loss", loss, prog_bar=True)

        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_outputs = self.shared_step(batch)

        metrics = self.compute_metrics(
            x_basal=batch_outputs["x_basal"],
            x_pred=batch_outputs["x_pred"],
            x_true=batch_outputs["x_true"],
        )

        # self.validation_outputs.append(
        #     (
        #         batch_outputs["x_basal"].detach().cpu(),
        #         batch_outputs["x_pred"].detach().cpu(),
        #         batch_outputs["x_true"].detach().cpu(),
        #     )
        # )

        self.log("val_loss", loss, prog_bar=True)

        for k, v in metrics.items():
            self.log(f"val_{k}", v, prog_bar=True)

        return loss

    # def validation_epoch_end(self, outputs):
    #     pass

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'pred', 'X', 'pert_name', etc.
        """
        loss, batch_outputs = self.shared_step(batch, truncate=False)

        batch = self.preprocess_batch(batch)

        gene_mask = self.gene_ids != self.pad_token_id

        if self.perturbation_type == "genetic":
            pert_ids = batch["pert_emb"].argmax(dim=1)
            cell_mask = (self.pert_flag_emb(pert_ids).sum(dim=1) > 0) | (
                pert_ids == self.control_pert_idx
            )
        else:
            cell_mask = torch.ones_like(
                batch["pert_emb"].argmax(dim=1), dtype=torch.bool
            )

        preds = batch_outputs["x_pred"].float()[cell_mask]
        X = batch["pert_cell_emb"].float()[cell_mask]

        pert_names = batch.get("pert_name", None)
        if isinstance(pert_names, torch.Tensor):
            pert_names = pert_names.cpu().numpy()[cell_mask]
        elif isinstance(pert_names, list):
            pert_names = [p for i, p in enumerate(pert_names) if cell_mask[i]]
        else:
            pert_names = None

        cell_types = batch.get("cell_type", None)
        if isinstance(cell_types, torch.Tensor):
            cell_types = cell_types.cpu().numpy()[cell_mask]
        elif isinstance(cell_types, list):
            cell_types = [c for i, c in enumerate(cell_types) if cell_mask[i]]
        else:
            cell_types = None

        batch_names = batch.get("batch_name", None)
        if isinstance(batch_names, torch.Tensor):
            batch_names = batch_names.cpu().numpy()[cell_mask]
        elif isinstance(batch_names, list):
            batch_names = [b for i, b in enumerate(batch_names) if cell_mask[i]]
        else:
            batch_names = None

        return {
            "preds": preds,
            "X": X[:, gene_mask],
            "pert_name": pert_names,
            "cell_type": cell_types,
            "batch_name": batch_names,
        }
