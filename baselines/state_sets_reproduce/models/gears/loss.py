import torch
import numpy as np


def uncertainty_loss_fct(
    pred, logvar, y, perts, reg=0.1, ctrl=None, direction_lambda=1e-3, dict_filter=None
):
    """
    Uncertainty loss function

    Args:
        pred (torch.tensor): predicted values
        logvar (torch.tensor): log variance
        y (torch.tensor): true values
        perts (list): list of perturbations
        reg (float): regularization parameter
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        if p != "ctrl":
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts == p)[0]][:, retain_idx]
            y_p = y[np.where(perts == p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts == p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts == p)[0]]
            y_p = y[np.where(perts == p)[0]]
            logvar_p = logvar[np.where(perts == p)[0]]

        # uncertainty based loss
        losses += (
            torch.sum(
                (pred_p - y_p) ** (2 + gamma)
                + reg * torch.exp(-logvar_p) * (pred_p - y_p) ** (2 + gamma)
            )
            / pred_p.shape[0]
            / pred_p.shape[1]
        )

        # direction loss
        if p != "ctrl":
            losses += (
                torch.sum(
                    direction_lambda
                    * (
                        torch.sign(y_p - ctrl[retain_idx])
                        - torch.sign(pred_p - ctrl[retain_idx])
                    )
                    ** 2
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
        else:
            losses += (
                torch.sum(
                    direction_lambda
                    * (torch.sign(y_p - ctrl) - torch.sign(pred_p - ctrl)) ** 2
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )

    return losses / (len(set(perts)))


def loss_fct(
    pred,
    y,
    x_basal,
    cell_lines,
    pert_indices,
    unique_pert,
    unique_cell_lines,
    direction_lambda=1e-3,
):
    """
    Main MSE Loss function, includes direction loss

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    num_perts = len(unique_pert)
    num_cell_lines = len(unique_cell_lines)

    num_existing_cats = 0
    for cell_line in unique_cell_lines:
        for p in unique_pert:
            pert_idx = torch.where((pert_indices == p) & (cell_lines == cell_line))[0]

            if pert_idx.shape[0] == 0:
                continue

            num_existing_cats += 1

            # during training, we remove the all zero genes into calculation of loss.
            # this gives a cleaner direction loss. empirically, the performance stays the same.

            X_pert = y[pert_idx, :]
            X_basal = x_basal[pert_idx, :].mean(dim=0)

            retain_idx = X_pert.sum(axis=0) > 0
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = X_pert[:, retain_idx]
            x_basal_p = X_basal[retain_idx]

            losses = (
                losses
                + torch.sum((pred_p - y_p) ** (2 + gamma))
                / pred_p.shape[0]
                / pred_p.shape[1]
            )

            losses = (
                losses
                + torch.sum(
                    direction_lambda
                    * (torch.sign(y_p - x_basal_p) - torch.sign(pred_p - x_basal_p))
                    ** 2
                )
                / pred_p.shape[0]
                / pred_p.shape[1]
            )
    return losses / num_existing_cats
