import torch.nn as nn

from geomloss import SamplesLoss


def get_loss_fn(
    name: str,
    **kwargs,
) -> nn.Module:
    """
    Get a loss function by name.
    """
    if name == "MSELoss":
        return nn.MSELoss()
    elif name == "L1Loss":
        return nn.L1Loss()
    elif name == "SamplesLoss":
        return SamplesLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function '{name}'")
