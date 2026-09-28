"""Teacher -> base maps f_{t->b}: a per-entry MLP on pair-representation channels.

Each map takes a teacher's pair representation (normalised per channel with the
teacher's statistics) and predicts the base model's pair representation in the base's
normalised space. It acts on every (i, j) entry independently, so it translates between
representation spaces without moving information between token pairs.

Architecture used in the paper: Linear(c_t, 1024) -> LayerNorm -> SiLU -> Linear(1024, c_b).
"""
import os

import torch
from torch import nn

MODELS = ("af3", "protenix", "opendde", "esmfold2")


class PairMap(nn.Module):
    def __init__(self, c_in, c_out, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, c_out))

    def forward(self, z):  # noqa: D102
        return self.net(z)


def load_map(weights_dir, teacher, base, device="cuda"):
    """weights/maps/<teacher>_to_<base>.pt -> PairMap in eval mode."""
    ck = torch.load(os.path.join(weights_dir, "maps", f"{teacher}_to_{base}.pt"),
                    map_location="cpu", weights_only=True)
    net = PairMap(ck["c_in"], ck["c_out"], ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()
