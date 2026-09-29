"""Trunk capture / trunk bypass for Protenix (2.0.0). OpenDDE (1.1.0) has the same trunk
interface and reuses these hooks (hooks/opendde.py).

Both models run the trunk in `get_pairformer_output`, which returns (s_inputs, s, z),
the diffusion module, distogram head and confidence head then consume only these three.
`s_inputs` is a pure function of the featurised input and is recomputed, so (s, z) is the
representation that is stored and injected.

Bypass stubs the trunk's three stages: the MSA module (which returns z by ASSIGNMENT, so
its stub is the identity on z), the template embedder (skipped via n_blocks = 0), and
the pairformer (returns the injected s, z at every recycle).
"""
import contextlib

import torch
from torch import nn

from .. import tokens


class _IdentityZ(nn.Module):
    n_blocks = 0

    def forward(self, input_feature_dict, z, *args, **kwargs):  # noqa: D102
        return z


class _NoTemplate(nn.Module):
    n_blocks = 0

    def forward(self, *args, **kwargs):  # noqa: D102
        raise AssertionError("template branch must be skipped when n_blocks == 0")


class _FixedPairformer(nn.Module):
    def __init__(self, s, z):
        super().__init__()
        self._s, self._z = s, z

    def forward(self, s, z, **kwargs):  # noqa: D102
        return self._s, self._z


@contextlib.contextmanager
def capture_trunk(model, box):
    """Record (s, z) as the trunk hands them over. The forward pass continues unchanged."""
    original = model.get_pairformer_output

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        box["s"], box["z"] = out[1], out[2]
        return out

    model.get_pairformer_output = wrapped
    try:
        yield
    finally:
        model.get_pairformer_output = original


def drop_template_features(feature_dict):
    """The trunk consumes (and deletes) the template features. With the trunk bypassed they
    would reach the diffusion cache and collide with the injected z."""
    for key in [k for k in feature_dict if "template_" in k]:
        del feature_dict[key]


@contextlib.contextmanager
def trunk_bypass(model, s, z):
    """Run the model with the trunk replaced by the given (s, z)."""
    saved = (model.msa_module, model.pairformer_stack, model.template_embedder)
    model.msa_module, model.template_embedder = _IdentityZ(), _NoTemplate()
    model.pairformer_stack = _FixedPairformer(s, z)
    try:
        yield
    finally:
        model.msa_module, model.pairformer_stack, model.template_embedder = saved


@contextlib.contextmanager
def seed_diffusion(model, seed):
    """Reseed right before the diffusion sampler, so a bypass run and a standalone run reach it
    with the same RNG state (the skipped trunk would otherwise consume random draws)."""
    original = model.sample_diffusion

    def wrapped(*args, **kwargs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return original(*args, **kwargs)

    model.sample_diffusion = wrapped
    try:
        yield
    finally:
        model.sample_diffusion = original


def token_layout(atom_array):
    """Token layout (tokens.py) from the atom array of the featurised input: a token is identified
    by its centre atom."""
    c = atom_array[atom_array.centre_atom_mask.astype(bool)]
    return tokens.rows(c.asym_id_int, c.res_id, c.element, c.atom_name)
