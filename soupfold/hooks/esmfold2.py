"""Trunk capture / trunk bypass for ESMFold2 (esm 3.4.0, `EsmFold2Model`).

ESMFold2's trunk produces a pair representation only (the structure head is called with
s_trunk=None). The stored representation is the post-parcae z, exactly the tensor passed
to `structure_head.sample(z_trunk=...)`. Injection therefore replaces the folding loop AND
both parcae stages, since feeding a post-parcae z through them again would transform it twice.
"""
import contextlib

import torch
from torch import nn

from .. import tokens


class _Identity(nn.Module):
    def forward(self, z, *args, **kwargs):  # noqa: D102
        return z


class StopAfterTrunk(Exception):
    """Raised by capture_trunk(stop=True) once z is recorded."""


@contextlib.contextmanager
def capture_trunk(model, box, stop=False):
    """Record z_trunk as it reaches the structure head. With stop=True the forward pass is
    aborted there (raises StopAfterTrunk), skipping diffusion and the confidence head."""
    head = model.structure_head
    original = head.sample

    def sample(*args, **kwargs):
        box["z"] = kwargs.get("z_trunk", args[0] if args else None)
        if stop:
            raise StopAfterTrunk
        return original(*args, **kwargs)

    head.sample = sample
    try:
        yield
    finally:
        head.sample = original


@contextlib.contextmanager
def trunk_bypass(model, z):
    """Feed a stored post-parcae z straight into the structure head."""
    saved = (model._run_one_loop, model.parcae_readout, model.parcae_coda)
    model._run_one_loop = lambda *a, **k: z
    model.parcae_readout, model.parcae_coda = _Identity(), _Identity()
    try:
        yield
    finally:
        model._run_one_loop, model.parcae_readout, model.parcae_coda = saved


@contextlib.contextmanager
def seed_diffusion(model, seed):
    """Reseed right before the structure head's sampler."""
    head = model.structure_head
    original = head.sample

    def sample(*args, **kwargs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return original(*args, **kwargs)

    head.sample = sample
    try:
        yield
    finally:
        head.sample = original


def token_layout(feats, chains):
    """Token layout (tokens.py) from the featurised input: a token is identified by its first atom."""
    start = {int(t.token_index): int(t.atom_start) for c in chains for t in vars(c)["tokens"]}
    atom = [start[i] for i in range(feats["residue_index"].shape[1])]
    element = feats["ref_element"][0].cpu().numpy()
    element = element.argmax(-1) if element.ndim == 2 else element
    names = feats["ref_atom_name_chars"][0].cpu().numpy()
    return tokens.rows(feats["asym_id"][0].tolist(), feats["residue_index"][0].tolist(),
                       element[atom], [tokens.atom_name(names[a]) for a in atom])
