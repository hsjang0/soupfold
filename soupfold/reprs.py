"""Trunk representations on disk: <root>/<model>/seed_<seed>/<system>.npz with
z [N, N, C_z] (float16, lossless for the bf16 trunk output) and
s [N, C_s] (float32, absent for ESMFold2, whose trunk has no single representation)."""
import json
import os

import numpy as np


def path(root, model, seed, system):
    return os.path.join(root, model, f"seed_{seed}", f"{system}.npz")


def save(root, model, seed, system, z, s=None, **meta):
    p = path(root, model, seed, system)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    to = lambda t: t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    arrs = {"z": to(z).astype(np.float16)}
    if s is not None:
        arrs["s"] = to(s).astype(np.float32)
    arrs["meta"] = np.array(json.dumps(dict(meta, seed=int(seed))))
    tmp = p + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return p


def load(root, model, seed, system):
    """-> (z float32, s float32 or None)."""
    with np.load(path(root, model, seed, system)) as f:
        return f["z"].astype(np.float32), (f["s"].astype(np.float32) if "s" in f.files else None)
