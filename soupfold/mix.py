"""SoupFold: mix the anchor model's pair representation with mapped peer representations.

    z' = denorm_a( (x_a + sum_p x_p) / (1 + P) ),   x_a = norm_a(z_a),   x_p = f_{p->a}(norm_p(z_p))

All sources live in the anchor model's per-channel normalised space and are weighted equally.

When the models tokenise a complex differently (a residue or ligand atom present in one
and not another), a peer contributes only on the sub-grid of tokens both models share
(see tokens.py), and the denominator becomes per-entry: entries no peer covers keep the
anchor representation unchanged.
"""
import torch


def normalise(z, stat, device):
    mu = torch.tensor(stat["mean"], dtype=torch.float32, device=device)
    sd = torch.tensor(stat["std"], dtype=torch.float32, device=device)
    return (z - mu) / sd, mu, sd


@torch.no_grad()
def soup(z_anchor, peers, maps, stats, anchor, token_maps=None, device="cuda", chunk_rows=1024):
    """Return z' [N, N, C_a].

    z_anchor    anchor pair representation, [N, N, C_a]
    peers       {name: peer pair representation [N_p, N_p, C_p]}
    maps        {name: PairMap peer -> anchor}
    stats       {model name: {"mean": [C], "std": [C]}} per-channel statistics
    anchor      anchor model name (key into stats)
    token_maps  {name: (idx_anchor, idx_peer)} shared tokens (tokens.align), None assumes
                identical token order and counts
    """
    h0, mu, sd = normalise(z_anchor.float().to(device), stats[anchor], device)
    N = h0.shape[0]
    num = h0.clone()
    den = torch.ones(N, N, 1, device=device) if token_maps is not None else 1.0
    for name, zt in peers.items():
        x = normalise(zt.float().to(device), stats[name], device)[0]
        if token_maps is not None:
            ib, it = (torch.as_tensor(v, device=device, dtype=torch.long) for v in token_maps[name])
            if ib.numel() == 0:
                raise ValueError(f"{anchor} vs {name}: no shared tokens")
            K = ib.numel()
            for r0 in range(0, K, chunk_rows):
                rb = ib[r0:r0 + chunk_rows]
                p = maps[name](x[it[r0:r0 + chunk_rows]][:, it])
                ri, ci = rb.unsqueeze(1).expand(-1, K), ib.unsqueeze(0).expand(rb.numel(), -1)
                num.index_put_((ri, ci), p, accumulate=True)
                den.index_put_((ri, ci), torch.ones_like(p[..., :1]), accumulate=True)
        else:
            if zt.shape[0] != N:
                raise ValueError(f"{anchor} vs {name}: token counts differ ({N} vs {zt.shape[0]}); pass token_maps")
            for r0 in range(0, N, chunk_rows):
                num[r0:r0 + chunk_rows] += maps[name](x[r0:r0 + chunk_rows])
            den = den + 1.0
    zp = (num / den) * sd + mu
    if not torch.isfinite(zp).all():
        raise FloatingPointError(f"{anchor}: z' contains non-finite values")
    return zp
