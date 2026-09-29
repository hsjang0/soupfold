"""Token alignment across co-folding models.

A model's token layout is read from any structure it predicted, in file order: one token
per polymer residue, one token per non-polymer (ligand) atom. A residue is identified by
(chain rank, residue offset within the chain), which is independent of how a model names its
chains or numbers its residues. Within a ligand residue, atoms are paired by position when both
models list the same elements in the same order, and by atom name otherwise. Models name the
atoms of a SMILES ligand differently (ESMFold2 does not use AF3's N1, C1, C2, ...) but keep the
SMILES atom order, and a CCD ligand whose atom set differs between models still pairs by name.

Token identity does not depend on MSA, templates or seed, so one prediction per model and
system is enough.
"""
import json
from collections import defaultdict

import numpy as np


def layout(cif_path):
    """[(chain, residue number, element, atom name)] in token order."""
    import gemmi
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    out = []
    for ch in st[0]:
        poly = ch.get_polymer()
        if poly.length() > 0:
            for res in poly:
                at = res[0] if len(res) else None
                out.append([ch.name, res.seqid.num, at.element.name if at else "", at.name if at else ""])
        else:
            for res in ch:
                for at in res:
                    out.append([ch.name, res.seqid.num, at.element.name, at.name])
    return out


def fingerprint(rows):
    """(chain rank, residue offset, atom name if the residue has several tokens) per token."""
    rank, first, count = {}, {}, defaultdict(int)
    for a, r, _e, _n in rows:
        rank.setdefault(a, len(rank))
        first[a] = min(first.get(a, r), r)
        count[(a, r)] += 1
    return [(rank[a], r - first[a], n if count[(a, r)] > 1 else None) for a, r, _e, n in rows]


def save_layout(cif_path, out_json):
    json.dump(layout(cif_path), open(out_json, "w"))


def _residues(rows, keys):
    """(chain rank, residue offset) -> token indices, for residues that span several tokens."""
    out = defaultdict(list)
    for i, k in enumerate(keys):
        if k[2] is not None:
            out[k[:2]].append(i)
    return out


def align(anchor_rows, peer_rows):
    """(idx_anchor, idx_peer): tokens present in both models, in corresponding order."""
    ka, kp = fingerprint(anchor_rows), fingerprint(peer_rows)
    single = {k: j for j, k in enumerate(kp) if k[2] is None}
    pairs = [(i, single[k]) for i, k in enumerate(ka) if k[2] is None and k in single]
    ra, rp = _residues(anchor_rows, ka), _residues(peer_rows, kp)
    for res, ia in ra.items():
        ip = rp.get(res)
        if not ip:
            continue
        if [anchor_rows[i][2] for i in ia] == [peer_rows[j][2] for j in ip]:
            pairs += zip(ia, ip)
        else:
            by_name = {peer_rows[j][3]: j for j in ip}
            pairs += [(i, by_name[anchor_rows[i][3]]) for i in ia if anchor_rows[i][3] in by_name]
    pairs.sort()
    ib = np.asarray([i for i, _ in pairs], dtype=np.int64)
    it = np.asarray([j for _, j in pairs], dtype=np.int64)
    return ib, it
