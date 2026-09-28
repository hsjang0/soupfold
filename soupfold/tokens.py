"""Token alignment across co-folding models.

A model's token layout is read from any structure it predicted, in file order: one token
per polymer residue, one token per non-polymer (ligand) atom. Each token gets the
fingerprint (chain rank, residue offset within the chain, atom name if the residue spans
several tokens), which is independent of how a model names its chains or numbers its
residues. Tokens with the same fingerprint in two models are the same token.

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


def align(base_rows, teacher_rows):
    """(idx_base, idx_teacher): tokens present in both models, in corresponding order."""
    kb, kt = fingerprint(base_rows), fingerprint(teacher_rows)
    tpos = {k: j for j, k in enumerate(kt)}
    pairs = [(i, tpos[k]) for i, k in enumerate(kb) if k in tpos]
    ib = np.asarray([i for i, _ in pairs], dtype=np.int64)
    it = np.asarray([j for _, j in pairs], dtype=np.int64)
    return ib, it
