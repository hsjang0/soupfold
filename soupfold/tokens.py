"""Token alignment across co-folding models.

A model's token layout is read from any structure it predicted, in file order: one token
per polymer residue, one token per non-polymer (ligand) atom. A residue is identified by
(chain rank, residue offset within the chain), which is independent of how a model names its
chains or numbers its residues. Ligand atoms pair by name for CCD ligands and by position for
SMILES ligands (see _pair_ligand). Which ligands are SMILES is read from the input (smiles_chains).

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


HYDROGEN = ("H", "D")


def _pair_ligand(ra, rp, ia, ip, smiles):
    """Token pairs within one ligand residue. ra, rp: layout rows. ia, ip: the residue's token indices.

    A CCD ligand has the same atom names in every model and pairs by name. A SMILES ligand is named
    differently by each model (ESMFold2 does not use AF3's N1, C1, C2, ...), but every model keeps the
    RDKit MolFromSmiles atom order, so it pairs by position when the elements agree, with or without
    the hydrogens a model keeps (ESMFold2 keeps isotopic ones). A SMILES ligand whose heavy atoms still
    differ between the models cannot be aligned and raises an error."""
    heavy = lambda rows, idx: [i for i in idx if rows[i][2] not in HYDROGEN]
    ha, hp = heavy(ra, ia), heavy(rp, ip)
    if smiles and {ra[i][3] for i in ha} != {rp[j][3] for j in hp}:
        if [ra[i][2] for i in ia] == [rp[j][2] for j in ip]:
            return list(zip(ia, ip))                               # same elements in order: every atom
        if [ra[i][2] for i in ha] == [rp[j][2] for j in hp]:
            return list(zip(ha, hp))                               # same once hydrogens are dropped
        raise ValueError(f"SMILES ligand in chain {ra[ia[0]][0]}: the models disagree on its heavy atoms "
                         f"({len(ha)} vs {len(hp)}), so its tokens cannot be aligned")
    by_name = {rp[j][3]: j for j in ip}
    return [(i, by_name[ra[i][3]]) for i in ia if ra[i][3] in by_name]


def smiles_chains(record):
    """Chain ranks of the ligands a Protenix-format record gives as SMILES. Every model lays out the
    chains in entity order, the copies of an entity in a row."""
    out, rank = set(), 0
    for ent in record["sequences"]:
        (kind, v), = ent.items()
        n = int(v.get("count", 1))
        if kind == "ligand" and not str(v["ligand"]).startswith("CCD_"):
            out.update(range(rank, rank + n))
        rank += n
    return out


def align(anchor_rows, peer_rows, smiles=()):
    """(idx_anchor, idx_peer): tokens present in both models, in corresponding order.
    smiles: chain ranks of SMILES ligands (smiles_chains)."""
    ka, kp = fingerprint(anchor_rows), fingerprint(peer_rows)
    single = {k: j for j, k in enumerate(kp) if k[2] is None}
    pairs = [(i, single[k]) for i, k in enumerate(ka) if k[2] is None and k in single]
    ra, rp = _residues(anchor_rows, ka), _residues(peer_rows, kp)
    for res, ia in ra.items():
        ip = rp.get(res)
        if not ip:
            continue
        pairs += _pair_ligand(anchor_rows, peer_rows, ia, ip, res[0] in smiles)
    pairs.sort()
    ib = np.asarray([i for i, _ in pairs], dtype=np.int64)
    it = np.asarray([j for _, j in pairs], dtype=np.int64)
    return ib, it
