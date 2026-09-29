"""Token alignment across co-folding models.

A model's token layout is read from any structure it predicted, in file order: one token per
standard polymer residue, one token per atom of everything else (ligands, ions, and the modified
residues inside a polymer chain, which every model tokenises per atom). A residue is identified by
(chain rank, residue index within the chain), which is independent of how a model names its
chains or numbers its residues. Ligand atoms pair by name for CCD ligands and by position for
SMILES ligands (see _pair_ligand). Which ligands are SMILES is read from the input (smiles_chains).

Token identity does not depend on MSA, templates or seed, so one prediction per model and
system is enough. A layout must have as many tokens as the model's representation (see load).
"""
import json
import os
from collections import defaultdict

import numpy as np


# Residues a model holds as one token when they are part of a polymer chain.
STANDARD = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK "
                     "A C G U N DA DC DG DT DN".split())


def layout(cif_path):
    """[(chain, residue index, element, atom name)] in token order.

    A residue is one token when it has a standard name and is written as ATOM. Anything else is one
    token per atom: a ligand or ion, a modified residue (HYP, SEP, ACE, ...), and an amino acid or
    nucleotide given as a ligand, which the models write as HETATM."""
    import gemmi
    out = []
    for ch in gemmi.read_structure(cif_path)[0]:
        k = 0
        for res in ch:
            if res.name in STANDARD and res.het_flag != "H":
                out.append([ch.name, k, res[0].element.name, res[0].name])
            else:
                seen = set()             # ESMFold2 writes a ligand of several components as one
                for at in res:           # residue: a repeated atom name starts the next component
                    if at.name in seen:
                        k, seen = k + 1, set()
                    seen.add(at.name)
                    out.append([ch.name, k, at.element.name, at.name])
            k += 1
    return out


def fingerprint(rows):
    """(chain rank, residue index, atom name if the residue has several tokens) per token."""
    rank, first, count = {}, {}, defaultdict(int)
    for a, r, _e, _n in rows:
        rank.setdefault(a, len(rank))
        first[a] = min(first.get(a, r), r)
        count[(a, r)] += 1
    return [(rank[a], r - first[a], n if count[(a, r)] > 1 else None) for a, r, _e, n in rows]


def save_layout(cif_path, out_json):
    json.dump(layout(cif_path), open(out_json, "w"))


def load(root, model, system, n_tokens):
    """The layout token_layout.py wrote for one model and system. n_tokens: size of the model's
    representation. A layout of another length would pair the wrong tokens, so it is refused."""
    rows = json.load(open(os.path.join(root, model, f"{system}.json")))
    if len(rows) != n_tokens:
        raise ValueError(f"{system}: the {model} token layout has {len(rows)} tokens, its representation "
                         f"{n_tokens}. Write the layout again with scripts/token_layout.py; if the two "
                         f"still differ, the tokens of this system cannot be aligned")
    return rows


def _residues(rows, keys):
    """(chain rank, residue index) -> token indices, for residues that span several tokens."""
    out = defaultdict(list)
    for i, k in enumerate(keys):
        if k[2] is not None:
            out[k[:2]].append(i)
    return out


HYDROGEN = ("H", "D")


def _pair_ligand(ra, rp, ia, ip, smiles):
    """Token pairs within one residue held per atom (a ligand, a modified residue). ra, rp: layout rows. ia, ip: the residue's token indices.

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
    if len(by_name) < len(ip) or len({ra[i][3] for i in ia}) < len(ia):
        raise ValueError(f"chain {ra[ia[0]][0]}: atom names repeat within a residue, so its tokens cannot "
                         f"be paired by name; write the layouts again with scripts/token_layout.py")
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
