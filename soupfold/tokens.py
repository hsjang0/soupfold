"""Token alignment across co-folding models.

Every standalone run writes the token layout of its model next to the representation, from the
features the model was given: one row [chain, residue, element, atom name] per token, at the token's
centre atom (the atom itself for a token that is one atom). Row i describes z[i].

Two layouts are aligned at the granularity the models agree on:

  chains    paired in order: every model is given the chains in the input's entity order. Each pair
            is checked to have the same residues, and a mismatch raises an error. A chain the peer
            leaves out (absent_chains) has no counterpart.
  residues  one to one within a matched chain.
  atoms     a residue both models hold per atom (a ligand, a modified residue) pairs atom by atom
            (pair_residue). A residue only one model holds per atom pairs its single token with
            the centre atom of the other.

A token without a counterpart is left out. There the anchor keeps its own representation (mix.py).
"""
import json
import os

import numpy as np

ELEMENTS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
            "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
            "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
HYDROGEN = ("H", "D")
CENTRE = ("CA", "C1'", "P", "C1*")


def atom_name(chars):
    """Atom name from the ref_atom_name_chars feature of one atom: 4 characters as ascii - 32,
    given as indices or one-hot."""
    a = np.asarray(chars)
    a = a.argmax(-1) if a.ndim == 2 else a
    return "".join(chr(int(c) + 32) for c in a).strip()


def rows(chain, residue, element, name):
    """Layout rows from per-token values. element: atomic number or symbol."""
    sym = lambda e: str(e).capitalize() if isinstance(e, str) else ELEMENTS[int(e)]
    return [[int(c), int(r), sym(e), str(n)] for c, r, e, n in zip(chain, residue, element, name)]


def path(root, model, system):
    return os.path.join(root, model, f"{system}.json")


def save(root, model, system, layout, n_tokens):
    """n_tokens: size of the model's representation."""
    if len(layout) != n_tokens:
        raise ValueError(f"{system}: the {model} token layout has {len(layout)} tokens, "
                         f"its representation {n_tokens}")
    os.makedirs(os.path.join(root, model), exist_ok=True)
    json.dump(layout, open(path(root, model, system), "w"))


def load(root, model, system, n_tokens):
    """The layout the standalone run wrote. n_tokens: size of the model's representation. A layout
    of another length would pair the wrong tokens, so it is refused."""
    layout = json.load(open(path(root, model, system)))
    if len(layout) != n_tokens:
        raise ValueError(f"{system}: the {model} token layout has {len(layout)} tokens, its representation "
                         f"{n_tokens}. Run the standalone step of {model} again for this system")
    return layout


def _chains(layout):
    """[[(residue, [(atom name, element, token index), ...]), ...], ...]: chains in layout order,
    residues numbered from the first residue of the chain."""
    out, prev = [], object()
    for i, (chain, res, el, name) in enumerate(layout):
        if chain != prev:
            out.append([])
            prev = chain
        if not out[-1] or out[-1][-1][0] != res:
            out[-1].append((res, []))
        out[-1][-1][1].append((name, el, i))
    return [[(r - min(r for r, _ in ch), atoms) for r, atoms in ch] for ch in out]


def _residues(chain):
    """The residues of a chain, to check that two paired chains agree. How a residue is split into
    tokens is left out: only one model may hold a modified residue per atom."""
    return tuple(r for r, _ in chain)


def pair_residue(ga, gp, smiles, where=""):
    """Token pairs of one residue. ga, gp: [(atom name, element, token index)] in the anchor and
    the peer. smiles: the residue is a ligand the input gives as SMILES.

    A CCD component has the same atom names in every model and pairs by name. A SMILES ligand is
    named differently by each model (ESMFold2 does not use AF3's N1, C1, C2, ...), but every model
    keeps the RDKit MolFromSmiles atom order, so it pairs by position when the elements agree, with
    or without the hydrogens a model keeps (ESMFold2 keeps isotopic ones). A SMILES ligand whose
    heavy atoms still differ between the models cannot be aligned and raises an error."""
    if len(ga) == 1 and len(gp) == 1:
        return [(ga[0][2], gp[0][2])]
    if len(ga) > 1 and len(gp) > 1:
        ha = [x for x in ga if x[1] not in HYDROGEN]
        hp = [x for x in gp if x[1] not in HYDROGEN]
        if smiles and {x[0] for x in ha} != {x[0] for x in hp}:
            if [x[1] for x in ga] == [x[1] for x in gp]:
                return [(x[2], y[2]) for x, y in zip(ga, gp)]      # same elements in order: every atom
            if [x[1] for x in ha] == [x[1] for x in hp]:
                return [(x[2], y[2]) for x, y in zip(ha, hp)]      # same once hydrogens are dropped
            raise ValueError(f"SMILES ligand{where}: the models disagree on its heavy atoms "
                             f"({len(ha)} vs {len(hp)}), so its tokens cannot be aligned")
        by_name = {name: i for name, _, i in gp}
        if len(by_name) < len(gp) or len({x[0] for x in ga}) < len(ga):
            raise ValueError(f"residue{where}: atom names repeat, so its tokens cannot be paired by name")
        return [(i, by_name[name]) for name, _, i in ga if name in by_name]
    one, many = (ga, gp) if len(ga) == 1 else (gp, ga)               # one model holds it per atom
    names = [x[0] for x in many]
    centre = many[next((names.index(c) for c in CENTRE if c in names), 0)][2]
    return [(one[0][2], centre)] if len(ga) == 1 else [(centre, one[0][2])]


def smiles_chains(record):
    """Chain ranks of the ligands a Protenix-format record gives as SMILES. Every model is given
    the chains in entity order, the copies of an entity in a row."""
    out, rank = set(), 0
    for ent in record["sequences"]:
        (kind, v), = ent.items()
        n = int(v.get("count", 1))
        if kind == "ligand" and not str(v["ligand"]).startswith("CCD_"):
            out.update(range(rank, rank + n))
        rank += n
    return out


def absent_chains(record, model):
    """Chain ranks of a Protenix-format record that the model leaves out. AlphaFold3 drops a polymer
    chain made only of unknown residues (X in a protein, N in DNA or RNA)."""
    out, rank = set(), 0
    for ent in record["sequences"]:
        (kind, v), = ent.items()
        n = int(v.get("count", 1))
        unknown = {"proteinChain": "X", "dnaSequence": "N", "rnaSequence": "N"}.get(kind)
        if model == "af3" and unknown and set(v["sequence"]) == {unknown}:
            out.update(range(rank, rank + n))
        rank += n
    return out


def align(anchor, peer, smiles=(), absent=()):
    """(idx_anchor, idx_peer): tokens present in both models, in corresponding order.
    anchor, peer: layouts. smiles: chain ranks of SMILES ligands (smiles_chains). absent: chain ranks
    the peer leaves out (absent_chains)."""
    ca, cp = _chains(anchor), _chains(peer)
    pairs, j = [], 0
    for rank, chain in enumerate(ca):
        if rank in absent:
            continue
        if j == len(cp) or _residues(cp[j]) != _residues(chain):
            raise ValueError(f"chain {rank + 1} of the anchor and chain {j + 1} of the peer differ, "
                             f"so the chains cannot be paired")
        other, j = cp[j], j + 1
        for (_, ga), (_, gp) in zip(chain, other):
            pairs += pair_residue(ga, gp, rank in smiles, f" in chain {rank + 1}")
    if j < len(cp):
        raise ValueError(f"the peer has {len(cp) - j} chain(s) the anchor lacks, so the chains cannot be paired")
    pairs.sort()
    return (np.asarray([i for i, _ in pairs], dtype=np.int64),
            np.asarray([j for _, j in pairs], dtype=np.int64))
