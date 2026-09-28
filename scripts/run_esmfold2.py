"""ESMFold2: native folding (saving the trunk representation) and SoupFold folding.

  native    stochastic trunk seeded by --seed (ESMC-embedding dropout 0.3 and MSA column
            masking 0.1, the per-seed diversity used in the paper), z captured at the
            structure head and saved to <repr-root>/esmfold2/seed_<seed>/<system>.npz, then
            diffusion from that z.
  soupfold  no trunk: this model's stored z and every teacher's stored z for the same seed
            are mixed (soupfold.mix.soup) and diffusion runs from z'.

  python scripts/run_esmfold2.py --mode native   --input 8JT6.json --seed 1 --ckpt ... --esmc ...
  python scripts/run_esmfold2.py --mode soupfold --input 8JT6.json --seed 1 \
         --teachers protenix,opendde,af3 --layouts layouts/ --ckpt ... --esmc ...

Run inside an esm 3.4.0 environment (unmodified). Inputs are the same Protenix-format JSON
as for the other models. The MSA files named there are converted to ESMFold2's keyed a3m
(paired rows keyed by row index, unpaired rows key=-1).
"""
import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from soupfold import maps as M, mix, reprs, tokens  # noqa: E402
from soupfold.hooks import esmfold2 as H  # noqa: E402

from esm.models.esmfold2.config import EsmFold2Config  # noqa: E402
from esm.models.esmfold2.model import EsmFold2Model  # noqa: E402
from esm.models.esmfold2.processor import ESMFold2InputBuilder  # noqa: E402
from esm.models.esmfold2.types import (DNAInput, LigandInput, Modification, ProteinInput,  # noqa: E402
                                       RNAInput, StructurePredictionInput)
from esm.utils.msa import MSA  # noqa: E402

CHAINS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _a3m(path):
    hdr, buf = None, []
    for line in open(path):
        if line.startswith(">"):
            if hdr is not None:
                yield "".join(buf)
            hdr, buf = line, []
        else:
            buf.append(line.strip())
    if hdr is not None:
        yield "".join(buf)


def keyed_msa(paired, unpaired, out, cap=2048):
    """Paired rows keyed by row index (so rows pair across chains), unpaired rows key=-1."""
    query = None
    with open(out, "w") as f:
        for i, seq in enumerate(_a3m(paired) if paired and os.path.exists(paired) else []):
            if i == 0:
                f.write(">query key=-1\n" + seq + "\n"); query = seq
            elif i <= cap:
                f.write(f">p{i} key={i}\n{seq}\n")
        for i, seq in enumerate(_a3m(unpaired) if unpaired and os.path.exists(unpaired) else []):
            if i == 0:
                if query is None:
                    f.write(">query key=-1\n" + seq + "\n"); query = seq
            elif i <= cap:
                f.write(f">u{i} key=-1\n{seq}\n")
    return MSA.from_a3m(out, max_sequences=2 * cap + 1) if query else None


def to_esm_input(record, workdir):
    """Protenix-format record -> StructurePredictionInput (chains A, B, ... in entity order)."""
    names, seqs, ci = iter(CHAINS), [], 0
    for ent in record["sequences"]:
        (kind, v), = ent.items()
        ids = [next(names) for _ in range(int(v.get("count", 1)))]
        if kind == "proteinChain":
            ci += 1
            msa = keyed_msa(v.get("pairedMsaPath"), v.get("unpairedMsaPath"), os.path.join(workdir, f"chain{ci}.a3m"))
            mods = [Modification(position=m["ptmPosition"] - 1, ccd=m["ptmType"].removeprefix("CCD_"))
                    for m in v.get("modifications") or []]
            seqs.append(ProteinInput(id=ids, sequence=v["sequence"], modifications=mods or None, msa=msa))
        elif kind == "dnaSequence":
            seqs.append(DNAInput(id=ids, sequence=v["sequence"]))
        elif kind == "rnaSequence":
            seqs.append(RNAInput(id=ids, sequence=v["sequence"], msa=None))
        elif kind in ("ligand", "ion"):
            lig = v.get("ligand") or v.get("ion")
            if lig.startswith("CCD_"):
                seqs.append(LigandInput(id=ids, ccd=lig[4:].split("_")))
            else:
                seqs.append(LigandInput(id=ids, smiles=lig))
        else:
            raise ValueError(f"unhandled entity {kind!r}")
    return StructurePredictionInput(sequences=seqs)


def load_model(ckpt, esmc, device):
    cfg = EsmFold2Config.from_pretrained(ckpt)
    cfg.esmc_id = esmc
    cfg.lm_encoder.lm_dropout = cfg.lm_dropout = 0.0
    return EsmFold2Model.from_pretrained(ckpt, load_esmc=True, config=cfg, device=device, dtype=torch.float32).eval()


def set_lm_dropout(model, p):
    model.config.lm_encoder.lm_dropout = p
    model.config.lm_dropout = p


def write(out, sid, results):
    detail = []
    for i, r in enumerate(results):
        open(os.path.join(out, f"{sid}__s{i}.cif"), "w").write(r.complex.to_mmcif())
        detail.append({k: float(getattr(r, k)) for k in ("iptm", "ptm") if getattr(r, k, None) is not None})
    json.dump({"conf": [d.get("iptm", 0.0) for d in detail], "metric": "iptm", "detail": detail},
              open(os.path.join(out, f"{sid}__conf.json"), "w"))
    return [d.get("iptm") for d in detail]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "soupfold"], required=True)
    ap.add_argument("--input", nargs="+", required=True, help="Protenix-format system JSON(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repr-root", required=True)
    ap.add_argument("--ckpt", required=True, help="ESMFold2 checkpoint directory")
    ap.add_argument("--esmc", required=True, help="ESMC-6B checkpoint directory")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=None, help="default 5 (soupfold), 1 (native)")
    ap.add_argument("--recycling", type=int, default=10)
    ap.add_argument("--sampling-steps", type=int, default=None, help="default 68 (soupfold), 2 (native)")
    ap.add_argument("--lm-dropout", type=float, default=0.3, help="native: trunk ESMC-embedding dropout")
    ap.add_argument("--msa-column-mask-rate", type=float, default=0.1, help="native: trunk MSA column masking")
    ap.add_argument("--teachers", default="")
    ap.add_argument("--layouts", default=None)
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
    a = ap.parse_args()
    # native runs exist to store the trunk representation. Their samples are not used, so by default they get
    # one sample and 2 diffusion steps (noise). Pass the paper values to get real native structures.
    a.samples = a.samples or (1 if a.mode == "native" else 5)
    a.sampling_steps = a.sampling_steps or (2 if a.mode == "native" else 68)
    device = "cuda"
    os.makedirs(a.out, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    model = load_model(a.ckpt, a.esmc, device)
    builder = ESMFold2InputBuilder(ccd_cache=a.ckpt)
    common = dict(num_loops=a.recycling, num_sampling_steps=a.sampling_steps)
    for jpath in a.input:
        rec = json.load(open(jpath))
        rec = rec[0] if isinstance(rec, list) else rec
        sid = rec["name"]
        with tempfile.TemporaryDirectory() as wd:
            inp = to_esm_input(rec, wd)
            feats, chains = builder.prepare_input(inp, seed=0 if a.mode == "native" else a.seed, device=device)
        if a.mode == "native":
            box = {}
            set_lm_dropout(model, a.lm_dropout)
            model.set_chunk_size(64)
            torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
            with torch.no_grad(), H.capture_trunk(model, box, stop=True):
                try:
                    model(**feats, **common, num_diffusion_samples=1, msa_column_mask_rate=a.msa_column_mask_rate)
                except H.StopAfterTrunk:
                    pass
            set_lm_dropout(model, 0.0)
            z = box["z"].float()
            z = z if z.dim() == 4 else z[None]
            reprs.save(a.repr_root, "esmfold2", a.seed, sid, z[0], lm_dropout=a.lm_dropout,
                       msa_column_mask_rate=a.msa_column_mask_rate, recycling=a.recycling)
        else:
            z0, _ = reprs.load(a.repr_root, "esmfold2", a.seed, sid)
            z0 = torch.from_numpy(z0).to(device)
            teach = [t for t in a.teachers.split(",") if t]
            if teach:
                stats = json.load(open(os.path.join(a.weights, "chan_stats.json")))
                zt = {t: torch.from_numpy(reprs.load(a.repr_root, t, a.seed, sid)[0]) for t in teach}
                fmap = {t: M.load_map(a.weights, t, "esmfold2", device) for t in teach}
                tmaps = None
                if a.layouts:
                    lay = lambda m: json.load(open(os.path.join(a.layouts, m, f"{sid}.json")))
                    tmaps = {t: tokens.align(lay("esmfold2"), lay(t)) for t in teach}
                z0 = mix.soup(z0, zt, fmap, stats, "esmfold2", token_maps=tmaps, device=device)
            z = z0[None]
        with torch.no_grad(), H.trunk_bypass(model, z), H.seed_diffusion(model, a.seed):
            out = model(**feats, **common, num_diffusion_samples=a.samples)
        res = builder.decode(out, feats, chains, num_diffusion_samples=a.samples, complex_id=sid)
        conf = write(a.out, sid, res if isinstance(res, list) else [res])
        print(f"{sid} [esmfold2 {a.mode} seed {a.seed}] iptm {[round(c, 4) for c in conf]}", flush=True)
        out = res = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
