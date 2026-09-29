"""ESMFold2: standalone run (stores the trunk representation) or SoupFold run.

  standalone  trunk seeded by --seed (ESMC-embedding dropout 0.3, MSA column masking 0.1 give per-seed
              diversity), z stored in <workdir>/reprs/esmfold2/seed_<k>/, then diffusion from that z.
  soupfold    no trunk. Mixes this model's stored z with the peers' stored z (same seed), then runs
              diffusion from z'.

  python scripts/run_esmfold2.py --mode standalone --input examples/9mnb/9mnb.json --ckpt <esmfold2> --esmc <esmc>
  python scripts/run_esmfold2.py --mode soupfold   --input examples/9mnb/9mnb.json --ckpt <esmfold2> --esmc <esmc>

Run in an esm 3.4.0 environment (unmodified).
"""
import gc
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from soupfold import cli, inputs, maps as M, mix, reprs, tokens  # noqa: E402
from soupfold.hooks import esmfold2 as H  # noqa: E402

from esm.models.esmfold2.config import EsmFold2Config  # noqa: E402
from esm.models.esmfold2.model import EsmFold2Model  # noqa: E402
from esm.models.esmfold2.processor import ESMFold2InputBuilder  # noqa: E402
from esm.models.esmfold2.types import (DNAInput, LigandInput, Modification, ProteinInput,  # noqa: E402
                                       RNAInput, StructurePredictionInput)
from esm.utils.msa import MSA  # noqa: E402

def chain_ids():
    """A, B, ..., Z, AA, AB, ..."""
    n = 0
    while True:
        s, k = "", n
        while k >= 0:
            s, k = chr(65 + k % 26) + s, k // 26 - 1
        yield s
        n += 1


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
    names, seqs, ci = chain_ids(), [], 0
    for ent in record["sequences"]:
        (kind, v), = ent.items()
        ids = [next(names) for _ in range(int(v.get("count", 1)))]
        if kind == "proteinChain":
            ci += 1
            msa = keyed_msa(v.get("pairedMsaPath"), v.get("unpairedMsaPath"), os.path.join(workdir, f"chain{ci}.a3m"))
            mods = [Modification(position=m["ptmPosition"] - 1, ccd=m["ptmType"].removeprefix("CCD_"))
                    for m in v.get("modifications") or []]
            seqs.append(ProteinInput(id=ids, sequence=v["sequence"], modifications=mods or None, msa=msa))
        elif kind in ("dnaSequence", "rnaSequence"):
            mods = [Modification(position=m["basePosition"] - 1, ccd=m["modificationType"].removeprefix("CCD_"))
                    for m in v.get("modifications") or []]
            if kind == "dnaSequence":
                seqs.append(DNAInput(id=ids, sequence=v["sequence"], modifications=mods or None))
            else:
                seqs.append(RNAInput(id=ids, sequence=v["sequence"], modifications=mods or None, msa=None))
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
    ap = cli.parser("esmfold2")
    ap.add_argument("--ckpt", required=True, help="ESMFold2 checkpoint directory")
    ap.add_argument("--esmc", required=True, help="ESMC-6B checkpoint directory")
    ap.add_argument("--lm-dropout", type=float, default=0.3, help="standalone: trunk ESMC-embedding dropout")
    ap.add_argument("--msa-column-mask-rate", type=float, default=0.1, help="standalone: trunk MSA column masking")
    a = cli.finish(ap.parse_args(), "esmfold2", steps=68)
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    model = load_model(a.ckpt, a.esmc, device)
    builder = ESMFold2InputBuilder(ccd_cache=a.ckpt)
    common = dict(num_loops=a.recycling, num_sampling_steps=a.sampling_steps)
    for jpath in a.input:
        rec = inputs.load(jpath)
        sid = rec["name"]
        with tempfile.TemporaryDirectory() as wd:
            feats, chains = builder.prepare_input(to_esm_input(rec, wd), seed=0 if a.mode == "standalone" else a.seed,
                                                  device=device)
        if a.mode == "standalone":
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
            tokens.save(a.layouts, "esmfold2", sid, H.token_layout(feats, chains), z.shape[1])
        else:
            z0 = torch.from_numpy(reprs.load(a.repr_root, "esmfold2", a.seed, sid)[0]).to(device)
            stats = json.load(open(os.path.join(a.weights, "chan_stats.json")))
            zt = {t: torch.from_numpy(reprs.load(a.repr_root, t, a.seed, sid)[0]) for t in a.peers}
            fmap = {t: M.load_map(a.weights, t, "esmfold2", device) for t in a.peers}
            lay = {m: tokens.load(a.layouts, m, sid, x.shape[0]) for m, x in {"esmfold2": z0, **zt}.items()}
            tmaps = {t: tokens.align(lay["esmfold2"], lay[t], tokens.smiles_chains(rec), tokens.absent_chains(rec, t)) for t in a.peers}
            z = mix.soup(z0, zt, fmap, stats, "esmfold2", token_maps=tmaps, device=device)[None]
            if "af3" in a.peers:
                reprs.copy_terms(a.repr_root, a.seed, a.out)
        # A large system can OOM with all samples in one batch. The batch is then halved and the
        # samples are collected over passes, pass i seeded with seed + i.
        res, cur, si = [], a.samples, 0
        while len(res) < a.samples:
            b = min(cur, a.samples - len(res))
            try:
                with torch.no_grad(), H.trunk_bypass(model, z), H.seed_diffusion(model, a.seed + si):
                    out = model(**feats, **common, num_diffusion_samples=b)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # ROCm reports a diffusion batch that does not fit as
                # "HIP error: invalid configuration argument", not as OutOfMemoryError.
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and \
                        "invalid configuration" not in str(exc).lower():
                    raise
                out = None
                gc.collect(); torch.cuda.empty_cache()
                if cur == 1:
                    raise
                cur //= 2
                print(f"{sid} OOM, batch -> {cur}", flush=True)
                continue
            r = builder.decode(out, feats, chains, num_diffusion_samples=b, complex_id=sid)
            res += r if isinstance(r, list) else [r]
            si += 1
            out = None
            gc.collect(); torch.cuda.empty_cache()
        conf = write(a.out, sid, res)
        print(f"{sid} [esmfold2 {a.mode} seed {a.seed}] iptm {[round(c, 4) for c in conf]} -> {a.out}", flush=True)
        res = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
