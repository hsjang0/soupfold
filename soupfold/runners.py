"""What the pack needs from each model's own inference path: Protenix and OpenDDE runners, ESMFold2 input."""
import copy
import glob
import json
import os
import re
import shutil

import torch

from soupfold.models.esm.models.esmfold2.types import (DNAInput, LigandInput, Modification, ProteinInput,
                                                       RNAInput, StructurePredictionInput)
from soupfold.models.esm.utils.msa import MSA


def build_runner(a):
    if a.model == "protenix":
        from soupfold.models.protenix.utils.seed import seed_everything
        from soupfold.models.protenix.utils.torch_utils import to_device
        from soupfold.models.protenix.data.inference.infer_dataloader import get_inference_dataloader
        from soupfold.models.protenix_runner.inference import update_inference_configs
        from soupfold.models.protenix_runner.batch_inference import get_default_runner
        kw = dict(model_name="protenix_base_default_v1.0.0", use_seeds_in_json=False)
    else:
        from soupfold.models.opendde.utils.seed import seed_everything
        from soupfold.models.opendde.utils.torch_utils import to_device
        from soupfold.models.opendde.data.inference.infer_dataloader import get_inference_dataloader
        from soupfold.models.opendde_runner.inference import update_inference_configs
        from soupfold.models.opendde_runner.batch_inference import get_default_runner
        kw = dict(model_name="opendde_v1", load_checkpoint_path=a.checkpoint)
    runner = get_default_runner(
        seeds=[a.seed], n_cycle=a.recycling, n_step=a.sampling_steps, n_sample=a.samples, dtype="bf16",
        use_msa=True, use_template=True, use_rna_msa=False, need_atom_confidence=False,
        trimul_kernel="torch", triatt_kernel="torch", **kw)
    c = runner.configs
    c.msa_pair_as_unpair = True
    if a.model == "protenix":
        c.mc_dropout_apply_rate = 0.0          # deterministic trunk (Protenix draws MC dropout by default)
    t = c.data.template
    t.prot_template_mmcif_dir, t.prot_template_cache_dir = a.template_mmcif_dir, None
    t.release_dates_path, t.obsolete_pdbs_path = a.release_dates, a.obsolete_pdbs
    t.fetch_remote = False
    if a.model == "opendde" and a.chunk_thresholds:   # smaller trunk chunks for GPUs under ~180 GB, same numerics
        c.infer_setting.chunk_size_thresholds = json.loads(a.chunk_thresholds)
    c.dump_dir = a.out
    runner.dumper.base_dir = a.out
    return runner, seed_everything, to_device, get_inference_dataloader, update_inference_configs


def predict(runner, data, to_device, model):
    if model == "opendde":                    # OpenDDE's runner handles device, precision and copies itself
        return runner.predict(data)
    data = to_device(copy.deepcopy(data), runner.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred, _, _ = runner.model(input_feature_dict=data["input_feature_dict"], label_full_dict=None,
                                  label_dict=None, mode="inference", mc_dropout_apply_rate=0.0)
    return pred


def flatten(out, sid, seed):
    """<out>/<sid>__s<i>.cif + <sid>__conf.json (per-sample ranking_score, iptm, ptm)."""
    d = os.path.join(out, sid, f"seed_{seed}", "predictions")
    cifs = sorted(glob.glob(os.path.join(d, f"{sid}_sample_*.cif")), key=lambda c: int(c.rsplit("_", 1)[1][:-4]))
    detail = []
    for k, c in enumerate(cifs):
        i = c.rsplit("_", 1)[1][:-4]
        sc = json.load(open(os.path.join(d, f"{sid}_summary_confidence_sample_{i}.json")))
        detail.append({key: float(sc[key]) for key in ("ranking_score", "iptm", "ptm") if key in sc})
        shutil.copyfile(c, os.path.join(out, f"{sid}__s{k}.cif"))
    shutil.rmtree(os.path.join(out, sid))
    json.dump({"conf": [x["ranking_score"] for x in detail], "metric": "ranking_score", "detail": detail},
              open(os.path.join(out, f"{sid}__conf.json"), "w"))
    return [x["ranking_score"] for x in detail]


def chain_ids():
    """A, B, ..., Z, AA, AB, ..."""
    n = 0
    while True:
        s, k = "", n
        while k >= 0:
            s, k = chr(65 + k % 26) + s, k // 26 - 1
        yield s
        n += 1


_TAXID = re.compile(r"\bTaxID=(-?\d+)|^>UniRef100_[^_\s]+_(\d+)")


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


def _taxonomy_msa(unpaired, out):
    """The a3m as ESMFold2 reads it if its headers name the taxonomy (key=<id>, TaxID=<id> or UniRef100_<acc>_<id>), else None."""
    lines = open(unpaired).readlines()
    headers = [line for line in lines if line.startswith(">")]
    if any(re.search(r"\bkey=-?\d+", h) for h in headers):
        return MSA.from_a3m(unpaired)
    if not any(_TAXID.search(h) for h in headers):
        return None
    with open(out, "w") as f:
        for line in lines:
            if line.startswith(">"):
                m = _TAXID.search(line)
                line = f"{line.rstrip()} key={(m.group(1) or m.group(2)) if m else -1}\n"
            f.write(line)
    return MSA.from_a3m(out)


def keyed_msa(paired, unpaired, out, cap=2048):
    """Taxonomy in the headers of a lone a3m pairs rows across chains. Otherwise paired rows are keyed by row index (so rows
    pair across chains) and unpaired rows key=-1."""
    if unpaired and os.path.exists(unpaired) and (not paired or os.path.realpath(paired) == os.path.realpath(unpaired)):
        msa = _taxonomy_msa(unpaired, out)
        if msa is not None:
            return msa
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


def set_lm_dropout(model, p):
    model.config.lm_encoder.lm_dropout = p
    model.config.lm_dropout = p
