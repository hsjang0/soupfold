"""AlphaFold3 (teacher only): native run that stores the trunk representation.

Builds the AF3 input from the same Protenix-format JSON: unpaired MSA = paired rows prepended to the
unpaired a3m (deduplicated), templates = AF3's parse of the hmmsearch a3m against the local mmCIF
store (cutoff 2021-09-30, at most 4). Stores z = pair_embeddings, s = single_embeddings in
<workdir>/reprs/af3/seed_<k>/.

  python scripts/run_af3.py --input examples/9y0a/9y0a.json --model-dir <af3> --template-mmcif-dir <mmcif>

Run with AlphaFold3 v3.0.1 on PYTHONPATH (repo root and src/), with the atom_layout patch applied.
"""
import datetime
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from soupfold import cli, inputs, reprs  # noqa: E402

import jax  # noqa: E402
from absl import flags  # noqa: E402
from alphafold3.common import folding_input  # noqa: E402
from alphafold3.constants import residue_names  # noqa: E402
from alphafold3.data import msa_config, structure_stores, templates as af3_templates  # noqa: E402
import run_alphafold as ra  # noqa: E402

if not flags.FLAGS.is_parsed():          # AF3 is driven as a library: mark absl flags parsed
    flags.FLAGS(sys.argv[:1])


def parse_a3m(text):
    out, hdr, buf = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if hdr is not None:
                out.append((hdr, "".join(buf)))
            hdr, buf = line, []
        elif hdr is not None:
            buf.append(line.strip())
    if hdr is not None:
        out.append((hdr, "".join(buf)))
    return out


def pair_as_unpair(paired, unpaired):
    if not paired:
        return unpaired or ""
    seen, keep = set(), []
    for h, s in parse_a3m(paired) + parse_a3m(unpaired or ""):
        if s not in seen:
            seen.add(s); keep.append(f"{h}\n{s}\n")
    return "".join(keep)


def fix_query_row(a3m, query):
    """AF3 checks the first a3m row against its query, in which modified residues are
    already mapped (e.g. PCA -> E, unknown -> X). Only that row is rewritten."""
    if not a3m:
        return a3m
    lines, i = a3m.split("\n"), 0
    while not lines[i].strip() or lines[i].startswith("#"):
        i += 1
    i += lines[i].startswith(">")
    lines[i] = query
    return "\n".join(lines)


def chain_ids():
    i = 0
    while True:
        s, k = "", i
        while True:
            s = chr(65 + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        yield s
        i += 1


def read(path):
    return open(path).read() if path and os.path.exists(path) else ""


def af3_record(rec, seed, store, cutoff):
    filt = msa_config.TemplateFilterConfig(max_subsequence_ratio=0.95, min_align_ratio=0.1, min_hit_length=10,
                                           deduplicate_sequences=True, max_hits=4, max_template_date=cutoff)
    ids, seqs = chain_ids(), []
    for ent in rec["sequences"]:
        (kind, v), = ent.items()
        cid = [next(ids) for _ in range(int(v.get("count", 1)))]
        cid = cid if len(cid) > 1 else cid[0]
        if kind == "proteinChain":
            mods = [{"ptmType": m["ptmType"].removeprefix("CCD_"), "ptmPosition": m["ptmPosition"]}
                    for m in v.get("modifications") or []]
            q = list(v["sequence"])
            for m in mods:
                q[m["ptmPosition"] - 1] = residue_names.letters_three_to_one(m["ptmType"], default="X")
            q = "".join(q)
            paired, unpaired = read(v.get("pairedMsaPath")), read(v.get("unpairedMsaPath"))
            if q != v["sequence"]:
                paired, unpaired = fix_query_row(paired, q), fix_query_row(unpaired, q)
            tt = af3_templates.Templates.from_hmmsearch_a3m(
                query_sequence=v["sequence"], a3m=read(v.get("templatesPath")), max_template_date=cutoff,
                structure_store=store, filter_config=filt)
            templates = [{"mmcif": st.to_mmcif(), "queryIndices": list(h.query_to_hit_mapping.keys()),
                          "templateIndices": list(h.query_to_hit_mapping.values())}
                         for h, st in tt.get_hits_with_structures()]
            seqs.append({"protein": {"id": cid, "sequence": v["sequence"], "modifications": mods,
                                     "unpairedMsa": pair_as_unpair(paired, unpaired), "pairedMsa": paired,
                                     "templates": templates}})
        elif kind in ("dnaSequence", "rnaSequence"):
            body = {"id": cid, "sequence": v["sequence"]}
            if kind == "rnaSequence":
                body["unpairedMsa"] = ""
            seqs.append({"dna" if kind == "dnaSequence" else "rna": body})
        elif kind in ("ligand", "ion"):
            lig = v.get("ligand") or v.get("ion")
            body = {"id": cid, **({"ccdCodes": lig[4:].split("_")} if lig.startswith("CCD_") else {"smiles": lig})}
            seqs.append({"ligand": body})
        else:
            raise ValueError(f"unhandled entity {kind!r}")
    return {"name": rec["name"], "modelSeeds": [seed], "dialect": "alphafold3", "version": 1, "sequences": seqs}


def main():
    ap = cli.parser("af3", soupfold=False)
    ap.add_argument("--model-dir", required=True, help="AF3 parameter directory")
    ap.add_argument("--template-mmcif-dir", required=True)
    ap.add_argument("--template-cutoff", default="2021-09-30")
    a = cli.finish(ap.parse_args(), "af3", steps=200)
    config = ra.make_model_config(num_recycles=a.recycling, return_embeddings=True, num_diffusion_samples=a.samples)
    config.heads.diffusion.eval.steps = a.sampling_steps
    runner = ra.ModelRunner(config=config, device=jax.local_devices()[0], model_dir=pathlib.Path(a.model_dir))
    store = structure_stores.StructureStore(a.template_mmcif_dir)
    cutoff = datetime.date.fromisoformat(a.template_cutoff)
    for jpath in a.input:
        rec = inputs.load(jpath)
        sid = rec["name"]
        fi = folding_input.Input.from_json(json.dumps(af3_record(rec, a.seed, store, cutoff)))
        res = ra.predict_structure(fi, runner, ref_max_modified_date=datetime.date(3000, 1, 1))
        emb = res[0].embeddings
        reprs.save(a.repr_root, "af3", a.seed, sid, np.asarray(emb["pair_embeddings"], np.float32),
                   np.asarray(emb["single_embeddings"], np.float32), recycling=a.recycling)
        conf = []
        for i, r in enumerate(res[0].inference_results):
            open(os.path.join(a.out, f"{sid}__s{i}.cif"), "w").write(r.predicted_structure.to_mmcif())
            conf.append(float(r.metadata["ranking_score"]))
        json.dump({"conf": conf, "metric": "ranking_score"}, open(os.path.join(a.out, f"{sid}__conf.json"), "w"))
        print(f"{sid} [af3 native seed {a.seed}] ranking_score {[round(c, 4) for c in conf]} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
