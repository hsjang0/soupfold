"""Protenix: native folding (saving the trunk representation) and SoupFold folding.
OpenDDE runs through this same code via scripts/run_opendde.py.

  native    full model (MSA + templates -> trunk -> diffusion). Writes the samples and the
            trunk representation (s, z) to <repr-root>/<model>/seed_<seed>/<system>.npz.
  soupfold  no trunk: loads this model's stored (s, z) and every teacher's stored z for
            the same seed, mixes z (soupfold.mix.soup), and runs diffusion + confidence
            from (s, z'). The base model's own s is used unchanged.

  python scripts/run_protenix.py --mode native   --input 8JT6.json --seed 1 ...
  python scripts/run_protenix.py --mode soupfold --input 8JT6.json --seed 1 \
         --teachers esmfold2,opendde,af3 --layouts layouts/ ...

Run inside the model's own environment (Protenix 2.0.0 or OpenDDE 1.1.0, with the
parser patch in patches/ applied). Inputs are Protenix-format JSON (a list holding one
record, with MSA and template paths), see the README.
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from soupfold import maps as M, mix, reprs, tokens  # noqa: E402
import importlib  # noqa: E402


def build_runner(a):
    if a.model == "protenix":
        from protenix.utils.seed import seed_everything
        from protenix.utils.torch_utils import to_device
        from protenix.data.inference.infer_dataloader import get_inference_dataloader
        from runner.inference import update_inference_configs
        from runner.batch_inference import get_default_runner
        kw = dict(model_name="protenix_base_default_v1.0.0", kalign_binary_path=a.kalign, use_seeds_in_json=False)
    else:
        from opendde.utils.seed import seed_everything
        from opendde.utils.torch_utils import to_device
        from opendde.data.inference.infer_dataloader import get_inference_dataloader
        from runner.inference import update_inference_configs
        from runner.batch_inference import get_default_runner
        os.environ["PATH"] = os.path.dirname(a.kalign) + ":" + os.environ.get("PATH", "")
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
    if a.model == "opendde":
        t.kalign_binary_path = a.kalign
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
    json.dump({"conf": [x["ranking_score"] for x in detail], "metric": "ranking_score", "detail": detail},
              open(os.path.join(out, f"{sid}__conf.json"), "w"))
    return [x["ranking_score"] for x in detail]


def soup_inputs(a, sid, device):
    z, s = reprs.load(a.repr_root, a.model, a.seed, sid)
    s = torch.from_numpy(s).to(device)
    z = torch.from_numpy(z).to(device)
    if not a.teachers:                        # fold from the stored representation alone
        return s, z
    teach = [t for t in a.teachers.split(",") if t]
    stats = json.load(open(os.path.join(a.weights, "chan_stats.json")))
    zt = {t: torch.from_numpy(reprs.load(a.repr_root, t, a.seed, sid)[0]) for t in teach}
    fmap = {t: M.load_map(a.weights, t, a.model, device) for t in teach}
    tmaps = None
    if a.layouts:
        lay = lambda m: json.load(open(os.path.join(a.layouts, m, f"{sid}.json")))
        tmaps = {t: tokens.align(lay(a.model), lay(t)) for t in teach}
    return s, mix.soup(z, zt, fmap, stats, a.model, token_maps=tmaps, device=device)


def main(model="protenix"):
    global H
    H = importlib.import_module(f"soupfold.hooks.{model}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "soupfold"], required=True)
    ap.add_argument("--input", nargs="+", required=True, help="Protenix-format system JSON(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repr-root", required=True, help="representation store (written by native, read by soupfold)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=None, help="default 5 (soupfold), 1 (native)")
    ap.add_argument("--recycling", type=int, default=10)
    ap.add_argument("--sampling-steps", type=int, default=None, help="default 200 (soupfold), 2 (native)")
    ap.add_argument("--teachers", default="", help="soupfold: comma list, e.g. esmfold2,opendde,af3")
    ap.add_argument("--layouts", default=None, help="token layouts (scripts/token_layout.py); omit only if all "
                                                    "models tokenise identically")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
    if model == "opendde":
        ap.add_argument("--checkpoint", required=True, help="OpenDDE checkpoint (opendde.pt)")
    ap.add_argument("--template-mmcif-dir", required=True)
    ap.add_argument("--release-dates", required=True, help="release_date_cache.json (Protenix common data)")
    ap.add_argument("--obsolete-pdbs", required=True, help="obsolete_to_successor.json (Protenix common data)")
    ap.add_argument("--kalign", required=True, help="path to the kalign binary")
    a = ap.parse_args()
    a.model = model
    # native runs exist to store the trunk representation. Their samples are not used, so by default they get
    # one sample and 2 diffusion steps (noise). Pass the paper values to get real native structures.
    a.samples = a.samples or (1 if a.mode == "native" else 5)
    a.sampling_steps = a.sampling_steps or (2 if a.mode == "native" else 200)
    # kalign (template featurisation) reads a non-terminal stdin and blocks forever on a pipe or socket
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.makedirs(a.out, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    runner, seed_everything, to_device, get_loader, update_cfg = build_runner(a)
    model, base_cfg = runner.model, copy.deepcopy(runner.configs)
    for jpath in a.input:
        runner.configs.input_json_path = jpath
        for batch in get_loader(configs=runner.configs):
            data, atom_array, err = batch[0]
            if err:
                raise RuntimeError(f"{jpath}: {err}")
            sid = data["sample_name"]
            runner.update_model_configs(update_cfg(base_cfg, data["N_token"].item()))
            seed_everything(seed=a.seed, deterministic=False)
            if a.mode == "native":
                box = {}
                with torch.no_grad(), H.capture_trunk(model, box):
                    pred = predict(runner, data, to_device, a.model)
                reprs.save(a.repr_root, a.model, a.seed, sid, box["z"].squeeze(0) if box["z"].dim() == 4 else box["z"],
                           box["s"].squeeze(0) if box["s"].dim() == 3 else box["s"], recycling=a.recycling)
            else:
                s, z = soup_inputs(a, sid, runner.device)
                if a.model == "opendde":           # OpenDDE consumes the trunk output in bf16
                    s, z = s.to(torch.bfloat16), z.to(torch.bfloat16)
                H.drop_template_features(data["input_feature_dict"])
                with torch.no_grad(), H.trunk_bypass(model, s, z), H.seed_diffusion(model, a.seed):
                    pred = predict(runner, data, to_device, a.model)
            name = {"dataset_name" if a.model == "protenix" else "group_name": ""}
            runner.dumper.dump(**name, pdb_id=sid, seed=a.seed, pred_dict=pred, atom_array=atom_array,
                               entity_poly_type={k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"})
            conf = flatten(a.out, sid, a.seed)
            print(f"{sid} [{a.model} {a.mode} seed {a.seed}] ranking_score {np.round(conf, 4).tolist()}", flush=True)
            pred = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
