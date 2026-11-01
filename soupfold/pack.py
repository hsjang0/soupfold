"""SoupFold OpenDDE pack: one model, one checkpoint, one environment.

`SoupOpenDDE` holds OpenDDE (anchor), ESMFold2 and Protenix (peers) and the two peer -> anchor maps as submodules of a
single nn.Module, saved as a single checkpoint. For each system the peers run their trunks, OpenDDE runs its trunk, its z is
mixed with the peers' z in memory as the trunk hands it over (mix.py, equal weights), and OpenDDE goes on to diffusion and
confidence in the same forward pass. No representation is written anywhere.

A model waits in CPU memory while another one runs, so the trunks do not compete for the GPU.
"""
import argparse
import copy
import gc
import json
import os
import re
import tempfile

import numpy as np
import torch
from rdkit import RDLogger
from torch import nn

from . import inputs, maps as M, mix, runners as R, tokens
from .hooks import esmfold2 as HE, opendde as HO
from .models.esm.models.esmfold2.config import EsmFold2Config
from .models.esm.models.esmfold2.model import EsmFold2Model
from .models.esm.models.esmfold2.processor import ESMFold2InputBuilder
from .models.esm.models.hub import init_empty_weights

PEERS = ("protenix", "esmfold2")

os.environ.setdefault("LAYERNORM_TYPE", "torch")
RDLogger.DisableLog("rdApp.*")                          # one version warning for each molecule in the bundled CCD


def _empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _build_runner(model, data, checkpoint=None):
    """Upstream's runner with its checkpoint step skipped: the weights come from the pack. The runner brings the
    featuriser, the configs and the structure writer; its model becomes a submodule of the pack."""
    if model == "protenix":
        from .models.protenix_runner import batch_inference as B, inference as I
    else:
        from .models.opendde_runner import batch_inference as B, inference as I
    for mod, fn in ((B, "download_inference_cache"), (I, "_download_inference_assets")):
        if hasattr(mod, fn):                            # would look for the checkpoint the pack already holds
            setattr(mod, fn, lambda configs: None)
    a = argparse.Namespace(model=model, seed=1, recycling=10, sampling_steps=200, samples=5,
                           template_mmcif_dir=data["template_mmcif_dir"],
                           release_dates=os.path.join(data["protenix_root"], "common", "release_date_cache.json"),
                           obsolete_pdbs=os.path.join(data["protenix_root"], "common", "obsolete_to_successor.json"),
                           checkpoint=checkpoint or "pack", chunk_thresholds=data.get("chunk_thresholds"),
                           out=data["workdir"])
    load = I.InferenceRunner.load_checkpoint
    I.InferenceRunner.load_checkpoint = lambda self: self.model.eval()
    try:
        return R.build_runner(a)
    finally:
        I.InferenceRunner.load_checkpoint = load


class SoupOpenDDE(nn.Module):
    """data: protenix_root, template_mmcif_dir, workdir, and optionally chunk_thresholds (see README)."""

    def __init__(self, esm_config, stats, map_shapes, ccd_pkl, data):
        super().__init__()
        os.environ.setdefault("PROTENIX_ROOT_DIR", data["protenix_root"])
        self.data, self.stats = data, stats
        torch.set_float32_matmul_precision("high")
        os.makedirs(data["workdir"], exist_ok=True)
        self._ccd = tempfile.TemporaryDirectory(prefix="soup_opendde_ccd_")
        with open(os.path.join(self._ccd.name, "ccd.pkl"), "wb") as f:
            f.write(ccd_pkl)
        self.builder = ESMFold2InputBuilder(ccd_cache=self._ccd.name)
        self.runners = {}
        self.runners["protenix"] = _build_runner("protenix", data)
        self.protenix = self.runners["protenix"][0].model
        self.runners["opendde"] = _build_runner("opendde", data)
        self.opendde = self.runners["opendde"][0].model
        with init_empty_weights():
            self.esmfold2 = EsmFold2Model(EsmFold2Config(**esm_config))
        self.maps = nn.ModuleDict({p: M.PairMap(*map_shapes[p]) for p in PEERS})
        self.esm_config, self.map_shapes, self.ccd_pkl = esm_config, map_shapes, ccd_pkl
        self.base_cfg = {k: copy.deepcopy(r[0].configs) for k, r in self.runners.items()}
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active = None

    def save_pack(self, path):
        tmp = path + ".tmp"
        torch.save({"state_dict": self.state_dict(), "esm_config": self.esm_config, "stats": self.stats,
                    "map_shapes": self.map_shapes, "ccd_pkl": self.ccd_pkl}, tmp)
        os.replace(tmp, path)

    @classmethod
    def from_pack(cls, path, data):
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        pack = cls(ck["esm_config"], ck["stats"], ck["map_shapes"], ck["ccd_pkl"], data)
        pack.load_weights(ck["state_dict"])
        return pack

    def load_weights(self, state_dict):
        missing = self.load_state_dict(state_dict, strict=False, assign=True)
        ignore = EsmFold2Model._keys_to_ignore_on_load_missing[0].replace("^", "^esmfold2\\.")
        left = [k for k in missing.missing_keys if not re.search(ignore, k)]
        if left or missing.unexpected_keys:
            raise RuntimeError(f"pack checkpoint does not match: missing {left[:8]}, unexpected {missing.unexpected_keys[:8]}")
        self.esmfold2._materialize_uninitialized(device="cpu")
        self.eval()
        self.to("cpu")
        self.maps.to(self.device_)
        self.active = None

    def _use(self, name):
        """Bring one of the three models to the GPU, sending the previous one to CPU memory."""
        if self.active == name:
            return
        if self.active:
            getattr(self, self.active).to("cpu")
            _empty_cache()
        getattr(self, name).to(self.device_)
        self.active = name

    @torch.no_grad()
    def _esmfold2_trunk(self, record, seed, recycling, lm_dropout=0.3, msa_column_mask_rate=0.1):
        self._use("esmfold2")
        model = self.esmfold2
        with tempfile.TemporaryDirectory() as wd:
            feats, chains = self.builder.prepare_input(R.to_esm_input(record, wd), seed=0, device=self.device_)
        box = {}
        R.set_lm_dropout(model, lm_dropout)
        model.set_chunk_size(64)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with HE.capture_trunk(model, box, stop=True):
            try:
                model(**feats, num_loops=recycling, num_sampling_steps=2, num_diffusion_samples=1,
                      msa_column_mask_rate=msa_column_mask_rate)
            except HE.StopAfterTrunk:
                pass
        R.set_lm_dropout(model, 0.0)
        z = box["z"].float()
        z = z[0] if z.dim() == 4 else z
        return z.half().cpu(), HE.token_layout(feats, chains)

    def _batch(self, name, jpath, workdir):
        """(data, atom_array) of one system from the model's own featuriser."""
        runner, _, _, get_loader, _ = self.runners[name]
        runner.configs.input_json_path = inputs.resolved(jpath, os.path.join(workdir, "logs", "inputs"))
        batches = list(get_loader(configs=runner.configs))
        if len(batches) != 1:
            raise ValueError(f"{jpath}: expected one system, got {len(batches)}")
        data, atom_array, err = batches[0][0]
        if err:
            raise RuntimeError(f"{jpath}: {err}")
        return data, atom_array

    @torch.no_grad()
    def _protenix_trunk(self, jpath, seed, workdir):
        self._use("protenix")
        runner, seed_everything, to_device, _, update_cfg = self.runners["protenix"]
        data, atom_array = self._batch("protenix", jpath, workdir)
        runner.update_model_configs(update_cfg(self.base_cfg["protenix"], data["N_token"].item()))
        seed_everything(seed=seed, deterministic=False)
        box = {}
        with HO.capture_trunk(self.protenix, box, stop=True):
            try:
                R.predict(runner, data, to_device, "protenix")
            except HO.StopAfterTrunk:
                pass
        z = box["z"].squeeze(0) if box["z"].dim() == 4 else box["z"]
        return z.float().half().cpu(), HO.token_layout(atom_array)

    @torch.no_grad()
    def predict(self, jpath, seed=1, out=None, workdir=None, recycling=10, return_z=False):
        """Fold one Protenix-format JSON. Writes <out>/<system>__s<i>.cif and <system>__conf.json, returns the sample
        confidences (with return_z: and the mixed z and the peers' z, for tests)."""
        workdir = workdir or self.data["workdir"]
        out = out or os.path.join(workdir, "samples", "soupfold", "opendde", f"seed_{seed}")
        os.makedirs(out, exist_ok=True)
        record = inputs.load(jpath)
        sid = record["name"]
        peers, layouts = {}, {}
        peers["esmfold2"], layouts["esmfold2"] = self._esmfold2_trunk(record, seed, recycling)
        peers["protenix"], layouts["protenix"] = self._protenix_trunk(jpath, seed, workdir)
        kept = {}
        self._use("opendde")
        runner, seed_everything, to_device, _, update_cfg = self.runners["opendde"]
        data, atom_array = self._batch("opendde", jpath, workdir)
        runner.update_model_configs(update_cfg(self.base_cfg["opendde"], data["N_token"].item()))
        runner.configs.dump_dir = runner.dumper.base_dir = out
        layout = HO.token_layout(atom_array)
        smiles = tokens.smiles_chains(record)

        def soup(z):
            with torch.autocast(device_type="cuda", enabled=False):
                z0 = z[0] if z.dim() == 4 else z
                tmaps = {p: tokens.align(layout, layouts[p], smiles, tokens.absent_chains(record, p)) for p in peers}
                zp = mix.soup(z0, peers, self.maps, self.stats, "opendde", token_maps=tmaps, device=self.device_)
            if return_z:
                kept["z_mixed"] = zp.half().cpu()
            return (zp[None] if z.dim() == 4 else zp).to(z.dtype)

        seed_everything(seed=seed, deterministic=False)
        with HO.soup_trunk(self.opendde, soup), HO.seed_diffusion(self.opendde, seed):
            pred = R.predict(runner, data, to_device, "opendde")
        runner.dumper.dump(group_name="", pdb_id=sid, seed=seed, pred_dict=pred, atom_array=atom_array,
                           entity_poly_type={k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"})
        conf = R.flatten(out, sid, seed)
        print(f"{sid} [SoupOpenDDE seed {seed}] ranking_score {np.round(conf, 4).tolist()} -> {out}", flush=True)
        pred = None
        _empty_cache()
        return (conf, kept, peers) if return_z else conf


def build_from_sources(protenix_pt, opendde_pt, esmfold2_dir, esmc_dir, weights_dir, data):
    """A SoupOpenDDE from the four upstream checkpoints and weights/ (only for scripts/build_soup_opendde_pack.py)."""
    cfg = EsmFold2Config.from_pretrained(esmfold2_dir)
    cfg.esmc_id = esmc_dir
    cfg.lm_encoder.lm_dropout = cfg.lm_dropout = 0.0
    esm_model = EsmFold2Model.from_pretrained(esmfold2_dir, load_esmc=True, config=cfg, device="cpu", dtype=torch.float32)
    esmc_config = esm_model.esmc.config.to_dict()
    esm_config = cfg.to_dict()
    esm_config["esmc_config"], esm_config["esmc_id"] = esmc_config, "biohub/ESMC-6B"
    stats = json.load(open(os.path.join(weights_dir, "chan_stats.json")))
    stats = {m: stats[m] for m in ("opendde",) + PEERS}
    maps, shapes = {}, {}
    for p in PEERS:
        maps[p] = M.load_map(weights_dir, p, "opendde", "cpu")
        ck = torch.load(os.path.join(weights_dir, "maps", f"{p}_to_opendde.pt"), map_location="cpu", weights_only=True)
        shapes[p] = (ck["c_in"], ck["c_out"], ck["hidden"])
    ccd = open(os.path.join(esmfold2_dir, "ccd.pkl"), "rb").read()
    pack = SoupOpenDDE(esm_config, stats, shapes, ccd, data)
    sd = {}
    for name, path in (("protenix", protenix_pt), ("opendde", opendde_pt)):
        ck = torch.load(path, map_location="cpu", weights_only=False)["model"]
        sd.update({f"{name}.{k.removeprefix('module.')}": v for k, v in ck.items()})
    sd.update({f"esmfold2.{k}": v for k, v in esm_model.state_dict().items()})
    for p, net in maps.items():
        sd.update({f"maps.{p}.{k}": v for k, v in net.state_dict().items()})
    pack.load_weights(sd)
    return pack
