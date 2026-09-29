"""Arguments shared by the per-model scripts."""
import argparse
import os

from .inputs import MODELS, out_dir

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")


def parser(model, soupfold=True):
    ap = argparse.ArgumentParser()
    if soupfold:
        ap.add_argument("--mode", choices=["standalone", "soupfold"], default="standalone")
    ap.add_argument("--input", nargs="+", required=True, help="Protenix-format system JSON(s)")
    ap.add_argument("--workdir", default=".", help="outputs go under samples/, logs/, reprs/, layouts/ here")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", action="store_true",
                    help="standalone: sample real structures into samples/standalone/ instead of logs/")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--sampling-steps", type=int, default=None)
    ap.add_argument("--recycling", type=int, default=10)
    ap.add_argument("--out", default=None, help="override the output directory")
    ap.add_argument("--repr-root", default=None, help="default <workdir>/reprs")
    ap.add_argument("--layouts", default=None, help="token layouts, default <workdir>/layouts")
    if soupfold:
        ap.add_argument("--peers", default=None, help="default: the other three models")
        ap.add_argument("--weights", default=WEIGHTS)
    return ap


def finish(a, model, steps):
    """Fill defaults. Standalone runs are for the representations: without --sample they draw one
    sample with 2 diffusion steps (noise) into logs/. SoupFold and --sample use 5 samples and the
    model's default steps."""
    a.model = model
    a.mode = getattr(a, "mode", "standalone")
    real = a.mode == "soupfold" or a.sample
    a.samples = a.samples or (5 if real else 1)
    a.sampling_steps = a.sampling_steps or (steps if real else 2)
    a.out = a.out or out_dir(a.workdir, model, a.mode, a.sample, a.seed)
    a.repr_root = a.repr_root or os.path.join(a.workdir, "reprs")
    a.layouts = a.layouts or os.path.join(a.workdir, "layouts")
    if a.mode == "soupfold":
        a.peers = a.peers.split(",") if a.peers else [m for m in MODELS if m != model]
    os.makedirs(a.out, exist_ok=True)
    return a
