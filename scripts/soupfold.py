"""SoupFold pipeline: standalone runs of all four models, token layouts, then SoupFold for each anchor.

  python scripts/soupfold.py --config config.json --input examples/9mnb/9mnb.json
  python scripts/soupfold.py --config config.json --input examples/9mnb/9mnb.json --seeds 1-5 --sample

Each model runs in its own environment (see config.example.json). Outputs under --workdir:
  reprs/<model>/seed_<k>/             trunk representations
  logs/<model>/seed_<k>/              standalone runs without --sample (1 sample, 2 steps)
  samples/standalone/<model>/seed_<k> standalone runs with --sample (5 samples, default steps)
  layouts/<model>/                    token layouts
  samples/soupfold/<anchor>/seed_<k>/ SoupFold results (5 samples, default steps)
  logs/run/                           stdout of every step
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from soupfold.inputs import MODELS, out_dir  # noqa: E402

SCRIPT = {"af3": "run_af3.py", "protenix": "run_protenix.py", "opendde": "run_opendde.py",
          "esmfold2": "run_esmfold2.py"}


def seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_config(path):
    """Relative paths in the config are taken relative to the config file."""
    cfg, base = json.load(open(path)), os.path.dirname(os.path.abspath(path))
    fix = lambda v: os.path.join(base, v) if isinstance(v, str) and v.startswith(".") else v
    for k, v in cfg.items():
        if isinstance(v, dict):
            v["python"] = fix(v["python"])
            v["args"] = [fix(x) for x in v.get("args", [])]
            v["pythonpath"] = [fix(x) for x in v.get("pythonpath", [])]
        else:
            cfg[k] = fix(v)
    return cfg


def run(cfg, model, args, log):
    c = cfg[model]
    env = dict(os.environ, **c.get("env", {}))
    if c.get("pythonpath"):
        env["PYTHONPATH"] = os.pathsep.join(c["pythonpath"] + [env.get("PYTHONPATH", "")])
    cmd = [c["python"], os.path.join(HERE, args[0])] + args[1:] + c.get("args", [])
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f"[soupfold] {model}: {' '.join(args[1:4])} ... > {log}", flush=True)
    with open(log, "w") as fh:
        if subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode:
            sys.exit(f"[soupfold] {model} failed, see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", nargs="+", required=True, help="Protenix-format system JSON(s)")
    ap.add_argument("--workdir", default="soupfold_out")
    ap.add_argument("--seeds", default="1", help="e.g. 1 or 1-5")
    ap.add_argument("--anchors", default="protenix,opendde,esmfold2")
    ap.add_argument("--sample", action="store_true", help="standalone runs also sample real structures")
    a = ap.parse_args()
    cfg = load_config(a.config)
    inp = [os.path.abspath(x) for x in a.input]
    W = os.path.abspath(a.workdir)
    data = ["--template-mmcif-dir", cfg["template_mmcif_dir"]]
    pdata = data + ["--release-dates", cfg["release_dates"], "--obsolete-pdbs", cfg["obsolete_pdbs"],
                    "--kalign", cfg["kalign"]]
    extra = {"af3": data, "protenix": pdata, "opendde": pdata, "esmfold2": []}
    common = ["--input", *inp, "--workdir", W] + (["--sample"] if a.sample else [])
    L = os.path.join(W, "logs", "run")
    ks = seeds(a.seeds)

    for k in ks:                                  # 1. standalone runs (representations)
        for m in MODELS:
            mode = [] if m == "af3" else ["--mode", "standalone"]
            run(cfg, m, [SCRIPT[m], *mode, "--seed", str(k), *common, *extra[m]], f"{L}/standalone_{m}_seed{k}.log")

    for m in MODELS:                              # 2. token layouts
        run(cfg, "protenix", ["token_layout.py", "--pred", out_dir(W, m, "standalone", a.sample, ks[0]), "--model", m,
                              "--out", os.path.join(W, "layouts")], f"{L}/layout_{m}.log")

    for k in ks:                                  # 3. SoupFold
        for b in a.anchors.split(","):
            run(cfg, b, [SCRIPT[b], "--mode", "soupfold", "--seed", str(k), "--input", *inp, "--workdir", W,
                         *extra[b]], f"{L}/soupfold_{b}_seed{k}.log")
    print(f"[soupfold] done. SoupFold samples in {W}/samples/soupfold/", flush=True)


if __name__ == "__main__":
    main()
