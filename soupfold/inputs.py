"""Input JSON (Protenix format) with MSA / template paths relative to the JSON file."""
import json
import os

PATH_KEYS = ("unpairedMsaPath", "pairedMsaPath", "templatesPath")
MODELS = ("af3", "protenix", "opendde", "esmfold2")


def load(json_path):
    """The single record of a Protenix-format input, with every path made absolute."""
    rec = json.load(open(json_path))
    rec = rec[0] if isinstance(rec, list) else rec
    base = os.path.dirname(os.path.abspath(json_path))
    for ent in rec["sequences"]:
        for v in ent.values():
            for k in PATH_KEYS:
                if v.get(k) and not os.path.isabs(v[k]):
                    v[k] = os.path.join(base, v[k])
    return rec


def resolved(json_path, workdir):
    """Write the input with absolute paths (for readers that take a file path) and return that path."""
    rec = load(json_path)
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, f"{rec['name']}.input.json")
    json.dump([rec], open(out, "w"), indent=1)
    return out


def out_dir(workdir, model, mode, sample, seed):
    """samples/soupfold/<model> for SoupFold, samples/standalone/<model> for sampled native runs,
    logs/<model> for native runs that only store representations."""
    if mode == "soupfold":
        sub = os.path.join("samples", "soupfold", model)
    else:
        sub = os.path.join("samples", "standalone", model) if sample else os.path.join("logs", model)
    return os.path.join(workdir, sub, f"seed_{seed}")
