"""SoupFold in one run from one checkpoint: OpenDDE anchor, ESMFold2 and Protenix peers.

  soupfold-pack --input examples/9mnb/9mnb.json

The checkpoint and the CCD data come from Hugging Face on the first run.
Outputs: <workdir>/samples/soupfold/opendde/seed_<k>/<system>__s<i>.cif, <system>__conf.json.
"""
import argparse
import os

CHECKPOINT = ("hyosoon0/soup-opendde-pack", "soup_opendde_pack.pt")
COMMON = ("aurekaresearch/OpenDDE", "eddd563ce96571f784012edd8f045181c8f8627d")  # common/: CCD data, release dates


def seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description="Fold complexes with SoupFold (OpenDDE + ESMFold2 + Protenix).")
    ap.add_argument("--input", nargs="+", required=True, help="Protenix-format system JSON(s)")
    ap.add_argument("--workdir", default="soupfold_out")
    ap.add_argument("--seeds", default="1", help="e.g. 1 or 1-5")
    ap.add_argument("--checkpoint", help=f"default: {CHECKPOINT[0]} on Hugging Face")
    ap.add_argument("--protenix-root", help=f"holds common/ (CCD data, release dates); default: {COMMON[0]} on Hugging Face")
    ap.add_argument("--template-mmcif-dir", help="default: mmcif/ next to the first input, if there is one")
    ap.add_argument("--chunk-thresholds", help='OpenDDE trunk chunk sizes by token count, e.g. \'{"1024":64,"2048":16}\'')
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("soupfold-pack needs an NVIDIA GPU: torch sees none.")
    from huggingface_hub import hf_hub_download, snapshot_download
    workdir = os.path.abspath(a.workdir)
    checkpoint = a.checkpoint or hf_hub_download(*CHECKPOINT)
    protenix_root = a.protenix_root or snapshot_download(COMMON[0], revision=COMMON[1], allow_patterns="common/*")
    mmcif = a.template_mmcif_dir or os.path.join(os.path.dirname(os.path.abspath(a.input[0])), "mmcif")
    if not os.path.isdir(mmcif):                         # OpenDDE asserts the directory exists, used or not
        mmcif = os.path.join(workdir, "template_mmcif")
        os.makedirs(mmcif, exist_ok=True)

    from soupfold.pack import SoupOpenDDE
    data = dict(protenix_root=protenix_root, template_mmcif_dir=os.path.abspath(mmcif), workdir=workdir,
                chunk_thresholds=a.chunk_thresholds)
    pack = SoupOpenDDE.from_pack(checkpoint, data)
    for k in seeds(a.seeds):
        for jpath in a.input:
            pack.predict(jpath, seed=k)
    print(f"done. Samples in {workdir}/samples/soupfold/opendde/", flush=True)


if __name__ == "__main__":
    main()
