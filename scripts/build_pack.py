"""Bundle OpenDDE, Protenix, ESMFold2 (with ESMC-6B) and the two maps into one checkpoint. Users do not need this: the
checkpoint is on Hugging Face.

  python scripts/build_pack.py --protenix <protenix_base_default_v1.0.0.pt> --opendde <opendde.pt> \
      --esmfold2 <esmfold2 dir> --esmc <esmc-6b dir> --protenix-root <protenix_root> --out soup_opendde_pack.pt
"""
import argparse
import os
import tempfile

from soupfold.pack import build_from_sources


def main():
    ap = argparse.ArgumentParser()
    for k in ("protenix", "opendde", "esmfold2", "esmc", "protenix-root", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as wd:
        data = dict(protenix_root=a.protenix_root, template_mmcif_dir=wd, workdir=wd)
        pack = build_from_sources(a.protenix, a.opendde, a.esmfold2, a.esmc, a.weights, data)
        pack.save_pack(a.out)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
