"""SoupFold in one run from one checkpoint: OpenDDE anchor, ESMFold2 and Protenix peers.

  python scripts/soup_opendde_pack.py --checkpoint soup_opendde_pack.pt --protenix-root <protenix_root> \
      --template-mmcif-dir examples/9mnb/mmcif --kalign <kalign> --input examples/9mnb/9mnb.json

Outputs: <workdir>/samples/soupfold/opendde/seed_<k>/<system>__s<i>.cif, <system>__conf.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from soupfold.pack import SoupOpenDDE  # noqa: E402


def seeds(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="from scripts/build_soup_opendde_pack.py")
    ap.add_argument("--input", nargs="+", required=True, help="Protenix-format system JSON(s)")
    ap.add_argument("--protenix-root", required=True, help="holds common/ (CCD data, release dates)")
    ap.add_argument("--template-mmcif-dir", required=True)
    ap.add_argument("--kalign", required=True)
    ap.add_argument("--workdir", default="soupfold_out")
    ap.add_argument("--seeds", default="1", help="e.g. 1 or 1-5")
    ap.add_argument("--chunk-thresholds", help='OpenDDE trunk chunk sizes by token count, e.g. \'{"1024":64,"2048":16}\'')
    a = ap.parse_args()
    # kalign (template featurisation) reads a non-terminal stdin and blocks forever on a pipe or socket
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    data = dict(protenix_root=a.protenix_root, template_mmcif_dir=os.path.abspath(a.template_mmcif_dir), kalign=a.kalign,
                workdir=os.path.abspath(a.workdir), chunk_thresholds=a.chunk_thresholds)
    pack = SoupOpenDDE.from_pack(a.checkpoint, data)
    for k in seeds(a.seeds):
        for jpath in a.input:
            pack.predict(jpath, seed=k)
    print(f"done. Samples in {data['workdir']}/samples/soupfold/opendde/", flush=True)


if __name__ == "__main__":
    main()
