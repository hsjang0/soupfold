"""Write each model's token layout from one of its predicted structures (any sample, any seed).

  python scripts/token_layout.py --pred out/protenix/seed_1 --model protenix --out layouts
  -> layouts/protenix/<system>.json for every <system>__s0.cif in the directory
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from soupfold import tokens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True, help="directory with <system>__s0.cif")
ap.add_argument("--model", required=True, choices=["af3", "protenix", "opendde", "esmfold2"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, a.model), exist_ok=True)
for cif in sorted(glob.glob(os.path.join(a.pred, "*__s0.cif"))):
    sid = os.path.basename(cif).split("__")[0]
    tokens.save_layout(cif, os.path.join(a.out, a.model, f"{sid}.json"))
    print(sid)
