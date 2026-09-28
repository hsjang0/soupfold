"""OpenDDE: native folding (saving the trunk representation) and SoupFold folding.

Same code path and flags as scripts/run_protenix.py (OpenDDE shares Protenix's trunk
interface), plus --checkpoint for the OpenDDE weights.

  python scripts/run_opendde.py --mode soupfold --checkpoint opendde.pt --input 8JT6.json --seed 1 \
         --teachers esmfold2,protenix,af3 --layouts layouts/ ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_protenix import main  # noqa: E402

if __name__ == "__main__":
    main("opendde")
